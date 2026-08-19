

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reliability_metrics import calculate_selective_metrics


DAY5_RELIABILITY_RUN_ID = "20260814T141442_497529-0700"
DAY5_RELIABILITY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "reliability"
    / DAY5_RELIABILITY_RUN_ID
    / "val_reliability_signals.csv"
).resolve()
EXPECTED_DAY5_SHA256 = (
    "a767b163b3565a9fe664068c5fd18e96e4f13bf4dea7040ee179f098b477edb8"
)

PREDICTION_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "predictions" / "selective_routing"
)
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "selective_routing"
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"

EXPECTED_EYES = 120
EXPECTED_SUBJECTS = 95
EXPECTED_NORMAL = 32
EXPECTED_ABNORMAL = 88
EXPECTED_CORRECT = 110
EXPECTED_INCORRECT = 10
EXPECTED_AGREEMENT = 112
TARGET_COVERAGE_PERCENTAGES = (50, 60, 70, 80, 90)
ALLOWED_LABELS = {"Normal", "Abnormal"}
FORBIDDEN_TEST_NAMES = {
    "test_images.csv",
    "test_eye_predictions.csv",
    "test_image_predictions.csv",
    "test_reliability_signals.csv",
}
LOADED_PATHS: set[Path] = set()

CONFIDENCE_FIELD = "PrimaryConfidenceMargin"
AGREEMENT_FIELD = "ModelAgreement"
VIEW_SD_FIELD = "Day3StdViewProbability"
FEATURE_PERCENTILE_FIELD = "PredictedClassReferencePercentile"

# only the signals below are allowed in the routing search
# view probability range stays out because it was nearly redundant with view sd
ROUTING_SIGNAL_FIELDS = {
    CONFIDENCE_FIELD,
    AGREEMENT_FIELD,
    VIEW_SD_FIELD,
    FEATURE_PERCENTILE_FIELD,
}

REQUIRED_COLUMNS = {
    "EyeExamID",
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "TrueLabel",
    "Day4EyeProbability",
    "Day4PredictedLabel",
    "Day4Correct",
    CONFIDENCE_FIELD,
    "NumberOfViews",
    "Day3PredictedLabel",
    AGREEMENT_FIELD,
    VIEW_SD_FIELD,
    FEATURE_PERCENTILE_FIELD,
}


class SafetyError(RuntimeError):
    """used when a locked input, validation, or test-protection check fails."""


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    require_agreement: bool
    use_view_sd: bool
    use_feature: bool
    complexity: int
    requested: bool = True

    @property
    def rule(self) -> str:
        clauses = [f"{CONFIDENCE_FIELD} >= ConfidenceThreshold"]
        if self.require_agreement:
            clauses.append(f"{AGREEMENT_FIELD} == True")
        if self.use_view_sd:
            clauses.append(f"{VIEW_SD_FIELD} <= ViewSDThreshold")
        if self.use_feature:
            clauses.append(
                f"{FEATURE_PERCENTILE_FIELD} <= FeaturePercentileThreshold"
            )
        return " AND ".join(clauses)


STRATEGIES = (
    StrategySpec(
        "confidence_only",
        "Confidence only",
        require_agreement=False,
        use_view_sd=False,
        use_feature=False,
        complexity=1,
    ),
    StrategySpec(
        "confidence_model_agreement",
        "Confidence + model agreement",
        require_agreement=True,
        use_view_sd=False,
        use_feature=False,
        complexity=2,
    ),
    StrategySpec(
        "confidence_view_sd",
        "Confidence + view SD",
        require_agreement=False,
        use_view_sd=True,
        use_feature=False,
        complexity=2,
    ),
    StrategySpec(
        "full_reliability",
        "Confidence + agreement + view SD + feature typicality",
        require_agreement=True,
        use_view_sd=True,
        use_feature=True,
        complexity=4,
    ),
)

# this is an ablation of the full rule, not a fifth candidate strategy
# it is only used to check whether feature typicality adds anything measurable
FEATURE_ABLATION = StrategySpec(
    "confidence_agreement_view_sd_ablation",
    "Confidence + agreement + view SD (feature ablation)",
    require_agreement=True,
    use_view_sd=True,
    use_feature=False,
    complexity=3,
    requested=False,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_test_path(path: Path) -> None:
    lowered_parts = [part.casefold() for part in path.parts]
    if (
        path.name.casefold() in FORBIDDEN_TEST_NAMES
        or path.name.casefold().startswith("test_")
        or any(part in {"test", "tests"} for part in lowered_parts)
    ):
        raise SafetyError(f"Test artifact access was attempted: {path}")


def read_locked_reliability_table(path: Path) -> pd.DataFrame:
    # only the hash-pinned day-5 validation table can be used here
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved != DAY5_RELIABILITY_PATH:
        raise SafetyError(f"Input is outside the locked Day-5 allowlist: {resolved}")
    if not resolved.is_file():
        raise SafetyError(f"Locked Day-5 reliability table is missing: {resolved}")
    observed_hash = file_sha256(resolved)
    if observed_hash != EXPECTED_DAY5_SHA256:
        raise SafetyError(
            "Locked Day-5 reliability table SHA-256 mismatch: "
            f"observed={observed_hash}, expected={EXPECTED_DAY5_SHA256}."
        )
    LOADED_PATHS.add(resolved)
    return pd.read_csv(
        resolved,
        dtype={
            "EyeExamID": "string",
            "ResearchSubjectID": "string",
            "EncounterID": "string",
            "Laterality": "string",
            "TrueLabel": "string",
            "Day4PredictedLabel": "string",
            "Day3PredictedLabel": "string",
        },
        float_precision="round_trip",
    )


def parse_bool_column(series: pd.Series, name: str) -> pd.Series:
    # reject anything other than explicit true or false values
    if series.dtype.kind == "b":
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = normalized[~normalized.isin(["true", "false"])]
    if not invalid.empty:
        values = sorted(invalid.dropna().unique().tolist())
        raise SafetyError(f"{name} contains non-boolean values: {values}")
    return normalized.eq("true")


def record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: str = "",
) -> None:
    checks.append(
        {
            "Check": name,
            "Status": "PASS" if passed else "FAIL",
            "Observed": observed,
            "Expected": expected,
            "Details": details,
        }
    )
    if not passed:
        raise SafetyError(f"{name}: observed={observed}; expected={expected}. {details}")


def bools_to_bitmask(values: np.ndarray) -> int:
    # bitmasks make repeated acceptance-set comparisons much faster
    mask = 0
    for index in np.flatnonzero(values):
        mask |= 1 << int(index)
    return mask


def bitmask_to_bools(mask: int, length: int) -> np.ndarray:
    return np.fromiter(
        ((mask >> index) & 1 == 1 for index in range(length)),
        dtype=bool,
        count=length,
    )


def make_threshold_masks(
    values: np.ndarray,
    comparator: str,
    include_reject_all: bool,
) -> list[tuple[float, int]]:
    # every observed value becomes a possible cutoff
    unique = np.unique(values.astype(float))
    candidates: list[float]
    if comparator == "ge":
        candidates = unique.tolist()
        if include_reject_all:
            candidates.append(math.inf)
        return [
            (float(cutoff), bools_to_bitmask(values >= cutoff))
            for cutoff in candidates
        ]
    if comparator == "le":
        candidates = unique.tolist()
        if include_reject_all:
            candidates.insert(0, -math.inf)
        return [
            (float(cutoff), bools_to_bitmask(values <= cutoff))
            for cutoff in candidates
        ]
    raise ValueError(f"Unknown comparator: {comparator}")


def threshold_sort_key(
    spec: StrategySpec,
    confidence_threshold: float,
    view_sd_threshold: float | None,
    feature_threshold: float | None,
) -> tuple[float, ...]:
    """prefer less restrictive extra gates when two rules behave the same."""
    key: list[float] = []
    if spec.use_feature:
        assert feature_threshold is not None
        key.append(-feature_threshold)
    if spec.use_view_sd:
        assert view_sd_threshold is not None
        key.append(-view_sd_threshold)
    key.append(confidence_threshold)
    return tuple(key)


def enumerate_rules(
    spec: StrategySpec,
    confidence_masks: list[tuple[float, int]],
    view_sd_masks: list[tuple[float, int]],
    feature_masks: list[tuple[float, int]],
    agreement_mask: int,
    all_mask: int,
) -> tuple[dict[int, dict[str, Any]], int]:
    """keep one representative threshold set for each unique accepted-eye set."""
    # different threshold combinations can produce the same accepted-eye set
    unique_rules: dict[int, dict[str, Any]] = {}
    raw_configuration_count = 0
    sd_iterable = view_sd_masks if spec.use_view_sd else [(None, all_mask)]
    feature_iterable = feature_masks if spec.use_feature else [(None, all_mask)]
    fixed_gate = agreement_mask if spec.require_agreement else all_mask

    for confidence_threshold, confidence_mask in confidence_masks:
        confidence_gate = confidence_mask & fixed_gate
        for view_sd_threshold, view_sd_mask in sd_iterable:
            partial_mask = confidence_gate & view_sd_mask
            for feature_threshold, feature_mask in feature_iterable:
                raw_configuration_count += 1
                accepted_mask = partial_mask & feature_mask
                candidate = {
                    "AcceptedBitmask": accepted_mask,
                    "ConfidenceThreshold": confidence_threshold,
                    "ViewSDThreshold": view_sd_threshold,
                    "FeaturePercentileThreshold": feature_threshold,
                }
                existing = unique_rules.get(accepted_mask)
                candidate_key = threshold_sort_key(
                    spec,
                    confidence_threshold,
                    view_sd_threshold,
                    feature_threshold,
                )
                if existing is None:
                    unique_rules[accepted_mask] = candidate
                    continue
                existing_key = threshold_sort_key(
                    spec,
                    float(existing["ConfidenceThreshold"]),
                    existing["ViewSDThreshold"],
                    existing["FeaturePercentileThreshold"],
                )
                if candidate_key < existing_key:
                    unique_rules[accepted_mask] = candidate
    return unique_rules, raw_configuration_count


def metrics_from_mask(
    accepted_mask: int,
    true_abnormal_mask: int,
    true_normal_mask: int,
    predicted_abnormal_mask: int,
    predicted_normal_mask: int,
) -> dict[str, int | float]:
    accepted_count = accepted_mask.bit_count()
    tp = (accepted_mask & true_abnormal_mask & predicted_abnormal_mask).bit_count()
    tn = (accepted_mask & true_normal_mask & predicted_normal_mask).bit_count()
    fp = (accepted_mask & true_normal_mask & predicted_abnormal_mask).bit_count()
    fn = (accepted_mask & true_abnormal_mask & predicted_normal_mask).bit_count()
    errors = fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )
    return {
        "AcceptedCount": accepted_count,
        "AcceptedErrors": errors,
        "AcceptedFalseNegatives": fn,
        "AcceptedTruePositives": tp,
        "AcceptedTrueNegatives": tn,
        "AcceptedFalsePositives": fp,
        "AcceptedSensitivity": sensitivity,
        "AcceptedSpecificity": specificity,
        "AcceptedBalancedAccuracy": balanced_accuracy,
    }


def accepted_eye_ids(mask: int, eye_ids: list[str]) -> tuple[str, ...]:
    return tuple(eye_ids[i] for i in range(len(eye_ids)) if (mask >> i) & 1)


def make_risk_envelope(
    spec: StrategySpec,
    frame: pd.DataFrame,
    confidence_masks: list[tuple[float, int]],
    view_sd_masks: list[tuple[float, int]],
    feature_masks: list[tuple[float, int]],
    agreement_mask: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    # for each accepted-eye count, keep the best rule observed on validation
    n = len(frame)
    all_mask = (1 << n) - 1
    true_abnormal_mask = bools_to_bitmask(frame["TrueLabel"].eq("Abnormal").to_numpy())
    true_normal_mask = all_mask ^ true_abnormal_mask
    predicted_abnormal_mask = bools_to_bitmask(
        frame["Day4PredictedLabel"].eq("Abnormal").to_numpy()
    )
    predicted_normal_mask = all_mask ^ predicted_abnormal_mask
    eye_ids = frame["EyeExamID"].astype(str).tolist()

    unique_rules, raw_configuration_count = enumerate_rules(
        spec,
        confidence_masks,
        view_sd_masks,
        feature_masks,
        agreement_mask,
        all_mask,
    )

    best_by_count: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for accepted_mask, thresholds in unique_rules.items():
        counts = metrics_from_mask(
            accepted_mask,
            true_abnormal_mask,
            true_normal_mask,
            predicted_abnormal_mask,
            predicted_normal_mask,
        )
        balanced_accuracy = float(counts["AcceptedBalancedAccuracy"])
        balanced_objective = (
            -balanced_accuracy if math.isfinite(balanced_accuracy) else math.inf
        )
        preference = threshold_sort_key(
            spec,
            float(thresholds["ConfidenceThreshold"]),
            thresholds["ViewSDThreshold"],
            thresholds["FeaturePercentileThreshold"],
        )
        # errors come first, then false negatives, then balanced accuracy
        objective: tuple[Any, ...] = (
            int(counts["AcceptedErrors"]),
            int(counts["AcceptedFalseNegatives"]),
            balanced_objective,
            *preference,
            accepted_eye_ids(accepted_mask, eye_ids),
        )
        accepted_count = int(counts["AcceptedCount"])
        existing = best_by_count.get(accepted_count)
        if existing is None or objective < existing[0]:
            best_by_count[accepted_count] = (
                objective,
                {**thresholds, **counts},
            )

    # pull one nearby operating point for each requested coverage target
    rows: list[dict[str, Any]] = []
    max_view_sd = float(frame[VIEW_SD_FIELD].max())
    max_feature = float(frame[FEATURE_PERCENTILE_FIELD].max())
    for accepted_count in sorted(best_by_count):
        result = best_by_count[accepted_count][1]
        confidence_threshold = float(result["ConfidenceThreshold"])
        view_sd_threshold = result["ViewSDThreshold"]
        feature_threshold = result["FeaturePercentileThreshold"]
        errors = int(result["AcceptedErrors"])
        row = {
            "StrategyID": spec.strategy_id,
            "Strategy": spec.display_name,
            "StrategyComplexity": spec.complexity,
            "RequestedStrategy": spec.requested,
            "AcceptanceRule": spec.rule,
            "RequireModelAgreement": spec.require_agreement,
            "UseViewSD": spec.use_view_sd,
            "UseFeatureTypicality": spec.use_feature,
            "ConfidenceThreshold": confidence_threshold,
            "ViewSDThreshold": view_sd_threshold,
            "FeaturePercentileThreshold": feature_threshold,
            "ViewSDGateActive": bool(
                spec.use_view_sd
                and view_sd_threshold is not None
                and float(view_sd_threshold) < max_view_sd
            ),
            "FeatureGateActive": bool(
                spec.use_feature
                and feature_threshold is not None
                and float(feature_threshold) < max_feature
            ),
            "TotalCount": n,
            "AcceptedCount": accepted_count,
            "DeferredCount": n - accepted_count,
            "Coverage": accepted_count / n,
            "CoveragePercent": 100.0 * accepted_count / n,
            "ReviewRate": (n - accepted_count) / n,
            "ReviewRatePercent": 100.0 * (n - accepted_count) / n,
            "AcceptedErrors": errors,
            "AcceptedErrorRate": errors / accepted_count if accepted_count else math.nan,
            "AcceptedErrorPercent": (
                100.0 * errors / accepted_count if accepted_count else math.nan
            ),
            "AcceptedAccuracy": (
                1.0 - errors / accepted_count if accepted_count else math.nan
            ),
            "AcceptedSensitivity": result["AcceptedSensitivity"],
            "AcceptedSpecificity": result["AcceptedSpecificity"],
            "AcceptedBalancedAccuracy": result["AcceptedBalancedAccuracy"],
            "FalseNegativesAmongAccepted": int(result["AcceptedFalseNegatives"]),
            "AcceptedTruePositives": int(result["AcceptedTruePositives"]),
            "AcceptedTrueNegatives": int(result["AcceptedTrueNegatives"]),
            "AcceptedFalsePositives": int(result["AcceptedFalsePositives"]),
            "AcceptedFalseNegatives": int(result["AcceptedFalseNegatives"]),
            "AcceptedBitmask": int(result["AcceptedBitmask"]),
        }
        rows.append(row)
    envelope = pd.DataFrame(rows)
    enumeration = {
        "RawThresholdConfigurations": raw_configuration_count,
        "UniqueAcceptanceSets": len(unique_rules),
        "AttainableAcceptedCounts": len(best_by_count),
        "MaximumAcceptedCount": int(envelope["AcceptedCount"].max()),
    }
    return envelope, enumeration


def make_acceptance_mask(frame: pd.DataFrame, row: pd.Series) -> np.ndarray:
    accepted = frame[CONFIDENCE_FIELD].to_numpy(float) >= float(
        row["ConfidenceThreshold"]
    )
    if bool(row["RequireModelAgreement"]):
        accepted &= frame[AGREEMENT_FIELD].to_numpy(bool)
    if bool(row["UseViewSD"]):
        accepted &= frame[VIEW_SD_FIELD].to_numpy(float) <= float(
            row["ViewSDThreshold"]
        )
    if bool(row["UseFeatureTypicality"]):
        accepted &= frame[FEATURE_PERCENTILE_FIELD].to_numpy(float) <= float(
            row["FeaturePercentileThreshold"]
        )
    return accepted


def pick_target_candidates(
    envelopes: dict[str, pd.DataFrame], specs: Iterable[StrategySpec]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_percent in TARGET_COVERAGE_PERCENTAGES:
        target_count = EXPECTED_EYES * target_percent / 100.0
        for spec in specs:
            envelope = envelopes[spec.strategy_id]
            candidates = envelope.assign(
                _distance=(envelope["AcceptedCount"] - target_count).abs(),
                _above=envelope["AcceptedCount"].gt(target_count),
            ).sort_values(
                ["_distance", "_above", "AcceptedCount"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            chosen = candidates.iloc[0].drop(labels=["_distance", "_above"]).to_dict()
            chosen["TargetCoveragePercent"] = target_percent
            chosen["TargetAcceptedCount"] = target_count
            chosen["CoverageDeviationPercentagePoints"] = abs(
                float(chosen["CoveragePercent"]) - target_percent
            )
            chosen["CandidateOperatingPointID"] = (
                f"{spec.strategy_id}__coverage_{target_percent:03d}"
            )
            rows.append(chosen)

    candidates = pd.DataFrame(rows)
    strategy_order = {spec.strategy_id: index for index, spec in enumerate(STRATEGIES)}
    candidates["LowestAcceptedErrorAtTarget"] = False
    candidates["FewestFalseNegativesAtTarget"] = False
    candidates["SimplestPreferredAtTarget"] = False
    candidates["LowestErrorStrategies"] = ""
    candidates["FewestFalseNegativeStrategies"] = ""
    candidates["PreferredStrategyAtTarget"] = ""
    for target_percent, group in candidates.groupby("TargetCoveragePercent", sort=True):
        min_errors = int(group["AcceptedErrors"].min())
        error_winners = group.loc[group["AcceptedErrors"].eq(min_errors)]
        min_fn = int(group["FalseNegativesAmongAccepted"].min())
        fn_winners = group.loc[group["FalseNegativesAmongAccepted"].eq(min_fn)]

        # among rules with the same best error behavior, prefer the simpler strategy
        # the fixed strategy order is only used as the final tie-break
        best_fn_within_error = int(error_winners["FalseNegativesAmongAccepted"].min())
        preferred_pool = error_winners.loc[
            error_winners["FalseNegativesAmongAccepted"].eq(best_fn_within_error)
        ].copy()
        preferred_pool["_order"] = preferred_pool["StrategyID"].map(strategy_order)
        preferred = preferred_pool.sort_values(
            ["StrategyComplexity", "_order"], kind="mergesort"
        ).iloc[0]

        index = group.index
        candidates.loc[index, "LowestAcceptedErrorAtTarget"] = group[
            "AcceptedErrors"
        ].eq(min_errors).to_numpy()
        candidates.loc[index, "FewestFalseNegativesAtTarget"] = group[
            "FalseNegativesAmongAccepted"
        ].eq(min_fn).to_numpy()
        candidates.loc[index, "SimplestPreferredAtTarget"] = group[
            "StrategyID"
        ].eq(preferred["StrategyID"]).to_numpy()
        candidates.loc[index, "LowestErrorStrategies"] = "; ".join(
            error_winners.sort_values("StrategyComplexity")["Strategy"].tolist()
        )
        candidates.loc[index, "FewestFalseNegativeStrategies"] = "; ".join(
            fn_winners.sort_values("StrategyComplexity")["Strategy"].tolist()
        )
        candidates.loc[index, "PreferredStrategyAtTarget"] = preferred["Strategy"]
    return candidates.sort_values(
        ["TargetCoveragePercent", "StrategyComplexity", "StrategyID"],
        kind="mergesort",
    ).reset_index(drop=True)


def make_strategy_comparison(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in candidates.groupby("TargetCoveragePercent", sort=True):
        row: dict[str, Any] = {
            "TargetCoveragePercent": target,
            "TargetAcceptedCount": int(group["TargetAcceptedCount"].iloc[0]),
            "LowestAcceptedErrorStrategies": group["LowestErrorStrategies"].iloc[0],
            "FewestAcceptedFalseNegativeStrategies": group[
                "FewestFalseNegativeStrategies"
            ].iloc[0],
            "SimplestPreferredStrategy": group["PreferredStrategyAtTarget"].iloc[0],
            "OperatingPointFinalized": False,
        }
        for _, candidate in group.iterrows():
            prefix = str(candidate["StrategyID"])
            row[f"{prefix}__ActualCoveragePercent"] = candidate["CoveragePercent"]
            row[f"{prefix}__ReviewRatePercent"] = candidate["ReviewRatePercent"]
            row[f"{prefix}__AcceptedErrors"] = int(candidate["AcceptedErrors"])
            row[f"{prefix}__AcceptedErrorPercent"] = candidate[
                "AcceptedErrorPercent"
            ]
            row[f"{prefix}__AcceptedAccuracy"] = candidate["AcceptedAccuracy"]
            row[f"{prefix}__AcceptedSensitivity"] = candidate[
                "AcceptedSensitivity"
            ]
            row[f"{prefix}__AcceptedSpecificity"] = candidate[
                "AcceptedSpecificity"
            ]
            row[f"{prefix}__AcceptedBalancedAccuracy"] = candidate[
                "AcceptedBalancedAccuracy"
            ]
            row[f"{prefix}__FalseNegativesAmongAccepted"] = int(
                candidate["FalseNegativesAmongAccepted"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_routing_predictions(
    frame: pd.DataFrame, candidates: pd.DataFrame, spec: StrategySpec
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    strategy_candidates = candidates.loc[candidates["StrategyID"].eq(spec.strategy_id)]
    identity_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "TrueLabel",
        "Day4PredictedLabel",
        "Day4EyeProbability",
        "Day4Correct",
        CONFIDENCE_FIELD,
        AGREEMENT_FIELD,
        VIEW_SD_FIELD,
        FEATURE_PERCENTILE_FIELD,
    ]
    for _, candidate in strategy_candidates.iterrows():
        accepted = make_acceptance_mask(frame, candidate)
        block = frame[identity_columns].copy()
        insert_values = {
            "CandidateOperatingPointID": candidate["CandidateOperatingPointID"],
            "StrategyID": spec.strategy_id,
            "Strategy": spec.display_name,
            "TargetCoveragePercent": candidate["TargetCoveragePercent"],
            "ActualCoveragePercent": candidate["CoveragePercent"],
            "ConfidenceThreshold": candidate["ConfidenceThreshold"],
            "RequireModelAgreement": spec.require_agreement,
            "ViewSDThreshold": candidate["ViewSDThreshold"],
            "FeaturePercentileThreshold": candidate["FeaturePercentileThreshold"],
            "CandidateOnly": True,
        }
        for position, (column, value) in enumerate(insert_values.items()):
            block.insert(position, column, value)
        block["Accepted"] = accepted
        block["Deferred"] = ~accepted
        block["RoutingDecision"] = np.where(accepted, "ACCEPT", "DEFER")
        block["AcceptedPredictionCorrect"] = np.where(
            accepted, block["Day4Correct"], pd.NA
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def make_error_count_budget_table(
    envelopes: dict[str, pd.DataFrame], specs: Iterable[StrategySpec]
) -> pd.DataFrame:
    # ask how much coverage each strategy can reach for a fixed error-count budget
    maximum_errors = max(
        int(envelope["AcceptedErrors"].max()) for envelope in envelopes.values()
    )
    rows: list[dict[str, Any]] = []
    for error_budget in range(maximum_errors + 1):
        budget_rows: list[dict[str, Any]] = []
        for spec in specs:
            eligible = envelopes[spec.strategy_id].loc[
                envelopes[spec.strategy_id]["AcceptedErrors"].le(error_budget)
            ]
            chosen = eligible.sort_values(
                ["AcceptedCount", "AcceptedErrorRate", "FalseNegativesAmongAccepted"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0]
            budget_rows.append(
                {
                    "AcceptedErrorCountBudget": error_budget,
                    "StrategyID": spec.strategy_id,
                    "Strategy": spec.display_name,
                    "HighestAcceptedCountWithinBudget": int(chosen["AcceptedCount"]),
                    "HighestCoverageWithinBudget": float(chosen["Coverage"]),
                    "HighestCoveragePercentWithinBudget": float(
                        chosen["CoveragePercent"]
                    ),
                    "ObservedAcceptedErrors": int(chosen["AcceptedErrors"]),
                    "ObservedAcceptedErrorRate": float(chosen["AcceptedErrorRate"])
                    if math.isfinite(float(chosen["AcceptedErrorRate"]))
                    else math.nan,
                    "FalseNegativesAmongAccepted": int(
                        chosen["FalseNegativesAmongAccepted"]
                    ),
                    "ConfidenceThreshold": chosen["ConfidenceThreshold"],
                    "ViewSDThreshold": chosen["ViewSDThreshold"],
                    "FeaturePercentileThreshold": chosen[
                        "FeaturePercentileThreshold"
                    ],
                }
            )
        maximum_count = max(row["HighestAcceptedCountWithinBudget"] for row in budget_rows)
        winners = [
            row["Strategy"]
            for row in budget_rows
            if row["HighestAcceptedCountWithinBudget"] == maximum_count
        ]
        for row in budget_rows:
            row["HighestCoverageWinner"] = (
                row["HighestAcceptedCountWithinBudget"] == maximum_count
            )
            row["HighestCoverageStrategies"] = "; ".join(winners)
            rows.append(row)
    return pd.DataFrame(rows)


def make_error_rate_budget_table(
    envelopes: dict[str, pd.DataFrame], specs: Iterable[StrategySpec]
) -> pd.DataFrame:
    # repeat the same idea using accepted error rate instead of raw error count
    observed_rates = sorted(
        {
            float(value)
            for envelope in envelopes.values()
            for value in envelope["AcceptedErrorRate"].to_numpy(float)
            if math.isfinite(float(value))
        }
    )
    rows: list[dict[str, Any]] = []
    for risk_budget in observed_rates:
        budget_rows: list[dict[str, Any]] = []
        for spec in specs:
            eligible = envelopes[spec.strategy_id].loc[
                envelopes[spec.strategy_id]["AcceptedErrorRate"].le(
                    risk_budget + 1e-15
                )
            ]
            if eligible.empty:
                continue
            chosen = eligible.sort_values(
                ["AcceptedCount", "AcceptedErrorRate", "FalseNegativesAmongAccepted"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0]
            budget_rows.append(
                {
                    "AcceptedErrorRateBudget": risk_budget,
                    "AcceptedErrorPercentBudget": 100.0 * risk_budget,
                    "StrategyID": spec.strategy_id,
                    "Strategy": spec.display_name,
                    "HighestAcceptedCountWithinBudget": int(chosen["AcceptedCount"]),
                    "HighestCoverageWithinBudget": float(chosen["Coverage"]),
                    "HighestCoveragePercentWithinBudget": float(
                        chosen["CoveragePercent"]
                    ),
                    "ObservedAcceptedErrors": int(chosen["AcceptedErrors"]),
                    "ObservedAcceptedErrorRate": float(chosen["AcceptedErrorRate"]),
                    "FalseNegativesAmongAccepted": int(
                        chosen["FalseNegativesAmongAccepted"]
                    ),
                }
            )
        if not budget_rows:
            continue
        maximum_count = max(row["HighestAcceptedCountWithinBudget"] for row in budget_rows)
        winners = [
            row["Strategy"]
            for row in budget_rows
            if row["HighestAcceptedCountWithinBudget"] == maximum_count
        ]
        for row in budget_rows:
            row["HighestCoverageWinner"] = (
                row["HighestAcceptedCountWithinBudget"] == maximum_count
            )
            row["HighestCoverageStrategies"] = "; ".join(winners)
            rows.append(row)
    return pd.DataFrame(rows)


def make_pareto_frontier(envelope: pd.DataFrame) -> pd.DataFrame:
    """keep the coverage improvements that form the empirical risk frontier."""
    nonempty = envelope.loc[envelope["AcceptedCount"].gt(0)].copy()
    rows: list[pd.Series] = []
    for risk_budget in sorted(nonempty["AcceptedErrorRate"].unique()):
        eligible = nonempty.loc[nonempty["AcceptedErrorRate"].le(risk_budget + 1e-15)]
        chosen = eligible.sort_values(
            ["AcceptedCount", "AcceptedErrorRate", "FalseNegativesAmongAccepted"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        if not rows or int(chosen["AcceptedCount"]) > int(rows[-1]["AcceptedCount"]):
            rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def compare_feature_ablation(
    full_envelope: pd.DataFrame, ablation_envelope: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # compare the full rule with the same rule after removing feature typicality
    columns = [
        "AcceptedCount",
        "Coverage",
        "AcceptedErrors",
        "AcceptedErrorRate",
        "FalseNegativesAmongAccepted",
        "AcceptedBalancedAccuracy",
        "FeaturePercentileThreshold",
        "FeatureGateActive",
    ]
    full = full_envelope[columns].add_prefix("Full__")
    ablation_columns = [column for column in columns if column not in {
        "FeaturePercentileThreshold", "FeatureGateActive"
    }]
    ablation = ablation_envelope[ablation_columns].add_prefix("Ablation__")
    comparison = full.merge(
        ablation,
        left_on="Full__AcceptedCount",
        right_on="Ablation__AcceptedCount",
        how="inner",
        validate="one_to_one",
    )
    comparison["FeatureImprovesAcceptedErrors"] = (
        comparison["Full__AcceptedErrors"] < comparison["Ablation__AcceptedErrors"]
    )
    comparison["FeatureImprovesFalseNegatives"] = (
        comparison["Full__FalseNegativesAmongAccepted"]
        < comparison["Ablation__FalseNegativesAmongAccepted"]
    )
    comparison["AcceptedErrorDeltaFullMinusAblation"] = (
        comparison["Full__AcceptedErrors"] - comparison["Ablation__AcceptedErrors"]
    )
    comparison["FalseNegativeDeltaFullMinusAblation"] = (
        comparison["Full__FalseNegativesAmongAccepted"]
        - comparison["Ablation__FalseNegativesAmongAccepted"]
    )
    summary = {
        "ComparedAcceptedCounts": int(len(comparison)),
        "CountsWithAcceptedErrorImprovement": int(
            comparison["FeatureImprovesAcceptedErrors"].sum()
        ),
        "CountsWithFalseNegativeImprovement": int(
            comparison["FeatureImprovesFalseNegatives"].sum()
        ),
        "CountsWithFeatureGateActiveOnChosenEnvelope": int(
            comparison["Full__FeatureGateActive"].sum()
        ),
    }
    summary["MeasurableIncrementalBenefitObserved"] = bool(
        summary["CountsWithAcceptedErrorImprovement"]
        or summary["CountsWithFalseNegativeImprovement"]
    )
    return comparison, summary


def save_float_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.17g",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if math.isinf(numeric):
            return "Infinity" if numeric > 0 else "-Infinity"
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def get_git_info() -> tuple[str | None, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        status = "unavailable"
    return commit, status


def save_risk_coverage_plot(
    envelopes: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    # show the full empirical envelopes and mark the common coverage candidates
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "D"]
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    for spec, color, marker in zip(STRATEGIES, colors, markers, strict=True):
        envelope = envelopes[spec.strategy_id]
        plotted = envelope.loc[envelope["AcceptedCount"].gt(0)].sort_values(
            "AcceptedCount"
        )
        axis.plot(
            100.0 * plotted["Coverage"],
            100.0 * plotted["AcceptedErrorRate"],
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=spec.display_name,
        )
        selected = candidates.loc[candidates["StrategyID"].eq(spec.strategy_id)]
        axis.scatter(
            selected["CoveragePercent"],
            selected["AcceptedErrorPercent"],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            s=45,
            zorder=4,
        )
    for target in TARGET_COVERAGE_PERCENTAGES:
        axis.axvline(target, color="#b8b8b8", linewidth=0.6, alpha=0.45, zorder=0)
    axis.set_xlabel("Validation coverage (%)")
    axis.set_ylabel("Accepted-prelabel error (%)")
    axis.set_title("Validation selective-routing risk–coverage envelopes")
    axis.grid(True, linewidth=0.5, alpha=0.25)
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    figure.text(
        0.995,
        0.012,
        "Conjunctive thresholds; circles/markers denote 50–90% candidate targets.\n"
        "Validation optimized; no operating point finalized.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def check_source_table(frame: pd.DataFrame, checks: list[dict[str, Any]]) -> pd.DataFrame:
    # confirm the validation table still matches the locked day-5 cohort and signals
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    record_check(checks, "Required Day-5 columns are present", not missing, missing, [])
    record_check(checks, "Validation row count", len(frame) == EXPECTED_EYES, len(frame), EXPECTED_EYES)
    record_check(
        checks,
        "Validation EyeExamID is unique",
        not frame["EyeExamID"].duplicated().any(),
        int(frame["EyeExamID"].duplicated().sum()),
        0,
    )
    for column in ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality"]:
        blank = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
        record_check(checks, f"{column} has no blank values", not blank.any(), int(blank.sum()), 0)
    record_check(
        checks,
        "Validation subject count",
        frame["ResearchSubjectID"].nunique() == EXPECTED_SUBJECTS,
        int(frame["ResearchSubjectID"].nunique()),
        EXPECTED_SUBJECTS,
    )
    record_check(
        checks,
        "Only OD/OS laterality",
        set(frame["Laterality"].dropna().astype(str)) <= {"OD", "OS"},
        sorted(set(frame["Laterality"].dropna().astype(str))),
        ["OD", "OS"],
    )
    invalid_true = sorted(set(frame["TrueLabel"].dropna().astype(str)) - ALLOWED_LABELS)
    invalid_pred = sorted(
        set(frame["Day4PredictedLabel"].dropna().astype(str)) - ALLOWED_LABELS
    )
    record_check(checks, "True labels are Normal/Abnormal", not invalid_true, invalid_true, [])
    record_check(
        checks,
        "Day-4 predictions are Normal/Abnormal",
        not invalid_pred,
        invalid_pred,
        [],
    )
    normal = int(frame["TrueLabel"].eq("Normal").sum())
    abnormal = int(frame["TrueLabel"].eq("Abnormal").sum())
    record_check(checks, "Validation Normal count", normal == EXPECTED_NORMAL, normal, EXPECTED_NORMAL)
    record_check(
        checks,
        "Validation Abnormal count",
        abnormal == EXPECTED_ABNORMAL,
        abnormal,
        EXPECTED_ABNORMAL,
    )

    frame = frame.copy()
    frame["Day4Correct"] = parse_bool_column(frame["Day4Correct"], "Day4Correct")
    frame[AGREEMENT_FIELD] = parse_bool_column(frame[AGREEMENT_FIELD], AGREEMENT_FIELD)
    reproduced_correct = frame["TrueLabel"].eq(frame["Day4PredictedLabel"])
    reproduced_agreement = frame["Day3PredictedLabel"].eq(frame["Day4PredictedLabel"])
    record_check(
        checks,
        "Day-4 Correct flags reproduce",
        np.array_equal(
            reproduced_correct.to_numpy(bool), frame["Day4Correct"].to_numpy(bool)
        ),
        int((reproduced_correct != frame["Day4Correct"]).sum()),
        0,
    )
    record_check(
        checks,
        "ModelAgreement flags reproduce",
        np.array_equal(
            reproduced_agreement.to_numpy(bool), frame[AGREEMENT_FIELD].to_numpy(bool)
        ),
        int((reproduced_agreement != frame[AGREEMENT_FIELD]).sum()),
        0,
    )
    correct = int(frame["Day4Correct"].sum())
    agreement = int(frame[AGREEMENT_FIELD].sum())
    record_check(checks, "Validation correct count", correct == EXPECTED_CORRECT, correct, EXPECTED_CORRECT)
    record_check(
        checks,
        "Validation incorrect count",
        len(frame) - correct == EXPECTED_INCORRECT,
        len(frame) - correct,
        EXPECTED_INCORRECT,
    )
    record_check(
        checks,
        "Validation model-agreement count",
        agreement == EXPECTED_AGREEMENT,
        agreement,
        EXPECTED_AGREEMENT,
    )

    numeric_fields = [
        "Day4EyeProbability",
        CONFIDENCE_FIELD,
        VIEW_SD_FIELD,
        FEATURE_PERCENTILE_FIELD,
    ]
    for column in numeric_fields:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(frame[column].to_numpy(float))
        record_check(
            checks,
            f"{column} is finite",
            bool(finite.all()),
            int((~finite).sum()),
            0,
        )
    record_check(
        checks,
        "Day-4 probabilities are within [0,1]",
        frame["Day4EyeProbability"].between(0.0, 1.0, inclusive="both").all(),
        f"{frame['Day4EyeProbability'].min():.17g}..{frame['Day4EyeProbability'].max():.17g}",
        "0..1",
    )
    record_check(
        checks,
        "Confidence margins are nonnegative",
        frame[CONFIDENCE_FIELD].ge(0.0).all(),
        float(frame[CONFIDENCE_FIELD].min()),
        ">=0",
    )
    record_check(
        checks,
        "View SD values are nonnegative",
        frame[VIEW_SD_FIELD].ge(0.0).all(),
        float(frame[VIEW_SD_FIELD].min()),
        ">=0",
    )
    record_check(
        checks,
        "Feature reference percentiles are within [0,100]",
        frame[FEATURE_PERCENTILE_FIELD].between(0.0, 100.0, inclusive="both").all(),
        f"{frame[FEATURE_PERCENTILE_FIELD].min():.17g}..{frame[FEATURE_PERCENTILE_FIELD].max():.17g}",
        "0..100",
    )
    record_check(
        checks,
        "View probability range is excluded from routing signals",
        "Day3ViewProbabilityRange" not in ROUTING_SIGNAL_FIELDS,
        sorted(ROUTING_SIGNAL_FIELDS),
        "Day3ViewProbabilityRange absent",
    )
    return frame.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)


def check_envelopes_and_candidates(
    frame: pd.DataFrame,
    envelopes: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    # agreement-based strategies can accept at most the 112 agreeing eyes
    expected_maximum = {
        "confidence_only": 120,
        "confidence_model_agreement": 112,
        "confidence_view_sd": 120,
        "full_reliability": 112,
        FEATURE_ABLATION.strategy_id: 112,
    }
    for strategy_id, envelope in envelopes.items():
        counts = envelope["AcceptedCount"].astype(int).tolist()
        expected_counts = list(range(expected_maximum[strategy_id] + 1))
        record_check(
            checks,
            f"{strategy_id}: every attainable accepted count is represented",
            counts == expected_counts,
            f"{min(counts)}..{max(counts)} ({len(counts)} rows)",
            f"0..{expected_maximum[strategy_id]} ({len(expected_counts)} rows)",
        )
        record_check(
            checks,
            f"{strategy_id}: coverage is monotone by accepted count",
            envelope["Coverage"].is_monotonic_increasing,
            bool(envelope["Coverage"].is_monotonic_increasing),
            True,
        )
        for _, row in envelope.iterrows():
            accepted = make_acceptance_mask(frame, row)
            if int(accepted.sum()) != int(row["AcceptedCount"]):
                raise SafetyError(
                    f"{strategy_id}: stored threshold rule does not reproduce "
                    f"AcceptedCount={row['AcceptedCount']}."
                )
        record_check(
            checks,
            f"{strategy_id}: all threshold rules reproduce accepted counts",
            True,
            "all rows",
            "all rows",
        )

    record_check(
        checks,
        "Exactly four requested strategies",
        set(candidates["StrategyID"].unique()) == {spec.strategy_id for spec in STRATEGIES},
        sorted(candidates["StrategyID"].unique().tolist()),
        sorted(spec.strategy_id for spec in STRATEGIES),
    )
    record_check(
        checks,
        "Five common targets per strategy",
        len(candidates) == len(STRATEGIES) * len(TARGET_COVERAGE_PERCENTAGES),
        len(candidates),
        len(STRATEGIES) * len(TARGET_COVERAGE_PERCENTAGES),
    )
    for _, row in candidates.iterrows():
        accepted = make_acceptance_mask(frame, row)
        metrics = calculate_selective_metrics(
            frame["Day4PredictedLabel"].astype(str).to_numpy(),
            frame["TrueLabel"].astype(str).to_numpy(),
            accepted.astype(bool),
        )
        integer_fields = {
            "AcceptedCount": "AcceptedCount",
            "DeferredCount": "DeferredCount",
            "FalseNegativesAmongAccepted": "FalseNegativesAmongAccepted",
            "AcceptedFalsePositives": "AcceptedFalsePositives",
        }
        for output_field, metric_field in integer_fields.items():
            if int(row[output_field]) != int(metrics[metric_field]):
                raise SafetyError(
                    f"Candidate {row['CandidateOperatingPointID']} does not reproduce "
                    f"{output_field}."
                )
        float_fields = {
            "Coverage": "Coverage",
            "ReviewRate": "ReviewRate",
            "AcceptedErrorRate": "AcceptedErrorRate",
            "AcceptedAccuracy": "AcceptedAccuracy",
            "AcceptedSensitivity": "AcceptedSensitivity",
            "AcceptedSpecificity": "AcceptedSpecificity",
            "AcceptedBalancedAccuracy": "AcceptedBalancedAccuracy",
        }
        for output_field, metric_field in float_fields.items():
            observed = float(row[output_field])
            expected = float(metrics[metric_field])
            if math.isnan(expected):
                if not math.isnan(observed):
                    raise SafetyError(
                        f"Candidate {row['CandidateOperatingPointID']} {output_field} "
                        "should be NaN."
                    )
            elif not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise SafetyError(
                    f"Candidate {row['CandidateOperatingPointID']} does not reproduce "
                    f"{output_field}: {observed} vs {expected}."
                )
    record_check(
        checks,
        "All candidate metrics independently reproduce",
        True,
        len(candidates),
        len(candidates),
    )
    record_check(
        checks,
        "No final operating point was selected",
        not bool(candidates.get("OperatingPointFinalized", pd.Series(False)).any()),
        False,
        False,
    )


def make_output_dirs(run_id: str) -> tuple[Path, Path, Path]:
    prediction_dir = PREDICTION_OUTPUT_ROOT / run_id
    figure_dir = FIGURE_OUTPUT_ROOT / run_id
    audit_dir = AUDIT_OUTPUT_ROOT / run_id
    for directory in (prediction_dir, figure_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=False)
    return prediction_dir, figure_dir, audit_dir


def main() -> None:
    # this stage only searches validation thresholds and does not finalize a rule
    checks: list[dict[str, Any]] = []
    source_hash_before = file_sha256(DAY5_RELIABILITY_PATH) if DAY5_RELIABILITY_PATH.is_file() else None
    record_check(
        checks,
        "Locked Day-5 reliability table exists",
        DAY5_RELIABILITY_PATH.is_file(),
        DAY5_RELIABILITY_PATH.is_file(),
        True,
    )
    record_check(
        checks,
        "Locked Day-5 reliability SHA-256",
        source_hash_before == EXPECTED_DAY5_SHA256,
        source_hash_before,
        EXPECTED_DAY5_SHA256,
    )
    # validate the frozen day-5 table before enumerating any candidate rules
    frame = check_source_table(read_locked_reliability_table(DAY5_RELIABILITY_PATH), checks)

    confidence = frame[CONFIDENCE_FIELD].to_numpy(float)
    view_sd = frame[VIEW_SD_FIELD].to_numpy(float)
    feature = frame[FEATURE_PERCENTILE_FIELD].to_numpy(float)
    agreement = frame[AGREEMENT_FIELD].to_numpy(bool)
    # precompute threshold masks once and reuse them across strategies
    confidence_masks = make_threshold_masks(confidence, "ge", include_reject_all=True)
    view_sd_masks = make_threshold_masks(view_sd, "le", include_reject_all=True)
    feature_masks = make_threshold_masks(feature, "le", include_reject_all=True)
    agreement_mask = bools_to_bitmask(agreement)

    envelopes: dict[str, pd.DataFrame] = {}
    enumeration: dict[str, dict[str, int]] = {}
    # enumerate the four requested strategies plus the matched feature ablation
    for spec in (*STRATEGIES, FEATURE_ABLATION):
        envelope, counts = make_risk_envelope(
            spec,
            frame,
            confidence_masks,
            view_sd_masks,
            feature_masks,
            agreement_mask,
        )
        envelopes[spec.strategy_id] = envelope
        enumeration[spec.strategy_id] = counts
        print(
            f"Enumerated {spec.display_name}: "
            f"{counts['RawThresholdConfigurations']:,} threshold configurations, "
            f"{counts['UniqueAcceptanceSets']:,} unique accepted sets."
        )

    # summarize the envelopes at the common 50, 60, 70, 80, and 90 percent targets
    candidates = pick_target_candidates(envelopes, STRATEGIES)
    comparison = make_strategy_comparison(candidates)
    count_budgets = make_error_count_budget_table(envelopes, STRATEGIES)
    rate_budgets = make_error_rate_budget_table(envelopes, STRATEGIES)
    pareto = pd.DataFrame.from_records(
        [
            row
            for spec in STRATEGIES
            for row in make_pareto_frontier(
                envelopes[spec.strategy_id]
            ).to_dict(orient="records")
        ]
    )
    # compare full versus ablated rules at every shared accepted-eye count
    feature_by_coverage, feature_summary = compare_feature_ablation(
        envelopes["full_reliability"], envelopes[FEATURE_ABLATION.strategy_id]
    )

    check_envelopes_and_candidates(frame, envelopes, candidates, checks)
    record_check(
        checks,
        "Feature matched ablation compared every agreement-attainable count",
        int(feature_summary["ComparedAcceptedCounts"]) == EXPECTED_AGREEMENT + 1,
        int(feature_summary["ComparedAcceptedCounts"]),
        EXPECTED_AGREEMENT + 1,
    )
    record_check(
        checks,
        "Feature accepted-error improvement count is internally valid",
        0
        <= int(feature_summary["CountsWithAcceptedErrorImprovement"])
        <= int(feature_summary["ComparedAcceptedCounts"]),
        int(feature_summary["CountsWithAcceptedErrorImprovement"]),
        f"0..{feature_summary['ComparedAcceptedCounts']}",
        "The observed value is a scientific result, not a pass/fail expectation.",
    )
    record_check(
        checks,
        "Feature false-negative improvement count is internally valid",
        0
        <= int(feature_summary["CountsWithFalseNegativeImprovement"])
        <= int(feature_summary["ComparedAcceptedCounts"]),
        int(feature_summary["CountsWithFalseNegativeImprovement"]),
        f"0..{feature_summary['ComparedAcceptedCounts']}",
        "The observed value is a scientific result, not a pass/fail expectation.",
    )
    record_check(
        checks,
        "Loaded-path registry contains only the locked validation table",
        LOADED_PATHS == {DAY5_RELIABILITY_PATH},
        sorted(str(path) for path in LOADED_PATHS),
        [str(DAY5_RELIABILITY_PATH)],
    )
    record_check(
        checks,
        "No loaded path is a test artifact",
        all(
            path.name.casefold() not in FORBIDDEN_TEST_NAMES
            and not path.name.casefold().startswith("test_")
            for path in LOADED_PATHS
        ),
        False,
        False,
    )
    # rehash the source table to confirm this analysis did not modify it
    source_hash_after = file_sha256(DAY5_RELIABILITY_PATH)
    record_check(
        checks,
        "Locked Day-5 table unchanged during execution",
        source_hash_after == source_hash_before,
        source_hash_after,
        source_hash_before,
    )

    # create output folders only after the main validation checks have passed
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    prediction_dir, figure_dir, audit_dir = make_output_dirs(run_id)

    curve_path = audit_dir / "day6_risk_coverage_envelope.csv"
    pareto_path = audit_dir / "day6_risk_coverage_pareto_frontier.csv"
    comparison_path = audit_dir / "day6_strategy_comparison.csv"
    candidate_path = audit_dir / "day6_candidate_operating_points.csv"
    count_budget_path = audit_dir / "day6_error_count_coverage.csv"
    rate_budget_path = audit_dir / "day6_error_rate_coverage.csv"
    feature_path = audit_dir / "day6_feature_incremental_analysis.csv"
    checks_path = audit_dir / "day6_selective_routing_checks.csv"
    metadata_path = prediction_dir / "day6_routing_metadata.json"
    figure_path = figure_dir / "risk_coverage_curve.png"

    requested_envelopes = pd.DataFrame.from_records(
        [
            row
            for spec in STRATEGIES
            for row in envelopes[spec.strategy_id].to_dict(orient="records")
        ]
    )
    # bitmasks are only used internally and are not written to the result tables
    requested_envelopes = requested_envelopes.drop(columns="AcceptedBitmask")
    candidate_output = candidates.drop(columns="AcceptedBitmask")
    save_float_csv(requested_envelopes, curve_path)
    save_float_csv(pareto.drop(columns="AcceptedBitmask"), pareto_path)
    save_float_csv(comparison, comparison_path)
    save_float_csv(candidate_output, candidate_path)
    save_float_csv(count_budgets, count_budget_path)
    save_float_csv(rate_budgets, rate_budget_path)
    save_float_csv(feature_by_coverage, feature_path)

    prediction_paths: dict[str, str] = {}
    validation_eye_ids = set(frame["EyeExamID"].astype(str))
    routing_frames: dict[str, pd.DataFrame] = {}
    for spec in STRATEGIES:
        routing = make_routing_predictions(frame, candidates, spec)
        routing_frames[spec.strategy_id] = routing
        path = prediction_dir / f"{spec.strategy_id}_validation_routing_predictions.csv"
        save_float_csv(routing, path)
        prediction_paths[spec.strategy_id] = str(path)

    save_risk_coverage_plot(envelopes, candidates, figure_path)

    record_check(
        checks,
        "Every validation routing file has five rows per eye",
        all(
            len(routing) == EXPECTED_EYES * len(TARGET_COVERAGE_PERCENTAGES)
            for routing in routing_frames.values()
        ),
        EXPECTED_EYES * len(TARGET_COVERAGE_PERCENTAGES),
        EXPECTED_EYES * len(TARGET_COVERAGE_PERCENTAGES),
        "Output reads only; no model-input or test artifact is accessed.",
    )
    record_check(
        checks,
        "Every routing candidate contains exactly the locked validation EyeExamIDs",
        all(
            set(group["EyeExamID"].astype(str)) == validation_eye_ids
            and not group["EyeExamID"].duplicated().any()
            for routing in routing_frames.values()
            for _, group in routing.groupby("CandidateOperatingPointID", sort=False)
        ),
        "all candidate ID sets",
        "exact locked validation ID set",
        "This proves output scope without loading test IDs.",
    )
    record_check(
        checks,
        "Routing Accepted flags reproduce candidate accepted counts",
        all(
            int(group["Accepted"].sum())
            == int(
                candidates.loc[
                    candidates["CandidateOperatingPointID"].eq(candidate_id),
                    "AcceptedCount",
                ].iloc[0]
            )
            for routing in routing_frames.values()
            for candidate_id, group in routing.groupby(
                "CandidateOperatingPointID", sort=False
            )
        ),
        "all 20 candidate masks",
        "all 20 candidate masks",
    )
    record_check(
        checks,
        "Risk-coverage figure exists and is nonempty",
        figure_path.is_file() and figure_path.stat().st_size > 0,
        figure_path.stat().st_size if figure_path.is_file() else 0,
        ">0 bytes",
    )

    git_commit, git_status = get_git_info()
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "Day 6 selective-routing threshold tuning",
        "data_scope": "locked validation reliability signals only",
        "source": {
            "path": str(DAY5_RELIABILITY_PATH),
            "day5_run_id": DAY5_RELIABILITY_RUN_ID,
            "sha256": source_hash_after,
            "rows": len(frame),
            "eyes": int(frame["EyeExamID"].nunique()),
            "subjects": int(frame["ResearchSubjectID"].nunique()),
            "normal": int(frame["TrueLabel"].eq("Normal").sum()),
            "abnormal": int(frame["TrueLabel"].eq("Abnormal").sum()),
            "correct": int(frame["Day4Correct"].sum()),
            "incorrect": int((~frame["Day4Correct"]).sum()),
        },
        "strategy_definitions": [
            {
                "strategy_id": spec.strategy_id,
                "display_name": spec.display_name,
                "acceptance_rule": spec.rule,
                "complexity": spec.complexity,
            }
            for spec in STRATEGIES
        ],
        "threshold_search": {
            "method": (
                "exhaustive inclusive cutoffs at every unique observed validation "
                "value; exact accepted-set deduplication"
            ),
            "comparators": {
                CONFIDENCE_FIELD: ">=",
                AGREEMENT_FIELD: "is True when enabled",
                VIEW_SD_FIELD: "<=",
                FEATURE_PERCENTILE_FIELD: "<=",
            },
            "view_probability_range_used": False,
            "zero_acceptance_sentinels": {
                "confidence": "+Infinity",
                "view_sd": "-Infinity",
                "feature_percentile": "-Infinity",
            },
            "envelope_tie_rule": [
                "fewest accepted errors",
                "fewest accepted false negatives",
                "highest accepted balanced accuracy",
                "least restrictive feature cutoff",
                "least restrictive view-SD cutoff",
                "lowest confidence cutoff",
                "lexicographic accepted EyeExamID signature",
            ],
            "candidate_target_rule": (
                "nearest attainable accepted-eye count; equidistant ties choose "
                "the lower count"
            ),
            "target_coverage_percentages": list(TARGET_COVERAGE_PERCENTAGES),
            "curves_are_validation_optimized_envelopes": True,
            "envelope_masks_may_be_nonnested": True,
            "risk_smoothing_applied": False,
            "enumeration": enumeration,
        },
        "simplicity_preference": {
            "essentially_same_definition": (
                "same target accepted count, accepted-error count, and accepted-FN count"
            ),
            "rule": "prefer fewer enabled reliability components",
            "strategy_order_after_complexity_tie": [
                spec.strategy_id for spec in STRATEGIES
            ],
        },
        "feature_incremental_analysis": {
            "matched_ablation": FEATURE_ABLATION.rule,
            **feature_summary,
            "interpretation": (
                "No measurable incremental routing benefit on this validation set"
                if not feature_summary["MeasurableIncrementalBenefitObserved"]
                else "A measurable validation-set improvement was observed"
            ),
            "final_routing_requirement_inferred": False,
        },
        "metrics": {
            "positive_class": "Abnormal",
            "accepted_metrics_are_conditional_on_accepted_eyes": True,
            "zero_accepted_conditional_metrics": "NaN",
            "risk_count_budget_table": str(count_budget_path),
            "risk_rate_budget_table": str(rate_budget_path),
        },
        "operating_point_finalized": False,
        "routing_thresholds_are_candidates_only": True,
        "test_data_loaded": False,
        "test_set_evaluated": False,
        "test_predictions_created": False,
        "loaded_paths": sorted(str(path) for path in LOADED_PATHS),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "utility_module": str(
            PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
        ),
        "utility_module_sha256": file_sha256(
            PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
        ),
        "git_commit_hash": git_commit,
        "git_worktree_status_at_completion": git_status,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "artifacts": {
            "risk_coverage_envelope": str(curve_path),
            "pareto_frontier": str(pareto_path),
            "strategy_comparison": str(comparison_path),
            "candidate_operating_points": str(candidate_path),
            "error_count_coverage": str(count_budget_path),
            "error_rate_coverage": str(rate_budget_path),
            "feature_incremental_analysis": str(feature_path),
            "validation_routing_predictions": prediction_paths,
            "validation_checks": str(checks_path),
            "risk_coverage_curve": str(figure_path),
        },
    }
    write_json(metadata_path, make_json_safe(metadata))
    pd.DataFrame(checks).to_csv(checks_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 108)
    print("DAY-6 VALIDATION-ONLY SELECTIVE-ROUTING CANDIDATES COMPLETE")
    print("=" * 108)
    display_columns = [
        "TargetCoveragePercent",
        "Strategy",
        "AcceptedCount",
        "CoveragePercent",
        "ReviewRatePercent",
        "AcceptedErrors",
        "AcceptedErrorPercent",
        "AcceptedAccuracy",
        "AcceptedSensitivity",
        "AcceptedSpecificity",
        "AcceptedBalancedAccuracy",
        "FalseNegativesAmongAccepted",
    ]
    print(candidates[display_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nLOWEST ACCEPTED ERROR / SIMPLEST TIED STRATEGY")
    for _, row in comparison.iterrows():
        print(
            f"  {int(row['TargetCoveragePercent'])}%: lowest error = "
            f"{row['LowestAcceptedErrorStrategies']}; simplicity preference = "
            f"{row['SimplestPreferredStrategy']}"
        )
    print("\nFEWEST ACCEPTED FALSE NEGATIVES")
    for _, row in comparison.iterrows():
        print(
            f"  {int(row['TargetCoveragePercent'])}%: "
            f"{row['FewestAcceptedFalseNegativeStrategies']}"
        )
    print("\nHIGHEST COVERAGE BY ACCEPTED-ERROR COUNT BUDGET")
    count_winners = count_budgets.loc[count_budgets["HighestCoverageWinner"]]
    for budget, group in count_winners.groupby("AcceptedErrorCountBudget", sort=True):
        print(
            f"  <= {int(budget)} errors: "
            f"{int(group['HighestAcceptedCountWithinBudget'].iloc[0])}/120 "
            f"({group['HighestCoveragePercentWithinBudget'].iloc[0]:.2f}%) — "
            f"{group['HighestCoverageStrategies'].iloc[0]}"
        )
    print("\nFEATURE TYPICALITY INCREMENTAL VALUE")
    print(
        "  Accepted-count levels with lower error versus matched no-feature ablation: "
        f"{feature_summary['CountsWithAcceptedErrorImprovement']}"
    )
    print(
        "  Accepted-count levels with fewer false negatives: "
        f"{feature_summary['CountsWithFalseNegativeImprovement']}"
    )
    print(
        "  Conclusion: no measurable incremental routing benefit on this validation set."
        if not feature_summary["MeasurableIncrementalBenefitObserved"]
        else "  Conclusion: a measurable validation-set improvement was observed."
    )
    print(f"\nStrategy comparison        : {comparison_path}")
    print(f"Candidate operating points : {candidate_path}")
    print(f"Risk-coverage curve        : {figure_path}")
    print(f"Validation predictions     : {prediction_dir}")
    print(f"Validation checks          : {checks_path}")
    print("\nNO OPERATING POINT WAS FINALIZED.")
    print("TEST SET WAS NOT LOADED OR EVALUATED.")


if __name__ == "__main__":
    try:
        main()
    except (SafetyError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
