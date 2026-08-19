
from __future__ import annotations

from datetime import datetime
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reliability_metrics import calculate_selective_metrics


DAY5_RUN_ID = "20260814T141442_497529-0700"
DAY6_RUN_ID = "20260814T152105_990701-0700"

DAY5_RELIABILITY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "reliability"
    / DAY5_RUN_ID
    / "val_reliability_signals.csv"
).resolve()
DAY6_AUDIT_DIR = (PROJECT_ROOT / "outputs" / "audits" / DAY6_RUN_ID).resolve()
DAY6_PREDICTION_DIR = (
    PROJECT_ROOT / "outputs" / "predictions" / "selective_routing" / DAY6_RUN_ID
).resolve()
DAY6_ENVELOPE_PATH = (DAY6_AUDIT_DIR / "day6_risk_coverage_envelope.csv").resolve()
DAY6_PARETO_PATH = (
    DAY6_AUDIT_DIR / "day6_risk_coverage_pareto_frontier.csv"
).resolve()
DAY6_METADATA_PATH = (DAY6_PREDICTION_DIR / "day6_routing_metadata.json").resolve()

EXPECTED_HASHES = {
    DAY5_RELIABILITY_PATH: (
        "a767b163b3565a9fe664068c5fd18e96e4f13bf4dea7040ee179f098b477edb8"
    ),
    DAY6_ENVELOPE_PATH: (
        "144911385308fbbf080b1a6ab5195f9d290db5b385139110c70cd0e9c55d3a7b"
    ),
    DAY6_PARETO_PATH: (
        "6d091933283cfebd8d5b77969da27e0864645e8f7f17179bedec2c41201dc784"
    ),
    DAY6_METADATA_PATH: (
        "eb85ed241e118e21ce45ce9d12dc2d440b3ad3dfc15298b115c87f0b7eb115e3"
    ),
}

CSV_INPUTS = {DAY5_RELIABILITY_PATH, DAY6_ENVELOPE_PATH, DAY6_PARETO_PATH}
JSON_INPUTS = {DAY6_METADATA_PATH}
LOADED_INPUT_PATHS: set[Path] = set()

OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "models" / "selective_routing").resolve()
FROZEN_JSON_PATH = (OUTPUT_DIR / "frozen_routing_rules.json").resolve()
FROZEN_CSV_PATH = (OUTPUT_DIR / "frozen_routing_rules.csv").resolve()
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"

EXPECTED_EYES = 120
EXPECTED_SUBJECTS = 95
EXPECTED_NORMAL = 32
EXPECTED_ABNORMAL = 88
BALANCED_MIN_COVERAGE = 0.70
BALANCED_MAX_COVERAGE = 0.80
EXPECTED_CONSERVATIVE_CONFIDENCE_THRESHOLD = 2.2084707824028404
EXPECTED_BALANCED_CONFIDENCE_THRESHOLD = 0.8317997606386935
EXPECTED_BALANCED_VIEW_SD_THRESHOLD = 0.33322164747457844
EXPECTED_CONSERVATIVE_ACCEPTED_ID_SHA256 = (
    "4d4f13d81de13a5787d5363a9c4d6ca9194e2762ad11ae8d7290707e0e9fb998"
)
EXPECTED_BALANCED_ACCEPTED_ID_SHA256 = (
    "57f0dc715d52965fef319727fe1e2de661a33ecc96edcabe87913699d38c41ec"
)
ALLOWED_LABELS = {"Normal", "Abnormal"}
FORBIDDEN_TEST_NAMES = {
    "test_images.csv",
    "test_eye_predictions.csv",
    "test_image_predictions.csv",
    "test_reliability_signals.csv",
}

RELIABILITY_REQUIRED_COLUMNS = {
    "EyeExamID",
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "TrueLabel",
    "Day4PredictedLabel",
    "Day4EyeProbability",
    "PrimaryConfidenceMargin",
    "ModelAgreement",
    "Day3StdViewProbability",
    "PredictedClassReferencePercentile",
}
ENVELOPE_REQUIRED_COLUMNS = {
    "AcceptanceRule",
    "StrategyID",
    "Strategy",
    "StrategyComplexity",
    "RequireModelAgreement",
    "UseViewSD",
    "UseFeatureTypicality",
    "ConfidenceThreshold",
    "ViewSDThreshold",
    "FeaturePercentileThreshold",
    "FeatureGateActive",
    "AcceptedCount",
    "DeferredCount",
    "Coverage",
    "ReviewRate",
    "AcceptedErrors",
    "AcceptedErrorRate",
    "AcceptedAccuracy",
    "AcceptedSensitivity",
    "AcceptedSpecificity",
    "AcceptedBalancedAccuracy",
    "FalseNegativesAmongAccepted",
    "AcceptedTruePositives",
    "AcceptedTrueNegatives",
    "AcceptedFalsePositives",
    "AcceptedFalseNegatives",
}


class SafetyError(RuntimeError):
    """used when an input, rule-selection, or lock check fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_test_path(path: Path) -> None:
    name = path.name.casefold()
    if name in FORBIDDEN_TEST_NAMES or name.startswith("test_"):
        raise SafetyError(f"Test artifact access was attempted: {path}")


def check_locked_input(path: Path) -> str:
    # every upstream input is hash-pinned before it can be used
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in EXPECTED_HASHES:
        raise SafetyError(f"Input is outside the exact frozen allowlist: {resolved}")
    if not resolved.is_file():
        raise SafetyError(f"Required locked input is missing: {resolved}")
    observed = file_sha256(resolved)
    expected = EXPECTED_HASHES[resolved]
    if observed != expected:
        raise SafetyError(
            f"Locked input hash mismatch for {resolved}: observed={observed}, "
            f"expected={expected}."
        )
    return observed


def read_allowed_csv(path: Path) -> pd.DataFrame:
    # only the locked validation csv files are allowed in this stage
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in CSV_INPUTS:
        raise SafetyError(f"CSV is outside the validation-only allowlist: {resolved}")
    check_locked_input(resolved)
    LOADED_INPUT_PATHS.add(resolved)
    return pd.read_csv(
        resolved,
        dtype={
            "EyeExamID": "string",
            "ResearchSubjectID": "string",
            "EncounterID": "string",
            "Laterality": "string",
            "TrueLabel": "string",
            "Day4PredictedLabel": "string",
            "StrategyID": "string",
            "Strategy": "string",
        },
        float_precision="round_trip",
    )


def read_allowed_json(path: Path) -> dict[str, Any]:
    # json inputs follow the same validation-only allowlist
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in JSON_INPUTS:
        raise SafetyError(f"JSON is outside the validation-only allowlist: {resolved}")
    check_locked_input(resolved)
    LOADED_INPUT_PATHS.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"Expected JSON object in {resolved}.")
    return value


def parse_bool_column(series: pd.Series, column: str) -> pd.Series:
    if series.dtype.kind == "b":
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = normalized[~normalized.isin(["true", "false"])]
    if not invalid.empty:
        raise SafetyError(
            f"{column} contains non-boolean values: "
            f"{sorted(invalid.dropna().unique().tolist())}"
        )
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


def check_finite_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise SafetyError(f"{column} contains a missing or nonfinite value.")


def check_inputs(
    reliability: pd.DataFrame,
    envelope: pd.DataFrame,
    pareto: pd.DataFrame,
    metadata: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # confirm the frozen day-5 and day-6 artifacts still have the expected schema
    missing_reliability = sorted(RELIABILITY_REQUIRED_COLUMNS - set(reliability.columns))
    missing_envelope = sorted(ENVELOPE_REQUIRED_COLUMNS - set(envelope.columns))
    missing_pareto = sorted(ENVELOPE_REQUIRED_COLUMNS - set(pareto.columns))
    record_check(
        checks,
        "Locked reliability schema",
        not missing_reliability,
        missing_reliability,
        [],
    )
    record_check(checks, "Locked envelope schema", not missing_envelope, missing_envelope, [])
    record_check(checks, "Locked Pareto schema", not missing_pareto, missing_pareto, [])

    reliability = reliability.copy()
    envelope = envelope.copy()
    pareto = pareto.copy()
    reliability["ModelAgreement"] = parse_bool_column(
        reliability["ModelAgreement"], "ModelAgreement"
    )
    for frame in (envelope, pareto):
        for column in [
            "RequireModelAgreement",
            "UseViewSD",
            "UseFeatureTypicality",
            "FeatureGateActive",
        ]:
            frame[column] = parse_bool_column(frame[column], column)

    record_check(checks, "Validation eye count", len(reliability) == EXPECTED_EYES, len(reliability), EXPECTED_EYES)
    record_check(
        checks,
        "Validation EyeExamID uniqueness",
        not reliability["EyeExamID"].duplicated().any(),
        int(reliability["EyeExamID"].duplicated().sum()),
        0,
    )
    record_check(
        checks,
        "Validation subject count",
        reliability["ResearchSubjectID"].nunique() == EXPECTED_SUBJECTS,
        int(reliability["ResearchSubjectID"].nunique()),
        EXPECTED_SUBJECTS,
    )
    invalid_true = sorted(set(reliability["TrueLabel"].dropna().astype(str)) - ALLOWED_LABELS)
    invalid_pred = sorted(
        set(reliability["Day4PredictedLabel"].dropna().astype(str)) - ALLOWED_LABELS
    )
    record_check(checks, "True label domain", not invalid_true, invalid_true, [])
    record_check(checks, "Prediction label domain", not invalid_pred, invalid_pred, [])
    normal = int(reliability["TrueLabel"].eq("Normal").sum())
    abnormal = int(reliability["TrueLabel"].eq("Abnormal").sum())
    record_check(checks, "Validation Normal count", normal == EXPECTED_NORMAL, normal, EXPECTED_NORMAL)
    record_check(
        checks,
        "Validation Abnormal count",
        abnormal == EXPECTED_ABNORMAL,
        abnormal,
        EXPECTED_ABNORMAL,
    )
    check_finite_numeric(
        reliability,
        [
            "Day4EyeProbability",
            "PrimaryConfidenceMargin",
            "Day3StdViewProbability",
            "PredictedClassReferencePercentile",
        ],
    )
    record_check(
        checks,
        "Reliability signals are finite and in valid ranges",
        reliability["Day4EyeProbability"].between(0, 1, inclusive="both").all()
        and reliability["PrimaryConfidenceMargin"].ge(0).all()
        and reliability["Day3StdViewProbability"].ge(0).all()
        and reliability["PredictedClassReferencePercentile"].between(
            0, 100, inclusive="both"
        ).all(),
        "valid",
        "valid",
    )

    record_check(
        checks,
        "Day-6 metadata identifies the locked run",
        metadata.get("run_id") == DAY6_RUN_ID,
        metadata.get("run_id"),
        DAY6_RUN_ID,
    )
    for flag in ["test_data_loaded", "test_set_evaluated", "test_predictions_created"]:
        record_check(
            checks,
            f"Upstream Day-6 {flag} is false",
            flag in metadata and metadata[flag] is False,
            metadata.get(flag, "missing"),
            False,
        )
    feature_analysis = metadata.get("feature_incremental_analysis", {})
    record_check(
        checks,
        "Day-6 feature analysis found no measurable incremental benefit",
        feature_analysis.get("MeasurableIncrementalBenefitObserved") is False,
        feature_analysis.get("MeasurableIncrementalBenefitObserved", "missing"),
        False,
    )
    record_check(
        checks,
        "Day-6 did not finalize an operating point",
        metadata.get("operating_point_finalized") is False,
        metadata.get("operating_point_finalized", "missing"),
        False,
    )
    return (
        reliability.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True),
        envelope.sort_values(["StrategyID", "AcceptedCount"], kind="mergesort").reset_index(drop=True),
        pareto.sort_values(["StrategyID", "AcceptedCount"], kind="mergesort").reset_index(drop=True),
    )


def choose_conservative_rule(envelope: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    # maximize zero-error coverage, then prefer the simpler tied strategy
    zero_error = envelope.loc[
        envelope["AcceptedCount"].gt(0) & envelope["AcceptedErrors"].eq(0)
    ].copy()
    if zero_error.empty:
        raise SafetyError("No nonempty zero-error validation rule exists.")
    maximum_count = int(zero_error["AcceptedCount"].max())
    maximum = zero_error.loc[zero_error["AcceptedCount"].eq(maximum_count)].copy()
    selected = maximum.sort_values(
        ["StrategyComplexity", "StrategyID"], kind="mergesort"
    ).iloc[0]
    context = {
        "zero_error_solution_count": int(len(zero_error)),
        "maximum_zero_error_accepted_count": maximum_count,
        "strategies_at_maximum_zero_error_coverage": maximum["Strategy"].tolist(),
        "selection_rule": (
            "maximum nonempty zero-error accepted count, then lowest strategy "
            "complexity, then StrategyID"
        ),
    }
    return selected, context


def choose_balanced_rule(envelope: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    # search only the prespecified 70 to 80 percent validation coverage window
    window = envelope.loc[
        envelope["Coverage"].between(
            BALANCED_MIN_COVERAGE, BALANCED_MAX_COVERAGE, inclusive="both"
        )
    ].copy()
    if window.empty:
        raise SafetyError("No validation envelope rows occur in the balanced window.")
    # errors come first, then false negatives, then coverage
    minimum_errors = int(window["AcceptedErrors"].min())
    error_best = window.loc[window["AcceptedErrors"].eq(minimum_errors)].copy()
    minimum_fn = int(error_best["FalseNegativesAmongAccepted"].min())
    error_fn_best = error_best.loc[
        error_best["FalseNegativesAmongAccepted"].eq(minimum_fn)
    ].copy()
    maximum_count = int(error_fn_best["AcceptedCount"].max())
    maximum_coverage = error_fn_best.loc[
        error_fn_best["AcceptedCount"].eq(maximum_count)
    ].copy()
    source_selected = maximum_coverage.sort_values(
        ["StrategyComplexity", "AcceptedBalancedAccuracy", "StrategyID"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    if not (
        str(source_selected["StrategyID"]) == "full_reliability"
        and bool(source_selected["RequireModelAgreement"])
        and bool(source_selected["UseViewSD"])
        and bool(source_selected["UseFeatureTypicality"])
        and not bool(source_selected["FeatureGateActive"])
    ):
        raise SafetyError(
            "The maximum-coverage one-error frontier row is not the expected "
            "full rule with a nonbinding feature gate."
        )

    # the feature cutoff did not change the accepted set
    # freeze the exact matched rule without that unused feature gate
    selected = source_selected.copy()
    selected["StrategyID"] = "confidence_model_agreement_view_sd"
    selected["Strategy"] = "Confidence + model agreement + view SD"
    selected["StrategyComplexity"] = 3
    selected["UseFeatureTypicality"] = False
    selected["FeaturePercentileThreshold"] = np.nan
    selected["FeatureGateActive"] = False
    selected["AcceptanceRule"] = (
        "PrimaryConfidenceMargin >= ConfidenceThreshold AND "
        "ModelAgreement == True AND "
        "Day3StdViewProbability <= ViewSDThreshold"
    )
    context = {
        "coverage_window": [BALANCED_MIN_COVERAGE, BALANCED_MAX_COVERAGE],
        "minimum_accepted_errors_in_window": minimum_errors,
        "minimum_false_negatives_among_minimum_error_solutions": minimum_fn,
        "maximum_accepted_count_with_those_results": maximum_count,
        "source_frontier_solution": {
            "strategy": str(source_selected["Strategy"]),
            "strategy_id": str(source_selected["StrategyID"]),
            "accepted_count": int(source_selected["AcceptedCount"]),
            "coverage": float(source_selected["Coverage"]),
            "accepted_errors": int(source_selected["AcceptedErrors"]),
            "false_negatives": int(source_selected["FalseNegativesAmongAccepted"]),
            "feature_percentile_threshold": float(
                source_selected["FeaturePercentileThreshold"]
            ),
            "feature_gate_active": bool(source_selected["FeatureGateActive"]),
        },
        "frozen_matched_ablation": "Confidence + model agreement + view SD",
        "next_count_envelope_solution_error_count": int(
            envelope.loc[
                envelope["StrategyID"].eq("full_reliability")
                & envelope["AcceptedCount"].eq(maximum_count + 1),
                "AcceptedErrors",
            ].iloc[0]
        ),
        "envelope_masks_may_be_nonnested": True,
        "simpler_nearby_rule": {
            "strategy": "Confidence + view SD",
            "accepted_count": 89,
            "coverage": 89 / EXPECTED_EYES,
            "accepted_errors": 1,
            "false_negatives": 1,
        },
        "selection_rule": (
            "within 70%-80% coverage: minimum accepted errors, then minimum "
            "accepted false negatives, then maximum coverage; remove the "
            "nonbinding feature-typicality gate to freeze the exact matched "
            "simpler ablation"
        ),
    }
    return selected, context


def as_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def make_acceptance_mask(reliability: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    # replay the frozen rule directly from its saved thresholds
    thresholds = rule["thresholds"]
    accepted = reliability["PrimaryConfidenceMargin"].to_numpy(float) >= float(
        thresholds["confidence_threshold"]
    )
    if bool(thresholds["model_agreement_required"]):
        accepted &= reliability["ModelAgreement"].to_numpy(bool)
    view_sd_threshold = thresholds["view_sd_threshold"]
    if view_sd_threshold is not None:
        accepted &= reliability["Day3StdViewProbability"].to_numpy(float) <= float(
            view_sd_threshold
        )
    feature_threshold = thresholds["feature_percentile_threshold"]
    if feature_threshold is not None:
        accepted &= reliability["PredictedClassReferencePercentile"].to_numpy(
            float
        ) <= float(feature_threshold)
    return accepted.astype(bool)


def accepted_ids_sha256(reliability: pd.DataFrame, accepted: np.ndarray) -> str:
    # hash the sorted accepted eye ids so the accepted set itself is locked
    identifiers = sorted(reliability.loc[accepted, "EyeExamID"].astype(str).tolist())
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode("utf-8")).hexdigest()


def check_number_match(observed: Any, expected: Any, name: str) -> None:
    observed_float = float(observed)
    expected_float = float(expected)
    if math.isnan(expected_float):
        if not math.isnan(observed_float):
            raise SafetyError(f"{name}: expected NaN, observed {observed_float}.")
    elif not math.isclose(observed_float, expected_float, rel_tol=0.0, abs_tol=1e-15):
        raise SafetyError(
            f"{name}: observed={observed_float:.17g}, expected={expected_float:.17g}."
        )


def make_frozen_rule(
    operating_point_id: str,
    operating_point_name: str,
    selected: pd.Series,
    reliability: pd.DataFrame,
    selection_rationale: str,
    day6_source_frontier_strategy_id: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    # store both the thresholds and the validation performance they reproduce
    rule = {
        "operating_point_id": operating_point_id,
        "operating_point_name": operating_point_name,
        "strategy_id": str(selected["StrategyID"]),
        "strategy_name": str(selected["Strategy"]),
        "day6_source_frontier_strategy_id": (
            day6_source_frontier_strategy_id or str(selected["StrategyID"])
        ),
        "acceptance_rule": str(selected["AcceptanceRule"]),
        "thresholds": {
            "confidence_threshold": float(selected["ConfidenceThreshold"]),
            "model_agreement_required": bool(selected["RequireModelAgreement"]),
            "view_sd_threshold": as_optional_float(selected["ViewSDThreshold"]),
            "feature_percentile_threshold": as_optional_float(
                selected["FeaturePercentileThreshold"]
            ),
        },
        "selection_rationale": selection_rationale,
    }
    accepted = make_acceptance_mask(reliability, rule)
    metrics = calculate_selective_metrics(
        reliability["Day4PredictedLabel"].astype(str).to_numpy(),
        reliability["TrueLabel"].astype(str).to_numpy(),
        accepted,
    )
    expected_metric_map = {
        "AcceptedCount": "AcceptedCount",
        "DeferredCount": "DeferredCount",
        "Coverage": "Coverage",
        "ReviewRate": "ReviewRate",
        "AcceptedErrorRate": "AcceptedErrorRate",
        "AcceptedAccuracy": "AcceptedAccuracy",
        "AcceptedSensitivity": "AcceptedSensitivity",
        "AcceptedSpecificity": "AcceptedSpecificity",
        "AcceptedBalancedAccuracy": "AcceptedBalancedAccuracy",
        "FalseNegativesAmongAccepted": "FalseNegativesAmongAccepted",
        "AcceptedTruePositives": "AcceptedTruePositives",
        "AcceptedTrueNegatives": "AcceptedTrueNegatives",
        "AcceptedFalsePositives": "AcceptedFalsePositives",
        "AcceptedFalseNegatives": "AcceptedFalseNegatives",
    }
    for envelope_field, metric_field in expected_metric_map.items():
        check_number_match(
            metrics[metric_field],
            selected[envelope_field],
            f"{operating_point_id} {metric_field}",
        )
    accepted_errors = int(metrics["AcceptedFalsePositives"]) + int(
        metrics["AcceptedFalseNegatives"]
    )
    if accepted_errors != int(selected["AcceptedErrors"]):
        raise SafetyError(f"{operating_point_id}: accepted-error count did not replay.")

    rule["validation_performance"] = {
        "total_eyes": int(metrics["TotalCount"]),
        "accepted_eyes": int(metrics["AcceptedCount"]),
        "deferred_eyes": int(metrics["DeferredCount"]),
        "coverage": float(metrics["Coverage"]),
        "coverage_percent": 100.0 * float(metrics["Coverage"]),
        "review_rate": float(metrics["ReviewRate"]),
        "review_rate_percent": 100.0 * float(metrics["ReviewRate"]),
        "accepted_error_rate": float(metrics["AcceptedErrorRate"]),
        "accepted_error_percent": 100.0 * float(metrics["AcceptedErrorRate"]),
        "accepted_error_count": accepted_errors,
        "accepted_accuracy": float(metrics["AcceptedAccuracy"]),
        "accepted_sensitivity": float(metrics["AcceptedSensitivity"]),
        "accepted_specificity": float(metrics["AcceptedSpecificity"]),
        "accepted_balanced_accuracy": float(metrics["AcceptedBalancedAccuracy"]),
        "accepted_false_negatives": int(metrics["FalseNegativesAmongAccepted"]),
        "accepted_true_positives": int(metrics["AcceptedTruePositives"]),
        "accepted_true_negatives": int(metrics["AcceptedTrueNegatives"]),
        "accepted_false_positives": int(metrics["AcceptedFalsePositives"]),
        "accepted_false_negatives_confusion": int(
            metrics["AcceptedFalseNegatives"]
        ),
        "accepted_eyeexamid_sha256": accepted_ids_sha256(reliability, accepted),
    }
    return rule, accepted


def is_on_pareto_frontier(rule: dict[str, Any], pareto: pd.DataFrame) -> bool:
    # confirm the frozen rule corresponds to the locked day-6 frontier
    performance = rule["validation_performance"]
    thresholds = rule["thresholds"]
    candidates = pareto.loc[
        pareto["StrategyID"].eq(rule["day6_source_frontier_strategy_id"])
        & pareto["AcceptedCount"].eq(performance["accepted_eyes"])
    ]
    for _, row in candidates.iterrows():
        if not math.isclose(
            float(row["ConfidenceThreshold"]),
            float(thresholds["confidence_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        row_sd = as_optional_float(row["ViewSDThreshold"])
        if row_sd != thresholds["view_sd_threshold"]:
            continue
        if (
            bool(row["UseFeatureTypicality"])
            and as_optional_float(row["FeaturePercentileThreshold"]) is not None
            and bool(row["FeatureGateActive"])
        ):
            continue
        return True
    return False


def make_rules_table(rules: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        thresholds = rule["thresholds"]
        performance = rule["validation_performance"]
        rows.append(
            {
                "OperatingPointID": rule["operating_point_id"],
                "OperatingPointName": rule["operating_point_name"],
                "StrategyID": rule["strategy_id"],
                "StrategyName": rule["strategy_name"],
                "Day6SourceFrontierStrategyID": rule[
                    "day6_source_frontier_strategy_id"
                ],
                "AcceptanceRule": rule["acceptance_rule"],
                "ConfidenceThreshold": thresholds["confidence_threshold"],
                "ModelAgreementRequired": thresholds[
                    "model_agreement_required"
                ],
                "ViewSDThreshold": thresholds["view_sd_threshold"],
                "FeaturePercentileThreshold": thresholds[
                    "feature_percentile_threshold"
                ],
                "ValidationTotalEyes": performance["total_eyes"],
                "ValidationAcceptedEyes": performance["accepted_eyes"],
                "ValidationDeferredEyes": performance["deferred_eyes"],
                "ValidationCoverage": performance["coverage"],
                "ValidationCoveragePercent": performance["coverage_percent"],
                "ValidationReviewRate": performance["review_rate"],
                "ValidationReviewRatePercent": performance["review_rate_percent"],
                "ValidationAcceptedErrorRate": performance[
                    "accepted_error_rate"
                ],
                "ValidationAcceptedErrorPercent": performance[
                    "accepted_error_percent"
                ],
                "ValidationAcceptedErrorCount": performance[
                    "accepted_error_count"
                ],
                "ValidationAcceptedAccuracy": performance["accepted_accuracy"],
                "ValidationAcceptedSensitivity": performance[
                    "accepted_sensitivity"
                ],
                "ValidationAcceptedSpecificity": performance[
                    "accepted_specificity"
                ],
                "ValidationAcceptedBalancedAccuracy": performance[
                    "accepted_balanced_accuracy"
                ],
                "ValidationAcceptedFalseNegatives": performance[
                    "accepted_false_negatives"
                ],
                "ValidationAcceptedEyeExamIDSHA256": performance[
                    "accepted_eyeexamid_sha256"
                ],
                "SelectionRationale": rule["selection_rationale"],
                "Frozen": True,
                "TestDerivedValuesUsed": False,
            }
        )
    return pd.DataFrame(rows)


def serialize_rules_csv(frame: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    frame.to_csv(
        stream,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.17g",
        lineterminator="\n",
    )
    return stream.getvalue()


def stable_lock_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(payload))
    stable.pop("frozen_at", None)
    stable.pop("freezer_provenance", None)
    return stable


def write_or_check_lock(
    payload: dict[str, Any], summary_frame: pd.DataFrame
) -> str:
    # both frozen files must either exist together or be created together
    json_exists = FROZEN_JSON_PATH.exists()
    csv_exists = FROZEN_CSV_PATH.exists()
    if json_exists != csv_exists:
        raise SafetyError(
            "The frozen routing output is partially present; refusing to alter the lock."
        )
    csv_text = serialize_rules_csv(summary_frame)
    # reruns may only reproduce the existing lock exactly
    if json_exists:
        with FROZEN_JSON_PATH.open("r", encoding="utf-8") as stream:
            existing_payload = json.load(stream)
        if stable_lock_payload(existing_payload) != stable_lock_payload(payload):
            raise SafetyError(
                "Recomputed frozen routing rules differ from the existing lock; "
                "existing files were not overwritten."
            )
        existing_csv = pd.read_csv(
            FROZEN_CSV_PATH, float_precision="round_trip"
        )
        expected_csv = pd.read_csv(io.StringIO(csv_text), float_precision="round_trip")
        try:
            pd.testing.assert_frame_equal(
                existing_csv,
                expected_csv,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise SafetyError(
                "Recomputed human-readable summary differs from the existing lock; "
                "existing files were not overwritten."
            ) from error
        return "REPRODUCIBLE_EXISTING_LOCK"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created_in_this_attempt: list[Path] = []
    try:
        with FROZEN_CSV_PATH.open("x", encoding="utf-8", newline="") as stream:
            created_in_this_attempt.append(FROZEN_CSV_PATH)
            stream.write(csv_text)
        with FROZEN_JSON_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            created_in_this_attempt.append(FROZEN_JSON_PATH)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        for created_path in reversed(created_in_this_attempt):
            created_path.unlink(missing_ok=True)
        raise
    return "CREATED_NEW_LOCK"


def replay_rules(
    reliability: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    # always replay the serialized rules after writing or verifying the lock
    with FROZEN_JSON_PATH.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    summary = pd.read_csv(FROZEN_CSV_PATH, float_precision="round_trip")
    rows: list[pd.DataFrame] = []
    for rule in payload.get("operating_points", []):
        accepted = make_acceptance_mask(reliability, rule)
        metrics = calculate_selective_metrics(
            reliability["Day4PredictedLabel"].astype(str).to_numpy(),
            reliability["TrueLabel"].astype(str).to_numpy(),
            accepted,
        )
        performance = rule["validation_performance"]
        comparisons = {
            "AcceptedCount": (metrics["AcceptedCount"], performance["accepted_eyes"]),
            "DeferredCount": (metrics["DeferredCount"], performance["deferred_eyes"]),
            "Coverage": (metrics["Coverage"], performance["coverage"]),
            "ReviewRate": (metrics["ReviewRate"], performance["review_rate"]),
            "AcceptedErrorRate": (
                metrics["AcceptedErrorRate"],
                performance["accepted_error_rate"],
            ),
            "AcceptedSensitivity": (
                metrics["AcceptedSensitivity"],
                performance["accepted_sensitivity"],
            ),
            "AcceptedSpecificity": (
                metrics["AcceptedSpecificity"],
                performance["accepted_specificity"],
            ),
            "AcceptedBalancedAccuracy": (
                metrics["AcceptedBalancedAccuracy"],
                performance["accepted_balanced_accuracy"],
            ),
            "FalseNegativesAmongAccepted": (
                metrics["FalseNegativesAmongAccepted"],
                performance["accepted_false_negatives"],
            ),
        }
        for name, (observed, expected) in comparisons.items():
            check_number_match(
                observed,
                expected,
                f"serialized {rule['operating_point_id']} {name}",
            )
        replay_id_hash = accepted_ids_sha256(reliability, accepted)
        if replay_id_hash != performance["accepted_eyeexamid_sha256"]:
            raise SafetyError(
                f"Serialized {rule['operating_point_id']} accepted-ID set did not replay."
            )
        block = reliability[
            [
                "EyeExamID",
                "ResearchSubjectID",
                "EncounterID",
                "Laterality",
                "TrueLabel",
                "Day4PredictedLabel",
                "Day4EyeProbability",
                "PrimaryConfidenceMargin",
                "ModelAgreement",
                "Day3StdViewProbability",
                "PredictedClassReferencePercentile",
            ]
        ].copy()
        block.insert(0, "OperatingPointID", rule["operating_point_id"])
        block.insert(1, "StrategyName", rule["strategy_name"])
        block["Accepted"] = accepted
        block["Deferred"] = ~accepted
        block["RoutingDecision"] = np.where(accepted, "ACCEPT", "DEFER")
        block["AcceptedPredictionCorrect"] = np.where(
            accepted,
            block["TrueLabel"].eq(block["Day4PredictedLabel"]),
            pd.NA,
        )
        rows.append(block)
    if len(rows) != 2:
        raise SafetyError("Frozen JSON must contain exactly two operating points.")
    return payload, summary, pd.concat(rows, ignore_index=True)


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


def main() -> None:
    # selection and freezing use validation artifacts only
    checks: list[dict[str, Any]] = []
    # verify every upstream hash before reading the selection inputs
    observed_hashes = {
        str(path): check_locked_input(path) for path in EXPECTED_HASHES
    }
    reliability = read_allowed_csv(DAY5_RELIABILITY_PATH)
    envelope = read_allowed_csv(DAY6_ENVELOPE_PATH)
    pareto = read_allowed_csv(DAY6_PARETO_PATH)
    day6_metadata = read_allowed_json(DAY6_METADATA_PATH)
    reliability, envelope, pareto = check_inputs(
        reliability, envelope, pareto, day6_metadata, checks
    )

    # choose the two operating points from the locked day-6 envelope
    conservative_row, conservative_context = choose_conservative_rule(envelope)
    balanced_row, balanced_context = choose_balanced_rule(envelope)
    conservative_rule, conservative_accepted = make_frozen_rule(
        "conservative_zero_error",
        "Conservative zero-error rule",
        conservative_row,
        reliability,
        (
            "Simplest strategy at the maximum observed nonempty zero-error "
            "validation coverage."
        ),
    )
    balanced_rule, balanced_accepted = make_frozen_rule(
        "balanced_agreement_view_sd",
        "Balanced agreement and view-SD rule",
        balanced_row,
        reliability,
        (
            "Within the Day-6 validation-optimized envelope, 90/120 is the "
            "highest accepted count between 70% and 80% coverage having one "
            "accepted error and one accepted false negative. The feature gate "
            "was nonbinding, so this is the exact matched no-feature ablation. "
            "The separately optimized 91/120 envelope solution has two accepted "
            "errors. Envelope masks are nonnested, so this is not interpreted "
            "as a one-eye incremental transition."
        ),
        day6_source_frontier_strategy_id="full_reliability",
    )
    balanced_source_row = envelope.loc[
        envelope["StrategyID"].eq("full_reliability")
        & envelope["AcceptedCount"].eq(
            balanced_rule["validation_performance"]["accepted_eyes"]
        )
    ].iloc[0]
    balanced_source_rule = {
        "thresholds": {
            "confidence_threshold": float(
                balanced_source_row["ConfidenceThreshold"]
            ),
            "model_agreement_required": bool(
                balanced_source_row["RequireModelAgreement"]
            ),
            "view_sd_threshold": as_optional_float(
                balanced_source_row["ViewSDThreshold"]
            ),
            "feature_percentile_threshold": as_optional_float(
                balanced_source_row["FeaturePercentileThreshold"]
            ),
        }
    }
    # prove that removing the nonbinding feature gate leaves the exact same mask
    balanced_source_accepted = make_acceptance_mask(reliability, balanced_source_rule)
    rules = [conservative_rule, balanced_rule]

    record_check(
        checks,
        "Conservative rule maximizes observed zero-error coverage",
        conservative_rule["validation_performance"]["accepted_eyes"]
        == conservative_context["maximum_zero_error_accepted_count"],
        conservative_rule["validation_performance"]["accepted_eyes"],
        conservative_context["maximum_zero_error_accepted_count"],
    )
    record_check(
        checks,
        "Conservative rule uses the simplest tied strategy",
        conservative_rule["strategy_id"] == "confidence_only",
        conservative_rule["strategy_id"],
        "confidence_only",
    )
    record_check(
        checks,
        "Conservative frozen confidence threshold",
        math.isclose(
            conservative_rule["thresholds"]["confidence_threshold"],
            EXPECTED_CONSERVATIVE_CONFIDENCE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        conservative_rule["thresholds"]["confidence_threshold"],
        EXPECTED_CONSERVATIVE_CONFIDENCE_THRESHOLD,
    )
    record_check(
        checks,
        "Conservative accepted-ID set",
        accepted_ids_sha256(reliability, conservative_accepted)
        == EXPECTED_CONSERVATIVE_ACCEPTED_ID_SHA256,
        accepted_ids_sha256(reliability, conservative_accepted),
        EXPECTED_CONSERVATIVE_ACCEPTED_ID_SHA256,
    )
    record_check(
        checks,
        "Balanced rule is inside the prespecified coverage window",
        BALANCED_MIN_COVERAGE
        <= balanced_rule["validation_performance"]["coverage"]
        <= BALANCED_MAX_COVERAGE,
        balanced_rule["validation_performance"]["coverage"],
        f"{BALANCED_MIN_COVERAGE}..{BALANCED_MAX_COVERAGE}",
    )
    record_check(
        checks,
        "Balanced rule reaches the minimum-error maximum coverage",
        balanced_rule["validation_performance"]["accepted_eyes"]
        == balanced_context["maximum_accepted_count_with_those_results"],
        balanced_rule["validation_performance"]["accepted_eyes"],
        balanced_context["maximum_accepted_count_with_those_results"],
    )
    record_check(
        checks,
        "Balanced rule is the confidence-agreement-view-SD matched ablation",
        balanced_rule["strategy_id"] == "confidence_model_agreement_view_sd"
        and balanced_rule["thresholds"]["model_agreement_required"]
        and balanced_rule["thresholds"]["view_sd_threshold"] is not None
        and balanced_rule["thresholds"]["feature_percentile_threshold"] is None,
        balanced_rule["strategy_id"],
        "confidence_model_agreement_view_sd without feature typicality",
    )
    record_check(
        checks,
        "Balanced frozen thresholds",
        math.isclose(
            balanced_rule["thresholds"]["confidence_threshold"],
            EXPECTED_BALANCED_CONFIDENCE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(balanced_rule["thresholds"]["view_sd_threshold"]),
            EXPECTED_BALANCED_VIEW_SD_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        (
            balanced_rule["thresholds"]["confidence_threshold"],
            balanced_rule["thresholds"]["view_sd_threshold"],
        ),
        (
            EXPECTED_BALANCED_CONFIDENCE_THRESHOLD,
            EXPECTED_BALANCED_VIEW_SD_THRESHOLD,
        ),
    )
    record_check(
        checks,
        "Balanced accepted-ID set",
        accepted_ids_sha256(reliability, balanced_accepted)
        == EXPECTED_BALANCED_ACCEPTED_ID_SHA256,
        accepted_ids_sha256(reliability, balanced_accepted),
        EXPECTED_BALANCED_ACCEPTED_ID_SHA256,
    )
    record_check(
        checks,
        "Balanced matched ablation exactly reproduces the source full-rule mask",
        np.array_equal(balanced_accepted, balanced_source_accepted),
        int(np.logical_xor(balanced_accepted, balanced_source_accepted).sum()),
        0,
        "The removed feature cutoff was empirically nonbinding.",
    )
    record_check(
        checks,
        "Both finalized rules are on the locked Day-6 Pareto frontier",
        all(is_on_pareto_frontier(rule, pareto) for rule in rules),
        "both",
        "both",
    )
    record_check(
        checks,
        "Feature typicality is not required by either rule",
        all(
            rule["thresholds"]["feature_percentile_threshold"] is None
            for rule in rules
        ),
        True,
        True,
    )

    git_commit, git_status = get_git_info()
    frozen_at = datetime.now().astimezone().isoformat()
    # record the rules, validation results, upstream hashes, and provenance
    payload = {
        "schema_version": 1,
        "artifact_status": "FROZEN",
        "frozen_at": frozen_at,
        "selection_data_scope": "locked validation data only",
        "positive_class": "Abnormal",
        "upstream": {
            "day5_reliability_run_id": DAY5_RUN_ID,
            "day6_strategy_run_id": DAY6_RUN_ID,
            "validation_reliability_table": str(DAY5_RELIABILITY_PATH),
            "day6_risk_coverage_envelope": str(DAY6_ENVELOPE_PATH),
            "day6_pareto_frontier": str(DAY6_PARETO_PATH),
            "day6_routing_metadata": str(DAY6_METADATA_PATH),
            "sha256": observed_hashes,
        },
        "selection_policy": {
            "conservative": conservative_context,
            "balanced": balanced_context,
            "feature_typicality_incremental_benefit_observed": False,
            "feature_typicality_required": False,
            "view_probability_range_used": False,
        },
        "operating_points": rules,
        "feature_typicality_required_by_any_rule": False,
        "test_protection": {
            "test_derived_values_used": False,
            "test_data_loaded": False,
            "test_set_evaluated": False,
            "test_predictions_created": False,
            "loaded_input_paths": sorted(str(path) for path in LOADED_INPUT_PATHS),
        },
        "freezer_provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "utility_module": str(
                PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
            ),
            "utility_module_sha256": file_sha256(
                PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
            ),
            "git_commit_hash": git_commit,
            "git_worktree_status": git_status,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }
    # write a json lock plus a small human-readable csv summary
    summary_frame = make_rules_table(rules)
    lock_action = write_or_check_lock(payload, summary_frame)

    # replay from disk instead of trusting only the in-memory objects
    replay_payload, replay_summary, replay_predictions = replay_rules(reliability)
    record_check(
        checks,
        "Frozen JSON contains exactly two operating points",
        len(replay_payload.get("operating_points", [])) == 2,
        len(replay_payload.get("operating_points", [])),
        2,
    )
    record_check(
        checks,
        "Human-readable frozen summary contains exactly two operating points",
        len(replay_summary) == 2,
        len(replay_summary),
        2,
    )
    for frozen_rule in replay_payload["operating_points"]:
        performance = frozen_rule["validation_performance"]
        record_check(
            checks,
            (
                f"Serialized {frozen_rule['operating_point_id']} rule reproduces "
                "validation metrics and accepted IDs"
            ),
            True,
            (
                f"accepted={performance['accepted_eyes']}, "
                f"errors={performance['accepted_error_count']}, "
                f"FN={performance['accepted_false_negatives']}, "
                f"ID_SHA256={performance['accepted_eyeexamid_sha256']}"
            ),
            "exact replay",
        )
    stored_provenance = replay_payload.get("freezer_provenance", {})
    current_script_hash = file_sha256(Path(__file__).resolve())
    current_utility_hash = file_sha256(
        PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
    )
    record_check(
        checks,
        "Current freezer script matches frozen provenance",
        stored_provenance.get("script_sha256") == current_script_hash,
        current_script_hash,
        stored_provenance.get("script_sha256", "missing"),
    )
    record_check(
        checks,
        "Current metric utility matches frozen provenance",
        stored_provenance.get("utility_module_sha256") == current_utility_hash,
        current_utility_hash,
        stored_provenance.get("utility_module_sha256", "missing"),
    )
    record_check(
        checks,
        "Exactly the locked validation inputs were loaded",
        LOADED_INPUT_PATHS == set(EXPECTED_HASHES),
        sorted(str(path) for path in LOADED_INPUT_PATHS),
        sorted(str(path) for path in EXPECTED_HASHES),
    )
    record_check(
        checks,
        "No loaded input path is a test artifact",
        all(
            path.name.casefold() not in FORBIDDEN_TEST_NAMES
            and not path.name.casefold().startswith("test_")
            for path in LOADED_INPUT_PATHS
        ),
        True,
        True,
    )
    record_check(
        checks,
        "Locked upstream inputs remained unchanged",
        all(file_sha256(path) == expected for path, expected in EXPECTED_HASHES.items()),
        "all hashes unchanged",
        "all hashes unchanged",
    )

    # save a separate audit trail for each freezer execution
    audit_run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_OUTPUT_ROOT / audit_run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    checks_path = audit_dir / "frozen_routing_rule_checks.csv"
    replay_path = audit_dir / "frozen_routing_validation_replay.csv"
    selection_path = audit_dir / "frozen_routing_selection_context.json"
    pd.DataFrame(checks).to_csv(checks_path, index=False, quoting=csv.QUOTE_MINIMAL)
    replay_predictions.to_csv(
        replay_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.17g",
    )
    selection_context = {
        "audit_run_id": audit_run_id,
        "lock_action": lock_action,
        "conservative": conservative_context,
        "balanced": balanced_context,
        "frozen_json_sha256": file_sha256(FROZEN_JSON_PATH),
        "frozen_csv_sha256": file_sha256(FROZEN_CSV_PATH),
        "checks": str(checks_path),
        "validation_replay": str(replay_path),
        "test_data_loaded": False,
        "test_set_evaluated": False,
    }
    with selection_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(selection_context, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    print("\n" + "=" * 100)
    print("FROZEN SELECTIVE-ROUTING OPERATING POINTS")
    print("=" * 100)
    for rule in replay_payload["operating_points"]:
        thresholds = rule["thresholds"]
        performance = rule["validation_performance"]
        print(f"\n{rule['operating_point_name']}")
        print(f"  Strategy                    : {rule['strategy_name']}")
        print(
            "  Confidence threshold        : "
            f"{thresholds['confidence_threshold']:.17g}"
        )
        print(
            "  Model agreement required    : "
            f"{thresholds['model_agreement_required']}"
        )
        print(f"  View-SD threshold           : {thresholds['view_sd_threshold']}")
        print(
            "  Feature-percentile threshold: "
            f"{thresholds['feature_percentile_threshold']}"
        )
        print(
            "  Validation coverage / review: "
            f"{performance['coverage_percent']:.6f}% / "
            f"{performance['review_rate_percent']:.6f}%"
        )
        print(
            "  Accepted error              : "
            f"{performance['accepted_error_count']} "
            f"({performance['accepted_error_percent']:.6f}%)"
        )
        print(
            "  Sensitivity / specificity   : "
            f"{performance['accepted_sensitivity']:.6f} / "
            f"{performance['accepted_specificity']:.6f}"
        )
        print(
            "  Balanced accuracy / FN      : "
            f"{performance['accepted_balanced_accuracy']:.6f} / "
            f"{performance['accepted_false_negatives']}"
        )
    print(f"\nLock action       : {lock_action}")
    print(f"Frozen JSON       : {FROZEN_JSON_PATH}")
    print(f"Frozen CSV        : {FROZEN_CSV_PATH}")
    print(f"Validation replay : {replay_path}")
    print(f"Validation checks : {checks_path}")
    print("\nTEST SET WAS NOT LOADED OR EVALUATED.")


if __name__ == "__main__":
    try:
        main()
    except (SafetyError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
