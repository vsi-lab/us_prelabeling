

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DAY5_FEATURE_RUN_ID = "20260814T130502_173490-0700"
DAY5_FEATURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "multiview_resnet18"
    / DAY5_FEATURE_RUN_ID
).resolve()
TRAIN_EMBEDDINGS_PATH = (DAY5_FEATURE_DIR / "train_eye_embeddings.csv").resolve()
VAL_EMBEDDINGS_PATH = (DAY5_FEATURE_DIR / "val_eye_embeddings.csv").resolve()
REFERENCE_EYES_PATH = (DAY5_FEATURE_DIR / "reference_eye_distances.csv").resolve()
REFERENCE_CENTROIDS_PATH = (DAY5_FEATURE_DIR / "reference_centroids.npz").resolve()
REFERENCE_METADATA_PATH = (DAY5_FEATURE_DIR / "reference_metadata.json").resolve()

DAY5_RELIABILITY_RUN_ID = "20260814T141442_497529-0700"
VAL_RELIABILITY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "reliability"
    / DAY5_RELIABILITY_RUN_ID
    / "val_reliability_signals.csv"
).resolve()

OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "audits").resolve()

EXPECTED_HASHES = {
    TRAIN_EMBEDDINGS_PATH: "4c51fdfd5e4bcc5583974c43d9e0563105ebb04e18f019c65236a236d06b77d8",
    VAL_EMBEDDINGS_PATH: "22563615dd5e2d61ad0cdbdc8225084abd2a5a7a878333eec5d946f05bd22045",
    REFERENCE_EYES_PATH: "6a7146a54ccd41ae5703dd5a6dd9844ac02825c073b3f5b5ca6c83748b688b51",
    REFERENCE_CENTROIDS_PATH: "bd92d9eb1a0361a2154c153ec75adc702be41b64745da6e36730b595740b9f5d",
    REFERENCE_METADATA_PATH: "0b16d43864228a527bc16067ded2e7425f12f249f894e95b41fb3550615a51bd",
    VAL_RELIABILITY_PATH: "a767b163b3565a9fe664068c5fd18e96e4f13bf4dea7040ee179f098b477edb8",
}
ALLOWED_INPUT_PATHS = frozenset(EXPECTED_HASHES)
LOADED_INPUT_PATHS: set[Path] = set()

EXPECTED_TRAIN_EYES = 560
EXPECTED_VAL_EYES = 120
EXPECTED_TRAIN_SUBJECTS = 179
EXPECTED_VAL_SUBJECTS = 95
EXPECTED_FEATURE_DIMENSION = 512
EXPECTED_REFERENCE_COUNTS = {"Normal": 61, "Abnormal": 194}
EXPECTED_VAL_LABEL_COUNTS = {"Normal": 32, "Abnormal": 88}
EXPECTED_VAL_CORRECT = 110
EXPECTED_VAL_INCORRECT = 10
ALLOWED_LABELS = {"Normal", "Abnormal"}
K_VALUES = (3, 5, 10)
RISK_BANDS = (10, 20, 25, 30)
TARGET_COVERAGES = (50, 60, 70, 75, 80, 90)
DATA_SCOPE = "TRAIN + VALIDATION ONLY"
TEST_STATUS = "TEST SET NOT LOADED OR EVALUATED"
NORM_EPSILON = 1e-12
CENTROID_TOLERANCE = 1e-12
K_SIMILAR_AUROC_TOLERANCE = 0.03
K_SIMILAR_CAPTURE_TOLERANCE = 1


class SafetyError(RuntimeError):
    """used when a locked input, scientific, or test-protection check fails."""


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    require_agreement: bool
    use_view_sd: bool
    use_knn: bool
    complexity: int
    matched_base_strategy_id: str | None = None

    @property
    def acceptance_rule(self) -> str:
        clauses = ["PrimaryConfidenceMargin >= ConfidenceThreshold"]
        if self.require_agreement:
            clauses.append("ModelAgreement == True")
        if self.use_view_sd:
            clauses.append("Day3StdViewProbability <= ViewSDThreshold")
        if self.use_knn:
            clauses.append("SelectedKNNReferencePercentile <= KNNPercentileThreshold")
        return " AND ".join(clauses)


BASE_STRATEGIES = (
    StrategySpec("c", "C", False, False, False, 1),
    StrategySpec("c_agreement", "C + Agreement", True, False, False, 2),
    StrategySpec("c_view_sd", "C + ViewSD", False, True, False, 2),
    StrategySpec(
        "c_agreement_view_sd",
        "C + Agreement + ViewSD",
        True,
        True,
        False,
        3,
    ),
)
KNN_STRATEGIES = (
    StrategySpec("c_knn", "C + KNN", False, False, True, 2, "c"),
    StrategySpec(
        "c_agreement_knn",
        "C + Agreement + KNN",
        True,
        False,
        True,
        3,
        "c_agreement",
    ),
    StrategySpec(
        "c_view_sd_knn",
        "C + ViewSD + KNN",
        False,
        True,
        True,
        3,
        "c_view_sd",
    ),
    StrategySpec(
        "c_agreement_view_sd_knn",
        "C + Agreement + ViewSD + KNN",
        True,
        True,
        True,
        4,
        "c_agreement_view_sd",
    ),
)
STRATEGIES = BASE_STRATEGIES + KNN_STRATEGIES


TRAIN_REQUIRED_COLUMNS = {
    "EyeExamID",
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "TrueLabel",
    "PredictedLabel",
    "EyeAbnormalProbability",
    "Correct",
    "PrimaryConfidenceMargin",
    "ReferenceSelected",
    "ReferenceRankWithinTrueClass",
    "DistanceToNormalCentroid",
    "DistanceToAbnormalCentroid",
}
VAL_REQUIRED_COLUMNS = {
    "EyeExamID",
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "TrueLabel",
    "PredictedLabel",
    "EyeAbnormalProbability",
    "Correct",
    "PrimaryConfidenceMargin",
    "DistanceToNormalCentroid",
    "DistanceToAbnormalCentroid",
    "FeatureTypicalityPredictedClass",
    "PredictedClassReferencePercentile",
}
RELIABILITY_REQUIRED_COLUMNS = {
    "EyeExamID",
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "TrueLabel",
    "Day4EyeProbability",
    "Day4PredictedLabel",
    "Day4Correct",
    "PrimaryConfidenceMargin",
    "Day3StdViewProbability",
    "ModelAgreement",
    "FeatureTypicalityDistance",
    "PredictedClassReferencePercentile",
}
REFERENCE_REQUIRED_COLUMNS = {
    "Class",
    "ReferenceRankWithinTrueClass",
    "EyeExamID",
    "ResearchSubjectID",
    "PrimaryConfidenceMargin",
    "CosineDistanceToClassCentroid",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_test_path(path: Path) -> None:
    """reject any path that looks like it belongs to the held-out test set."""
    lowered = [part.casefold() for part in path.parts]
    name = path.name.casefold()
    if name.startswith("test_") or name.endswith("_test.csv") or any(
        part in {"test", "tests"} for part in lowered
    ):
        raise SafetyError(f"Held-out test artifact access was attempted: {path}")


def check_allowed_input(path: Path) -> Path:
    # every input must come from the fixed train and validation allowlist
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in ALLOWED_INPUT_PATHS:
        raise SafetyError(f"Input is outside the fixed train/validation allowlist: {resolved}")
    if not resolved.is_file():
        raise SafetyError(f"Required locked input is missing: {resolved}")
    observed = file_sha256(resolved)
    expected = EXPECTED_HASHES[resolved]
    if observed != expected:
        raise SafetyError(
            f"Locked input SHA-256 mismatch for {resolved}: "
            f"observed={observed}, expected={expected}."
        )
    LOADED_INPUT_PATHS.add(resolved)
    return resolved


def read_locked_csv(path: Path, string_columns: Iterable[str]) -> pd.DataFrame:
    resolved = check_allowed_input(path)
    dtype = {column: "string" for column in string_columns}
    return pd.read_csv(resolved, dtype=dtype, float_precision="round_trip")


def read_locked_json(path: Path) -> dict[str, Any]:
    resolved = check_allowed_input(path)
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"Expected a JSON object in {resolved}")
    return value


def read_locked_centroids(path: Path) -> dict[str, np.ndarray]:
    resolved = check_allowed_input(path)
    with np.load(resolved, allow_pickle=False) as archive:
        missing = ALLOWED_LABELS - set(archive.files)
        if missing:
            raise SafetyError(f"Reference centroid archive is missing: {sorted(missing)}")
        return {
            label: np.asarray(archive[label], dtype=np.float64).copy()
            for label in sorted(ALLOWED_LABELS)
        }


def parse_bool_column(series: pd.Series, name: str) -> pd.Series:
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
            "DataScope": DATA_SCOPE,
            "TestStatus": TEST_STATUS,
        }
    )
    if not passed:
        raise SafetyError(f"{name}: observed={observed}; expected={expected}. {details}")


def check_required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SafetyError(f"{name} is missing required columns: {missing}")


def normalize_features(features: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    # cosine comparisons use l2-normalized feature vectors
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SafetyError(f"{name} must be a finite two-dimensional feature matrix.")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= NORM_EPSILON):
        raise SafetyError(f"{name} contains non-finite or near-zero feature norms.")
    return values / norms[:, None], norms


def centroid_cosine_distance(features: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    similarities = np.clip(features @ centroid, -1.0, 1.0)
    return 1.0 - similarities


def reference_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    # percentile is the fraction of reference distances at or below the query value
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    queries = np.asarray(values, dtype=np.float64)
    return 100.0 * np.searchsorted(ordered, queries, side="right") / len(ordered)


def compute_error_auroc(truth_is_error: np.ndarray, risk_scores: np.ndarray) -> float:
    truth = np.asarray(truth_is_error, dtype=bool)
    scores = np.asarray(risk_scores, dtype=np.float64)
    if truth.ndim != 1 or scores.ndim != 1 or len(truth) != len(scores):
        raise ValueError("AUROC inputs must be equal-length one-dimensional arrays.")
    if not np.isfinite(scores).all() or truth.sum() == 0 or (~truth).sum() == 0:
        raise ValueError("AUROC requires finite scores and both outcome classes.")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    return float(
        (ranks[truth].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def compute_spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Spearman inputs must be equal-length one-dimensional arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must be finite.")
    xr = pd.Series(x).rank(method="average").to_numpy(float)
    yr = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def select_high_risk_indices(
    frame: pd.DataFrame,
    field: str,
    percentage: int,
    ascending: bool,
) -> np.ndarray:
    # use eye id as the deterministic tie-break inside each risk band
    count = int(math.ceil(len(frame) * percentage / 100.0))
    ranked = frame[["EyeExamID", field]].sort_values(
        [field, "EyeExamID"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    return ranked.index[:count].to_numpy(int)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "Mean": float(np.mean(array)),
        "Median": float(np.median(array)),
        "StandardDeviation": float(np.std(array, ddof=0)),
        "Percentile25": float(np.percentile(array, 25.0, method="linear")),
        "Percentile75": float(np.percentile(array, 75.0, method="linear")),
    }


def check_identity_mapping(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_predicted: str,
    right_predicted: str,
) -> tuple[bool, float]:
    columns = ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel"]
    a = left.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    b = right.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    identity_ok = a[columns].astype(str).equals(b[columns].astype(str))
    prediction_ok = a[left_predicted].astype(str).equals(b[right_predicted].astype(str))
    max_probability_difference = float(
        np.max(
            np.abs(
                a["EyeAbnormalProbability"].to_numpy(float)
                - b["Day4EyeProbability"].to_numpy(float)
            )
        )
    )
    return bool(identity_ok and prediction_ok), max_probability_difference


def compute_knn_scores(
    reference_features: dict[str, np.ndarray],
    validation_features: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    validation_results: dict[int, dict[str, np.ndarray]] = {
        k: {
            "distance": np.full(len(validation_features), np.nan, dtype=np.float64),
            "percentile": np.full(len(validation_features), np.nan, dtype=np.float64),
        }
        for k in K_VALUES
    }
    reference_distributions: dict[str, dict[int, np.ndarray]] = {}

    # compute local distances separately within each predicted class
    for label in sorted(ALLOWED_LABELS):
        refs = reference_features[label]
        # leave each reference eye out of its own neighborhood calibration
        pairwise = 1.0 - np.clip(refs @ refs.T, -1.0, 1.0)
        np.fill_diagonal(pairwise, np.inf)
        reference_distributions[label] = {}
        class_validation_indices = np.flatnonzero(predicted_labels == label)
        query_distances = 1.0 - np.clip(
            validation_features[class_validation_indices] @ refs.T,
            -1.0,
            1.0,
        )
        for k in K_VALUES:
            if len(refs) <= k:
                raise SafetyError(
                    f"{label} has {len(refs)} references, insufficient for leave-one-out k={k}."
                )
            ref_mean = np.partition(pairwise, kth=k - 1, axis=1)[:, :k].mean(axis=1)
            query_mean = np.partition(query_distances, kth=k - 1, axis=1)[:, :k].mean(axis=1)
            reference_distributions[label][k] = ref_mean
            validation_results[k]["distance"][class_validation_indices] = query_mean
            validation_results[k]["percentile"][class_validation_indices] = (
                reference_percentiles(ref_mean, query_mean)
            )

    for k in K_VALUES:
        for field, values in validation_results[k].items():
            if not np.isfinite(values).all():
                raise SafetyError(f"k={k} validation {field} contains NaN/inf.")
    return validation_results, reference_distributions


def compute_optional_mahalanobis(
    reference_features: dict[str, np.ndarray],
    validation_features: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray], str]:
    """optionally calculate ledoit-wolf mahalanobis distances when available."""
    try:
        from sklearn.covariance import LedoitWolf
    except Exception as error:  # environment-dependent fallback
        return None, {}, f"Skipped: scikit-learn LedoitWolf unavailable ({error})."

    distances = np.full(len(validation_features), np.nan, dtype=np.float64)
    percentiles = np.full(len(validation_features), np.nan, dtype=np.float64)
    reference_distances: dict[str, np.ndarray] = {}
    try:
        # fit one shrinkage covariance model per class
        for label in sorted(ALLOWED_LABELS):
            refs = reference_features[label]
            estimator = LedoitWolf(assume_centered=False, store_precision=True).fit(refs)
            centered_refs = refs - estimator.location_
            ref_sq = np.einsum(
                "ij,jk,ik->i", centered_refs, estimator.precision_, centered_refs
            )
            ref_distance = np.sqrt(np.maximum(ref_sq, 0.0))
            reference_distances[label] = ref_distance

            indices = np.flatnonzero(predicted_labels == label)
            centered_queries = validation_features[indices] - estimator.location_
            query_sq = np.einsum(
                "ij,jk,ik->i",
                centered_queries,
                estimator.precision_,
                centered_queries,
            )
            query_distance = np.sqrt(np.maximum(query_sq, 0.0))
            distances[indices] = query_distance
            percentiles[indices] = reference_percentiles(ref_distance, query_distance)
    except Exception as error:  # stable fallback is part of the protocol
        return None, {}, f"Skipped: Ledoit-Wolf calculation was unstable ({error})."

    if not np.isfinite(distances).all() or not np.isfinite(percentiles).all():
        return None, {}, "Skipped: Ledoit-Wolf produced non-finite values."
    if np.unique(percentiles).size < 2:
        return (
            None,
            {},
            "Skipped: the numerically stable Ledoit-Wolf fit yielded a degenerate in-sample-calibrated validation percentile distribution; a defensible out-of-fold calibration would add substantial methodology beyond this focused comparison.",
        )
    return (
        {"distance": distances, "percentile": percentiles},
        reference_distances,
        "Calculated as an experimental secondary method using class-specific Ledoit-Wolf shrinkage covariance on L2-normalized references.",
    )


def make_analysis_row(**values: Any) -> dict[str, Any]:
    row = {
        "AnalysisType": "",
        "Method": "",
        "Measure": "",
        "PredictionGroup": "",
        "RiskBandPercent": math.nan,
        "Count": math.nan,
        "Mean": math.nan,
        "Median": math.nan,
        "StandardDeviation": math.nan,
        "Percentile25": math.nan,
        "Percentile75": math.nan,
        "ErrorDetectionAUROC": math.nan,
        "CapturedErrors": math.nan,
        "TotalErrors": EXPECTED_VAL_INCORRECT,
        "ErrorCapturePercent": math.nan,
        "CorrelationSignal": "",
        "SpearmanRho": math.nan,
        "UniqueErrorsBeyondConfidenceAndViewSD": math.nan,
        "Notes": "",
        "DataScope": DATA_SCOPE,
        "TestStatus": TEST_STATUS,
    }
    row.update(values)
    return row


def make_error_analysis(
    comparison: pd.DataFrame,
    methods: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    rows: list[dict[str, Any]] = []
    # treat an incorrect day-4 prediction as the positive error outcome
    is_error = ~comparison["Day4Correct"].to_numpy(bool)
    method_summary: dict[str, dict[str, float | int]] = {}

    for method, (distance_field, percentile_field) in methods.items():
        for measure, field in (("Distance", distance_field), ("ReferencePercentile", percentile_field)):
            for group_name, group_mask in (
                ("Correct", ~is_error),
                ("Incorrect", is_error),
            ):
                statistics = summarize_values(comparison.loc[group_mask, field].to_numpy(float))
                rows.append(
                    make_analysis_row(
                        AnalysisType="GroupSummary",
                        Method=method,
                        Measure=measure,
                        PredictionGroup=group_name,
                        Count=int(group_mask.sum()),
                        **statistics,
                        Notes="SD is population SD (ddof=0), matching the existing Day-5 convention; percentiles use NumPy linear interpolation.",
                    )
                )

        risk = comparison[percentile_field].to_numpy(float)
        auc = compute_error_auroc(is_error, risk)
        method_summary[method] = {"ErrorDetectionAUROC": auc}
        rows.append(
            make_analysis_row(
                AnalysisType="ErrorDetection",
                Method=method,
                Measure="ReferencePercentile",
                Count=len(comparison),
                ErrorDetectionAUROC=auc,
                Notes="Incorrect Day-4 prediction is the positive outcome; higher atypicality is higher risk.",
            )
        )

        for signal_name, signal_field in (
            ("PrimaryConfidenceMargin", "PrimaryConfidenceMargin"),
            ("Day3StdViewProbability", "Day3StdViewProbability"),
        ):
            rho = compute_spearman(
                risk, comparison[signal_field].to_numpy(float)
            )
            rows.append(
                make_analysis_row(
                    AnalysisType="Correlation",
                    Method=method,
                    Measure="ReferencePercentile",
                    Count=len(comparison),
                    CorrelationSignal=signal_name,
                    SpearmanRho=rho,
                    Notes="Average ranks; no imputation and no inferential p-value.",
                )
            )

        for band in RISK_BANDS:
            indices = select_high_risk_indices(
                comparison, percentile_field, band, ascending=False
            )
            captured = int(is_error[indices].sum())
            method_summary[method][f"Top{band}PercentCapturedErrors"] = captured

            confidence_indices = select_high_risk_indices(
                comparison, "PrimaryConfidenceMargin", band, ascending=True
            )
            view_indices = select_high_risk_indices(
                comparison, "Day3StdViewProbability", band, ascending=False
            )
            baseline_error_indices = {
                int(index)
                for index in np.concatenate([confidence_indices, view_indices])
                if is_error[int(index)]
            }
            method_error_indices = {int(index) for index in indices if is_error[int(index)]}
            unique_errors = len(method_error_indices - baseline_error_indices)
            rows.append(
                make_analysis_row(
                    AnalysisType="ErrorCapture",
                    Method=method,
                    Measure="ReferencePercentile",
                    RiskBandPercent=band,
                    Count=len(indices),
                    CapturedErrors=captured,
                    ErrorCapturePercent=100.0 * captured / int(is_error.sum()),
                    UniqueErrorsBeyondConfidenceAndViewSD=unique_errors,
                    Notes=(
                        "Exact ceil(N*band/100) validation eyes; risk score descending then EyeExamID ascending. "
                        "Incremental errors are absent from the union of equally sized low-confidence and high-view-SD bands."
                    ),
                )
            )

    return pd.DataFrame(rows), method_summary


def choose_knn_k(method_summary: dict[str, dict[str, float | int]]) -> tuple[int, bool, str]:
    # prefer k=5 when the tested k values are practically similar
    aucs = {k: float(method_summary[f"knn_k{k}"]["ErrorDetectionAUROC"]) for k in K_VALUES}
    captures = {
        k: int(method_summary[f"knn_k{k}"]["Top25PercentCapturedErrors"])
        for k in K_VALUES
    }
    similar = (
        max(aucs.values()) - min(aucs.values()) <= K_SIMILAR_AUROC_TOLERANCE
        and max(captures.values()) - min(captures.values()) <= K_SIMILAR_CAPTURE_TOLERANCE
    )
    if similar:
        return (
            5,
            True,
            "k=3,5,10 met the prespecified similarity rubric (AUROC spread <=0.03 and top-25% error-capture spread <=1); k=5 was preferred without aggressive validation tuning.",
        )
    selected = max(
        K_VALUES,
        key=lambda k: (aucs[k], captures[k], -abs(k - 5), -k),
    )
    return (
        selected,
        False,
        "k values did not meet the similarity rubric; selected lexicographically by error-detection AUROC, top-25% error capture, then closeness to k=5.",
    )


def bools_to_bitmask(values: np.ndarray) -> int:
    # bitmasks make repeated accepted-set comparisons cheaper
    mask = 0
    for index in np.flatnonzero(values):
        mask |= 1 << int(index)
    return mask


def bitmask_to_bools(mask: int, length: int) -> np.ndarray:
    return np.fromiter(
        (((mask >> index) & 1) == 1 for index in range(length)),
        dtype=bool,
        count=length,
    )


def make_threshold_masks(values: np.ndarray, comparator: str) -> list[tuple[float, int]]:
    # every observed validation value becomes a candidate routing cutoff
    finite = np.asarray(values, dtype=np.float64)
    if not np.isfinite(finite).all():
        raise SafetyError("Routing threshold values must be finite.")
    unique = np.unique(finite)
    if comparator == "ge":
        reject_all = float(np.nextafter(unique.max(), math.inf))
        return [
            (float(cutoff), bools_to_bitmask(finite >= cutoff))
            for cutoff in [*unique.tolist(), reject_all]
        ]
    if comparator == "le":
        reject_all = float(np.nextafter(unique.min(), -math.inf))
        return [
            (float(cutoff), bools_to_bitmask(finite <= cutoff))
            for cutoff in [reject_all, *unique.tolist()]
        ]
    raise ValueError(f"Unknown comparator: {comparator}")


def threshold_sort_key(
    spec: StrategySpec,
    confidence_threshold: float,
    view_sd_threshold: float | None,
    knn_threshold: float | None,
) -> tuple[float, ...]:
    key: list[float] = []
    if spec.use_knn:
        assert knn_threshold is not None
        key.append(-float(knn_threshold))
    if spec.use_view_sd:
        assert view_sd_threshold is not None
        key.append(-float(view_sd_threshold))
    key.append(float(confidence_threshold))
    return tuple(key)


def enumerate_rules(
    spec: StrategySpec,
    confidence_masks: list[tuple[float, int]],
    view_sd_masks: list[tuple[float, int]],
    knn_masks: list[tuple[float, int]],
    agreement_mask: int,
    all_mask: int,
) -> tuple[dict[int, dict[str, Any]], int]:
    unique_rules: dict[int, dict[str, Any]] = {}
    raw_count = 0
    sd_iterable = view_sd_masks if spec.use_view_sd else [(None, all_mask)]
    knn_iterable = knn_masks if spec.use_knn else [(None, all_mask)]
    fixed = agreement_mask if spec.require_agreement else all_mask

    for confidence_threshold, confidence_mask in confidence_masks:
        confidence_gate = confidence_mask & fixed
        for view_sd_threshold, view_sd_mask in sd_iterable:
            partial = confidence_gate & view_sd_mask
            for knn_threshold, knn_mask in knn_iterable:
                raw_count += 1
                accepted = partial & knn_mask
                candidate = {
                    "AcceptedBitmask": accepted,
                    "ConfidenceThreshold": float(confidence_threshold),
                    "ViewSDThreshold": view_sd_threshold,
                    "KNNPercentileThreshold": knn_threshold,
                }
                existing = unique_rules.get(accepted)
                candidate_key = threshold_sort_key(
                    spec, confidence_threshold, view_sd_threshold, knn_threshold
                )
                if existing is None:
                    unique_rules[accepted] = candidate
                    continue
                existing_key = threshold_sort_key(
                    spec,
                    float(existing["ConfidenceThreshold"]),
                    existing["ViewSDThreshold"],
                    existing["KNNPercentileThreshold"],
                )
                if candidate_key < existing_key:
                    unique_rules[accepted] = candidate
    return unique_rules, raw_count


def routing_counts(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, int | float]:
    truth = frame["TrueLabel"].astype(str).to_numpy()
    prediction = frame["Day4PredictedLabel"].astype(str).to_numpy()
    accepted_truth = truth[mask]
    accepted_prediction = prediction[mask]
    tp = int(((accepted_truth == "Abnormal") & (accepted_prediction == "Abnormal")).sum())
    tn = int(((accepted_truth == "Normal") & (accepted_prediction == "Normal")).sum())
    fp = int(((accepted_truth == "Normal") & (accepted_prediction == "Abnormal")).sum())
    fn = int(((accepted_truth == "Abnormal") & (accepted_prediction == "Normal")).sum())
    accepted = int(mask.sum())
    errors = fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced = (
        (sensitivity + specificity) / 2.0
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )
    return {
        "AcceptedCount": accepted,
        "AcceptedErrors": errors,
        "AcceptedErrorRate": errors / accepted if accepted else math.nan,
        "AcceptedAccuracy": 1.0 - errors / accepted if accepted else math.nan,
        "AcceptedSensitivity": sensitivity,
        "AcceptedSpecificity": specificity,
        "AcceptedBalancedAccuracy": balanced,
        "AcceptedTruePositives": tp,
        "AcceptedTrueNegatives": tn,
        "AcceptedFalsePositives": fp,
        "AcceptedFalseNegatives": fn,
        "FalseNegativesAmongAccepted": fn,
    }


def accepted_eye_ids(bitmask: int, eye_ids: list[str]) -> tuple[str, ...]:
    return tuple(
        eye_ids[index]
        for index in range(len(eye_ids))
        if ((bitmask >> index) & 1) == 1
    )


def build_envelope(
    spec: StrategySpec,
    frame: pd.DataFrame,
    confidence_masks: list[tuple[float, int]],
    view_sd_masks: list[tuple[float, int]],
    knn_masks: list[tuple[float, int]],
    agreement_mask: int,
    selected_k: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    n = len(frame)
    all_mask = (1 << n) - 1
    unique_rules, raw_count = enumerate_rules(
        spec,
        confidence_masks,
        view_sd_masks,
        knn_masks,
        agreement_mask,
        all_mask,
    )
    eye_ids = frame["EyeExamID"].astype(str).tolist()
    best_by_count: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    for accepted_bitmask, thresholds in unique_rules.items():
        accepted = bitmask_to_bools(accepted_bitmask, n)
        metrics = routing_counts(frame, accepted)
        balanced = float(metrics["AcceptedBalancedAccuracy"])
        objective: tuple[Any, ...] = (
            int(metrics["AcceptedErrors"]),
            int(metrics["AcceptedFalseNegatives"]),
            -balanced if math.isfinite(balanced) else math.inf,
            *threshold_sort_key(
                spec,
                float(thresholds["ConfidenceThreshold"]),
                thresholds["ViewSDThreshold"],
                thresholds["KNNPercentileThreshold"],
            ),
            accepted_eye_ids(accepted_bitmask, eye_ids),
        )
        accepted_count = int(metrics["AcceptedCount"])
        existing = best_by_count.get(accepted_count)
        payload = {**thresholds, **metrics}
        if existing is None or objective < existing[0]:
            best_by_count[accepted_count] = (objective, payload)

    max_sd = float(frame["Day3StdViewProbability"].max())
    max_knn = float(frame["SelectedKNNReferencePercentile"].max())
    rows: list[dict[str, Any]] = []
    for accepted_count in sorted(best_by_count):
        result = best_by_count[accepted_count][1]
        sd_threshold = result["ViewSDThreshold"]
        knn_threshold = result["KNNPercentileThreshold"]
        rows.append(
            {
                "StrategyID": spec.strategy_id,
                "Strategy": spec.display_name,
                "StrategyComplexity": spec.complexity,
                "MatchedBaseStrategyID": spec.matched_base_strategy_id or "",
                "AcceptanceRule": spec.acceptance_rule,
                "RequireModelAgreement": spec.require_agreement,
                "UseViewSD": spec.use_view_sd,
                "UseKNN": spec.use_knn,
                "SelectedK": selected_k,
                "ConfidenceThreshold": result["ConfidenceThreshold"],
                "ViewSDThreshold": sd_threshold,
                "KNNPercentileThreshold": knn_threshold,
                "ViewSDGateActive": bool(
                    spec.use_view_sd
                    and sd_threshold is not None
                    and float(sd_threshold) < max_sd
                ),
                "KNNGateActive": bool(
                    spec.use_knn
                    and knn_threshold is not None
                    and float(knn_threshold) < max_knn
                ),
                "TotalCount": n,
                "AcceptedCount": accepted_count,
                "DeferredCount": n - accepted_count,
                "Coverage": accepted_count / n,
                "CoveragePercent": 100.0 * accepted_count / n,
                "ReviewRate": (n - accepted_count) / n,
                "ReviewRatePercent": 100.0 * (n - accepted_count) / n,
                **{key: value for key, value in result.items() if key.startswith("Accepted") or key == "FalseNegativesAmongAccepted"},
                "AcceptedBitmask": int(result["AcceptedBitmask"]),
                "DataScope": DATA_SCOPE,
                "TestStatus": TEST_STATUS,
            }
        )
    envelope = pd.DataFrame(rows)
    return envelope, {
        "RawThresholdConfigurations": raw_count,
        "UniqueAcceptanceSets": len(unique_rules),
        "AttainableAcceptedCounts": len(best_by_count),
        "MaximumAcceptedCount": int(envelope["AcceptedCount"].max()),
    }


def replay_acceptance(frame: pd.DataFrame, row: pd.Series) -> np.ndarray:
    mask = frame["PrimaryConfidenceMargin"].to_numpy(float) >= float(
        row["ConfidenceThreshold"]
    )
    if bool(row["RequireModelAgreement"]):
        mask &= frame["ModelAgreement"].to_numpy(bool)
    if bool(row["UseViewSD"]):
        mask &= frame["Day3StdViewProbability"].to_numpy(float) <= float(
            row["ViewSDThreshold"]
        )
    if bool(row["UseKNN"]):
        mask &= frame["SelectedKNNReferencePercentile"].to_numpy(float) <= float(
            row["KNNPercentileThreshold"]
        )
    return mask


def select_target_candidates(
    envelopes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_percent in TARGET_COVERAGES:
        target_count = EXPECTED_VAL_EYES * target_percent / 100.0
        for spec in STRATEGIES:
            candidates = envelopes[spec.strategy_id].assign(
                _distance=lambda x: (x["AcceptedCount"] - target_count).abs(),
                _above=lambda x: x["AcceptedCount"].gt(target_count),
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
            rows.append(chosen)

    output = pd.DataFrame(rows)
    output["MatchedBaseAcceptedErrors"] = math.nan
    output["MatchedBaseFalseNegatives"] = math.nan
    output["ErrorDifferenceVsMatchedBase"] = math.nan
    output["FalseNegativeDifferenceVsMatchedBase"] = math.nan
    output["StrictImprovementAtMatchedAcceptedCount"] = False
    output["BaseMaxCoverageAtSameErrorConstraint"] = math.nan
    output["KNNMaxCoverageAtSameErrorConstraint"] = math.nan
    output["StrictHigherCoverageAtSameErrorConstraint"] = False
    output["AnyStrictKNNImprovement"] = False

    for index, row in output.loc[output["UseKNN"].eq(True)].iterrows():
        base_id = str(row["MatchedBaseStrategyID"])
        accepted_count = int(row["AcceptedCount"])
        base_match = envelopes[base_id].loc[
            envelopes[base_id]["AcceptedCount"].eq(accepted_count)
        ]
        if len(base_match) != 1:
            raise SafetyError(
                f"No unique matched base rule for {row['StrategyID']} at {accepted_count} eyes."
            )
        base_row = base_match.iloc[0]
        error_budget = int(row["AcceptedErrors"])
        base_budget = envelopes[base_id].loc[
            envelopes[base_id]["AcceptedErrors"].le(error_budget)
        ]
        knn_budget = envelopes[str(row["StrategyID"])].loc[
            envelopes[str(row["StrategyID"])]["AcceptedErrors"].le(error_budget)
        ]
        base_max = int(base_budget["AcceptedCount"].max())
        knn_max = int(knn_budget["AcceptedCount"].max())
        error_difference = int(base_row["AcceptedErrors"]) - int(row["AcceptedErrors"])
        fn_difference = int(base_row["AcceptedFalseNegatives"]) - int(
            row["AcceptedFalseNegatives"]
        )
        same_count_strict = error_difference > 0
        coverage_strict = knn_max > base_max
        output.loc[index, "MatchedBaseAcceptedErrors"] = int(base_row["AcceptedErrors"])
        output.loc[index, "MatchedBaseFalseNegatives"] = int(
            base_row["AcceptedFalseNegatives"]
        )
        output.loc[index, "ErrorDifferenceVsMatchedBase"] = error_difference
        output.loc[index, "FalseNegativeDifferenceVsMatchedBase"] = fn_difference
        output.loc[index, "StrictImprovementAtMatchedAcceptedCount"] = same_count_strict
        output.loc[index, "BaseMaxCoverageAtSameErrorConstraint"] = base_max / EXPECTED_VAL_EYES
        output.loc[index, "KNNMaxCoverageAtSameErrorConstraint"] = knn_max / EXPECTED_VAL_EYES
        output.loc[index, "StrictHigherCoverageAtSameErrorConstraint"] = coverage_strict
        output.loc[index, "AnyStrictKNNImprovement"] = same_count_strict or coverage_strict
    return output


def add_pareto_flag(envelope: pd.DataFrame) -> pd.DataFrame:
    output = envelope.copy()
    flags: list[bool] = []
    for _, row in output.iterrows():
        accepted = int(row["AcceptedCount"])
        risk = float(row["AcceptedErrorRate"])
        if accepted == 0 or not math.isfinite(risk):
            flags.append(False)
            continue
        dominated = False
        for _, other in output.iterrows():
            other_accepted = int(other["AcceptedCount"])
            other_risk = float(other["AcceptedErrorRate"])
            if other_accepted == 0 or not math.isfinite(other_risk):
                continue
            if (
                other_accepted >= accepted
                and other_risk <= risk
                and (other_accepted > accepted or other_risk < risk)
            ):
                dominated = True
                break
        flags.append(not dominated)
    output["ParetoFrontier"] = flags
    return output


def strict_benefit_summary(
    envelopes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, bool, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    any_strict = False
    strict_by_strategy: dict[str, bool] = {}
    for spec in KNN_STRATEGIES:
        base_id = str(spec.matched_base_strategy_id)
        base = envelopes[base_id].set_index("AcceptedCount")
        knn = envelopes[spec.strategy_id].set_index("AcceptedCount")
        common_counts = sorted(set(base.index) & set(knn.index))
        strict_counts = [
            int(count)
            for count in common_counts
            if int(knn.loc[count, "AcceptedErrors"])
            < int(base.loc[count, "AcceptedErrors"])
        ]
        maximum_error = max(
            int(base["AcceptedErrors"].max()), int(knn["AcceptedErrors"].max())
        )
        coverage_improvement_budgets: list[int] = []
        for budget in range(maximum_error + 1):
            base_eligible = base.loc[base["AcceptedErrors"].le(budget)]
            knn_eligible = knn.loc[knn["AcceptedErrors"].le(budget)]
            base_max = int(base_eligible.index.max()) if len(base_eligible) else 0
            knn_max = int(knn_eligible.index.max()) if len(knn_eligible) else 0
            if knn_max > base_max:
                coverage_improvement_budgets.append(budget)
        strict = bool(strict_counts or coverage_improvement_budgets)
        strict_by_strategy[spec.strategy_id] = strict
        any_strict = any_strict or strict
        rows.append(
            {
                "AnalysisType": "RoutingStrictBenefit",
                "Method": spec.strategy_id,
                "Measure": "MatchedBaseComparison",
                "PredictionGroup": "",
                "RiskBandPercent": math.nan,
                "Count": len(common_counts),
                "Mean": math.nan,
                "Median": math.nan,
                "StandardDeviation": math.nan,
                "Percentile25": math.nan,
                "Percentile75": math.nan,
                "ErrorDetectionAUROC": math.nan,
                "CapturedErrors": math.nan,
                "TotalErrors": EXPECTED_VAL_INCORRECT,
                "ErrorCapturePercent": math.nan,
                "CorrelationSignal": "",
                "SpearmanRho": math.nan,
                "UniqueErrorsBeyondConfidenceAndViewSD": math.nan,
                "Notes": (
                    f"Matched base={base_id}; strict lower-error accepted counts={strict_counts}; "
                    f"strict higher-coverage error budgets={coverage_improvement_budgets}; strict={strict}."
                ),
                "DataScope": DATA_SCOPE,
                "TestStatus": TEST_STATUS,
            }
        )
    return pd.DataFrame(rows), any_strict, strict_by_strategy


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    started = datetime.now().astimezone()
    run_id = started.strftime("%Y%m%dT%H%M%S_%f%z")
    output_dir = OUTPUT_ROOT / run_id
    checks: list[dict[str, Any]] = []

    # perform the path-only test-artifact guard before hashing or opening any
    # source. this also protects against an unexpected symlink replacement.
    for path in ALLOWED_INPUT_PATHS:
        check_not_test_path(path)
    source_hashes_before = {
        str(path): file_sha256(path) if path.is_file() else None
        for path in ALLOWED_INPUT_PATHS
    }
    train = read_locked_csv(
        TRAIN_EMBEDDINGS_PATH,
        ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel", "PredictedLabel"],
    )
    validation = read_locked_csv(
        VAL_EMBEDDINGS_PATH,
        ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel", "PredictedLabel"],
    )
    references_saved = read_locked_csv(
        REFERENCE_EYES_PATH,
        ["Class", "EyeExamID", "ResearchSubjectID"],
    )
    reliability = read_locked_csv(
        VAL_RELIABILITY_PATH,
        [
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "TrueLabel",
            "Day4PredictedLabel",
            "Day3PredictedLabel",
        ],
    )
    reference_metadata = read_locked_json(REFERENCE_METADATA_PATH)
    saved_centroids = read_locked_centroids(REFERENCE_CENTROIDS_PATH)

    check_required_columns(train, TRAIN_REQUIRED_COLUMNS, "training embeddings")
    check_required_columns(validation, VAL_REQUIRED_COLUMNS, "validation embeddings")
    check_required_columns(reliability, RELIABILITY_REQUIRED_COLUMNS, "validation reliability table")
    check_required_columns(references_saved, REFERENCE_REQUIRED_COLUMNS, "reference eye table")

    for metadata_flag in (
        "test_manifest_loaded",
        "test_set_evaluated",
        "test_embeddings_extracted",
        "test_predictions_created",
    ):
        record_check(
            checks,
            f"Locked Day-5 metadata records {metadata_flag}=False",
            reference_metadata.get(metadata_flag) is False,
            reference_metadata.get(metadata_flag),
            False,
        )

    train["Correct"] = parse_bool_column(train["Correct"], "training Correct")
    train["ReferenceSelected"] = parse_bool_column(
        train["ReferenceSelected"], "training ReferenceSelected"
    )
    validation["Correct"] = parse_bool_column(validation["Correct"], "validation Correct")
    reliability["Day4Correct"] = parse_bool_column(
        reliability["Day4Correct"], "reliability Day4Correct"
    )
    reliability["ModelAgreement"] = parse_bool_column(
        reliability["ModelAgreement"], "reliability ModelAgreement"
    )

    feature_columns = [f"Feature_{index:03d}" for index in range(EXPECTED_FEATURE_DIMENSION)]
    actual_train_feature_columns = [
        column for column in train.columns if column.startswith("Feature_")
    ]
    actual_val_feature_columns = [
        column for column in validation.columns if column.startswith("Feature_")
    ]
    record_check(
        checks,
        "Training feature schema contains exactly Feature_000 through Feature_511",
        actual_train_feature_columns == feature_columns,
        len(actual_train_feature_columns),
        EXPECTED_FEATURE_DIMENSION,
    )
    record_check(
        checks,
        "Validation feature schema contains exactly Feature_000 through Feature_511",
        actual_val_feature_columns == feature_columns,
        len(actual_val_feature_columns),
        EXPECTED_FEATURE_DIMENSION,
    )
    record_check(checks, "Training eye count", len(train) == EXPECTED_TRAIN_EYES, len(train), EXPECTED_TRAIN_EYES)
    record_check(checks, "Validation eye count", len(validation) == EXPECTED_VAL_EYES, len(validation), EXPECTED_VAL_EYES)
    record_check(
        checks,
        "Training EyeExamID is unique",
        train["EyeExamID"].is_unique,
        int(train["EyeExamID"].nunique()),
        EXPECTED_TRAIN_EYES,
    )
    record_check(
        checks,
        "Validation EyeExamID is unique",
        validation["EyeExamID"].is_unique,
        int(validation["EyeExamID"].nunique()),
        EXPECTED_VAL_EYES,
    )
    record_check(
        checks,
        "Training subject count",
        train["ResearchSubjectID"].nunique() == EXPECTED_TRAIN_SUBJECTS,
        int(train["ResearchSubjectID"].nunique()),
        EXPECTED_TRAIN_SUBJECTS,
    )
    record_check(
        checks,
        "Validation subject count",
        validation["ResearchSubjectID"].nunique() == EXPECTED_VAL_SUBJECTS,
        int(validation["ResearchSubjectID"].nunique()),
        EXPECTED_VAL_SUBJECTS,
    )
    subject_overlap = set(train["ResearchSubjectID"].astype(str)) & set(
        validation["ResearchSubjectID"].astype(str)
    )
    eye_overlap = set(train["EyeExamID"].astype(str)) & set(validation["EyeExamID"].astype(str))
    record_check(checks, "Training and validation subjects are disjoint", not subject_overlap, len(subject_overlap), 0)
    record_check(checks, "Training and validation EyeExamIDs are disjoint", not eye_overlap, len(eye_overlap), 0)

    for name, frame, label_column, predicted_column in (
        ("training", train, "TrueLabel", "PredictedLabel"),
        ("validation", validation, "TrueLabel", "PredictedLabel"),
        ("reliability", reliability, "TrueLabel", "Day4PredictedLabel"),
    ):
        observed_labels = set(frame[label_column].dropna().astype(str))
        observed_predictions = set(frame[predicted_column].dropna().astype(str))
        record_check(
            checks,
            f"{name.title()} labels are Normal/Abnormal",
            observed_labels <= ALLOWED_LABELS and observed_predictions <= ALLOWED_LABELS,
            sorted(observed_labels | observed_predictions),
            sorted(ALLOWED_LABELS),
        )

    record_check(
        checks,
        "Validation label counts reproduce the locked cohort",
        validation["TrueLabel"].value_counts().to_dict() == EXPECTED_VAL_LABEL_COUNTS,
        validation["TrueLabel"].value_counts().to_dict(),
        EXPECTED_VAL_LABEL_COUNTS,
    )
    record_check(
        checks,
        "Validation correct/incorrect counts reproduce Day 4",
        int(validation["Correct"].sum()) == EXPECTED_VAL_CORRECT,
        {"correct": int(validation["Correct"].sum()), "incorrect": int((~validation["Correct"]).sum())},
        {"correct": EXPECTED_VAL_CORRECT, "incorrect": EXPECTED_VAL_INCORRECT},
    )

    validation_ids = set(validation["EyeExamID"].astype(str))
    reliability_ids = set(reliability["EyeExamID"].astype(str))
    record_check(
        checks,
        "Validation embeddings and reliability table contain identical EyeExamIDs",
        validation_ids == reliability_ids and reliability["EyeExamID"].is_unique,
        len(validation_ids ^ reliability_ids),
        0,
    )
    identity_ok, probability_difference = check_identity_mapping(
        validation, reliability, "PredictedLabel", "Day4PredictedLabel"
    )
    record_check(
        checks,
        "Validation identity and predictions match the locked reliability table",
        identity_ok,
        identity_ok,
        True,
    )
    record_check(
        checks,
        "Validation Day-4 probabilities match the locked reliability table",
        probability_difference <= 1e-15,
        probability_difference,
        "<= 1e-15",
    )

    validation = validation.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    reliability = reliability.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    train = train.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    references_saved = references_saved.sort_values(
        ["Class", "ReferenceRankWithinTrueClass", "EyeExamID"], kind="mergesort"
    ).reset_index(drop=True)

    # the hard no-test-id check is implemented without loading test ids: every
    # analyzed eye must be a member of one of the exact hash-pinned day-5 train
    # or validation tables, and the validation/reliability sets must match.
    locked_development_ids = set(train["EyeExamID"].astype(str)) | validation_ids
    analyzed_ids = (
        set(train["EyeExamID"].astype(str))
        | set(validation["EyeExamID"].astype(str))
        | set(reliability["EyeExamID"].astype(str))
        | set(references_saved["EyeExamID"].astype(str))
    )
    record_check(
        checks,
        "No out-of-allowlist EyeExamID (including any test EyeExamID) enters analysis",
        analyzed_ids <= locked_development_ids,
        len(analyzed_ids - locked_development_ids),
        0,
        "Operationalized against exact hash-pinned train/validation ID sets so no test-ID file is loaded.",
    )

    train_features, _ = normalize_features(
        train[feature_columns].to_numpy(float), "training pooled features"
    )
    val_features, _ = normalize_features(
        validation[feature_columns].to_numpy(float), "validation pooled features"
    )
    record_check(
        checks,
        "All normalized pooled embeddings are finite",
        np.isfinite(train_features).all() and np.isfinite(val_features).all(),
        bool(np.isfinite(train_features).all() and np.isfinite(val_features).all()),
        True,
    )

    metadata_reference_ids = reference_metadata.get("reference_selection", {}).get(
        "reference_eye_ids", {}
    )
    reference_indices: dict[str, np.ndarray] = {}
    reference_features: dict[str, np.ndarray] = {}
    for label in sorted(ALLOWED_LABELS):
        correct_class = train.loc[
            train["TrueLabel"].eq(label) & train["Correct"]
        ].sort_values(
            ["PrimaryConfidenceMargin", "EyeExamID"],
            ascending=[False, True],
            kind="mergesort",
        )
        expected_count = int(math.ceil(len(correct_class) * 0.5))
        recomputed_ids = correct_class.iloc[:expected_count]["EyeExamID"].astype(str).tolist()
        saved_ids = references_saved.loc[
            references_saved["Class"].eq(label), "EyeExamID"
        ].astype(str).tolist()
        train_selected_ids = train.loc[
            train["ReferenceSelected"] & train["TrueLabel"].eq(label), "EyeExamID"
        ].astype(str).tolist()
        metadata_ids = [str(value) for value in metadata_reference_ids.get(label, [])]
        count_ok = expected_count == EXPECTED_REFERENCE_COUNTS[label]
        ids_ok = set(recomputed_ids) == set(saved_ids) == set(train_selected_ids) == set(metadata_ids)
        record_check(
            checks,
            f"{label} frozen reference count",
            count_ok,
            expected_count,
            EXPECTED_REFERENCE_COUNTS[label],
        )
        record_check(
            checks,
            f"{label} frozen reference IDs reproduce the Day-5 rule",
            ids_ok,
            len(set(recomputed_ids) ^ set(saved_ids)),
            0,
            "Correct training eyes only; confidence descending; EyeExamID tie-break; ceil(top 50%).",
        )
        index_by_id = {str(value): index for index, value in enumerate(train["EyeExamID"])}
        indices = np.asarray([index_by_id[value] for value in saved_ids], dtype=int)
        reference_indices[label] = indices
        reference_features[label] = train_features[indices]

    reference_id_set = set(references_saved["EyeExamID"].astype(str))
    record_check(
        checks,
        "No validation eye enters the reference database",
        not (reference_id_set & validation_ids),
        len(reference_id_set & validation_ids),
        0,
    )

    recomputed_centroids: dict[str, np.ndarray] = {}
    centroid_reference_distances: dict[str, np.ndarray] = {}
    max_centroid_vector_difference = 0.0
    max_reference_distance_difference = 0.0
    for label in sorted(ALLOWED_LABELS):
        mean_vector = reference_features[label].mean(axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if not math.isfinite(norm) or norm <= NORM_EPSILON:
            raise SafetyError(f"{label} reference centroid is non-finite or near zero.")
        centroid = mean_vector / norm
        recomputed_centroids[label] = centroid
        max_centroid_vector_difference = max(
            max_centroid_vector_difference,
            float(np.max(np.abs(centroid - saved_centroids[label]))),
        )
        centroid_reference_distances[label] = centroid_cosine_distance(
            reference_features[label], centroid
        )
        saved_class_distances = references_saved.loc[
            references_saved["Class"].eq(label), "CosineDistanceToClassCentroid"
        ].to_numpy(float)
        max_reference_distance_difference = max(
            max_reference_distance_difference,
            float(
                np.max(
                    np.abs(
                        centroid_reference_distances[label] - saved_class_distances
                    )
                )
            ),
        )
    record_check(
        checks,
        "Recomputed class centroids match frozen Day-5 centroids",
        max_centroid_vector_difference <= CENTROID_TOLERANCE,
        max_centroid_vector_difference,
        f"<= {CENTROID_TOLERANCE}",
    )
    record_check(
        checks,
        "Recomputed reference-eye centroid distances match Day 5",
        max_reference_distance_difference <= CENTROID_TOLERANCE,
        max_reference_distance_difference,
        f"<= {CENTROID_TOLERANCE}",
    )

    distance_to_normal = centroid_cosine_distance(
        val_features, recomputed_centroids["Normal"]
    )
    distance_to_abnormal = centroid_cosine_distance(
        val_features, recomputed_centroids["Abnormal"]
    )
    predicted_labels = validation["PredictedLabel"].astype(str).to_numpy()
    centroid_primary = np.where(
        predicted_labels == "Normal", distance_to_normal, distance_to_abnormal
    )
    centroid_percentile = np.empty(len(validation), dtype=np.float64)
    for label in sorted(ALLOWED_LABELS):
        indices = np.flatnonzero(predicted_labels == label)
        centroid_percentile[indices] = reference_percentiles(
            centroid_reference_distances[label], centroid_primary[indices]
        )
    centroid_replay_difference = float(
        np.max(
            np.abs(
                centroid_primary
                - validation["FeatureTypicalityPredictedClass"].to_numpy(float)
            )
        )
    )
    centroid_percentile_difference = float(
        np.max(
            np.abs(
                centroid_percentile
                - validation["PredictedClassReferencePercentile"].to_numpy(float)
            )
        )
    )
    record_check(
        checks,
        "Centroid cosine typicality reproduces Day 5",
        centroid_replay_difference <= CENTROID_TOLERANCE,
        centroid_replay_difference,
        f"<= {CENTROID_TOLERANCE}",
    )
    record_check(
        checks,
        "Centroid empirical percentiles reproduce Day 5",
        centroid_percentile_difference <= CENTROID_TOLERANCE,
        centroid_percentile_difference,
        f"<= {CENTROID_TOLERANCE}",
    )

    knn_results, knn_reference_distributions = compute_knn_scores(
        reference_features, val_features, predicted_labels
    )
    mahalanobis_results, mahalanobis_reference_distances, mahalanobis_note = (
        compute_optional_mahalanobis(reference_features, val_features, predicted_labels)
    )

    comparison = reliability[
        [
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "TrueLabel",
            "Day4PredictedLabel",
            "Day4Correct",
            "PrimaryConfidenceMargin",
            "Day3StdViewProbability",
            "ModelAgreement",
        ]
    ].copy()
    comparison.insert(0, "DataScope", DATA_SCOPE)
    comparison.insert(1, "TestStatus", TEST_STATUS)
    comparison["CentroidCosineDistance"] = centroid_primary
    comparison["CentroidReferencePercentile"] = centroid_percentile
    methods: dict[str, tuple[str, str]] = {
        "centroid_baseline": (
            "CentroidCosineDistance",
            "CentroidReferencePercentile",
        )
    }
    for k in K_VALUES:
        distance_field = f"KNN{k}MeanCosineDistance"
        percentile_field = f"KNN{k}ReferencePercentile"
        comparison[distance_field] = knn_results[k]["distance"]
        comparison[percentile_field] = knn_results[k]["percentile"]
        methods[f"knn_k{k}"] = (distance_field, percentile_field)
    if mahalanobis_results is not None:
        comparison["ShrinkageMahalanobisDistance"] = mahalanobis_results["distance"]
        comparison["ShrinkageMahalanobisReferencePercentile"] = mahalanobis_results[
            "percentile"
        ]
        methods["shrinkage_mahalanobis_experimental"] = (
            "ShrinkageMahalanobisDistance",
            "ShrinkageMahalanobisReferencePercentile",
        )

    numeric_comparison = comparison[
        [column for pair in methods.values() for column in pair]
    ].to_numpy(float)
    record_check(
        checks,
        "All feature-distribution comparison values are finite",
        np.isfinite(numeric_comparison).all(),
        bool(np.isfinite(numeric_comparison).all()),
        True,
    )
    percentile_columns = [pair[1] for pair in methods.values()]
    percentile_bounds_ok = all(
        comparison[column].between(0.0, 100.0, inclusive="both").all()
        for column in percentile_columns
    )
    record_check(
        checks,
        "All reference percentiles are within [0,100]",
        percentile_bounds_ok,
        percentile_bounds_ok,
        True,
    )

    error_analysis, method_summary = make_error_analysis(comparison, methods)
    selected_k, k_values_similar, k_selection_reason = choose_knn_k(method_summary)
    selected_percentile_field = f"KNN{selected_k}ReferencePercentile"
    comparison["SelectedK"] = selected_k
    comparison["SelectedKNNMeanCosineDistance"] = comparison[
        f"KNN{selected_k}MeanCosineDistance"
    ]
    comparison["SelectedKNNReferencePercentile"] = comparison[
        selected_percentile_field
    ]

    routing = reliability.copy()
    routing["SelectedKNNReferencePercentile"] = comparison[
        "SelectedKNNReferencePercentile"
    ].to_numpy(float)
    routing = routing.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    agreement_mask = bools_to_bitmask(routing["ModelAgreement"].to_numpy(bool))
    confidence_masks = make_threshold_masks(
        routing["PrimaryConfidenceMargin"].to_numpy(float), "ge"
    )
    view_sd_masks = make_threshold_masks(
        routing["Day3StdViewProbability"].to_numpy(float), "le"
    )
    knn_masks = make_threshold_masks(
        routing["SelectedKNNReferencePercentile"].to_numpy(float), "le"
    )

    envelopes: dict[str, pd.DataFrame] = {}
    enumeration: dict[str, dict[str, int]] = {}
    for spec in STRATEGIES:
        envelope, counts = build_envelope(
            spec,
            routing,
            confidence_masks,
            view_sd_masks,
            knn_masks,
            agreement_mask,
            selected_k,
        )
        envelopes[spec.strategy_id] = envelope
        enumeration[spec.strategy_id] = counts

    for spec in KNN_STRATEGIES:
        base = envelopes[str(spec.matched_base_strategy_id)].set_index("AcceptedCount")
        knn = envelopes[spec.strategy_id].set_index("AcceptedCount")
        common = sorted(set(base.index) & set(knn.index))
        noninferior = all(
            int(knn.loc[count, "AcceptedErrors"])
            <= int(base.loc[count, "AcceptedErrors"])
            for count in common
        )
        record_check(
            checks,
            f"{spec.display_name} envelope includes a nonbinding-KNN reproduction of its base",
            noninferior,
            noninferior,
            True,
        )

    routing_ablation = select_target_candidates(envelopes)
    # build from records rather than concatenating heterogeneous all-na
    # threshold columns; this preserves stable dtypes without pandas' evolving
    # all-na concatenation behavior.
    frontier = pd.DataFrame(
        [
            row
            for spec in STRATEGIES
            for row in add_pareto_flag(envelopes[spec.strategy_id]).to_dict("records")
        ]
    )
    strict_rows, any_strict_routing_benefit, strict_benefit_by_strategy = (
        strict_benefit_summary(envelopes)
    )
    error_analysis = pd.concat([error_analysis, strict_rows], ignore_index=True)

    target_exact = routing_ablation["CoverageDeviationPercentagePoints"].eq(0.0).all()
    record_check(
        checks,
        "All requested routing coverage targets are exactly attainable",
        bool(target_exact),
        bool(target_exact),
        True,
    )
    replay_ok = True
    for _, row in routing_ablation.iterrows():
        mask = replay_acceptance(routing, row)
        metrics = routing_counts(routing, mask)
        replay_ok = replay_ok and int(metrics["AcceptedCount"]) == int(row["AcceptedCount"])
        replay_ok = replay_ok and int(metrics["AcceptedErrors"]) == int(row["AcceptedErrors"])
        replay_ok = replay_ok and int(metrics["AcceptedFalseNegatives"]) == int(
            row["AcceptedFalseNegatives"]
        )
        replay_ok = replay_ok and int(metrics["AcceptedFalsePositives"]) == int(
            row["AcceptedFalsePositives"]
        )
    record_check(
        checks,
        "All candidate routing thresholds replay their saved validation metrics",
        replay_ok,
        replay_ok,
        True,
    )

    selected_method = f"knn_k{selected_k}"
    selected_auc = float(method_summary[selected_method]["ErrorDetectionAUROC"])
    correct_median = float(
        comparison.loc[
            comparison["Day4Correct"], "SelectedKNNReferencePercentile"
        ].median()
    )
    incorrect_median = float(
        comparison.loc[
            ~comparison["Day4Correct"], "SelectedKNNReferencePercentile"
        ].median()
    )
    selected_capture_rows = error_analysis.loc[
        error_analysis["Method"].eq(selected_method)
        & error_analysis["AnalysisType"].eq("ErrorCapture")
    ]
    selected_unique_errors = int(
        selected_capture_rows["UniqueErrorsBeyondConfidenceAndViewSD"].max()
    )
    strict_benefit_beyond_view_sd = bool(
        strict_benefit_by_strategy["c_view_sd_knn"]
        or strict_benefit_by_strategy["c_agreement_view_sd_knn"]
    )
    if strict_benefit_beyond_view_sd or selected_unique_errors > 0:
        conclusion_code = "A"
        conclusion = "KNN feature typicality provides measurable incremental routing benefit beyond the existing confidence/view-disagreement signals on validation."
    elif selected_auc > 0.5 and incorrect_median > correct_median:
        conclusion_code = "B"
        conclusion = "KNN feature typicality separates validation errors descriptively and can improve simpler confidence-based rules, but adds little beyond the existing view-disagreement signals."
    else:
        conclusion_code = "C"
        conclusion = "KNN feature typicality provides no useful improvement on this validation cohort."

    reference_output_rows: list[dict[str, Any]] = []
    mahal_ref_lookup = mahalanobis_reference_distances
    for label in sorted(ALLOWED_LABELS):
        class_saved = references_saved.loc[references_saved["Class"].eq(label)].reset_index(drop=True)
        for position, row in class_saved.iterrows():
            output_row = {
                "DataScope": DATA_SCOPE,
                "TestStatus": TEST_STATUS,
                "Class": label,
                "EyeExamID": row["EyeExamID"],
                "ResearchSubjectID": row["ResearchSubjectID"],
                "CentroidCosineDistance": centroid_reference_distances[label][position],
            }
            for k in K_VALUES:
                output_row[f"KNN{k}LeaveOneOutMeanCosineDistance"] = (
                    knn_reference_distributions[label][k][position]
                )
            if mahalanobis_results is not None:
                output_row["ShrinkageMahalanobisInSampleDistance"] = mahal_ref_lookup[
                    label
                ][position]
            reference_output_rows.append(output_row)
    reference_output = pd.DataFrame(reference_output_rows)

    source_hashes_after = {str(path): file_sha256(path) for path in ALLOWED_INPUT_PATHS}
    record_check(
        checks,
        "Locked input artifacts were unchanged during analysis",
        source_hashes_after == source_hashes_before,
        source_hashes_after == source_hashes_before,
        True,
    )
    record_check(
        checks,
        "Loaded input registry contains only the six fixed train/validation artifacts",
        LOADED_INPUT_PATHS == set(ALLOWED_INPUT_PATHS),
        len(LOADED_INPUT_PATHS),
        len(ALLOWED_INPUT_PATHS),
    )
    no_test_paths = all(
        not any(part.casefold() in {"test", "tests"} for part in path.parts)
        and not path.name.casefold().startswith("test_")
        for path in LOADED_INPUT_PATHS
    )
    record_check(
        checks,
        "No test artifact path was loaded",
        no_test_paths,
        no_test_paths,
        True,
    )
    record_check(
        checks,
        "Test set was not loaded or evaluated",
        True,
        True,
        True,
        "The script has no test input path and uses a fixed allowlist of hash-pinned train/validation artifacts.",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = output_dir / "feature_distribution_comparison.csv"
    error_analysis_path = output_dir / "feature_distribution_error_analysis.csv"
    routing_ablation_path = output_dir / "feature_distribution_routing_ablation.csv"
    frontier_path = output_dir / "feature_distribution_frontier.csv"
    checks_path = output_dir / "feature_distribution_checks.csv"
    reference_output_path = output_dir / "feature_distribution_reference_distances.csv"
    metadata_path = output_dir / "feature_distribution_metadata.json"
    summary_path = output_dir / "feature_distribution_summary.md"

    write_csv(comparison, comparison_path)
    write_csv(error_analysis, error_analysis_path)
    write_csv(routing_ablation.drop(columns=["AcceptedBitmask"]), routing_ablation_path)
    write_csv(frontier.drop(columns=["AcceptedBitmask"]), frontier_path)
    write_csv(pd.DataFrame(checks), checks_path)
    write_csv(reference_output, reference_output_path)

    completed = datetime.now().astimezone()
    metadata = {
        "stage": "Day 5 feature-distribution improvement before held-out evaluation",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "data_scope": DATA_SCOPE,
        "test_status": TEST_STATUS,
        "test_data_loaded": False,
        "test_set_evaluated": False,
        "model_retrained": False,
        "checkpoint_loaded": False,
        "frozen_routing_rules_modified": False,
        "feature_dimension": EXPECTED_FEATURE_DIMENSION,
        "reference_rule": "Within true training class: correct Day-4 predictions only; PrimaryConfidenceMargin descending then EyeExamID ascending; ceil(top 50%).",
        "reference_counts": EXPECTED_REFERENCE_COUNTS,
        "centroid_baseline": "Cosine distance from L2-normalized eye feature to unit-normalized predicted-class reference centroid; right-continuous empirical class percentile.",
        "knn_method": {
            "k_values": list(K_VALUES),
            "distance": "Mean cosine distance to k nearest L2-normalized references from the Day-4 predicted class.",
            "reference_distribution": "For each reference eye, mean distance to k nearest other same-class references (leave-one-out).",
            "percentile": "100 * count(reference LOO distance <= query distance) / class reference count.",
            "selected_k": selected_k,
            "k_values_similar": k_values_similar,
            "selection_reason": k_selection_reason,
        },
        "mahalanobis": {
            "calculated": mahalanobis_results is not None,
            "status": mahalanobis_note,
            "primary_method": False,
        },
        "routing": {
            "data": "validation only",
            "targets_percent": list(TARGET_COVERAGES),
            "comparators": {
                "confidence": ">=",
                "agreement": "== True",
                "view_sd": "<=",
                "knn_percentile": "<=",
            },
            "envelope_objective": "At each accepted count: fewest errors, then fewest false negatives, then highest balanced accuracy, then least-restrictive extra gates and deterministic EyeExamID tie-break.",
            "envelope_masks_may_be_nonnested": True,
            "strict_improvement": "Same accepted count with fewer errors OR same accepted-error constraint with higher maximum coverage.",
            "enumeration": enumeration,
            "any_strict_knn_benefit": any_strict_routing_benefit,
            "strict_benefit_by_strategy": strict_benefit_by_strategy,
            "strict_benefit_beyond_view_sd": strict_benefit_beyond_view_sd,
            "selected_knn_unique_errors_beyond_confidence_and_view_sd": selected_unique_errors,
        },
        "method_conclusion": {
            "code": conclusion_code,
            "statement": conclusion,
            "validation_only": True,
            "no_statistical_significance_claim": True,
        },
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): EXPECTED_HASHES[path]
            for path in sorted(ALLOWED_INPUT_PATHS, key=str)
        },
        "loaded_input_paths": [
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted(LOADED_INPUT_PATHS, key=str)
        ],
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "outputs": {
            "comparison": str(comparison_path),
            "error_analysis": str(error_analysis_path),
            "routing_ablation": str(routing_ablation_path),
            "frontier": str(frontier_path),
            "checks": str(checks_path),
            "reference_distances": str(reference_output_path),
            "summary": str(summary_path),
        },
    }
    save_json(metadata_path, make_json_safe(metadata))

    selected_stats = {}
    for group_name, group_mask in (
        ("Correct", comparison["Day4Correct"]),
        ("Incorrect", ~comparison["Day4Correct"]),
    ):
        selected_stats[group_name] = summarize_values(
            comparison.loc[group_mask, "SelectedKNNReferencePercentile"].to_numpy(float)
        )
    summary_lines = [
        "# Day-5 feature-distribution improvement",
        "",
        f"**Data scope:** {DATA_SCOPE}",
        f"**Safety status:** {TEST_STATUS}",
        "",
        "No model was retrained. The frozen Day-5 reference populations were reused: "
        f"Normal={EXPECTED_REFERENCE_COUNTS['Normal']}, Abnormal={EXPECTED_REFERENCE_COUNTS['Abnormal']}.",
        "",
        "## Baseline reproduction",
        "",
        f"- Maximum centroid-distance replay difference: {centroid_replay_difference:.3g}",
        f"- Maximum centroid-percentile replay difference: {centroid_percentile_difference:.3g}",
        "",
        "## Local kNN results",
        "",
        "| Method | Error-detection AUROC | Errors captured in highest-risk 25% |",
        "|---|---:|---:|",
    ]
    for k in K_VALUES:
        summary_lines.append(
            f"| k={k} | {float(method_summary[f'knn_k{k}']['ErrorDetectionAUROC']):.6f} | "
            f"{int(method_summary[f'knn_k{k}']['Top25PercentCapturedErrors'])}/{EXPECTED_VAL_INCORRECT} |"
        )
    summary_lines += [
        "",
        f"Selected development default: **k={selected_k}**. {k_selection_reason}",
        "",
        f"Correct predictions: selected-k percentile mean={selected_stats['Correct']['Mean']:.3f}, "
        f"median={selected_stats['Correct']['Median']:.3f}.",
        f"Incorrect predictions: selected-k percentile mean={selected_stats['Incorrect']['Mean']:.3f}, "
        f"median={selected_stats['Incorrect']['Median']:.3f}.",
        "",
        "## Routing ablation",
        "",
        "Rules were optimized independently at each accepted count on validation only; envelope masks may be nonnested.",
        "",
        "| Strategy | Target coverage | Actual coverage | Errors | Error rate | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in routing_ablation.iterrows():
        summary_lines.append(
            f"| {row['Strategy']} | {float(row['TargetCoveragePercent']):.0f}% | "
            f"{float(row['CoveragePercent']):.1f}% | {int(row['AcceptedErrors'])} | "
            f"{100.0 * float(row['AcceptedErrorRate']):.2f}% | "
            f"{int(row['AcceptedFalseNegatives'])} | {int(row['AcceptedFalsePositives'])} |"
        )
    summary_lines += [
        "",
        "## Conclusion",
        "",
        f"**{conclusion_code}. {conclusion}**",
        "",
        f"Strict KNN improvement for at least one simpler route: **{any_strict_routing_benefit}**. "
        f"Strict improvement after including view SD: **{strict_benefit_beyond_view_sd}**. "
        f"Selected-k unique errors beyond matched confidence/view-SD risk bands: **{selected_unique_errors}**.",
        "",
        "This is descriptive validation-only development analysis. It does not establish statistical significance and does not alter the frozen Day-6 operating points.",
        "",
        f"Optional shrinkage Mahalanobis: {mahalanobis_note}",
        "",
        f"Output directory: `{output_dir}`",
        "",
        f"**{TEST_STATUS}**",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nDay-5 feature-distribution improvement complete")
    print(f"Data scope                 : {DATA_SCOPE}")
    print(f"Training / validation eyes : {len(train)} / {len(validation)}")
    print(
        "Reference eyes             : "
        f"Normal={EXPECTED_REFERENCE_COUNTS['Normal']}, "
        f"Abnormal={EXPECTED_REFERENCE_COUNTS['Abnormal']}"
    )
    print(f"Feature dimension           : {EXPECTED_FEATURE_DIMENSION}")
    print(
        "Centroid replay max diff   : "
        f"distance={centroid_replay_difference:.3g}, "
        f"percentile={centroid_percentile_difference:.3g}"
    )
    print("kNN error-detection AUROC   :")
    for k in K_VALUES:
        print(
            f"  k={k:<2} {float(method_summary[f'knn_k{k}']['ErrorDetectionAUROC']):.6f} "
            f"(top-25% errors {int(method_summary[f'knn_k{k}']['Top25PercentCapturedErrors'])}/{EXPECTED_VAL_INCORRECT})"
        )
    print(f"Selected development k      : {selected_k}")
    print(f"Strict routing benefit      : {any_strict_routing_benefit}")
    print(f"Benefit beyond view SD      : {strict_benefit_beyond_view_sd}")
    print(f"Conclusion                  : {conclusion_code}. {conclusion}")
    print(f"Output directory            : {output_dir}")
    print(f"Checks                      : {checks_path}")
    print(TEST_STATUS)


if __name__ == "__main__":
    try:
        main()
    except SafetyError as error:
        print(f"HARD FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
