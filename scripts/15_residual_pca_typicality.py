
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "multiview_resnet18"
    / "20260814T130502_173490-0700"
)
TRAIN_EMBEDDINGS = FEATURE_RUN / "train_eye_embeddings.csv"
VAL_EMBEDDINGS = FEATURE_RUN / "val_eye_embeddings.csv"
VAL_RELIABILITY = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "reliability"
    / "20260814T141442_497529-0700"
    / "val_reliability_signals.csv"
)
DAY4_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "multiview_resnet18"
    / "20260813T170240_569268-0700"
    / "best_model.pt"
)
FROZEN_ROUTING = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "selective_routing"
    / "frozen_routing_rules.json"
)
KNN_FOLLOWUP_DIR = PROJECT_ROOT / "outputs" / "audits" / "20260817T125819_273455-0700"
KNN_COMPARISON = KNN_FOLLOWUP_DIR / "feature_distribution_comparison.csv"
KNN_METADATA = KNN_FOLLOWUP_DIR / "feature_distribution_metadata.json"

EXPECTED_SHA256 = {
    TRAIN_EMBEDDINGS: "4c51fdfd5e4bcc5583974c43d9e0563105ebb04e18f019c65236a236d06b77d8",
    VAL_EMBEDDINGS: "22563615dd5e2d61ad0cdbdc8225084abd2a5a7a878333eec5d946f05bd22045",
    VAL_RELIABILITY: "a767b163b3565a9fe664068c5fd18e96e4f13bf4dea7040ee179f098b477edb8",
    DAY4_CHECKPOINT: "3c6d402ddbd2d1055cb68458bc6f5a7c880ac683a4527ff5171cc6f2408c754b",
    FROZEN_ROUTING: "d042f0032068e0528a64122cc5fde967fe3d7240db95be7606ac3b7390c0ced3",
    KNN_COMPARISON: "bb976c0d8ce461be6164caf4ca97e93b09142d4c07a0b9500f996180e4a1ea30",
    KNN_METADATA: "192cea121197165bbe1a3ecb41d55d4b4eb373b9d18a06237c79ee410631fc24",
}

FEATURE_COLUMNS = [f"Feature_{index:03d}" for index in range(512)]
ALLOWED_LABELS = {"Normal", "Abnormal"}
CLASS_TO_INDEX = {"Normal": 0, "Abnormal": 1}
EXPECTED_TRAIN_EYES = 560
EXPECTED_VAL_EYES = 120
EXPECTED_CORRECT_REFERENCES = {"Normal": 121, "Abnormal": 388}
N_COMPONENTS_MAX = 32
N_NEIGHBORS = 10
RESIDUAL_PERCENTILE_CUTOFF = 95.0
SEED = 42
NORM_EPSILON = 1e-12
DATA_SCOPE = "TRAIN + VALIDATION ONLY"
TEST_STATUS = "TEST SET NOT LOADED OR EVALUATED"


class SafetyError(RuntimeError):
    """used when a hard protocol, provenance, or numerical check fails."""


CHECKS: list[dict[str, Any]] = []
LOADED_INPUTS: set[Path] = set()


def record_hard_check(name: str, passed: bool, observed: Any, expected: Any) -> None:
    CHECKS.append(
        {
            "Check": name,
            "Status": "PASS" if bool(passed) else "FAIL",
            "Observed": observed,
            "Expected": expected,
            "Severity": "HARD FAIL",
            "DataScope": DATA_SCOPE,
            "TestStatus": TEST_STATUS,
        }
    )
    if not passed:
        raise SafetyError(f"{name}: observed={observed!r}; expected={expected!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_input_path(path: Path) -> Path:
    # every input is fixed by path and sha-256 before it can be used
    resolved = path.resolve()
    if resolved not in {item.resolve() for item in EXPECTED_SHA256}:
        raise SafetyError(f"Input is outside the fixed validation-only allowlist: {resolved}")
    if "test" in resolved.name.casefold():
        raise SafetyError(f"Held-out test artifact access attempted: {resolved}")
    expected = EXPECTED_SHA256[path]
    record_hard_check(f"input exists: {path.name}", resolved.is_file(), resolved.is_file(), True)
    observed = file_sha256(resolved)
    record_hard_check(f"input SHA-256: {path.name}", observed == expected, observed, expected)
    LOADED_INPUTS.add(resolved)
    return resolved


def read_locked_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(check_input_path(path), low_memory=False, **kwargs)


def read_locked_json(path: Path) -> dict[str, Any]:
    with check_input_path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"JSON root is not an object: {path}")
    return value


def parse_bool_column(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.casefold()
    invalid = ~values.isin({"true", "false"})
    if invalid.any():
        raise SafetyError(f"{field} has invalid booleans: {sorted(values[invalid].unique())}")
    return values.eq("true")


def check_required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    record_hard_check(f"{name} required columns", not missing, missing, [])


def empirical_reference_percentiles(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    # use the right-continuous empirical percentile against training distances
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(query, dtype=np.float64)
    return 100.0 * np.searchsorted(ordered, values, side="right") / len(ordered)


def compute_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = pd.Series(np.asarray(left, dtype=np.float64)).rank(method="average")
    right_ranks = pd.Series(np.asarray(right, dtype=np.float64)).rank(method="average")
    return float(left_ranks.corr(right_ranks, method="pearson"))


def euclidean_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # compute all pairwise distances without a python loop
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def mean_k_nearest(distances: np.ndarray, k: int) -> np.ndarray:
    # typicality uses the mean distance to the k nearest references
    if distances.shape[1] < k:
        raise SafetyError(f"Only {distances.shape[1]} candidate neighbors for k={k}.")
    return np.partition(distances, kth=k - 1, axis=1)[:, :k].mean(axis=1)


def compute_routing_metrics(frame: pd.DataFrame, accepted: np.ndarray) -> dict[str, Any]:
    truth = frame["TrueLabel"].astype(str).to_numpy()[accepted]
    predicted = frame["Day4PredictedLabel"].astype(str).to_numpy()[accepted]
    tp = int(((truth == "Abnormal") & (predicted == "Abnormal")).sum())
    tn = int(((truth == "Normal") & (predicted == "Normal")).sum())
    fp = int(((truth == "Normal") & (predicted == "Abnormal")).sum())
    fn = int(((truth == "Abnormal") & (predicted == "Normal")).sum())
    accepted_count = int(accepted.sum())
    errors = fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced = (
        (sensitivity + specificity) / 2.0
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else math.nan
    return {
        "TotalEyes": len(frame),
        "AcceptedEyes": accepted_count,
        "DeferredEyes": len(frame) - accepted_count,
        "Coverage": accepted_count / len(frame),
        "ReviewRate": 1.0 - accepted_count / len(frame),
        "AcceptedCorrect": accepted_count - errors,
        "AcceptedErrors": errors,
        "AcceptedErrorRate": errors / accepted_count if accepted_count else math.nan,
        "AcceptedTP": tp,
        "AcceptedTN": tn,
        "AcceptedFP": fp,
        "AcceptedFN": fn,
        "AcceptedSensitivity": sensitivity,
        "AcceptedSpecificity": specificity,
        "AcceptedBalancedAccuracy": balanced,
        "AcceptedF1": f1,
    }


def get_classifier_direction() -> tuple[np.ndarray, dict[str, Any]]:
    # use the abnormal-minus-normal day-4 classifier weights as the decision direction
    checkpoint_path = check_input_path(DAY4_CHECKPOINT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    record_hard_check(
        "Day-4 checkpoint class mapping",
        checkpoint.get("class_to_index") == CLASS_TO_INDEX,
        checkpoint.get("class_to_index"),
        CLASS_TO_INDEX,
    )
    state = checkpoint.get("model_state_dict", {})
    record_hard_check("Day-4 classifier weight exists", "classifier.weight" in state, "classifier.weight" in state, True)
    weights = state["classifier.weight"].detach().cpu().numpy().astype(np.float64)
    record_hard_check("Day-4 classifier weight shape", weights.shape == (2, 512), weights.shape, (2, 512))
    direction = weights[CLASS_TO_INDEX["Abnormal"]] - weights[CLASS_TO_INDEX["Normal"]]
    norm = float(np.linalg.norm(direction))
    record_hard_check("classifier direction finite and nonzero", np.isfinite(direction).all() and norm > NORM_EPSILON, norm, "> 1e-12")
    return direction / norm, {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "classifier_direction_norm_before_normalization": norm,
    }


def remove_classifier_direction(features: np.ndarray, class_mean: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # center within class and remove the component aligned with the classifier
    centered = np.asarray(features, dtype=np.float64) - class_mean
    projection = (centered @ direction)[:, None] * direction[None, :]
    residual = centered - projection
    return residual


def main() -> int:
    # this is a validation-only follow-up with no retraining or image inference
    started = datetime.now().astimezone()
    np.random.seed(SEED)

    # load only the hash-pinned development artifacts used by this follow-up
    train = read_locked_csv(TRAIN_EMBEDDINGS, float_precision="round_trip")
    validation = read_locked_csv(VAL_EMBEDDINGS, float_precision="round_trip")
    reliability = read_locked_csv(VAL_RELIABILITY, float_precision="round_trip")
    existing = read_locked_csv(KNN_COMPARISON, float_precision="round_trip")
    knn_metadata = read_locked_json(KNN_METADATA)
    routing = read_locked_json(FROZEN_ROUTING)
    classifier_direction, checkpoint_info = get_classifier_direction()

    embedding_required = {
        "EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality",
        "TrueLabel", "PredictedLabel", "Correct", *FEATURE_COLUMNS,
    }
    reliability_required = {
        "EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality",
        "TrueLabel", "Day4PredictedLabel", "Day4Correct",
        "PrimaryConfidenceMargin", "Day3StdViewProbability", "ModelAgreement",
        "PredictedClassReferencePercentile",
    }
    existing_required = {
        "EyeExamID", "CentroidReferencePercentile",
        "SelectedK", "SelectedKNNReferencePercentile",
    }
    check_required_columns(train, embedding_required, "training embeddings")
    check_required_columns(validation, embedding_required, "validation embeddings")
    check_required_columns(reliability, reliability_required, "validation reliability")
    check_required_columns(existing, existing_required, "existing feature comparison")

    train["Correct"] = parse_bool_column(train["Correct"], "training Correct")
    validation["Correct"] = parse_bool_column(validation["Correct"], "validation Correct")
    reliability["Day4Correct"] = parse_bool_column(reliability["Day4Correct"], "Day4Correct")
    reliability["ModelAgreement"] = parse_bool_column(reliability["ModelAgreement"], "ModelAgreement")
    record_hard_check("training eye count", len(train) == EXPECTED_TRAIN_EYES, len(train), EXPECTED_TRAIN_EYES)
    record_hard_check("validation eye count", len(validation) == EXPECTED_VAL_EYES, len(validation), EXPECTED_VAL_EYES)
    record_hard_check("training EyeExamID unique", not train["EyeExamID"].duplicated().any(), int(train["EyeExamID"].duplicated().sum()), 0)
    record_hard_check("validation EyeExamID unique", not validation["EyeExamID"].duplicated().any(), int(validation["EyeExamID"].duplicated().sum()), 0)
    record_hard_check("train/validation eye disjoint", not (set(train["EyeExamID"]) & set(validation["EyeExamID"])), len(set(train["EyeExamID"]) & set(validation["EyeExamID"])), 0)
    record_hard_check("train/validation subject disjoint", not (set(train["ResearchSubjectID"]) & set(validation["ResearchSubjectID"])), len(set(train["ResearchSubjectID"]) & set(validation["ResearchSubjectID"])), 0)
    record_hard_check("labels restricted to Normal/Abnormal", set(train["TrueLabel"]) == ALLOWED_LABELS and set(validation["TrueLabel"]) == ALLOWED_LABELS, sorted(set(train["TrueLabel"]) | set(validation["TrueLabel"])), sorted(ALLOWED_LABELS))
    record_hard_check("saved training correctness reproduces", np.array_equal(train["Correct"].to_numpy(bool), train["TrueLabel"].eq(train["PredictedLabel"]).to_numpy(bool)), "all rows", "exact")
    record_hard_check("saved validation correctness reproduces", np.array_equal(validation["Correct"].to_numpy(bool), validation["TrueLabel"].eq(validation["PredictedLabel"]).to_numpy(bool)), "all rows", "exact")

    train_features = train[FEATURE_COLUMNS].to_numpy(np.float64)
    val_features = validation[FEATURE_COLUMNS].to_numpy(np.float64)
    record_hard_check("training feature matrix finite 560x512", train_features.shape == (560, 512) and np.isfinite(train_features).all(), train_features.shape, (560, 512))
    record_hard_check("validation feature matrix finite 120x512", val_features.shape == (120, 512) and np.isfinite(val_features).all(), val_features.shape, (120, 512))

    # align every validation artifact by eye before comparing scores
    identity_columns = ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel"]
    val_sorted = validation.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    reliability_sorted = reliability.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    existing_sorted = existing.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    record_hard_check("validation/reliability identity mapping", val_sorted[identity_columns].astype(str).equals(reliability_sorted[identity_columns].astype(str)), "compared", "exact equality")
    record_hard_check("validation/reliability predictions match", val_sorted["PredictedLabel"].astype(str).equals(reliability_sorted["Day4PredictedLabel"].astype(str)), "compared", "exact equality")
    record_hard_check("validation/existing comparison eye set", val_sorted["EyeExamID"].astype(str).equals(existing_sorted["EyeExamID"].astype(str)), "compared", "exact equality")
    record_hard_check("existing selected k is 10", set(existing_sorted["SelectedK"].astype(int)) == {10} and int(knn_metadata["knn_method"]["selected_k"]) == 10, sorted(set(existing_sorted["SelectedK"].astype(int))), [10])
    record_hard_check("existing centroid percentile reproduces reliability", np.allclose(existing_sorted["CentroidReferencePercentile"].to_numpy(float), reliability_sorted["PredictedClassReferencePercentile"].to_numpy(float), rtol=0.0, atol=0.0), float(np.max(np.abs(existing_sorted["CentroidReferencePercentile"].to_numpy(float) - reliability_sorted["PredictedClassReferencePercentile"].to_numpy(float)))), 0.0)

    reference_indices: dict[str, np.ndarray] = {}
    class_means: dict[str, np.ndarray] = {}
    pca_models: dict[str, PCA] = {}
    train_coordinates: dict[str, np.ndarray] = {}
    train_distances: dict[str, np.ndarray] = {}
    train_reference_rows: list[dict[str, Any]] = []
    components_by_class: dict[str, int] = {}

    # build separate residual-pca reference spaces for normal and abnormal eyes
    for label in sorted(ALLOWED_LABELS):
        indices = np.flatnonzero(train["Correct"].to_numpy(bool) & train["TrueLabel"].eq(label).to_numpy(bool))
        reference_indices[label] = indices
        record_hard_check(f"{label} correctly classified reference count", len(indices) == EXPECTED_CORRECT_REFERENCES[label], len(indices), EXPECTED_CORRECT_REFERENCES[label])
        features = train_features[indices]
        class_mean = features.mean(axis=0)
        class_means[label] = class_mean
        # remove the classifier-aligned direction before fitting pca
        residual = remove_classifier_direction(features, class_mean, classifier_direction)
        orthogonality = float(np.max(np.abs(residual @ classifier_direction)))
        record_hard_check(f"{label} residuals orthogonal to classifier direction", orthogonality <= 1e-10, orthogonality, "<= 1e-10")
        components = min(N_COMPONENTS_MAX, len(indices) - 1, residual.shape[1])
        # whitened coordinates make local euclidean distance comparable across components
        pca = PCA(n_components=components, whiten=True, svd_solver="full")
        coordinates = pca.fit_transform(residual)
        record_hard_check(f"{label} PCA uses 32 whitened components", components == 32 and pca.whiten is True, components, 32)
        record_hard_check(f"{label} PCA coordinates finite", np.isfinite(coordinates).all(), bool(np.isfinite(coordinates).all()), True)
        pairwise = euclidean_distance_matrix(coordinates, coordinates)
        # leave each training reference out of its own 10-neighbor calibration
        np.fill_diagonal(pairwise, np.inf)
        loo = mean_k_nearest(pairwise, N_NEIGHBORS)
        record_hard_check(f"{label} LOO 10-NN distances finite", np.isfinite(loo).all() and np.all(loo > 0), f"min={loo.min()}, max={loo.max()}", "finite and >0")
        pca_models[label] = pca
        train_coordinates[label] = coordinates
        train_distances[label] = loo
        components_by_class[label] = components
        reference_percentiles = empirical_reference_percentiles(loo, loo)
        for position, train_index in enumerate(indices):
            row = train.iloc[int(train_index)]
            train_reference_rows.append(
                {
                    "EyeExamID": row["EyeExamID"],
                    "ResearchSubjectID": row["ResearchSubjectID"],
                    "TrueLabel": label,
                    "Correct": True,
                    "ResidualPCAComponents": components,
                    "ResidualPCA10NNLeaveOneOutDistance": loo[position],
                    "ResidualPCAReferencePercentile": reference_percentiles[position],
                    "DataScope": DATA_SCOPE,
                    "TestStatus": TEST_STATUS,
                }
            )

    # validation typicality always uses the day-4 predicted class
    predicted_labels = val_sorted["PredictedLabel"].astype(str).to_numpy()
    validation_distance = np.full(len(val_sorted), np.nan, dtype=np.float64)
    validation_percentile = np.full(len(val_sorted), np.nan, dtype=np.float64)
    val_sorted_features = val_sorted[FEATURE_COLUMNS].to_numpy(np.float64)
    for label in sorted(ALLOWED_LABELS):
        indices = np.flatnonzero(predicted_labels == label)
        residual = remove_classifier_direction(val_sorted_features[indices], class_means[label], classifier_direction)
        coordinates = pca_models[label].transform(residual)
        distances = euclidean_distance_matrix(coordinates, train_coordinates[label])
        query = mean_k_nearest(distances, N_NEIGHBORS)
        validation_distance[indices] = query
        validation_percentile[indices] = empirical_reference_percentiles(train_distances[label], query)
    record_hard_check("validation residual-PCA distances finite", np.isfinite(validation_distance).all(), bool(np.isfinite(validation_distance).all()), True)
    record_hard_check("validation residual-PCA percentiles in [0,100]", np.isfinite(validation_percentile).all() and np.all((validation_percentile >= 0) & (validation_percentile <= 100)), f"min={validation_percentile.min()}, max={validation_percentile.max()}", "[0,100]")

    scores = reliability_sorted[identity_columns + ["Day4PredictedLabel", "Day4Correct", "PrimaryConfidenceMargin", "Day3StdViewProbability", "ModelAgreement"]].copy()
    scores["CentroidReferencePercentile"] = existing_sorted["CentroidReferencePercentile"].to_numpy(float)
    scores["KNN10ReferencePercentile"] = existing_sorted["SelectedKNNReferencePercentile"].to_numpy(float)
    scores["ResidualPCA10NNDistance"] = validation_distance
    scores["ResidualPCAReferencePercentile"] = validation_percentile
    errors = ~scores["Day4Correct"].to_numpy(bool)
    record_hard_check("validation contains the locked 10 Day-4 errors", int(errors.sum()) == 10, int(errors.sum()), 10)

    # compare residual-pca against the earlier centroid and knn typicality scores
    method_fields = {
        "Centroid cosine": "CentroidReferencePercentile",
        "Existing kNN (k=10)": "KNN10ReferencePercentile",
        "Residual PCA (32-D whitened, k=10)": "ResidualPCAReferencePercentile",
    }
    comparison_rows: list[dict[str, Any]] = []
    for method, field in method_fields.items():
        values = scores[field].to_numpy(float)
        comparison_rows.append(
            {
                "Method": method,
                "RiskScore": field,
                "SpearmanWithPrimaryConfidenceMargin": compute_spearman(values, scores["PrimaryConfidenceMargin"].to_numpy(float)),
                "ErrorDetectionAUROC": float(roc_auc_score(errors.astype(int), values)),
                "CorrectMeanPercentile": float(np.mean(values[~errors])),
                "CorrectMedianPercentile": float(np.median(values[~errors])),
                "IncorrectMeanPercentile": float(np.mean(values[errors])),
                "IncorrectMedianPercentile": float(np.median(values[errors])),
                "DataScope": DATA_SCOPE,
                "TestStatus": TEST_STATUS,
            }
        )
    method_comparison = pd.DataFrame(comparison_rows)

    record_hard_check("frozen routing artifact status", routing.get("artifact_status") == "FROZEN", routing.get("artifact_status"), "FROZEN")
    rule_lookup = {rule["operating_point_id"]: rule for rule in routing["operating_points"]}
    record_hard_check("frozen balanced rule exists", "balanced_agreement_view_sd" in rule_lookup, sorted(rule_lookup), "contains balanced_agreement_view_sd")
    balanced_rule = rule_lookup["balanced_agreement_view_sd"]
    thresholds = balanced_rule["thresholds"]
    record_hard_check("frozen balanced rule has no feature gate", thresholds.get("feature_percentile_threshold") is None, thresholds.get("feature_percentile_threshold"), None)
    base_accepted = (
        scores["PrimaryConfidenceMargin"].to_numpy(float) >= float(thresholds["confidence_threshold"])
    )
    if bool(thresholds["model_agreement_required"]):
        base_accepted &= scores["ModelAgreement"].to_numpy(bool)
    if thresholds["view_sd_threshold"] is not None:
        base_accepted &= scores["Day3StdViewProbability"].to_numpy(float) <= float(thresholds["view_sd_threshold"])
    base_metrics = compute_routing_metrics(scores, base_accepted)
    expected_performance = balanced_rule["validation_performance"]
    record_hard_check("frozen balanced rule accepted count reproduces", base_metrics["AcceptedEyes"] == expected_performance["accepted_eyes"], base_metrics["AcceptedEyes"], expected_performance["accepted_eyes"])
    record_hard_check("frozen balanced rule error count reproduces", base_metrics["AcceptedErrors"] == expected_performance["accepted_error_count"], base_metrics["AcceptedErrors"], expected_performance["accepted_error_count"])
    record_hard_check("frozen balanced rule FN reproduces", base_metrics["AcceptedFN"] == expected_performance["accepted_false_negatives"], base_metrics["AcceptedFN"], expected_performance["accepted_false_negatives"])

    # evaluate exactly one prespecified follow-up gate at the 95th percentile
    augmented_accepted = base_accepted & (validation_percentile <= RESIDUAL_PERCENTILE_CUTOFF)
    augmented_metrics = compute_routing_metrics(scores, augmented_accepted)
    scores["FrozenBalancedAccepted"] = base_accepted
    scores["ResidualPCAAbove95"] = validation_percentile > RESIDUAL_PERCENTILE_CUTOFF
    scores["BalancedPlusResidualPCAAccepted"] = augmented_accepted
    scores["AdditionalDeferralByResidualPCA"] = base_accepted & ~augmented_accepted
    scores["DataScope"] = DATA_SCOPE
    scores["TestStatus"] = TEST_STATUS

    balanced_error_mask = base_accepted & errors
    record_hard_check("frozen balanced rule has exactly one accepted error", int(balanced_error_mask.sum()) == 1, int(balanced_error_mask.sum()), 1)
    error_row = scores.loc[balanced_error_mask].copy()
    caught = bool(error_row["ResidualPCAAbove95"].iloc[0])
    error_check = error_row[
        [
            "EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel",
            "Day4PredictedLabel", "PrimaryConfidenceMargin", "Day3StdViewProbability",
            "ModelAgreement", "CentroidReferencePercentile", "KNN10ReferencePercentile",
            "ResidualPCA10NNDistance", "ResidualPCAReferencePercentile",
            "FrozenBalancedAccepted", "ResidualPCAAbove95",
            "BalancedPlusResidualPCAAccepted", "DataScope", "TestStatus",
        ]
    ].copy()
    error_check["CaughtByPrespecifiedResidualPCA95Gate"] = caught

    routing_rows = []
    for analysis, accepted, metrics in [
        ("Frozen balanced rule", base_accepted, base_metrics),
        ("Frozen balanced rule + defer ResidualPCA percentile >95", augmented_accepted, augmented_metrics),
    ]:
        routing_rows.append(
            {
                "Analysis": analysis,
                "ConfidenceThreshold": float(thresholds["confidence_threshold"]),
                "ModelAgreementRequired": bool(thresholds["model_agreement_required"]),
                "ViewSDThreshold": float(thresholds["view_sd_threshold"]),
                "ResidualPCAPercentileGateUsed": analysis != "Frozen balanced rule",
                "ResidualPCAPercentileComparator": "<=",
                "ResidualPCAPercentileThreshold": RESIDUAL_PERCENTILE_CUTOFF if analysis != "Frozen balanced rule" else math.nan,
                **metrics,
                "AdditionalDeferredVsFrozen": int(base_accepted.sum() - accepted.sum()),
                "ErrorsRemovedVsFrozen": int(base_metrics["AcceptedErrors"] - metrics["AcceptedErrors"]),
                "FNRemovedVsFrozen": int(base_metrics["AcceptedFN"] - metrics["AcceptedFN"]),
                "FPRemovedVsFrozen": int(base_metrics["AcceptedFP"] - metrics["AcceptedFP"]),
                "DataScope": DATA_SCOPE,
                "TestStatus": TEST_STATUS,
            }
        )
    routing_check = pd.DataFrame(routing_rows)
    record_hard_check("exactly one additional routing gate evaluated", len(routing_check) == 2 and int(routing_check["ResidualPCAPercentileGateUsed"].sum()) == 1, f"rows={len(routing_check)}, additional={routing_check['ResidualPCAPercentileGateUsed'].sum()}", "2 rows: frozen baseline plus exactly 1 follow-up")

    output_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    output_dir = PROJECT_ROOT / "outputs" / "audits" / output_id
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "scores": output_dir / "residual_pca_validation_scores.csv",
        "references": output_dir / "residual_pca_training_reference_distances.csv",
        "comparison": output_dir / "residual_pca_method_comparison.csv",
        "routing": output_dir / "residual_pca_routing_check.csv",
        "accepted_error": output_dir / "residual_pca_balanced_accepted_error_check.csv",
        "checks": output_dir / "residual_pca_checks.csv",
        "parameters": output_dir / "residual_pca_parameters.npz",
        "metadata": output_dir / "residual_pca_metadata.json",
        "summary": output_dir / "residual_pca_summary.md",
    }
    scores.to_csv(paths["scores"], index=False, float_format="%.17g")
    pd.DataFrame(train_reference_rows).sort_values(["TrueLabel", "EyeExamID"], kind="mergesort").to_csv(paths["references"], index=False, float_format="%.17g")
    method_comparison.to_csv(paths["comparison"], index=False, float_format="%.17g")
    routing_check.to_csv(paths["routing"], index=False, float_format="%.17g")
    error_check.to_csv(paths["accepted_error"], index=False, float_format="%.17g")
    np.savez_compressed(
        paths["parameters"],
        classifier_direction=classifier_direction,
        Normal_class_mean=class_means["Normal"],
        Abnormal_class_mean=class_means["Abnormal"],
        Normal_pca_components=pca_models["Normal"].components_,
        Abnormal_pca_components=pca_models["Abnormal"].components_,
        Normal_pca_mean=pca_models["Normal"].mean_,
        Abnormal_pca_mean=pca_models["Abnormal"].mean_,
        Normal_pca_explained_variance=pca_models["Normal"].explained_variance_,
        Abnormal_pca_explained_variance=pca_models["Abnormal"].explained_variance_,
    )

    # rehash every locked input to confirm the analysis did not modify anything
    hashes_after = {path: file_sha256(path) for path in EXPECTED_SHA256}
    record_hard_check("all locked inputs unchanged during analysis", all(hashes_after[path] == expected for path, expected in EXPECTED_SHA256.items()), "all compared", "all pinned SHA-256 unchanged")
    record_hard_check("loaded input set equals fixed allowlist", LOADED_INPUTS == {path.resolve() for path in EXPECTED_SHA256}, len(LOADED_INPUTS), len(EXPECTED_SHA256))
    record_hard_check("no loaded input path names a test artifact", not any("test" in path.name.casefold() for path in LOADED_INPUTS), sorted(path.name for path in LOADED_INPUTS), "no test-named inputs")
    record_hard_check("no model retraining or image inference performed", True, True, True)
    record_hard_check("frozen routing artifact not modified", hashes_after[FROZEN_ROUTING] == EXPECTED_SHA256[FROZEN_ROUTING], hashes_after[FROZEN_ROUTING], EXPECTED_SHA256[FROZEN_ROUTING])
    pd.DataFrame(CHECKS).to_csv(paths["checks"], index=False)

    residual_row = method_comparison.loc[method_comparison["Method"].str.startswith("Residual PCA")].iloc[0]
    centroid_row = method_comparison.loc[method_comparison["Method"].eq("Centroid cosine")].iloc[0]
    knn_row = method_comparison.loc[method_comparison["Method"].eq("Existing kNN (k=10)")].iloc[0]
    # save the exact method, provenance, and comparison results for reproducibility
    metadata = {
        "run_id": output_id,
        "started_at": started.isoformat(),
        "completed_at": datetime.now().astimezone().isoformat(),
        "stage": "Prof. Ahn residual-PCA feature-typicality follow-up",
        "data_scope": DATA_SCOPE,
        "test_status": TEST_STATUS,
        "model_retrained": False,
        "image_inference_performed": False,
        "frozen_routing_rules_modified": False,
        "method": {
            "classifier_direction": "Day-4 Abnormal classifier weight minus Normal classifier weight; unit normalized",
            "class_reference_rule": "all correctly classified training eyes within true class; no confidence filter",
            "class_reference_counts": EXPECTED_CORRECT_REFERENCES,
            "centering": "subtract predicted/true class correctly-classified training mean",
            "residualization": "remove orthogonal projection onto the unit classifier direction",
            "pca": "separate per class; sklearn PCA whiten=True; svd_solver=full; up to 32 components",
            "components_by_class": components_by_class,
            "distance": "mean Euclidean distance in whitened residual-PCA space to 10 same-class references",
            "training_calibration": "leave-one-out neighbors; self-distance set to infinity",
            "validation_class_choice": "Day-4 predicted class only",
            "percentile": "right-continuous empirical percentile: 100 * count(training LOO distance <= query distance) / class reference count",
            "routing_followup": "exactly one prespecified gate: frozen balanced rule AND ResidualPCAReferencePercentile <= 95",
        },
        "checkpoint": checkpoint_info,
        "input_sha256": {str(path.relative_to(PROJECT_ROOT)): value for path, value in hashes_after.items()},
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "comparison": {
            "centroid_error_detection_auroc": float(centroid_row["ErrorDetectionAUROC"]),
            "knn10_error_detection_auroc": float(knn_row["ErrorDetectionAUROC"]),
            "residual_pca_error_detection_auroc": float(residual_row["ErrorDetectionAUROC"]),
            "residual_pca_spearman_with_confidence": float(residual_row["SpearmanWithPrimaryConfidenceMargin"]),
        },
        "balanced_accepted_error": {
            "eye_exam_id": str(error_row["EyeExamID"].iloc[0]),
            "residual_pca_percentile": float(error_row["ResidualPCAReferencePercentile"].iloc[0]),
            "caught_by_percentile_above_95": caught,
        },
        "outputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()},
        "test_data_loaded": False,
        "test_set_evaluated": False,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    summary_lines = [
        "# Residual-PCA feature typicality — validation-only follow-up",
        "",
        f"Data scope: **{DATA_SCOPE}**  ",
        f"Test status: **{TEST_STATUS}**",
        "",
        "## Method",
        "",
        "The frozen Day-4 classifier direction was removed from class-centered 512-D pooled features. Separate 32-component whitened PCAs were fit using all correctly classified training eyes in each true class (121 Normal, 388 Abnormal). Typicality is the mean Euclidean distance to 10 same-class training references. Training calibration distances exclude the query eye itself; validation uses the Day-4 predicted class.",
        "",
        "## Score comparison",
        "",
        "| Method | Spearman with confidence | Error-detection AUROC |",
        "|---|---:|---:|",
    ]
    for _, row in method_comparison.iterrows():
        summary_lines.append(f"| {row['Method']} | {row['SpearmanWithPrimaryConfidenceMargin']:.6f} | {row['ErrorDetectionAUROC']:.6f} |")
    summary_lines += [
        "",
        "## Frozen balanced-rule check",
        "",
        f"The one error accepted by the frozen balanced validation rule had residual-PCA percentile **{float(error_row['ResidualPCAReferencePercentile'].iloc[0]):.6f}**. It **{'was' if caught else 'was not'}** caught by the prespecified `>95` defer gate.",
        "",
        "| Rule | Coverage | Accepted errors | Accepted FN | Accepted FP |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in routing_check.iterrows():
        summary_lines.append(f"| {row['Analysis']} | {100*row['Coverage']:.2f}% | {int(row['AcceptedErrors'])} | {int(row['AcceptedFN'])} | {int(row['AcceptedFP'])} |")
    summary_lines += [
        "",
        "This is a fixed validation-only follow-up. No cutoff was searched, no model was retrained, and the frozen routing artifact was not modified.",
        "",
        TEST_STATUS + ".",
        "",
    ]
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")

    print("RESIDUAL-PCA FEATURE TYPICALITY — VALIDATION ONLY")
    print(f"Output directory            : {output_dir}")
    print(f"Training references         : Normal={EXPECTED_CORRECT_REFERENCES['Normal']}, Abnormal={EXPECTED_CORRECT_REFERENCES['Abnormal']}")
    print(f"PCA components              : Normal={components_by_class['Normal']}, Abnormal={components_by_class['Abnormal']}")
    print("\nMETHOD COMPARISON")
    for _, row in method_comparison.iterrows():
        print(f"  {row['Method']}: rho(confidence)={row['SpearmanWithPrimaryConfidenceMargin']:.6f}, error AUROC={row['ErrorDetectionAUROC']:.6f}")
    print("\nFROZEN BALANCED RULE")
    print(f"  Before: coverage={100*base_metrics['Coverage']:.2f}%, errors={base_metrics['AcceptedErrors']}, FN={base_metrics['AcceptedFN']}, FP={base_metrics['AcceptedFP']}")
    print(f"  After >95 defer gate: coverage={100*augmented_metrics['Coverage']:.2f}%, errors={augmented_metrics['AcceptedErrors']}, FN={augmented_metrics['AcceptedFN']}, FP={augmented_metrics['AcceptedFP']}")
    print(f"  Original accepted error residual-PCA percentile={float(error_row['ResidualPCAReferencePercentile'].iloc[0]):.6f}; caught={caught}")
    print(f"\nChecks: {len(CHECKS)}/{len(CHECKS)} PASS")
    print(TEST_STATUS + ".")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyError as error:
        print(f"HARD FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
