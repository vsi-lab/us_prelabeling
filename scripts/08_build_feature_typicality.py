
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

# set this before torch loads so cuda behavior matches the locked day-4 run
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAY4_RUN_ID = "20260813T170240_569268-0700"
DAY4_RUN_DIR = (
    PROJECT_ROOT / "outputs" / "models" / "multiview_resnet18" / DAY4_RUN_ID
).resolve()
DAY4_SCRIPT_PATH = (PROJECT_ROOT / "scripts" / "07_train_multiview.py").resolve()
CHECKPOINT_PATH = (DAY4_RUN_DIR / "best_model.pt").resolve()
CONFIG_USED_PATH = (DAY4_RUN_DIR / "config_used.yaml").resolve()
THRESHOLD_PATH = (DAY4_RUN_DIR / "classification_threshold.json").resolve()
LOCKED_VAL_PREDICTIONS_PATH = (DAY4_RUN_DIR / "val_eye_predictions.csv").resolve()
RUN_METADATA_PATH = (DAY4_RUN_DIR / "run_metadata.json").resolve()

TRAIN_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_inputs" / "train_images.csv"
).resolve()
VAL_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_inputs" / "val_images.csv"
).resolve()
ALLOWED_MANIFEST_PATHS = {TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH}
FORBIDDEN_TEST_BASENAME = "test_images.csv"

FEATURE_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "features" / "multiview_resnet18"
)
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "feature_typicality"
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"

EXPECTED_DAY4_SCRIPT_SHA256 = (
    "058de002bf0599602b9787bce887e2478571a9865e7a78c7c6fce3a279ab577c"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "3c6d402ddbd2d1055cb68458bc6f5a7c880ac683a4527ff5171cc6f2408c754b"
)
EXPECTED_CONFIG_USED_SHA256 = (
    "983120e1e0fba2bb9e56e3ac3fa0f774eda60822dea1ceb10adc01b5f12466bf"
)
EXPECTED_VAL_PREDICTIONS_SHA256 = (
    "194257a476648e630870e1a343b4dfcbe82897a949b1774c07df6ffdddac1234"
)
EXPECTED_THRESHOLD_SHA256 = (
    "369ae63c078a1aac105e5481c66740f80bf1b6cffde3bfb0253fd70d7dd07d41"
)
EXPECTED_RUN_METADATA_SHA256 = (
    "ed6edf2cb4ce415e6d965fadd5ad9fb413a79600aeffe47a0d6791edec1f5d10"
)
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "5df6765a3df29ca3f107758369fd1c17a92b1fcd336ec79991c823482711c4ad"
)
EXPECTED_VAL_MANIFEST_SHA256 = (
    "91846df6c0f3d1a153e1c8a03da4e92be561cab29736236774ac32ef9802e567"
)
EXPECTED_TRAIN_EYES = 560
EXPECTED_VAL_EYES = 120
EXPECTED_FEATURE_DIMENSION = 512
MINIMUM_REFERENCE_EYES_PER_CLASS = 20
REFERENCE_FRACTION = 0.50
CLASS_TO_INDEX = {"Normal": 0, "Abnormal": 1}
INDEX_TO_CLASS = {0: "Normal", 1: "Abnormal"}
PROBABILITY_REPRODUCTION_RTOL = 1e-6
PROBABILITY_REPRODUCTION_ATOL = 1e-7
NORM_EPSILON = 1e-12
REFERENCE_PERCENTILES = [90.0, 95.0, 97.5]
VALIDATION_CONFIDENCE_PERCENTILES = [50.0, 75.0, 90.0, 95.0]


class SafetyError(RuntimeError):
    """used when a locked input or integrity check fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_locked_file(path: Path, expected_sha256: str, description: str) -> str:
    # fail immediately if a supposedly frozen artifact has changed
    """check that a locked file exists and still has the expected hash."""
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError(f"Held-out test manifest access attempted as {description}.")
    if not resolved.is_file():
        raise SafetyError(f"Missing locked {description}: {resolved}")
    observed = file_sha256(resolved)
    if observed != expected_sha256:
        raise SafetyError(
            f"Locked {description} SHA-256 mismatch: observed={observed}, "
            f"expected={expected_sha256}."
        )
    return observed


def check_allowed_manifest_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError("Held-out test manifest access was attempted.")
    if resolved not in ALLOWED_MANIFEST_PATHS:
        raise SafetyError(f"Manifest is outside the train/validation allowlist: {resolved}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"Expected a JSON object: {path}")
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def import_locked_day4():
    # reuse the exact day-4 classes and preprocessing instead of duplicating them
    """import the locked day-4 code so the exact model definitions can be reused."""
    spec = importlib.util.spec_from_file_location(
        "locked_day4_multiview_for_typicality", DAY4_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise SafetyError("Could not construct an import specification for Day 4.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_locked_config() -> dict[str, Any]:
    with CONFIG_USED_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise SafetyError("Locked Day-4 config_used.yaml is not a mapping.")
    # these settings describe the day-4 run that produced the frozen checkpoint
    exact_requirements = {
        "model": "resnet18",
        "pretrained": True,
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "pooling": "max",
        "image_size": 224,
        "batch_size": 4,
        "seed": 42,
        "num_workers": 0,
        "max_views": 6,
    }
    mismatches = {
        key: {"observed": config.get(key), "expected": expected}
        for key, expected in exact_requirements.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise SafetyError(f"Locked Day-4 configuration mismatch: {mismatches}")
    return config


def check_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    # confirm that the checkpoint matches the expected architecture and config
    required = {
        "model_state_dict",
        "class_to_index",
        "architecture",
        "pooling",
        "feature_dimension",
        "pretrained_weights",
        "config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise SafetyError(f"Locked checkpoint is missing fields: {missing}")
    if checkpoint["architecture"] != "resnet18":
        raise SafetyError("Checkpoint architecture is not ResNet-18.")
    if checkpoint["pooling"] != "feature-wise max":
        raise SafetyError("Checkpoint pooling is not feature-wise max.")
    if int(checkpoint["feature_dimension"]) != EXPECTED_FEATURE_DIMENSION:
        raise SafetyError("Checkpoint feature dimension is not 512.")
    if checkpoint["class_to_index"] != CLASS_TO_INDEX:
        raise SafetyError("Checkpoint class mapping differs from Normal=0/Abnormal=1.")
    if checkpoint["pretrained_weights"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise SafetyError("Checkpoint pretrained-weights identifier is unexpected.")
    if checkpoint["config"] != config:
        raise SafetyError("Checkpoint config differs from locked config_used.yaml.")


def check_run_metadata(metadata: dict[str, Any], threshold: float) -> None:
    expected_values = {
        "run_timestamp": DAY4_RUN_ID,
        "model_architecture": "resnet18",
        "pooling": "feature-wise MAX across real view feature vectors",
        "feature_dimension": EXPECTED_FEATURE_DIMENSION,
        "classification_threshold": threshold,
        "training_manifest_sha256": EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": EXPECTED_VAL_MANIFEST_SHA256,
        "training_script_sha256": EXPECTED_DAY4_SCRIPT_SHA256,
        "config_used_sha256": EXPECTED_CONFIG_USED_SHA256,
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise SafetyError(f"Locked Day-4 run metadata mismatch: {mismatches}")
    for field in [
        "test_manifest_loaded",
        "test_set_evaluated",
        "test_predictions_created",
    ]:
        if metadata.get(field) is not False:
            raise SafetyError(f"Locked Day-4 metadata does not confirm {field}=false.")


def make_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")


def get_git_info() -> tuple[str | None, list[str]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        status = []
    return commit, status


def create_loader(day4, records: list[dict], transform, config: dict, device):
    # use deterministic validation preprocessing for both train and validation embeddings
    dataset = day4.EyeExamDataset(records, transform)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + 1)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=day4.seed_worker,
        collate_fn=day4.collate_eye_batch,
        generator=generator,
    )


def extract_embeddings(
    model,
    loader,
    records: list[dict],
    device: torch.device,
    classification_threshold: float,
    threshold_logit: float,
    split_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """extract frozen pooled eye features and the matching predictions."""
    rows: list[dict[str, Any]] = []
    features_by_index: dict[int, np.ndarray] = {}
    # feature extraction is inference-only; gradients are never needed here
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            eye_logits, pooled_features, _ = model(
                images,
                view_mask,
                return_diagnostics=True,
            )
            probabilities = torch.softmax(eye_logits, dim=1)[:, 1]
            logits_cpu = eye_logits.detach().cpu().numpy().astype(np.float64)
            probability_cpu = probabilities.detach().cpu().numpy().astype(np.float64)
            pooled_cpu = pooled_features.detach().cpu().numpy().astype(np.float64)
            eye_indices = batch["eye_indices"].detach().cpu().numpy().astype(int)

            if pooled_cpu.ndim != 2 or pooled_cpu.shape[1] != EXPECTED_FEATURE_DIMENSION:
                raise SafetyError(
                    f"{split_name}: unexpected pooled feature shape {pooled_cpu.shape}."
                )
            if not np.isfinite(pooled_cpu).all() or not np.isfinite(logits_cpu).all():
                raise SafetyError(f"{split_name}: nonfinite feature or logit encountered.")
            if not np.isfinite(probability_cpu).all():
                raise SafetyError(f"{split_name}: nonfinite probability encountered.")

            for offset, eye_index in enumerate(eye_indices):
                if eye_index in features_by_index:
                    raise SafetyError(
                        f"{split_name}: Eye record index was extracted more than once: "
                        f"{eye_index}."
                    )
                record = records[eye_index]
                probability = float(probability_cpu[offset])
                if not 0.0 <= probability <= 1.0:
                    raise SafetyError(
                        f"{split_name}: probability outside [0,1] for "
                        f"{record['EyeExamID']}."
                    )
                score = float(logits_cpu[offset, 1] - logits_cpu[offset, 0])
                probability_prediction = int(probability >= classification_threshold)
                logit_prediction = int(score >= threshold_logit)
                if probability_prediction != logit_prediction:
                    raise SafetyError(
                        f"{split_name}: probability and logit threshold rules disagree "
                        f"for {record['EyeExamID']}."
                    )
                predicted_label = INDEX_TO_CLASS[probability_prediction]
                correct = predicted_label == record["EyeLabel"]
                rows.append(
                    {
                        "_RecordIndex": int(eye_index),
                        "EyeExamID": record["EyeExamID"],
                        "ResearchSubjectID": record["ResearchSubjectID"],
                        "EncounterID": record["EncounterID"],
                        "Laterality": record["Laterality"],
                        "TrueLabel": record["EyeLabel"],
                        "PredictedLabel": predicted_label,
                        "EyeAbnormalProbability": probability,
                        "Correct": bool(correct),
                        "PrimaryConfidenceMargin": abs(score - threshold_logit),
                        "NumberOfViews": int(record["NumberOfViews"]),
                    }
                )
                features_by_index[int(eye_index)] = pooled_cpu[offset].copy()

            if batch_index % 25 == 0 or batch_index == len(loader):
                completed = min(batch_index * int(loader.batch_size), len(records))
                print(
                    f"  {split_name}: extracted {completed}/{len(records)} eyes",
                    flush=True,
                )

    if len(rows) != len(records) or set(features_by_index) != set(range(len(records))):
        raise SafetyError(
            f"{split_name}: expected {len(records)} embeddings, observed {len(rows)}."
        )
    rows_frame = pd.DataFrame(rows).sort_values("_RecordIndex", kind="mergesort")
    ordered_features = np.stack(
        [features_by_index[index] for index in rows_frame["_RecordIndex"].astype(int)]
    )
    rows_frame = rows_frame.drop(columns="_RecordIndex").reset_index(drop=True)
    if rows_frame["EyeExamID"].duplicated().any():
        raise SafetyError(f"{split_name}: duplicate pooled EyeExamID embeddings.")
    return rows_frame, ordered_features


def normalize_features(features: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    # normalization is only for distance calculations; raw features are still saved
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != EXPECTED_FEATURE_DIMENSION:
        raise SafetyError(f"{name}: inconsistent feature dimension {values.shape}.")
    if not np.isfinite(values).all():
        raise SafetyError(f"{name}: pooled embedding contains NaN or infinity.")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or (norms <= NORM_EPSILON).any():
        raise SafetyError(f"{name}: nonfinite or near-zero feature norm.")
    return values / norms[:, None], norms


def cosine_distance(normalized_features: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    similarities = np.clip(normalized_features @ centroid, -1.0, 1.0)
    return 1.0 - similarities


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "Mean": float(np.mean(values)),
        "StandardDeviation": float(np.std(values, ddof=0)),
        "Median": float(np.median(values)),
        "Percentile90": float(np.percentile(values, 90.0, method="linear")),
        "Percentile95": float(np.percentile(values, 95.0, method="linear")),
        "Percentile97_5": float(np.percentile(values, 97.5, method="linear")),
        "Maximum": float(np.max(values)),
    }


def reference_percentile(reference_distances: np.ndarray, value: float) -> float:
    ordered = np.sort(np.asarray(reference_distances, dtype=np.float64))
    rank = int(np.searchsorted(ordered, value, side="right"))
    return 100.0 * rank / len(ordered)


def choose_reference_eyes(
    train_rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    selections: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    # build each reference set from correctly classified training eyes only
    for label in ["Normal", "Abnormal"]:
        class_rows = train_rows.loc[train_rows["TrueLabel"].eq(label)]
        correct = class_rows.loc[class_rows["Correct"]].copy()
        correct = correct.sort_values(
            ["PrimaryConfidenceMargin", "EyeExamID"],
            ascending=[False, True],
            kind="mergesort",
        )
        # keep the most confident half, rounding up when the count is odd
        reference_count = (len(correct) + 1) // 2
        reference_indices = correct.index.to_numpy(dtype=int)[:reference_count]
        if reference_count < MINIMUM_REFERENCE_EYES_PER_CLASS:
            raise SafetyError(
                f"{label} reference set has {reference_count} eyes; minimum is "
                f"{MINIMUM_REFERENCE_EYES_PER_CLASS}."
            )
        selections[label] = reference_indices
        counts[label] = {
            "total_training_eyes": int(len(class_rows)),
            "correctly_classified_training_eyes": int(len(correct)),
            "reference_eyes": int(reference_count),
        }
    return selections, counts


def make_class_centroids(
    normalized_train: np.ndarray,
    selections: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for label in ["Normal", "Abnormal"]:
        selected = normalized_train[selections[label]]
        if selected.size == 0:
            raise SafetyError(f"{label} reference class is missing.")
        mean_vector = selected.mean(axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if not np.isfinite(mean_vector).all() or not np.isfinite(norm) or norm <= NORM_EPSILON:
            raise SafetyError(f"{label} reference centroid is nonfinite or near zero.")
        centroids[label] = mean_vector / norm
    return centroids


def check_validation_reproduction(
    extracted: pd.DataFrame,
    locked: pd.DataFrame,
) -> dict[str, float]:
    required = {
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "TrueLabel",
        "EyeAbnormalProbability",
        "PredictedLabel",
        "Correct",
    }
    missing = sorted(required - set(locked.columns))
    if missing:
        raise SafetyError(f"Locked Day-4 validation predictions missing: {missing}")
    if len(locked) != EXPECTED_VAL_EYES or locked["EyeExamID"].duplicated().any():
        raise SafetyError("Locked Day-4 validation predictions are not 120 unique eyes.")
    # the fresh extraction should reproduce the frozen day-4 validation output
    left = extracted.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    right = locked.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    if not left["EyeExamID"].equals(right["EyeExamID"].astype(str)):
        raise SafetyError("Extracted validation EyeExamIDs differ from locked Day-4 output.")
    for field in [
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "TrueLabel",
        "PredictedLabel",
    ]:
        if not left[field].astype(str).equals(right[field].astype(str)):
            raise SafetyError(f"Extracted validation {field} differs from locked Day 4.")
    locked_correct = right["Correct"]
    if locked_correct.dtype != bool:
        normalized = locked_correct.astype("string").str.casefold()
        if not normalized.isin(["true", "false"]).all():
            raise SafetyError("Locked validation Correct column contains invalid values.")
        locked_correct = normalized.eq("true")
    if not left["Correct"].astype(bool).equals(locked_correct.astype(bool)):
        raise SafetyError("Extracted validation correctness differs from locked Day 4.")
    observed = left["EyeAbnormalProbability"].to_numpy(dtype=np.float64)
    expected = pd.to_numeric(
        right["EyeAbnormalProbability"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(expected).all():
        raise SafetyError("Locked Day-4 validation probabilities are nonfinite.")
    absolute = np.abs(observed - expected)
    if not np.allclose(
        observed,
        expected,
        rtol=PROBABILITY_REPRODUCTION_RTOL,
        atol=PROBABILITY_REPRODUCTION_ATOL,
    ):
        raise SafetyError(
            "Frozen feature extraction did not reproduce Day-4 validation "
            f"probabilities; max_abs_difference={float(absolute.max())}."
        )
    return {
        "maximum_absolute_probability_difference": float(absolute.max()),
        "mean_absolute_probability_difference": float(absolute.mean()),
        "rtol": PROBABILITY_REPRODUCTION_RTOL,
        "atol": PROBABILITY_REPRODUCTION_ATOL,
    }


def attach_features(frame: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
    columns = [f"Feature_{index:03d}" for index in range(features.shape[1])]
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(features, columns=columns)],
        axis=1,
    )


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for correct_value, label in [(True, "Correct"), (False, "Incorrect")]:
        group = validation.loc[validation["Correct"].eq(correct_value)]
        distances = group["FeatureTypicalityPredictedClass"].to_numpy(dtype=float)
        percentiles = group["PredictedClassReferencePercentile"].to_numpy(dtype=float)
        rows.append(
            {
                "PredictionGroup": label,
                "Count": int(len(group)),
                "TypicalityDistanceMean": float(np.mean(distances)),
                "TypicalityDistanceMedian": float(np.median(distances)),
                "TypicalityDistanceStandardDeviation": float(np.std(distances, ddof=0)),
                "ReferencePercentileMean": float(np.mean(percentiles)),
                "ReferencePercentileMedian": float(np.median(percentiles)),
            }
        )
    return pd.DataFrame(rows)


def make_high_confidence_grid(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # this grid is descriptive only and does not define a routing rule
    margins = validation["PrimaryConfidenceMargin"].to_numpy(dtype=np.float64)
    quantile_rows = []
    grid_rows = []
    incorrect = validation.loc[~validation["Correct"]]
    for confidence_percentile in VALIDATION_CONFIDENCE_PERCENTILES:
        cutoff = float(
            np.percentile(margins, confidence_percentile, method="linear")
        )
        high_confidence_incorrect = incorrect.loc[
            incorrect["PrimaryConfidenceMargin"].ge(cutoff)
        ]
        quantile_rows.append(
            {
                "ValidationConfidencePercentile": confidence_percentile,
                "PrimaryConfidenceMarginCutoff": cutoff,
                "AllValidationEyesAtOrAbove": int(
                    validation["PrimaryConfidenceMargin"].ge(cutoff).sum()
                ),
                "IncorrectEyesAtOrAbove": int(len(high_confidence_incorrect)),
            }
        )
        for atypical_percentile in REFERENCE_PERCENTILES:
            joint = high_confidence_incorrect.loc[
                high_confidence_incorrect[
                    "PredictedClassReferencePercentile"
                ].ge(atypical_percentile)
            ]
            grid_rows.append(
                {
                    "ValidationConfidencePercentile": confidence_percentile,
                    "PrimaryConfidenceMarginCutoff": cutoff,
                    "ReferenceDistancePercentileCutoff": atypical_percentile,
                    "IncorrectHighConfidenceEyes": int(len(high_confidence_incorrect)),
                    "IncorrectHighConfidenceAndAtypicalEyes": int(len(joint)),
                    "MaximumReferencePercentileAmongIncorrectHighConfidence": (
                        float(
                            high_confidence_incorrect[
                                "PredictedClassReferencePercentile"
                            ].max()
                        )
                        if len(high_confidence_incorrect)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(quantile_rows), pd.DataFrame(grid_rows)


def save_typicality_plot(validation: pd.DataFrame, path: Path) -> None:
    correct = validation.loc[
        validation["Correct"], "FeatureTypicalityPredictedClass"
    ].to_numpy(dtype=float)
    incorrect = validation.loc[
        ~validation["Correct"], "FeatureTypicalityPredictedClass"
    ].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    boxes = axis.boxplot(
        [correct, incorrect],
        tick_labels=[f"Correct\n(n={len(correct)})", f"Incorrect\n(n={len(incorrect)})"],
        patch_artist=True,
        showmeans=True,
    )
    boxes["boxes"][0].set_facecolor("#4C78A8")
    boxes["boxes"][1].set_facecolor("#E45756")
    for box in boxes["boxes"]:
        box.set_alpha(0.75)
    axis.set_ylabel("Cosine distance to predicted-class centroid")
    axis.set_title("Validation feature typicality by Day-4 prediction correctness")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confidence_plot(validation: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    for correct_value, label, color, marker in [
        (True, "Correct", "#4C78A8", "o"),
        (False, "Incorrect", "#E45756", "X"),
    ]:
        group = validation.loc[validation["Correct"].eq(correct_value)]
        axis.scatter(
            group["PrimaryConfidenceMargin"],
            group["FeatureTypicalityPredictedClass"],
            label=f"{label} (n={len(group)})",
            color=color,
            marker=marker,
            alpha=0.78,
            edgecolors="none",
        )
    axis.set_xlabel("Primary confidence margin (log-odds from frozen boundary)")
    axis.set_ylabel("Cosine distance to predicted-class centroid")
    axis.set_title("Validation confidence versus feature typicality")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def get_environment_info(day4, device: torch.device) -> dict[str, Any]:
    metadata = day4.environment_metadata(device)
    metadata["platform"] = platform.platform()
    return metadata


def main() -> None:
    # this stage only performs frozen inference and feature-space analysis
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat()
    audit_rows: list[dict[str, str]] = []

    def hard_check(name: str, condition: bool, details: str) -> None:
        audit_rows.append(
            {
                "Check": name,
                "Severity": "HARD FAIL",
                "Status": "PASS" if condition else "FAIL",
                "Details": details,
            }
        )
        if not condition:
            raise SafetyError(f"{name}: {details}")

    print("DAY-5 FEATURE TYPICALITY")
    print(f"Locked Day-4 run: {DAY4_RUN_ID}")
    print("Mode: frozen inference only; no model training")
    print("Input allowlist: training and validation manifests only")

    # verify all locked day-4 files before loading the model or metadata
    script_hash = check_locked_file(
        DAY4_SCRIPT_PATH, EXPECTED_DAY4_SCRIPT_SHA256, "Day-4 training script"
    )
    checkpoint_hash = check_locked_file(
        CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256, "Day-4 checkpoint"
    )
    config_hash = check_locked_file(
        CONFIG_USED_PATH, EXPECTED_CONFIG_USED_SHA256, "Day-4 config_used"
    )
    val_predictions_hash = check_locked_file(
        LOCKED_VAL_PREDICTIONS_PATH,
        EXPECTED_VAL_PREDICTIONS_SHA256,
        "Day-4 validation predictions",
    )
    threshold_hash = check_locked_file(
        THRESHOLD_PATH, EXPECTED_THRESHOLD_SHA256, "Day-4 classification threshold"
    )
    run_metadata_hash = check_locked_file(
        RUN_METADATA_PATH, EXPECTED_RUN_METADATA_SHA256, "Day-4 run metadata"
    )
    check_allowed_manifest_path(TRAIN_MANIFEST_PATH)
    check_allowed_manifest_path(VAL_MANIFEST_PATH)
    train_manifest_hash = check_locked_file(
        TRAIN_MANIFEST_PATH, EXPECTED_TRAIN_MANIFEST_SHA256, "training manifest"
    )
    val_manifest_hash = check_locked_file(
        VAL_MANIFEST_PATH, EXPECTED_VAL_MANIFEST_SHA256, "validation manifest"
    )
    for check_name, observed_hash in [
        ("Day-4 script hash locked", script_hash),
        ("Day-4 checkpoint hash locked", checkpoint_hash),
        ("Day-4 config_used hash locked", config_hash),
        ("Day-4 validation predictions hash locked", val_predictions_hash),
        ("Day-4 classification threshold hash locked", threshold_hash),
        ("Day-4 run metadata hash locked", run_metadata_hash),
        ("training manifest hash locked", train_manifest_hash),
        ("validation manifest hash locked", val_manifest_hash),
    ]:
        hard_check(check_name, True, f"sha256={observed_hash}")

    # load the exact day-4 implementation after all file hashes have been checked
    day4 = import_locked_day4()
    config = read_locked_config()
    threshold_document = read_json(THRESHOLD_PATH)
    locked_run_metadata = read_json(RUN_METADATA_PATH)
    threshold = float(threshold_document.get("classification_threshold", math.nan))
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise SafetyError("Locked Day-4 classification threshold is invalid.")
    if threshold_document.get("test_data_used") is not False:
        raise SafetyError("Locked threshold metadata does not confirm test isolation.")
    threshold_logit = float(math.log(threshold / (1.0 - threshold)))
    check_run_metadata(locked_run_metadata, threshold)
    hard_check(
        "locked Day-4 metadata and threshold consistent",
        True,
        f"run={DAY4_RUN_ID}; threshold={threshold:.17g}",
    )

    day4.set_reproducibility(int(config["seed"]))
    train_records, val_records, preflight_report = day4.preflight(config)
    hard_check(
        "training eyes == 560",
        len(train_records) == EXPECTED_TRAIN_EYES,
        f"observed={len(train_records)}, expected={EXPECTED_TRAIN_EYES}",
    )
    hard_check(
        "validation eyes == 120",
        len(val_records) == EXPECTED_VAL_EYES,
        f"observed={len(val_records)}, expected={EXPECTED_VAL_EYES}",
    )
    train_subjects = {record["ResearchSubjectID"] for record in train_records}
    val_subjects = {record["ResearchSubjectID"] for record in val_records}
    train_eye_ids = [record["EyeExamID"] for record in train_records]
    val_eye_ids = [record["EyeExamID"] for record in val_records]
    hard_check(
        "train/validation subjects disjoint",
        not (train_subjects & val_subjects),
        f"intersection_count={len(train_subjects & val_subjects)}",
    )
    hard_check(
        "training EyeExamIDs unique",
        len(train_eye_ids) == len(set(train_eye_ids)),
        f"unique={len(set(train_eye_ids))}, rows={len(train_eye_ids)}",
    )
    hard_check(
        "validation EyeExamIDs unique",
        len(val_eye_ids) == len(set(val_eye_ids)),
        f"unique={len(set(val_eye_ids))}, rows={len(val_eye_ids)}",
    )

    device = day4.choose_device(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SafetyError("Locked Day-4 checkpoint is not a mapping.")
    check_checkpoint(checkpoint, config)
    hard_check(
        "checkpoint architecture and config match locked Day 4",
        True,
        "ResNet-18; masked feature-wise MAX; Linear(512,2); strict config match",
    )
    # reconstruct the architecture, load the frozen weights, then disable gradients
    model = day4.FeatureMaxResNet18(weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hard_check(
        "frozen checkpoint feature dimension == 512",
        int(model.feature_dim) == EXPECTED_FEATURE_DIMENSION,
        f"observed={model.feature_dim}, expected={EXPECTED_FEATURE_DIMENSION}",
    )
    hard_check(
        "model parameters frozen",
        not any(parameter.requires_grad for parameter in model.parameters()),
        "all requires_grad flags must be False",
    )

    _, deterministic_transform = day4.build_transforms(config)
    train_loader = create_loader(day4, train_records, deterministic_transform, config, device)
    val_loader = create_loader(day4, val_records, deterministic_transform, config, device)

    print(f"Device: {device}")
    print("Extracting training embeddings with deterministic validation preprocessing...")
    train_rows, train_features = extract_embeddings(
        model,
        train_loader,
        train_records,
        device,
        threshold,
        threshold_logit,
        "training",
    )
    print("Extracting validation embeddings with identical deterministic preprocessing...")
    val_rows, val_features = extract_embeddings(
        model,
        val_loader,
        val_records,
        device,
        threshold,
        threshold_logit,
        "validation",
    )

    hard_check(
        "training pooled EyeExamIDs unique",
        len(train_rows) == EXPECTED_TRAIN_EYES and not train_rows["EyeExamID"].duplicated().any(),
        f"rows={len(train_rows)}, unique={train_rows['EyeExamID'].nunique()}",
    )
    hard_check(
        "validation pooled EyeExamIDs unique",
        len(val_rows) == EXPECTED_VAL_EYES and not val_rows["EyeExamID"].duplicated().any(),
        f"rows={len(val_rows)}, unique={val_rows['EyeExamID'].nunique()}",
    )
    hard_check(
        "feature dimensions consistent",
        train_features.shape == (EXPECTED_TRAIN_EYES, EXPECTED_FEATURE_DIMENSION)
        and val_features.shape == (EXPECTED_VAL_EYES, EXPECTED_FEATURE_DIMENSION),
        f"train={train_features.shape}, validation={val_features.shape}",
    )
    hard_check(
        "training embeddings finite",
        np.isfinite(train_features).all(),
        f"nonfinite_values={int((~np.isfinite(train_features)).sum())}",
    )
    hard_check(
        "validation embeddings finite",
        np.isfinite(val_features).all(),
        f"nonfinite_values={int((~np.isfinite(val_features)).sum())}",
    )
    hard_check(
        "all probabilities in [0,1]",
        train_rows["EyeAbnormalProbability"].between(0.0, 1.0).all()
        and val_rows["EyeAbnormalProbability"].between(0.0, 1.0).all(),
        "checked frozen training and validation eye probabilities",
    )

    locked_val = pd.read_csv(
        LOCKED_VAL_PREDICTIONS_PATH,
        dtype={
            "EyeExamID": "string",
            "ResearchSubjectID": "string",
            "EncounterID": "string",
            "Laterality": "string",
            "TrueLabel": "string",
            "PredictedLabel": "string",
        },
        float_precision="round_trip",
        low_memory=False,
    )
    reproduction = check_validation_reproduction(val_rows, locked_val)
    hard_check(
        "locked Day-4 validation probabilities reproduced",
        True,
        (
            f"max_abs_difference={reproduction['maximum_absolute_probability_difference']:.12g}; "
            f"rtol={PROBABILITY_REPRODUCTION_RTOL}; atol={PROBABILITY_REPRODUCTION_ATOL}"
        ),
    )

    # cosine distances use l2-normalized features, not the raw stored vectors
    normalized_train, train_norms = normalize_features(train_features, "training")
    normalized_val, val_norms = normalize_features(val_features, "validation")
    hard_check(
        "all pooled embedding norms positive",
        bool((train_norms > NORM_EPSILON).all() and (val_norms > NORM_EPSILON).all()),
        f"minimum_norm={min(float(train_norms.min()), float(val_norms.min())):.12g}",
    )

    # reference centroids are built from reliable training eyes, never validation eyes
    selections, reference_counts = choose_reference_eyes(train_rows)
    hard_check(
        "Normal reference class present and sufficiently large",
        len(selections["Normal"]) >= MINIMUM_REFERENCE_EYES_PER_CLASS,
        f"reference_eyes={len(selections['Normal'])}, minimum={MINIMUM_REFERENCE_EYES_PER_CLASS}",
    )
    hard_check(
        "Abnormal reference class present and sufficiently large",
        len(selections["Abnormal"]) >= MINIMUM_REFERENCE_EYES_PER_CLASS,
        f"reference_eyes={len(selections['Abnormal'])}, minimum={MINIMUM_REFERENCE_EYES_PER_CLASS}",
    )
    centroids = make_class_centroids(normalized_train, selections)
    hard_check(
        "reference centroids finite",
        all(np.isfinite(centroid).all() for centroid in centroids.values()),
        "Normal and Abnormal unit centroids checked",
    )
    centroid_norms = {label: float(np.linalg.norm(value)) for label, value in centroids.items()}
    hard_check(
        "reference centroids unit-normalized",
        all(np.isclose(norm, 1.0, rtol=0.0, atol=1e-12) for norm in centroid_norms.values()),
        f"centroid_norms={centroid_norms}",
    )

    train_rows["ReferenceSelected"] = False
    train_rows["ReferenceRankWithinTrueClass"] = pd.Series(
        [pd.NA] * len(train_rows), dtype="Int64"
    )
    reference_distance_rows: list[dict[str, Any]] = []
    reference_distances: dict[str, np.ndarray] = {}
    reference_stats: dict[str, dict[str, float]] = {}
    for label in ["Normal", "Abnormal"]:
        selected_indices = selections[label]
        train_rows.loc[selected_indices, "ReferenceSelected"] = True
        train_rows.loc[selected_indices, "ReferenceRankWithinTrueClass"] = np.arange(
            1, len(selected_indices) + 1
        )
        distances = cosine_distance(normalized_train[selected_indices], centroids[label])
        if not np.isfinite(distances).all() or (distances < -1e-12).any() or (distances > 2.0 + 1e-12).any():
            raise SafetyError(f"{label} reference cosine distances are invalid.")
        reference_distances[label] = distances
        reference_stats[label] = summarize_distribution(distances)
        for rank, (row_index, distance) in enumerate(
            zip(selected_indices, distances), start=1
        ):
            reference_distance_rows.append(
                {
                    "Class": label,
                    "ReferenceRankWithinTrueClass": rank,
                    "EyeExamID": train_rows.loc[row_index, "EyeExamID"],
                    "ResearchSubjectID": train_rows.loc[row_index, "ResearchSubjectID"],
                    "PrimaryConfidenceMargin": float(
                        train_rows.loc[row_index, "PrimaryConfidenceMargin"]
                    ),
                    "CosineDistanceToClassCentroid": float(distance),
                }
            )

    train_distance_normal = cosine_distance(normalized_train, centroids["Normal"])
    train_distance_abnormal = cosine_distance(normalized_train, centroids["Abnormal"])
    train_rows["DistanceToNormalCentroid"] = train_distance_normal
    train_rows["DistanceToAbnormalCentroid"] = train_distance_abnormal
    train_rows["TrueClassCentroidDistance"] = np.where(
        train_rows["TrueLabel"].eq("Normal"),
        train_distance_normal,
        train_distance_abnormal,
    )

    val_distance_normal = cosine_distance(normalized_val, centroids["Normal"])
    val_distance_abnormal = cosine_distance(normalized_val, centroids["Abnormal"])
    val_rows["DistanceToNormalCentroid"] = val_distance_normal
    val_rows["DistanceToAbnormalCentroid"] = val_distance_abnormal
    # primary typicality always uses the centroid of the model-predicted class
    val_rows["FeatureTypicalityPredictedClass"] = np.where(
        val_rows["PredictedLabel"].eq("Normal"),
        val_distance_normal,
        val_distance_abnormal,
    )
    val_rows["PredictedClassReferencePercentile"] = [
        reference_percentile(
            reference_distances[predicted_label], float(distance)
        )
        for predicted_label, distance in zip(
            val_rows["PredictedLabel"],
            val_rows["FeatureTypicalityPredictedClass"],
        )
    ]
    expected_primary_distance = np.where(
        val_rows["PredictedLabel"].eq("Normal"),
        val_rows["DistanceToNormalCentroid"],
        val_rows["DistanceToAbnormalCentroid"],
    )
    hard_check(
        "primary typicality uses predicted class",
        np.array_equal(
            val_rows["FeatureTypicalityPredictedClass"].to_numpy(),
            np.asarray(expected_primary_distance),
        ),
        "true validation labels were not used to select the primary centroid",
    )
    hard_check(
        "validation cosine distances finite and bounded",
        np.isfinite(
            val_rows[
                [
                    "DistanceToNormalCentroid",
                    "DistanceToAbnormalCentroid",
                    "FeatureTypicalityPredictedClass",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and val_rows["FeatureTypicalityPredictedClass"].between(-1e-12, 2.0 + 1e-12).all(),
        "expected cosine-distance interval is [0,2] up to numerical tolerance",
    )
    hard_check(
        "reference percentiles finite and in [0,100]",
        np.isfinite(val_rows["PredictedClassReferencePercentile"]).all()
        and val_rows["PredictedClassReferencePercentile"].between(0.0, 100.0).all(),
        "right-continuous empirical class-reference ECDF",
    )

    # validation labels are used here only to describe correct versus incorrect predictions
    validation_summary = summarize_validation(val_rows)
    confidence_quantiles, high_confidence_grid = make_high_confidence_grid(val_rows)
    reference_summary = pd.DataFrame(
        [{"Class": label, **reference_stats[label]} for label in ["Normal", "Abnormal"]]
    )
    reference_eye_distances = pd.DataFrame(reference_distance_rows)
    correct_summary = validation_summary.loc[
        validation_summary["PredictionGroup"].eq("Correct")
    ].iloc[0]
    incorrect_summary = validation_summary.loc[
        validation_summary["PredictionGroup"].eq("Incorrect")
    ].iloc[0]
    raw_distance_comparison = (
        "higher mean and median"
        if (
            incorrect_summary["TypicalityDistanceMean"]
            > correct_summary["TypicalityDistanceMean"]
            and incorrect_summary["TypicalityDistanceMedian"]
            > correct_summary["TypicalityDistanceMedian"]
        )
        else "lower mean and median"
        if (
            incorrect_summary["TypicalityDistanceMean"]
            < correct_summary["TypicalityDistanceMean"]
            and incorrect_summary["TypicalityDistanceMedian"]
            < correct_summary["TypicalityDistanceMedian"]
        )
        else "mixed mean/median"
    )
    percentile_comparison = (
        "higher mean and median"
        if (
            incorrect_summary["ReferencePercentileMean"]
            > correct_summary["ReferencePercentileMean"]
            and incorrect_summary["ReferencePercentileMedian"]
            > correct_summary["ReferencePercentileMedian"]
        )
        else "lower mean and median"
        if (
            incorrect_summary["ReferencePercentileMean"]
            < correct_summary["ReferencePercentileMean"]
            and incorrect_summary["ReferencePercentileMedian"]
            < correct_summary["ReferencePercentileMedian"]
        )
        else "mixed mean/median"
    )
    any_high_confidence_incorrect_atypical = bool(
        high_confidence_grid[
            "IncorrectHighConfidenceAndAtypicalEyes"
        ].gt(0).any()
    )

    # recheck the protected inputs after extraction
    # the imported day-4 code must also show that only train and validation were loaded
    train_hash_after = file_sha256(check_allowed_manifest_path(TRAIN_MANIFEST_PATH))
    val_hash_after = file_sha256(check_allowed_manifest_path(VAL_MANIFEST_PATH))
    checkpoint_hash_after = file_sha256(CHECKPOINT_PATH)
    hard_check(
        "locked manifests unchanged during extraction",
        train_hash_after == train_manifest_hash and val_hash_after == val_manifest_hash,
        f"train={train_hash_after}; validation={val_hash_after}",
    )
    hard_check(
        "frozen checkpoint unchanged during extraction",
        checkpoint_hash_after == checkpoint_hash,
        f"checkpoint_sha256={checkpoint_hash_after}",
    )
    loaded_manifests = {Path(path).resolve() for path in day4.LOADED_MANIFEST_PATHS}
    hard_check(
        "only train/validation manifests loaded",
        loaded_manifests == ALLOWED_MANIFEST_PATHS,
        f"loaded={sorted(map(str, loaded_manifests))}",
    )
    hard_check(
        "test data not loaded or evaluated",
        all(
            path.name.casefold() != FORBIDDEN_TEST_BASENAME.casefold()
            for path in loaded_manifests
        ),
        "test_manifest_loaded=False; test_set_evaluated=False; test_embeddings_extracted=False",
    )

    # keep the raw 512-dimensional pooled features in the exported embedding files
    train_output = attach_features(train_rows, train_features)
    val_output = attach_features(val_rows, val_features)
    hard_check(
        "feature artifacts contain no duplicate EyeExamIDs",
        not train_output["EyeExamID"].duplicated().any()
        and not val_output["EyeExamID"].duplicated().any(),
        "one pooled representation per eye in each embedding CSV",
    )
    hard_check(
        "feature artifacts have exactly 512 feature columns",
        len([column for column in train_output if column.startswith("Feature_")])
        == EXPECTED_FEATURE_DIMENSION
        and len([column for column in val_output if column.startswith("Feature_")])
        == EXPECTED_FEATURE_DIMENSION,
        "expected Feature_000 through Feature_511",
    )

    # create a new output folder only after all feature and integrity checks pass
    run_id = make_run_id()
    feature_dir = FEATURE_OUTPUT_ROOT / run_id
    figure_dir = FIGURE_OUTPUT_ROOT / run_id
    audit_dir = AUDIT_OUTPUT_ROOT / run_id
    for directory in [feature_dir, figure_dir, audit_dir]:
        directory.mkdir(parents=True, exist_ok=False)

    train_embedding_path = feature_dir / "train_eye_embeddings.csv"
    val_embedding_path = feature_dir / "val_eye_embeddings.csv"
    centroid_path = feature_dir / "reference_centroids.npz"
    reference_metadata_path = feature_dir / "reference_metadata.json"
    reference_distances_path = feature_dir / "reference_eye_distances.csv"
    reference_summary_path = feature_dir / "reference_distance_summary.csv"
    validation_summary_path = feature_dir / "validation_typicality_summary.csv"
    confidence_quantiles_path = feature_dir / "validation_confidence_percentiles.csv"
    high_confidence_grid_path = feature_dir / "high_confidence_incorrect_grid.csv"
    audit_path = audit_dir / "day5_feature_typicality_checks.csv"

    train_output.to_csv(train_embedding_path, index=False, float_format="%.17g")
    val_output.to_csv(val_embedding_path, index=False, float_format="%.17g")
    reference_eye_distances.to_csv(
        reference_distances_path, index=False, float_format="%.17g"
    )
    reference_summary.to_csv(reference_summary_path, index=False, float_format="%.17g")
    validation_summary.to_csv(
        validation_summary_path, index=False, float_format="%.17g"
    )
    confidence_quantiles.to_csv(
        confidence_quantiles_path, index=False, float_format="%.17g"
    )
    high_confidence_grid.to_csv(
        high_confidence_grid_path, index=False, float_format="%.17g"
    )
    np.savez(
        centroid_path,
        Normal=centroids["Normal"],
        Abnormal=centroids["Abnormal"],
        class_names=np.asarray(["Normal", "Abnormal"]),
        feature_dimension=np.asarray(EXPECTED_FEATURE_DIMENSION, dtype=np.int64),
    )
    save_typicality_plot(
        val_rows, figure_dir / "typicality_correct_vs_incorrect.png"
    )
    save_confidence_plot(val_rows, figure_dir / "confidence_vs_typicality.png")

    # save enough provenance to reproduce exactly which frozen artifacts were used
    git_commit, git_status = get_git_info()
    feature_script_hash = file_sha256(Path(__file__).resolve())
    metadata = {
        "run_timestamp": run_id,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": float(time.monotonic() - started),
        "objective": "predicted-class feature-space typicality for validation eyes",
        "model_was_retrained": False,
        "model_parameters_frozen": True,
        "model_run": DAY4_RUN_ID,
        "model_run_directory": str(DAY4_RUN_DIR),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": checkpoint_hash,
        "day4_script": str(DAY4_SCRIPT_PATH),
        "day4_script_sha256": script_hash,
        "feature_typicality_script": str(Path(__file__).resolve()),
        "feature_typicality_script_sha256": feature_script_hash,
        "config_used": str(CONFIG_USED_PATH),
        "config_used_sha256": config_hash,
        "locked_validation_predictions": str(LOCKED_VAL_PREDICTIONS_PATH),
        "locked_validation_predictions_sha256": val_predictions_hash,
        "classification_threshold_sha256": threshold_hash,
        "locked_run_metadata": str(RUN_METADATA_PATH),
        "locked_run_metadata_sha256": run_metadata_hash,
        "training_manifest": str(TRAIN_MANIFEST_PATH),
        "training_manifest_sha256": train_manifest_hash,
        "validation_manifest": str(VAL_MANIFEST_PATH),
        "validation_manifest_sha256": val_manifest_hash,
        "loaded_manifest_paths": sorted(map(str, loaded_manifests)),
        "test_manifest_loaded": False,
        "test_set_evaluated": False,
        "test_embeddings_extracted": False,
        "test_predictions_created": False,
        "feature_definition": (
            "feature-wise MAX of real-view ResNet-18 512-dimensional outputs, "
            "immediately before the frozen shared eye classifier"
        ),
        "feature_dimension": EXPECTED_FEATURE_DIMENSION,
        "stored_feature_values": "raw unnormalized pooled float32 values serialized to CSV",
        "distance_calculation_dtype": "float64",
        "feature_normalization": (
            "per-eye L2 normalization for distance calculations only; stored raw "
            "feature columns are not normalized"
        ),
        "primary_typicality": (
            "cosine distance to the unit-normalized centroid of the predicted "
            "class's reliable training references"
        ),
        "centroid_definition": (
            "arithmetic mean of class-reference L2-normalized features, then "
            "renormalized to unit length"
        ),
        "classification_threshold": threshold,
        "classification_threshold_logit": threshold_logit,
        "classification_rule": "Abnormal if probability >= frozen threshold; otherwise Normal",
        "primary_confidence_margin": (
            "absolute value of ((Abnormal logit - Normal logit) - logit(frozen "
            "classification threshold)); units are log-odds"
        ),
        "reference_selection": {
            "data": "training eyes only",
            "within_true_class": True,
            "correct_predictions_only": True,
            "ranking": "PrimaryConfidenceMargin descending, then EyeExamID ascending",
            "tie_handling": "stable deterministic EyeExamID tie-break; boundary ties are not expanded",
            "fraction": REFERENCE_FRACTION,
            "count_rule": "ceil(correctly classified class eyes * 0.50)",
            "minimum_per_class": MINIMUM_REFERENCE_EYES_PER_CLASS,
            "counts": reference_counts,
            "reference_eye_ids": {
                label: train_rows.loc[selections[label], "EyeExamID"].astype(str).tolist()
                for label in ["Normal", "Abnormal"]
            },
        },
        "reference_distance_summary": reference_stats,
        "reference_percentile_definition": (
            "100 * count(reference distances <= query distance) / number of "
            "class references (right-continuous empirical CDF)"
        ),
        "percentile_method_for_descriptive_quantiles": "NumPy linear",
        "reference_centroid_norms": centroid_norms,
        "preprocessing": {
            "same_for_training_and_validation_embeddings": True,
            "random_augmentation_applied": False,
            "pipeline": (
                "Pillow EXIF transpose; RGB conversion; centered black square "
                "padding; bilinear antialiased resize to 224x224; tensor; "
                "ImageNet IMAGENET1K_V1 mean/std normalization"
            ),
            "view_order": "SelectedViewIndex ascending, ImageRelativePath tie-break",
            "batch_size": int(config["batch_size"]),
        },
        "validation_probability_reproduction": reproduction,
        "descriptive_high_confidence_analysis": {
            "confidence_distribution": "all 120 validation confidence margins",
            "confidence_percentiles": VALIDATION_CONFIDENCE_PERCENTILES,
            "atypicality_reference_percentiles": REFERENCE_PERCENTILES,
            "validation_labels_used_for_description_only": True,
            "routing_rule_created": False,
            "any_incorrect_eye_in_reported_high_confidence_and_atypicality_bands": (
                any_high_confidence_incorrect_atypical
            ),
        },
        "descriptive_validation_interpretation": {
            "raw_cosine_distance_incorrect_vs_correct": raw_distance_comparison,
            "predicted_class_reference_percentile_incorrect_vs_correct": (
                percentile_comparison
            ),
            "conclusion": (
                "Incorrect predictions appear more atypical relative to their "
                "predicted-class reference distributions. This is descriptive "
                "only and is not a statistical-significance claim."
                if percentile_comparison == "higher mean and median"
                else "The descriptive typicality comparison is mixed or does not "
                "show greater atypicality among incorrect predictions."
            ),
        },
        "routing_thresholds_selected": False,
        "statistical_significance_claimed": False,
        "optional_shrinkage_mahalanobis_calculated": False,
        "seed": int(config["seed"]),
        "configuration": config,
        "environment": get_environment_info(day4, device),
        "git_commit_hash": git_commit,
        "git_worktree_status_at_completion": git_status,
        "artifacts": {
            "train_eye_embeddings": str(train_embedding_path),
            "val_eye_embeddings": str(val_embedding_path),
            "reference_centroids": str(centroid_path),
            "reference_eye_distances": str(reference_distances_path),
            "reference_distance_summary": str(reference_summary_path),
            "validation_typicality_summary": str(validation_summary_path),
            "validation_confidence_percentiles": str(confidence_quantiles_path),
            "high_confidence_incorrect_grid": str(high_confidence_grid_path),
            "audit_checks": str(audit_path),
            "figure_directory": str(figure_dir),
        },
    }
    save_json(reference_metadata_path, metadata)

    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 88)
    print("DAY-5 FEATURE TYPICALITY COMPLETE")
    print("=" * 88)
    print(f"Device                              : {device}")
    print(f"Feature dimension                   : {EXPECTED_FEATURE_DIMENSION}")
    print(f"Frozen classification threshold     : {threshold:.17g}")
    print(f"Frozen threshold logit              : {threshold_logit:.12f}")
    print("\nTraining reference sizes:")
    for label in ["Normal", "Abnormal"]:
        counts = reference_counts[label]
        print(
            f"  {label:<8} total / correct / reference: "
            f"{counts['total_training_eyes']} / "
            f"{counts['correctly_classified_training_eyes']} / "
            f"{counts['reference_eyes']}"
        )
    print(
        "\nValidation correct / incorrect       : "
        f"{int(correct_summary['Count'])} / {int(incorrect_summary['Count'])}"
    )
    print("\nPredicted-class cosine distance:")
    print(
        f"  Correct mean / median              : "
        f"{correct_summary['TypicalityDistanceMean']:.6f} / "
        f"{correct_summary['TypicalityDistanceMedian']:.6f}"
    )
    print(
        f"  Incorrect mean / median            : "
        f"{incorrect_summary['TypicalityDistanceMean']:.6f} / "
        f"{incorrect_summary['TypicalityDistanceMedian']:.6f}"
    )
    print("\nPredicted-class reference percentile:")
    print(
        f"  Correct mean / median              : "
        f"{correct_summary['ReferencePercentileMean']:.3f} / "
        f"{correct_summary['ReferencePercentileMedian']:.3f}"
    )
    print(
        f"  Incorrect mean / median            : "
        f"{incorrect_summary['ReferencePercentileMean']:.3f} / "
        f"{incorrect_summary['ReferencePercentileMedian']:.3f}"
    )
    print("\nReference distance distributions:")
    print(reference_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nExploratory incorrect high-confidence / atypicality grid:")
    print(high_confidence_grid.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nDescriptive interpretation:")
    print(
        "  Raw predicted-class cosine distance: "
        f"incorrect predictions have {raw_distance_comparison} versus correct predictions."
    )
    if percentile_comparison == "higher mean and median":
        print(
            "  Relative to class-specific reference distributions, incorrect "
            "predictions have higher mean and median percentiles and therefore "
            "appear more atypical descriptively."
        )
    else:
        print(
            "  Class-reference percentiles are mixed or lower for incorrect "
            "predictions; greater atypicality is not descriptively established."
        )
    if any_high_confidence_incorrect_atypical:
        print(
            "  At least one incorrect prediction is present in a reported "
            "high-confidence and high-distance-percentile band."
        )
    else:
        print(
            "  No incorrect prediction reached any reported high-confidence band; "
            "none was simultaneously high-confidence and high-distance-percentile."
        )
    print(f"\nTraining embeddings                 : {train_embedding_path}")
    print(f"Validation embeddings               : {val_embedding_path}")
    print(f"Reference centroids                 : {centroid_path}")
    print(f"Reference metadata                  : {reference_metadata_path}")
    print(f"Validation checks                   : {audit_path}")
    print("\nDescriptive comparisons do not establish statistical significance.")
    print("NO ROUTING THRESHOLDS WERE SELECTED.")
    print("TEST SET WAS NOT LOADED OR EVALUATED")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "CUDA out of memory during frozen feature extraction; batch size was "
            "not silently changed."
        ) from error
    except (SafetyError, RuntimeError) as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
