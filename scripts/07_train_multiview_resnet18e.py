

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

# set this before cuda starts so matrix operations stay deterministic
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (PROJECT_ROOT / "data" / "raw").resolve()
TRAIN_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_inputs" / "train_images.csv"
).resolve()
VAL_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_inputs" / "val_images.csv"
).resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "multiview_resnet18.yaml"
CONFIG_ROOT = (PROJECT_ROOT / "configs").resolve()
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "multiview_resnet18"
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "multiview_resnet18"
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"
BASELINE_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "models" / "baseline_resnet18"
).resolve()

ALLOWED_MANIFEST_PATHS = {TRAIN_INPUT_PATH, VAL_INPUT_PATH}
FORBIDDEN_TEST_BASENAME = "test_images.csv"
LOADED_MANIFEST_PATHS: set[Path] = set()
LOADED_REPORT_PATHS: set[Path] = set()
CLASS_TO_INDEX = {"Normal": 0, "Abnormal": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
WEIGHTS_ENUM = ResNet18_Weights.IMAGENET1K_V1
EXPECTED_TRAIN_EYES = 560
EXPECTED_TRAIN_NORMAL = 149
EXPECTED_TRAIN_ABNORMAL = 411
EXPECTED_VALIDATION_EYES = 120
EXPECTED_VALIDATION_NORMAL = 32
EXPECTED_VALIDATION_ABNORMAL = 88
EXPECTED_TRAIN_IMAGES = 3324
EXPECTED_VALIDATION_IMAGES = 716
EXPECTED_TRAIN_SUBJECTS = 179
EXPECTED_VALIDATION_SUBJECTS = 95
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "5df6765a3df29ca3f107758369fd1c17a92b1fcd336ec79991c823482711c4ad"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "91846df6c0f3d1a153e1c8a03da4e92be561cab29736236774ac32ef9802e567"
)


class SafetyError(RuntimeError):
    "used when an input, integrity, or protocol check fails."


class CenterSquarePad:
    "pad an image to a centered square without stretching it."

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        left = horizontal // 2
        right = horizontal - left
        top = vertical // 2
        bottom = vertical - top
        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=self.fill,
        )


class EyeExamDataset(Dataset):
    "return all selected views for one eye as a single sample."

    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        views = []
        for image_path in record["ResolvedImagePaths"]:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            views.append(self.transform(image))
        if not views:
            raise SafetyError(f"Eye has zero views at dataset access: {record['EyeExamID']}")
        return (
            torch.stack(views, dim=0),
            CLASS_TO_INDEX[record["EyeLabel"]],
            index,
        )


def collate_eye_batch(batch):
    "pad each eye to the largest view count in the current batch."
    view_tensors, labels, eye_indices = zip(*batch)
    view_counts = [int(tensor.shape[0]) for tensor in view_tensors]
    if min(view_counts) < 1:
        raise SafetyError("A collated eye contains zero real views.")
    maximum = max(view_counts)
    channels, height, width = view_tensors[0].shape[1:]
    padded = view_tensors[0].new_zeros(
        (len(view_tensors), maximum, channels, height, width)
    )
    view_mask = torch.zeros((len(view_tensors), maximum), dtype=torch.bool)
    for batch_index, tensor in enumerate(view_tensors):
        count = tensor.shape[0]
        if tensor.shape[1:] != (channels, height, width):
            raise SafetyError("Transformed image shapes differ within a batch.")
        padded[batch_index, :count] = tensor
        view_mask[batch_index, :count] = True
    return {
        "images": padded,
        "view_mask": view_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "eye_indices": torch.tensor(eye_indices, dtype=torch.long),
    }


class FeatureMaxResNet18(nn.Module):

    def __init__(self, weights):
        super().__init__()
        backbone = resnet18(weights=weights)
        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, len(CLASS_TO_INDEX))

    def forward(self, images, view_mask, return_diagnostics=False):
        if images.ndim != 5:
            raise SafetyError(f"Expected [B,V,C,H,W], received shape {images.shape}.")
        if view_mask.ndim != 2 or tuple(view_mask.shape) != tuple(images.shape[:2]):
            raise SafetyError("View mask shape does not match [batch, views].")
        if view_mask.dtype != torch.bool:
            raise SafetyError("View mask must be boolean.")
        if (~view_mask.any(dim=1)).any():
            raise SafetyError("An eye has zero unmasked views.")

        batch_size, padded_views = images.shape[:2]
        flat_images = images.flatten(0, 1)
        flat_mask = view_mask.flatten()
        valid_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)

        # only real views go through the backbone
        valid_features = self.backbone(flat_images.index_select(0, valid_positions))
        if valid_features.ndim != 2 or valid_features.shape[1] != self.feature_dim:
            raise SafetyError("Unexpected ResNet-18 feature shape.")

        feature_grid = valid_features.new_full(
            (batch_size * padded_views, self.feature_dim),
            float("-inf"),
        )
        feature_grid = feature_grid.index_copy(0, valid_positions, valid_features)
        view_features = feature_grid.view(batch_size, padded_views, self.feature_dim)
        pooled_features = view_features.amax(dim=1)
        if not torch.isfinite(pooled_features).all():
            raise SafetyError("Masked feature MAX produced nonfinite pooled features.")
        eye_logits = self.classifier(pooled_features)

        if not return_diagnostics:
            return eye_logits

        valid_view_logits = self.classifier(valid_features)
        view_logit_grid = valid_view_logits.new_full(
            (batch_size * padded_views, len(CLASS_TO_INDEX)),
            float("nan"),
        )
        view_logit_grid = view_logit_grid.index_copy(
            0, valid_positions, valid_view_logits
        )
        view_logits = view_logit_grid.view(
            batch_size, padded_views, len(CLASS_TO_INDEX)
        )
        return eye_logits, pooled_features, view_logits


def parse_args():
    # preflight-only checks the full setup without starting training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run all manifest, source-image, architecture, and masked-padding "
            "checks without training or creating a model run."
        ),
    )
    return parser.parse_args()


def read_allowed_manifest(path: Path) -> pd.DataFrame:
    # block accidental access to the held-out test manifest
    "read one of the approved train or validation manifests."
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError("Held-out test manifest access was attempted.")
    if resolved not in ALLOWED_MANIFEST_PATHS:
        raise SafetyError(f"Manifest is not allowlisted: {resolved}")
    LOADED_MANIFEST_PATHS.add(resolved)
    identifier_types = {
        "EyeExamID": "string",
        "ResearchSubjectID": "string",
        "EncounterID": "string",
        "Laterality": "string",
        "ImageRelativePath": "string",
        "SHA256": "string",
    }
    return pd.read_csv(
        resolved,
        dtype=identifier_types,
        low_memory=False,
    )


def check_allowed_manifest_path(path: Path) -> Path:
    "check that a manifest path is one of the approved inputs."
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError("Held-out test manifest access was attempted.")
    if resolved not in ALLOWED_MANIFEST_PATHS:
        raise SafetyError(f"Manifest is not allowlisted: {resolved}")
    return resolved


def check_test_protection():
    # both approved manifests must be loaded and nothing else
    if LOADED_MANIFEST_PATHS != ALLOWED_MANIFEST_PATHS:
        raise SafetyError(
            "Loaded manifest set differs from the exact train/validation allowlist: "
            f"{sorted(map(str, LOADED_MANIFEST_PATHS))}"
        )
    if any(
        path.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold()
        for path in LOADED_MANIFEST_PATHS
    ):
        raise SafetyError("Held-out test manifest was loaded.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError("Held-out test manifest cannot be used as configuration.")
    try:
        resolved.relative_to(CONFIG_ROOT)
    except ValueError as error:
        raise SafetyError(
            f"Configuration must be a YAML file under {CONFIG_ROOT}."
        ) from error
    if resolved.suffix.casefold() not in {".yaml", ".yml"}:
        raise SafetyError("Configuration must have a .yaml or .yml suffix.")
    if not resolved.is_file():
        raise SafetyError(f"Configuration does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise SafetyError("Configuration root must be a YAML mapping.")
    required = {
        "model",
        "pretrained",
        "pretrained_weights",
        "pooling",
        "image_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "early_stopping_patience",
        "seed",
        "num_workers",
        "device",
        "max_views",
        "baseline_validation_metrics",
        "augmentation",
    }
    missing = sorted(required - set(config or {}))
    if missing:
        raise SafetyError(f"Configuration is missing required fields: {missing}")
    if config["model"] != "resnet18":
        raise SafetyError("Day 4 supports only model=resnet18.")
    if config["pretrained"] is not True:
        raise SafetyError("Day 4 requires pretrained=true.")
    if config["pretrained_weights"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise SafetyError(
            "pretrained_weights must be ResNet18_Weights.IMAGENET1K_V1."
        )
    if config["pooling"] != "max":
        raise SafetyError("The primary Day-4 model requires pooling=max.")
    if config["optimizer"] != "AdamW":
        raise SafetyError("The requested optimizer is AdamW.")
    if int(config["image_size"]) != 224:
        raise SafetyError("The locked Day-4 image size must remain 224.")
    if int(config["max_views"]) != 6:
        raise SafetyError("The locked Day-4 maximum-view rule must remain six.")
    if int(config["batch_size"]) not in {2, 4}:
        raise SafetyError("Day-4 batch_size must be the configured value 4 or manual fallback 2.")
    for field in ["epochs", "batch_size", "early_stopping_patience"]:
        if int(config[field]) <= 0:
            raise SafetyError(f"{field} must be a positive integer.")
    if int(config["num_workers"]) < 0:
        raise SafetyError("num_workers must be nonnegative.")
    if int(config["epochs"]) != 20:
        raise SafetyError("The configured Day-4 epoch limit must remain 20.")
    if int(config["early_stopping_patience"]) != 5:
        raise SafetyError("The Day-4 early-stopping patience must remain 5.")
    if int(config["seed"]) != 42:
        raise SafetyError("The fixed Day-4 seed must remain 42.")
    if float(config["learning_rate"]) != 0.0001:
        raise SafetyError("The locked Day-4 learning rate must remain 0.0001.")
    if float(config["weight_decay"]) < 0:
        raise SafetyError("weight_decay must be nonnegative.")
    if float(config["weight_decay"]) != 0.0001:
        raise SafetyError("The configured Day-4 weight decay must remain 0.0001.")

    augmentation_required = {
        "rotation_degrees",
        "translate_fraction",
        "scale_min",
        "scale_max",
        "brightness",
        "contrast",
    }
    augmentation = config["augmentation"] or {}
    missing_augmentation = sorted(augmentation_required - set(augmentation))
    if missing_augmentation:
        raise SafetyError(f"Augmentation config is missing: {missing_augmentation}")
    rotation = float(augmentation["rotation_degrees"])
    translation = float(augmentation["translate_fraction"])
    scale_min = float(augmentation["scale_min"])
    scale_max = float(augmentation["scale_max"])
    brightness = float(augmentation["brightness"])
    contrast = float(augmentation["contrast"])
    if not (0 <= rotation <= 5):
        raise SafetyError("rotation_degrees must be within [0, 5].")
    if not (0 <= translation <= 0.1):
        raise SafetyError("translate_fraction must be within [0, 0.1].")
    if not (0 < scale_min <= scale_max <= 1.1):
        raise SafetyError("scale range must satisfy 0 < min <= max <= 1.1.")
    if not (0 <= brightness <= 0.2 and 0 <= contrast <= 0.2):
        raise SafetyError("brightness and contrast must each be within [0, 0.2].")
    return config


def get_baseline_metrics_path(config: dict) -> Path:
    value = Path(str(config["baseline_validation_metrics"]))
    if value.is_absolute():
        raise SafetyError("baseline_validation_metrics must be project-relative.")
    # the day-3 comparison must come from the locked baseline output folder
    resolved = (PROJECT_ROOT / value).resolve()
    try:
        resolved.relative_to(BASELINE_OUTPUT_ROOT)
    except ValueError as error:
        raise SafetyError("Baseline metrics path must stay under the Day-3 output root.") from error
    if resolved.name != "val_metrics.csv" or not resolved.is_file():
        raise SafetyError(f"Locked Day-3 validation metrics not found: {resolved}")
    return resolved


def read_baseline_metrics(path: Path) -> tuple[dict, str]:
    "read the locked day-3 validation metrics for comparison only."
    resolved = path.resolve()
    LOADED_REPORT_PATHS.add(resolved)
    frame = pd.read_csv(resolved, low_memory=False)
    required = {
        "EvaluationUnit",
        "AUROC",
        "BalancedAccuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "Accuracy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SafetyError(f"Day-3 metrics missing required columns: {missing}")
    primary = frame.loc[frame["EvaluationUnit"].eq("eye_primary")]
    if len(primary) != 1:
        raise SafetyError("Day-3 metrics must have exactly one eye_primary row.")
    metrics = {
        name: float(primary.iloc[0][name])
        for name in [
            "AUROC",
            "BalancedAccuracy",
            "Sensitivity",
            "Specificity",
            "F1",
            "Accuracy",
        ]
    }
    if not all(np.isfinite(list(metrics.values()))):
        raise SafetyError("Day-3 primary metrics contain nonfinite values.")
    return metrics, file_sha256(resolved)


def parse_bool_column(series: pd.Series, field: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    if normalized.isna().any():
        raise SafetyError(f"Blank boolean values in {field}.")
    invalid = ~normalized.isin(["true", "false"])
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique())
        raise SafetyError(f"Invalid boolean values in {field}: {values}")
    return normalized.eq("true")


def check_nonblank_fields(frame: pd.DataFrame, fields: list[str], name: str):
    for field in fields:
        values = frame[field].astype("string")
        blank = values.isna() | values.str.strip().eq("")
        if blank.any():
            raise SafetyError(f"{name}: blank {field} rows={int(blank.sum())}.")


def parse_integer_column(frame: pd.DataFrame, field: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce")
    invalid = values.isna() | ~np.isclose(values, np.round(values))
    if invalid.any():
        raise SafetyError(f"{name}: non-integer {field} rows={int(invalid.sum())}.")
    return values.astype(int)


def get_image_path(relative_value: object) -> Path:
    text = str(relative_value).strip()
    if not text:
        raise SafetyError("Blank ImageRelativePath encountered.")
    relative = Path(text)
    if relative.is_absolute():
        raise SafetyError(f"Absolute source image path is forbidden: {text}")
    # keep every resolved image path inside data/raw
    resolved = (RAW_ROOT / relative).resolve()
    try:
        resolved.relative_to(RAW_ROOT)
    except ValueError as error:
        raise SafetyError(f"Image path escapes data/raw: {text}") from error
    if not resolved.is_file():
        raise SafetyError(f"Selected source image is missing: {resolved}")
    return resolved


def check_eye_consistency(frame: pd.DataFrame, name: str):
    consistency_fields = [
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "DatasetSplit",
        "SelectedImageCount",
        "AvailableUniqueImageCount",
    ]
    for field in consistency_fields:
        inconsistent = frame.groupby("EyeExamID")[field].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise SafetyError(
                f"{name}: {int(inconsistent.sum())} EyeExamIDs have inconsistent {field}."
            )


def check_manifest(
    frame: pd.DataFrame,
    name: str,
    expected_split: str,
    max_views: int,
) -> pd.DataFrame:
    required = {
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "DatasetSplit",
        "ImageRelativePath",
        "SHA256",
        "IncludeModelInput",
        "SelectedViewIndex",
        "SelectedImageCount",
        "AvailableUniqueImageCount",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SafetyError(f"{name}: missing required columns: {missing}")
    if frame.empty:
        raise SafetyError(f"{name}: manifest is empty.")
    frame = frame.copy()
    check_nonblank_fields(
        frame,
        [
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "EyeLabel",
            "DatasetSplit",
            "ImageRelativePath",
            "SHA256",
        ],
        name,
    )
    frame["SelectedViewIndex"] = parse_integer_column(
        frame, "SelectedViewIndex", name
    )
    frame["SelectedImageCount"] = parse_integer_column(
        frame, "SelectedImageCount", name
    )
    frame["AvailableUniqueImageCount"] = parse_integer_column(
        frame, "AvailableUniqueImageCount", name
    )

    labels = set(frame["EyeLabel"].astype(str).unique())
    if not labels.issubset(CLASS_TO_INDEX):
        raise SafetyError(f"{name}: labels outside Normal/Abnormal: {sorted(labels)}")
    if frame["Laterality"].eq("UNKNOWN").any():
        raise SafetyError(f"{name}: UNKNOWN laterality is forbidden.")
    invalid_laterality = ~frame["Laterality"].isin(["OD", "OS"])
    if invalid_laterality.any():
        raise SafetyError(
            f"{name}: invalid laterality rows={int(invalid_laterality.sum())}."
        )
    split_values = set(frame["DatasetSplit"].astype(str).unique())
    if split_values != {expected_split}:
        raise SafetyError(
            f"{name}: expected DatasetSplit={expected_split}, observed={split_values}."
        )
    include_input = parse_bool_column(frame["IncludeModelInput"], "IncludeModelInput")
    if not include_input.all():
        raise SafetyError(
            f"{name}: {int((~include_input).sum())} rows are not IncludeModelInput."
        )
    if frame.duplicated(["EyeExamID", "ImageRelativePath"], keep=False).any():
        raise SafetyError(f"{name}: duplicate image path within an EyeExamID.")
    if frame.duplicated(["EyeExamID", "SHA256"], keep=False).any():
        raise SafetyError(f"{name}: duplicate SHA-256 selected within an EyeExamID.")
    invalid_sha = ~frame["SHA256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}")
    if invalid_sha.any():
        raise SafetyError(f"{name}: invalid SHA-256 rows={int(invalid_sha.sum())}.")

    check_eye_consistency(frame, name)
    group_sizes = frame.groupby("EyeExamID").size()
    if group_sizes.empty or int(group_sizes.min()) < 1:
        raise SafetyError(f"{name}: an eye has zero selected views.")
    if int(group_sizes.max()) > max_views:
        raise SafetyError(
            f"{name}: an eye contains more than {max_views} selected views."
        )
    selected_counts = frame.groupby("EyeExamID")["SelectedImageCount"].first()
    mismatch = selected_counts.ne(group_sizes)
    if mismatch.any():
        raise SafetyError(
            f"{name}: SelectedImageCount mismatch eyes={int(mismatch.sum())}."
        )
    if (frame["AvailableUniqueImageCount"] < frame["SelectedImageCount"]).any():
        raise SafetyError(f"{name}: available unique count is below selected count.")
    should_retain_all = frame["AvailableUniqueImageCount"].le(max_views)
    truncated_below_limit = should_retain_all & frame[
        "SelectedImageCount"
    ].ne(frame["AvailableUniqueImageCount"])
    if truncated_below_limit.any():
        raise SafetyError(
            f"{name}: eyes with <= {max_views} usable images did not retain all views."
        )
    should_reduce_to_limit = frame["AvailableUniqueImageCount"].gt(max_views)
    wrong_reduced_count = should_reduce_to_limit & frame[
        "SelectedImageCount"
    ].ne(max_views)
    if wrong_reduced_count.any():
        raise SafetyError(
            f"{name}: eyes with > {max_views} usable images do not have exactly "
            f"{max_views} selected views."
        )

    bad_indices = []
    for eye_id, group in frame.groupby("EyeExamID", sort=False):
        observed = sorted(group["SelectedViewIndex"].tolist())
        expected = list(range(1, len(group) + 1))
        if observed != expected:
            bad_indices.append(str(eye_id))
    if bad_indices:
        raise SafetyError(
            f"{name}: noncontiguous or duplicate SelectedViewIndex eyes={len(bad_indices)}."
        )

    expected_eye_id = (
        frame["ResearchSubjectID"].astype(str)
        + "__"
        + frame["EncounterID"].astype(str)
        + "__"
        + frame["Laterality"].astype(str)
    )
    if not frame["EyeExamID"].astype(str).eq(expected_eye_id).all():
        raise SafetyError(f"{name}: EyeExamID does not match the canonical composite key.")

    frame["ResolvedImagePath"] = frame["ImageRelativePath"].map(get_image_path)
    # confirm that the source image bytes still match the locked manifest
    observed_hashes = frame["ResolvedImagePath"].map(file_sha256)
    hash_mismatch = ~observed_hashes.str.casefold().eq(
        frame["SHA256"].astype(str).str.casefold()
    )
    if hash_mismatch.any():
        raise SafetyError(
            f"{name}: source-image SHA-256 mismatch rows={int(hash_mismatch.sum())}."
        )
    return frame


def make_eye_records(frame: pd.DataFrame) -> list[dict]:
    # one record represents one eye and all of its selected views
    records = []
    ordered = frame.sort_values(
        ["EyeExamID", "SelectedViewIndex", "ImageRelativePath"],
        kind="mergesort",
    )
    for eye_id, group in ordered.groupby("EyeExamID", sort=True):
        first = group.iloc[0]
        records.append(
            {
                "EyeExamID": str(eye_id),
                "ResearchSubjectID": str(first["ResearchSubjectID"]),
                "EncounterID": str(first["EncounterID"]),
                "Laterality": str(first["Laterality"]),
                "EyeLabel": str(first["EyeLabel"]),
                "ImageRelativePaths": group["ImageRelativePath"].astype(str).tolist(),
                "ResolvedImagePaths": group["ResolvedImagePath"].tolist(),
                "SelectedViewIndices": group["SelectedViewIndex"].astype(int).tolist(),
                "SHA256Values": group["SHA256"].astype(str).tolist(),
                "NumberOfViews": int(len(group)),
            }
        )
    return records


def run_preflight(config: dict) -> tuple[list[dict], list[dict], dict]:
    # verify the locked manifests before reading any training data
    check_allowed_manifest_path(TRAIN_INPUT_PATH)
    check_allowed_manifest_path(VAL_INPUT_PATH)
    train_manifest_hash = file_sha256(TRAIN_INPUT_PATH)
    validation_manifest_hash = file_sha256(VAL_INPUT_PATH)
    if train_manifest_hash != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise SafetyError(
            "Locked training manifest SHA-256 differs from the committed Day-4 input."
        )
    if validation_manifest_hash != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise SafetyError(
            "Locked validation manifest SHA-256 differs from the committed Day-4 input."
        )
    train = read_allowed_manifest(TRAIN_INPUT_PATH)
    validation = read_allowed_manifest(VAL_INPUT_PATH)
    check_test_protection()
    train = check_manifest(train, "train", "train", int(config["max_views"]))
    validation = check_manifest(
        validation,
        "validation",
        "validation",
        int(config["max_views"]),
    )

    # train and validation must not share subjects, eyes, paths, or image hashes
    train_subjects = set(train["ResearchSubjectID"].astype(str))
    validation_subjects = set(validation["ResearchSubjectID"].astype(str))
    subject_intersection = train_subjects & validation_subjects
    if subject_intersection:
        raise SafetyError(
            "Train/validation ResearchSubjectID intersection is nonempty: "
            f"{len(subject_intersection)}"
        )
    train_eyes = set(train["EyeExamID"].astype(str))
    validation_eyes = set(validation["EyeExamID"].astype(str))
    if train_eyes & validation_eyes:
        raise SafetyError("Train/validation EyeExamID intersection is nonempty.")
    train_paths = set(train["ImageRelativePath"].astype(str))
    validation_paths = set(validation["ImageRelativePath"].astype(str))
    if train_paths & validation_paths:
        raise SafetyError("Train/validation ImageRelativePath intersection is nonempty.")
    train_hashes = set(train["SHA256"].astype(str).str.casefold())
    validation_hashes = set(validation["SHA256"].astype(str).str.casefold())
    if train_hashes & validation_hashes:
        raise SafetyError("Train/validation image SHA-256 intersection is nonempty.")

    if len(train) != EXPECTED_TRAIN_IMAGES:
        raise SafetyError(
            f"Training images={len(train)}, expected={EXPECTED_TRAIN_IMAGES}."
        )
    if len(validation) != EXPECTED_VALIDATION_IMAGES:
        raise SafetyError(
            f"Validation images={len(validation)}, expected={EXPECTED_VALIDATION_IMAGES}."
        )
    if len(train_subjects) != EXPECTED_TRAIN_SUBJECTS:
        raise SafetyError(
            f"Training subjects={len(train_subjects)}, expected={EXPECTED_TRAIN_SUBJECTS}."
        )
    if len(validation_subjects) != EXPECTED_VALIDATION_SUBJECTS:
        raise SafetyError(
            "Validation subjects="
            f"{len(validation_subjects)}, expected={EXPECTED_VALIDATION_SUBJECTS}."
        )

    train_records = make_eye_records(train)
    validation_records = make_eye_records(validation)
    if len(train_records) != EXPECTED_TRAIN_EYES:
        raise SafetyError(
            f"Training eyes={len(train_records)}, expected={EXPECTED_TRAIN_EYES}."
        )
    if len(validation_records) != EXPECTED_VALIDATION_EYES:
        raise SafetyError(
            f"Validation eyes={len(validation_records)}, "
            f"expected={EXPECTED_VALIDATION_EYES}."
        )
    train_counts = pd.Series(
        [record["EyeLabel"] for record in train_records]
    ).value_counts()
    normal_count = int(train_counts.get("Normal", 0))
    abnormal_count = int(train_counts.get("Abnormal", 0))
    if normal_count != EXPECTED_TRAIN_NORMAL:
        raise SafetyError(
            f"Training Normal eyes={normal_count}, expected={EXPECTED_TRAIN_NORMAL}."
        )
    if abnormal_count != EXPECTED_TRAIN_ABNORMAL:
        raise SafetyError(
            f"Training Abnormal eyes={abnormal_count}, "
            f"expected={EXPECTED_TRAIN_ABNORMAL}."
        )
    total = normal_count + abnormal_count
    # class weights use eye counts because one eye is one training sample
    class_weights = [
        total / (2.0 * normal_count),
        total / (2.0 * abnormal_count),
    ]
    validation_labels = {record["EyeLabel"] for record in validation_records}
    if validation_labels != set(CLASS_TO_INDEX):
        raise SafetyError("Validation eyes must contain both Normal and Abnormal.")
    validation_counts = pd.Series(
        [record["EyeLabel"] for record in validation_records]
    ).value_counts()
    validation_normal = int(validation_counts.get("Normal", 0))
    validation_abnormal = int(validation_counts.get("Abnormal", 0))
    if validation_normal != EXPECTED_VALIDATION_NORMAL:
        raise SafetyError(
            "Validation Normal eyes="
            f"{validation_normal}, expected={EXPECTED_VALIDATION_NORMAL}."
        )
    if validation_abnormal != EXPECTED_VALIDATION_ABNORMAL:
        raise SafetyError(
            "Validation Abnormal eyes="
            f"{validation_abnormal}, expected={EXPECTED_VALIDATION_ABNORMAL}."
        )

    report = {
        "training_images": int(len(train)),
        "validation_images": int(len(validation)),
        "training_eyes": int(len(train_records)),
        "validation_eyes": int(len(validation_records)),
        "training_subjects": int(len(train_subjects)),
        "validation_subjects": int(len(validation_subjects)),
        "subject_intersection": 0,
        "training_normal_eyes": normal_count,
        "training_abnormal_eyes": abnormal_count,
        "validation_normal_eyes": validation_normal,
        "validation_abnormal_eyes": validation_abnormal,
        "class_weights_normal_abnormal": class_weights,
        "training_view_count_distribution": dict(
            sorted(pd.Series([r["NumberOfViews"] for r in train_records]).value_counts().items())
        ),
        "validation_view_count_distribution": dict(
            sorted(
                pd.Series(
                    [r["NumberOfViews"] for r in validation_records]
                ).value_counts().items()
            )
        ),
        "training_manifest_sha256_before": train_manifest_hash,
        "validation_manifest_sha256_before": validation_manifest_hash,
        "loaded_manifest_paths": sorted(map(str, LOADED_MANIFEST_PATHS)),
        "test_manifest_loaded": False,
    }
    return train_records, validation_records, report


def set_random_seeds(seed: int):
    # seed every random source used by training and data loading
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_data_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(config: dict) -> torch.device:
    requested = str(config.get("device", "auto")).casefold()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SafetyError("device=cuda requested but CUDA is unavailable.")
    if requested not in {"cuda", "cpu"}:
        raise SafetyError(f"Unsupported device value: {requested}")
    return torch.device(requested)


def make_transforms(config: dict):
    image_size = int(config["image_size"])
    augmentation = config["augmentation"]
    # train and validation share the same base preprocessing
    common_start = [
        CenterSquarePad(fill=0),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    train_transform = transforms.Compose(
        [
            *common_start,
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                translate=(
                    float(augmentation["translate_fraction"]),
                    float(augmentation["translate_fraction"]),
                ),
                scale=(
                    float(augmentation["scale_min"]),
                    float(augmentation["scale_max"]),
                ),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=float(augmentation["brightness"]),
                contrast=float(augmentation["contrast"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            *common_start,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )
    return train_transform, validation_transform


def make_datasets_and_loaders(train_records, validation_records, config, device):
    train_transform, validation_transform = make_transforms(config)
    train_dataset = EyeExamDataset(train_records, train_transform)
    validation_dataset = EyeExamDataset(validation_records, validation_transform)

    train_generator = torch.Generator()
    train_generator.manual_seed(int(config["seed"]))
    validation_generator = torch.Generator()
    validation_generator.manual_seed(int(config["seed"]) + 1)
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_data_worker,
        "collate_fn": collate_eye_batch,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        **common,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader


def check_masked_pooling(
    model,
    validation_dataset,
    device,
    max_views,
    padding_rtol=0.0,
    padding_atol=0.0,
    mixed_batch_rtol=1e-4,
    mixed_batch_atol=1e-5,
) -> dict:
    "confirm that masked padding does not change the pooled eye features."
    # use a real short-view eye to prove masked padding cannot affect it
    candidate_index = next(
        (
            index
            for index, record in enumerate(validation_dataset.records)
            if record["NumberOfViews"] < max_views
        ),
        None,
    )
    if candidate_index is None:
        raise SafetyError("No validation eye is available for a real padded-view check.")
    real_views, _, _ = validation_dataset[candidate_index]
    real_count = int(real_views.shape[0])
    if not (1 <= real_count < max_views):
        raise SafetyError("Invalid eye selected for masked-padding invariance check.")
    companion_index = next(
        (
            index
            for index, record in enumerate(validation_dataset.records)
            if index != candidate_index and record["NumberOfViews"] == max_views
        ),
        None,
    )
    if companion_index is None:
        raise SafetyError(
            "No full-view validation eye is available for the padded batch check."
        )

    alone_images = real_views.unsqueeze(0).to(device)
    alone_mask = torch.ones((1, real_count), dtype=torch.bool, device=device)
    padded_shape = (1, max_views, *real_views.shape[1:])
    zero_padded = real_views.new_zeros(padded_shape)
    zero_padded[0, :real_count] = real_views
    padded_mask = torch.zeros((1, max_views), dtype=torch.bool)
    padded_mask[0, :real_count] = True
    poisoned = zero_padded.clone()
    poisoned[0, real_count:] = 12345.0
    zero_padded = zero_padded.to(device)
    poisoned = poisoned.to(device)
    padded_mask = padded_mask.to(device)
    companion_sample = validation_dataset[companion_index]
    mixed_batch = collate_eye_batch(
        [
            (real_views, CLASS_TO_INDEX[validation_dataset.records[candidate_index]["EyeLabel"]], candidate_index),
            companion_sample,
        ]
    )
    mixed_images = mixed_batch["images"].to(device)
    mixed_mask = mixed_batch["view_mask"].to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        alone_logits, alone_pooled, _ = model(
            alone_images, alone_mask, return_diagnostics=True
        )
        zero_logits, zero_pooled, _ = model(
            zero_padded, padded_mask, return_diagnostics=True
        )
        poison_logits, poison_pooled, _ = model(
            poisoned, padded_mask, return_diagnostics=True
        )
        mixed_logits, mixed_pooled, _ = model(
            mixed_images, mixed_mask, return_diagnostics=True
        )
    if was_training:
        model.train()

    try:
        torch.testing.assert_close(
            alone_pooled,
            zero_pooled,
            rtol=padding_rtol,
            atol=padding_atol,
        )
        torch.testing.assert_close(
            alone_pooled,
            poison_pooled,
            rtol=padding_rtol,
            atol=padding_atol,
        )
        torch.testing.assert_close(
            alone_logits,
            zero_logits,
            rtol=padding_rtol,
            atol=padding_atol,
        )
        torch.testing.assert_close(
            alone_logits,
            poison_logits,
            rtol=padding_rtol,
            atol=padding_atol,
        )
    except AssertionError as error:
        raise SafetyError(
            "Zero or poisoned masked padding changed the pooled representation."
        ) from error
    try:
        torch.testing.assert_close(
            alone_pooled,
            mixed_pooled[:1],
            rtol=mixed_batch_rtol,
            atol=mixed_batch_atol,
        )
        torch.testing.assert_close(
            alone_logits,
            mixed_logits[:1],
            rtol=mixed_batch_rtol,
            atol=mixed_batch_atol,
        )
    except AssertionError as error:
        raise SafetyError(
            "Processing an eye alone versus in a mixed padded batch exceeded the "
            "documented floating-point tolerance."
        ) from error

    zero_difference = float((alone_pooled - zero_pooled).abs().max().item())
    poison_difference = float((alone_pooled - poison_pooled).abs().max().item())
    mixed_batch_difference = float(
        (alone_pooled - mixed_pooled[:1]).abs().max().item()
    )
    return {
        "status": "PASS",
        "EyeExamID": validation_dataset.records[candidate_index]["EyeExamID"],
        "real_view_count": real_count,
        "padded_view_count": max_views,
        "zero_padding_max_abs_feature_difference": zero_difference,
        "poisoned_padding_max_abs_feature_difference": poison_difference,
        "mixed_eye_padded_batch_max_abs_feature_difference": mixed_batch_difference,
        "companion_EyeExamID": validation_dataset.records[companion_index][
            "EyeExamID"
        ],
        "padding_rtol": padding_rtol,
        "padding_atol": padding_atol,
        "mixed_batch_rtol": mixed_batch_rtol,
        "mixed_batch_atol": mixed_batch_atol,
        "backbone_received_only_unmasked_views": True,
    }


def labels_to_targets(labels: pd.Series) -> np.ndarray:
    mapped = labels.map(CLASS_TO_INDEX)
    if mapped.isna().any():
        raise SafetyError("Prediction labels contain an unexpected class.")
    return mapped.to_numpy(dtype=np.int64)


def compute_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(targets)) != 2:
        raise SafetyError("AUROC requires both Normal and Abnormal validation labels.")
    if not np.isfinite(probabilities).all():
        raise SafetyError("Nonfinite validation probabilities encountered.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise SafetyError("Validation probabilities fall outside [0, 1].")
    return float(roc_auc_score(targets, probabilities))


def evaluate(model, loader, dataset, criterion, device, max_views):
    # save one eye probability plus per-view diagnostic probabilities
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            eye_indices = batch["eye_indices"].tolist()
            eye_logits, _, view_logits = model(
                images,
                view_mask,
                return_diagnostics=True,
            )
            loss_vector = criterion(eye_logits, targets)
            sample_weights = criterion.weight[targets]
            loss_numerator += float(loss_vector.sum().item())
            loss_denominator += float(sample_weights.sum().item())
            eye_probabilities = torch.softmax(eye_logits, dim=1)[:, 1]
            view_probabilities = torch.softmax(view_logits, dim=2)[:, :, 1]

            for batch_index, eye_index in enumerate(eye_indices):
                record = dataset.records[eye_index]
                real_mask = view_mask[batch_index]
                real_view_probabilities = (
                    view_probabilities[batch_index][real_mask].cpu().numpy().astype(float)
                )
                if len(real_view_probabilities) != record["NumberOfViews"]:
                    raise SafetyError("Diagnostic view count differs from locked eye views.")
                if not np.isfinite(real_view_probabilities).all():
                    raise SafetyError("Nonfinite per-view diagnostic probability.")
                row = {
                    "EyeExamID": record["EyeExamID"],
                    "ResearchSubjectID": record["ResearchSubjectID"],
                    "EncounterID": record["EncounterID"],
                    "Laterality": record["Laterality"],
                    "TrueLabel": record["EyeLabel"],
                    "NumberOfViews": record["NumberOfViews"],
                    "EyeAbnormalProbability": float(
                        eye_probabilities[batch_index].cpu().item()
                    ),
                }
                for view_number in range(1, max_views + 1):
                    row[f"View{view_number}AbnormalProbability"] = np.nan
                for view_number, probability in zip(
                    record["SelectedViewIndices"], real_view_probabilities
                ):
                    row[f"View{view_number}AbnormalProbability"] = float(probability)
                row["MeanViewProbability"] = float(np.mean(real_view_probabilities))
                row["MaxViewProbability"] = float(np.max(real_view_probabilities))
                row["MinViewProbability"] = float(np.min(real_view_probabilities))
                row["StdViewProbability"] = float(
                    np.std(real_view_probabilities, ddof=0)
                )
                row["ViewProbabilityRange"] = float(
                    np.max(real_view_probabilities) - np.min(real_view_probabilities)
                )
                prediction_rows.append(row)
    if loss_denominator <= 0:
        raise SafetyError("Validation loss denominator is nonpositive.")
    predictions = pd.DataFrame(prediction_rows).sort_values(
        "EyeExamID", kind="mergesort"
    ).reset_index(drop=True)
    if len(predictions) != len(dataset) or not predictions["EyeExamID"].is_unique:
        raise SafetyError("Validation predictions are not exactly one row per eye.")
    targets = labels_to_targets(predictions["TrueLabel"])
    probabilities = predictions["EyeAbnormalProbability"].to_numpy(dtype=float)
    auroc = compute_auroc(targets, probabilities)
    balanced_accuracy_at_0_5 = float(
        balanced_accuracy_score(targets, (probabilities >= 0.5).astype(int))
    )
    return {
        "loss": loss_numerator / loss_denominator,
        "auroc": auroc,
        "balanced_accuracy_at_0_5": balanced_accuracy_at_0_5,
        "predictions": predictions,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    # optimization happens once per eye, not once per image
    model.train()
    loss_numerator = 0.0
    loss_denominator = 0.0
    eye_count = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        eye_logits = model(images, view_mask)
        loss_vector = criterion(eye_logits, targets)
        # each eye contributes one weighted loss term
        sample_weights = criterion.weight[targets]
        loss = loss_vector.sum() / sample_weights.sum()
        loss.backward()
        optimizer.step()
        loss_numerator += float(loss_vector.detach().sum().item())
        loss_denominator += float(sample_weights.detach().sum().item())
        eye_count += int(targets.numel())
    if loss_denominator <= 0 or eye_count != len(loader.dataset):
        raise SafetyError("Training did not process exactly one loss term per eye.")
    return loss_numerator / loss_denominator


def choose_classification_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict]:
    compute_auroc(targets, probabilities)
    # tune the classification threshold using validation eyes only
    unique_probabilities = sorted(set(float(value) for value in probabilities))
    # use midpoints so the threshold does not sit exactly on an observed score
    # this keeps the frozen decisions stable when probabilities are read back later
    interval_midpoints = []
    for lower, upper in zip(unique_probabilities[:-1], unique_probabilities[1:]):
        midpoint = lower + (upper - lower) / 2.0
        if not lower < midpoint < upper:
            midpoint = float(np.nextafter(lower, upper))
        if lower < midpoint < upper:
            interval_midpoints.append(midpoint)
    candidates = sorted(set(interval_midpoints) | {0.0, 0.5, 1.0})
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    results = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(
            targets, predicted, labels=[0, 1]
        ).ravel()
        exact_score_numerator = int(tp * negative_count + tn * positive_count)
        results.append(
            {
                "threshold": threshold,
                "score_numerator": exact_score_numerator,
                "distance_from_0_5": abs(threshold - 0.5),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    best = min(
        results,
        key=lambda row: (
            -row["score_numerator"],
            row["distance_from_0_5"],
            row["threshold"],
        ),
    )
    denominator = 2 * positive_count * negative_count
    best["balanced_accuracy"] = best["score_numerator"] / denominator
    best["candidate_count"] = len(candidates)
    distances = np.abs(probabilities - float(best["threshold"]))
    best["minimum_distance_to_observed_probability"] = float(distances.min())
    if best["minimum_distance_to_observed_probability"] <= 0:
        raise SafetyError(
            "Selected classification threshold lies on an observed probability."
        )
    return float(best["threshold"]), best


def compute_metrics(targets, probabilities, threshold):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "AUROC": compute_auroc(targets, probabilities),
        "BalancedAccuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1": float(f1_score(targets, predictions, zero_division=0)),
        "Accuracy": float(accuracy_score(targets, predictions)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def save_history(history, path: Path):
    columns = [
        "Epoch",
        "TrainingLoss",
        "ValidationLoss",
        "ValidationEyeAUROCPrimary",
        "ValidationBalancedAccuracyAt0_5",
        "CheckpointImproved",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history, path: Path):
    epochs = [row["Epoch"] for row in history]
    train_loss = [row["TrainingLoss"] for row in history]
    validation_loss = [row["ValidationLoss"] for row in history]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, train_loss, marker="o", label="Training loss")
    axis.plot(epochs, validation_loss, marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Class-weighted cross-entropy")
    axis.set_title("Multi-view ResNet-18 feature-MAX loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_eye_roc_plot(targets, probabilities, auroc, path: Path):
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Feature-MAX eye AUROC = {auroc:.3f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Validation eye-level ROC")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_plot(metrics, path: Path):
    matrix = np.asarray(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    )
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=12,
            )
    axis.set_xticks([0, 1], ["Normal", "Abnormal"])
    axis.set_yticks([0, 1], ["Normal", "Abnormal"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Validation eye-level confusion matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def get_git_info():
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


def save_json(path: Path, value: dict):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def get_environment_info(device: torch.device) -> dict:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        properties = torch.cuda.get_device_properties(device)
        device_memory_bytes = int(properties.total_memory)
    else:
        device_name = platform.processor() or "CPU"
        device_memory_bytes = None
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": Image.__version__,
        "matplotlib_version": matplotlib.__version__,
        "pyyaml_version": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": device_name,
        "device_total_memory_bytes": device_memory_bytes,
    }


def make_baseline_comparison(baseline_metrics: dict, multiview_metrics: dict):
    rows = []
    for metric in [
        "AUROC",
        "BalancedAccuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "Accuracy",
    ]:
        baseline = float(baseline_metrics[metric])
        multiview = float(multiview_metrics[metric])
        rows.append(
            {
                "Metric": metric,
                "PerImageBaseline": baseline,
                "MultiViewFeatureMax": multiview,
                "Difference": multiview - baseline,
            }
        )
    return pd.DataFrame(rows)


def main():
    # run data and protocol checks before creating a model run
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    set_random_seeds(int(config["seed"]))

    print("=" * 82)
    print("DAY-4 MULTI-VIEW EYE-LEVEL RESNET-18: FEATURE-WISE MAX POOLING")
    print("=" * 82)
    print(f"Configuration       : {config_path}")
    print(f"Train manifest      : {TRAIN_INPUT_PATH}")
    print(f"Validation manifest : {VAL_INPUT_PATH}")
    print("Test manifest       : NOT LOADED; TEST SET WILL NOT BE EVALUATED")
    print("Training unit       : one EyeExamID with 1-6 locked real views")
    print("Primary model       : ResNet features -> feature-wise MAX -> eye classifier")
    print("Loss                : one class-weighted cross-entropy term per eye")
    print(
        "View diagnostics    : shared classifier on each view feature; "
        "not used for loss/model selection\n"
    )

    # day-3 metrics are loaded for reporting only, not model selection
    train_records, validation_records, preflight_report = run_preflight(config)
    baseline_metrics_path = get_baseline_metrics_path(config)
    baseline_metrics, baseline_metrics_hash = read_baseline_metrics(
        baseline_metrics_path
    )
    print("MANIFEST PREFLIGHT: PASS")
    print(
        f"Train images/eyes/subjects           : "
        f"{preflight_report['training_images']:,} / "
        f"{preflight_report['training_eyes']:,} / "
        f"{preflight_report['training_subjects']:,}"
    )
    print(
        f"Validation images/eyes/subjects      : "
        f"{preflight_report['validation_images']:,} / "
        f"{preflight_report['validation_eyes']:,} / "
        f"{preflight_report['validation_subjects']:,}"
    )
    print(
        "Training Normal / Abnormal eyes      : "
        f"{preflight_report['training_normal_eyes']:,} / "
        f"{preflight_report['training_abnormal_eyes']:,}"
    )
    print(
        "Class weights [Normal, Abnormal]     : "
        f"{preflight_report['class_weights_normal_abnormal']}"
    )
    print(f"Configured batch size                : {config['batch_size']} (unchanged)")
    print(f"Locked Day-3 metrics                 : {baseline_metrics_path}\n")

    device = get_device(config)
    train_dataset, validation_dataset, train_loader, validation_loader = (
        make_datasets_and_loaders(
            train_records,
            validation_records,
            config,
            device,
        )
    )
    print(f"Device                               : {device}")
    if device.type == "cuda":
        print(f"GPU                                  : {torch.cuda.get_device_name(device)}")
    print("Loading fresh ImageNet ResNet18_Weights.IMAGENET1K_V1 ...")
    # train the imagenet backbone and the eye classifier together
    model = FeatureMaxResNet18(weights=WEIGHTS_ENUM).to(device)
    if model.feature_dim != 512:
        raise SafetyError(f"Expected ResNet-18 feature dimension 512, got {model.feature_dim}.")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise SafetyError("All backbone and classifier parameters must be trainable.")
    masking_check = check_masked_pooling(
        model,
        validation_dataset,
        device,
        int(config["max_views"]),
    )
    print(
        "MASKED-PADDING INVARIANCE: PASS "
        f"({masking_check['EyeExamID']}, {masking_check['real_view_count']} real -> "
        f"{masking_check['padded_view_count']} padded; zero/poison max diff "
        f"{max(masking_check['zero_padding_max_abs_feature_difference'], masking_check['poisoned_padding_max_abs_feature_difference']):.3g}; "
        f"mixed-batch max diff "
        f"{masking_check['mixed_eye_padded_batch_max_abs_feature_difference']:.3g} "
        f"within rtol={masking_check['mixed_batch_rtol']:.0e}, "
        f"atol={masking_check['mixed_batch_atol']:.0e})\n"
    )

    if args.preflight_only:
        check_test_protection()
        print("Preflight-only mode complete. No run directory or checkpoint was created.")
        print("TEST SET WAS NOT LOADED OR EVALUATED")
        return

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    model_dir = MODEL_OUTPUT_ROOT / timestamp
    figure_dir = FIGURE_OUTPUT_ROOT / timestamp
    audit_dir = AUDIT_OUTPUT_ROOT / timestamp
    model_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    audit_dir.mkdir(parents=True, exist_ok=False)

    best_model_path = model_dir / "best_model.pt"
    config_used_path = model_dir / "config_used.yaml"
    metadata_path = model_dir / "run_metadata.json"
    history_path = model_dir / "training_history.csv"
    val_predictions_path = model_dir / "val_eye_predictions.csv"
    metrics_path = model_dir / "val_metrics.csv"
    threshold_path = model_dir / "classification_threshold.json"
    comparison_path = audit_dir / "day4_baseline_vs_multiview.csv"
    with config_used_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(f"Model run directory                  : {model_dir}")
    print(f"Figure run directory                 : {figure_dir}")
    print(f"Audit run directory                  : {audit_dir}\n")

    class_weights = torch.tensor(
        preflight_report["class_weights_normal_abnormal"],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = []
    best_auroc = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    started_at = datetime.now().astimezone()
    wall_start = time.perf_counter()
    # keep the checkpoint with the best validation eye-level auroc
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        training_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            validation_dataset,
            criterion,
            device,
            int(config["max_views"]),
        )
        current_auroc = validation_result["auroc"]
        improved = current_auroc > best_auroc
        if improved:
            best_auroc = current_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_eye_auroc": best_auroc,
                    "class_to_index": CLASS_TO_INDEX,
                    "architecture": "resnet18",
                    "pooling": "feature-wise max",
                    "feature_dimension": model.feature_dim,
                    "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
                    "day3_checkpoint_loaded": False,
                    "config": config,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "Epoch": epoch,
                "TrainingLoss": training_loss,
                "ValidationLoss": validation_result["loss"],
                "ValidationEyeAUROCPrimary": current_auroc,
                "ValidationBalancedAccuracyAt0_5": validation_result[
                    "balanced_accuracy_at_0_5"
                ],
                "CheckpointImproved": improved,
            }
        )
        save_history(history, history_path)
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train loss {training_loss:.5f} | "
            f"val loss {validation_result['loss']:.5f} | "
            f"val eye AUROC {current_auroc:.4f} | "
            f"val balanced accuracy@0.5 "
            f"{validation_result['balanced_accuracy_at_0_5']:.4f} | "
            f"{'BEST' if improved else 'no improvement'} | "
            f"{time.perf_counter() - epoch_start:.1f}s"
        )
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            print(
                "Early stopping: validation eye-level AUROC did not improve for "
                f"{config['early_stopping_patience']} consecutive epochs."
            )
            break

    completed_epochs = len(history)
    if best_epoch is None or not best_model_path.is_file():
        raise SafetyError("No best checkpoint was produced.")

    # reload the best checkpoint before creating final validation outputs
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate(
        model,
        validation_loader,
        validation_dataset,
        criterion,
        device,
        int(config["max_views"]),
    )
    if not np.isclose(final_validation["auroc"], best_auroc, rtol=0, atol=1e-12):
        raise SafetyError(
            "Reloaded best checkpoint AUROC does not match the recorded best AUROC."
        )

    val_predictions = final_validation["predictions"]
    targets = labels_to_targets(val_predictions["TrueLabel"])
    probabilities = val_predictions["EyeAbnormalProbability"].to_numpy(float)
    # freeze the final threshold from validation predictions only
    threshold, threshold_details = choose_classification_threshold(
        targets,
        probabilities,
    )
    predicted_indices = (probabilities >= threshold).astype(np.int64)
    val_predictions["PredictedLabel"] = [
        INDEX_TO_CLASS[index] for index in predicted_indices
    ]
    val_predictions["ClassificationThreshold"] = threshold
    val_predictions["Correct"] = val_predictions["PredictedLabel"].eq(
        val_predictions["TrueLabel"]
    )
    ordered_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "TrueLabel",
        "NumberOfViews",
        "EyeAbnormalProbability",
        "PredictedLabel",
        "ClassificationThreshold",
        "Correct",
        *[
            f"View{view}AbnormalProbability"
            for view in range(1, int(config["max_views"]) + 1)
        ],
        "MeanViewProbability",
        "MaxViewProbability",
        "MinViewProbability",
        "StdViewProbability",
        "ViewProbabilityRange",
    ]
    val_predictions = val_predictions[ordered_columns]
    metrics = compute_metrics(targets, probabilities, threshold)

    check_test_protection()
    train_manifest_hash_after = file_sha256(TRAIN_INPUT_PATH)
    validation_manifest_hash_after = file_sha256(VAL_INPUT_PATH)
    if train_manifest_hash_after != preflight_report["training_manifest_sha256_before"]:
        raise SafetyError("Training manifest changed during the model run.")
    if (
        validation_manifest_hash_after
        != preflight_report["validation_manifest_sha256_before"]
    ):
        raise SafetyError("Validation manifest changed during the model run.")
    if file_sha256(baseline_metrics_path) != baseline_metrics_hash:
        raise SafetyError("Locked Day-3 validation metrics changed during the run.")

    val_predictions.to_csv(val_predictions_path, index=False, na_rep="")
    pd.DataFrame(
        [
            {
                "EvaluationUnit": "eye_primary",
                "Architecture": "ResNet-18",
                "Pooling": "feature-wise MAX",
                "Threshold": threshold,
                "ThresholdSource": "validation balanced accuracy optimum",
                "PrimaryForModelSelection": True,
                **metrics,
            }
        ]
    ).to_csv(metrics_path, index=False)

    threshold_payload = {
        "classification_threshold": threshold,
        "selected_on": "validation eye-level pooled-feature probabilities only",
        "objective": "maximize balanced accuracy",
        "prediction_rule": (
            "Abnormal if EyeAbnormalProbability >= threshold; otherwise Normal"
        ),
        "tie_breaking_rule": "closest to 0.5, then lower threshold",
        "candidate_rule": (
            "strict midpoints between consecutive distinct finite validation-eye "
            "probabilities, plus 0.0, 0.5, and 1.0"
        ),
        "candidate_count": threshold_details["candidate_count"],
        "minimum_distance_to_observed_probability": threshold_details[
            "minimum_distance_to_observed_probability"
        ],
        "balanced_accuracy": threshold_details["balanced_accuracy"],
        "tp": threshold_details["tp"],
        "tn": threshold_details["tn"],
        "fp": threshold_details["fp"],
        "fn": threshold_details["fn"],
        "distinct_from_future_confidence_abstention_thresholds": True,
        "test_data_used": False,
    }
    save_json(threshold_path, threshold_payload)

    # this comparison is descriptive and does not change the selected model
    comparison = make_baseline_comparison(baseline_metrics, metrics)
    comparison.to_csv(comparison_path, index=False)
    save_loss_plot(history, figure_dir / "training_loss_curve.png")
    save_eye_roc_plot(
        targets,
        probabilities,
        metrics["AUROC"],
        figure_dir / "validation_eye_roc_curve.png",
    )
    save_confusion_plot(metrics, figure_dir / "validation_confusion_matrix.png")

    # save enough run information to reproduce the experiment later
    git_commit, git_status = get_git_info()
    completed_at = datetime.now().astimezone()
    metadata = {
        "run_timestamp": timestamp,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_time_seconds": time.perf_counter() - wall_start,
        "seed": int(config["seed"]),
        "git_commit_hash": git_commit,
        "git_worktree_status_at_completion": git_status,
        "training_script": str(Path(__file__).resolve()),
        "training_script_sha256": file_sha256(Path(__file__).resolve()),
        "configuration_input": str(config_path),
        "configuration_input_sha256": file_sha256(config_path),
        "config_used_sha256": file_sha256(config_used_path),
        "training_manifest": str(TRAIN_INPUT_PATH),
        "training_manifest_sha256": train_manifest_hash_after,
        "validation_manifest": str(VAL_INPUT_PATH),
        "validation_manifest_sha256": validation_manifest_hash_after,
        "loaded_manifest_paths": sorted(map(str, LOADED_MANIFEST_PATHS)),
        "loaded_reporting_paths": sorted(map(str, LOADED_REPORT_PATHS)),
        "test_manifest_loaded": False,
        "test_set_evaluated": False,
        "test_predictions_created": False,
        "task": "eye-examination-level Normal vs Abnormal classification",
        "training_unit": "one EyeExamID with 1-6 locked selected views",
        "loss_unit": "one class-weighted cross-entropy term per EyeExamID",
        "model_architecture": "resnet18",
        "pretrained": True,
        "pretrained_weights_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "pretrained_weights_url": WEIGHTS_ENUM.url,
        "day3_checkpoint_loaded": False,
        "initialization": "fresh ImageNet initialization; not the Day-3 checkpoint",
        "fine_tuned_backbone": True,
        "feature_dimension": model.feature_dim,
        "pooling": "feature-wise MAX across real view feature vectors",
        "pooling_is_not_probability_max": True,
        "padded_view_handling": (
            "boolean mask; padded tensors are excluded from backbone and assigned "
            "negative infinity before feature-wise MAX"
        ),
        "masked_padding_invariance_check": masking_check,
        "classifier": "shared linear 512-to-2 Normal/Abnormal head",
        "optimizer": "AdamW",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "batch_size_silently_changed": False,
        "configured_epochs": int(config["epochs"]),
        "completed_epochs": completed_epochs,
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "best_epoch": best_epoch,
        "best_validation_eye_auroc": best_auroc,
        "classification_threshold": threshold,
        "checkpoint_selection": (
            "strict improvement in validation eye-level AUROC; earliest epoch "
            "retained on equal AUROC"
        ),
        "training_eye_class_counts": {
            "Normal": preflight_report["training_normal_eyes"],
            "Abnormal": preflight_report["training_abnormal_eyes"],
        },
        "class_weights_normal_abnormal": preflight_report[
            "class_weights_normal_abnormal"
        ],
        "class_weight_formula": "N / (2 * class_count), training eyes only",
        "training_view_count_distribution": preflight_report[
            "training_view_count_distribution"
        ],
        "validation_view_count_distribution": preflight_report[
            "validation_view_count_distribution"
        ],
        "view_order": "locked SelectedViewIndex ascending; ImageRelativePath tie-break",
        "preprocessing": {
            "load": "Pillow EXIF transpose then three-channel RGB conversion",
            "aspect_ratio": "centered black square padding before resize",
            "resize": [int(config["image_size"]), int(config["image_size"])],
            "resize_interpolation": "bilinear, antialias=True",
            "imagenet_mean": IMAGE_NET_MEAN,
            "imagenet_std": IMAGE_NET_STD,
            "validation_deterministic": True,
            "training_augmentation": {
                "random_rotation_degrees": [
                    -float(config["augmentation"]["rotation_degrees"]),
                    float(config["augmentation"]["rotation_degrees"]),
                ],
                "translation_fraction": float(
                    config["augmentation"]["translate_fraction"]
                ),
                "scale_range": [
                    float(config["augmentation"]["scale_min"]),
                    float(config["augmentation"]["scale_max"]),
                ],
                "brightness": float(config["augmentation"]["brightness"]),
                "contrast": float(config["augmentation"]["contrast"]),
                "horizontal_flip": False,
                "vertical_flip": False,
                "crop": False,
            },
        },
        "view_diagnostics": {
            "purpose": "secondary descriptive diagnostics for future work only",
            "head": "same shared eye classifier applied to each valid view feature",
            "used_for_training_loss": False,
            "used_for_model_selection": False,
            "used_for_routing": False,
            "standard_deviation": "population standard deviation, ddof=0",
            "eye_probability_definition": (
                "softmax Abnormal probability from classifier(feature-wise MAX(features))"
            ),
            "max_view_probability_definition": (
                "maximum shared-head Abnormal probability across individual views"
            ),
            "eye_probability_need_not_equal_max_view_probability": True,
        },
        "determinism": {
            "python_seeded": True,
            "numpy_seeded": True,
            "torch_seeded": True,
            "cuda_seeded": torch.cuda.is_available(),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "dataloader_num_workers": int(config["num_workers"]),
        },
        "environment": get_environment_info(device),
        "validation_metrics": metrics,
        "validation_diagnostics_summary": {
            "mean_number_of_views": float(val_predictions["NumberOfViews"].mean()),
            "mean_view_probability_standard_deviation": float(
                val_predictions["StdViewProbability"].mean()
            ),
            "mean_view_probability_range": float(
                val_predictions["ViewProbabilityRange"].mean()
            ),
        },
        "day3_baseline_comparison": {
            "metrics_path": str(baseline_metrics_path),
            "metrics_sha256": baseline_metrics_hash,
            "difference_definition": "MultiViewFeatureMax - PerImageBaseline",
            "validation_comparison_only": True,
            "no_superiority_claim_from_numeric_difference_alone": True,
        },
        "artifacts": {
            "best_model": str(best_model_path),
            "config_used": str(config_used_path),
            "training_history": str(history_path),
            "val_eye_predictions": str(val_predictions_path),
            "val_metrics": str(metrics_path),
            "classification_threshold": str(threshold_path),
            "baseline_comparison": str(comparison_path),
            "figure_directory": str(figure_dir),
        },
    }
    save_json(metadata_path, metadata)

    mean_views = float(val_predictions["NumberOfViews"].mean())
    mean_view_std = float(val_predictions["StdViewProbability"].mean())
    mean_view_range = float(val_predictions["ViewProbabilityRange"].mean())
    print("\n" + "=" * 82)
    print("DAY-4 MULTI-VIEW TRAINING COMPLETE")
    print("=" * 82)
    print(f"Device                               : {device}")
    print(f"Batch size                           : {config['batch_size']}")
    print(f"Completed epochs                     : {completed_epochs}")
    print(f"Best epoch                           : {best_epoch}")
    print(
        "Training Normal / Abnormal eyes      : "
        f"{preflight_report['training_normal_eyes']} / "
        f"{preflight_report['training_abnormal_eyes']}"
    )
    print(
        "Class weights [Normal, Abnormal]     : "
        f"{preflight_report['class_weights_normal_abnormal']}"
    )
    print(f"Best validation eye AUROC            : {best_auroc:.6f}")
    print(f"Frozen classification threshold      : {threshold:.10f}")
    print("\nValidation eye-level metrics:")
    for name in [
        "AUROC",
        "BalancedAccuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "Accuracy",
    ]:
        print(f"  {name:<18}: {metrics[name]:.6f}")
    print(
        f"  TP / TN / FP / FN : {metrics['TP']} / {metrics['TN']} / "
        f"{metrics['FP']} / {metrics['FN']}"
    )
    print("\nDAY-3 PER-IMAGE BASELINE VS DAY-4 MULTI-VIEW")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nDescriptive per-view diagnostics (not used for selection/routing):")
    print(f"  Mean validation views per eye       : {mean_views:.6f}")
    print(f"  Mean view-probability std (ddof=0)  : {mean_view_std:.6f}")
    print(f"  Mean view-probability range         : {mean_view_range:.6f}")
    print(f"\nBest model                           : {best_model_path}")
    print(f"Validation eye predictions           : {val_predictions_path}")
    print(f"Validation metrics                   : {metrics_path}")
    print(f"Baseline comparison                  : {comparison_path}")
    print(f"Run metadata                         : {metadata_path}")
    print("\nValidation comparison only; numerical differences do not establish superiority.")
    print("TEST SET WAS NOT LOADED OR EVALUATED")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "CUDA out of memory. The configured batch size was not changed. "
            "Report this failure before manually changing batch_size from 4 to 2."
        ) from error
    except SafetyError as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
