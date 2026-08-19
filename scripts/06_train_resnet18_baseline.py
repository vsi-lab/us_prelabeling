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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_resnet18.yaml"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "baseline_resnet18"
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "baseline_resnet18"

ALLOWED_INPUT_PATHS = {TRAIN_INPUT_PATH, VAL_INPUT_PATH}
FORBIDDEN_TEST_BASENAME = "test_images.csv"
LOADED_CSV_PATHS: set[Path] = set()
CLASS_TO_INDEX = {"Normal": 0, "Abnormal": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
WEIGHTS_ENUM = ResNet18_Weights.IMAGENET1K_V1


class SafetyError(RuntimeError):
    "used when an input or safety check fails."


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


class BscanImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(row["ResolvedImagePath"])
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image = self.transform(image)
        target = CLASS_TO_INDEX[row["EyeLabel"]]
        return image, target, index


def parse_args():
    # preflight-only is useful for checking the data without starting training
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
        help="Run all input/safety checks without loading weights or training.",
    )
    return parser.parse_args()


def read_allowed_manifest(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    if resolved.name.casefold() == FORBIDDEN_TEST_BASENAME.casefold():
        raise SafetyError("Held-out test manifest access was attempted.")
    if resolved not in ALLOWED_INPUT_PATHS:
        raise SafetyError(f"CSV input is not allowlisted: {resolved}")
    LOADED_CSV_PATHS.add(resolved)
    return pd.read_csv(resolved, low_memory=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "model",
        "pretrained",
        "pretrained_weights",
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
        "augmentation",
    }
    missing = sorted(required - set(config or {}))
    if missing:
        raise SafetyError(f"Configuration is missing required fields: {missing}")
    if config["model"] != "resnet18":
        raise SafetyError("This baseline supports only model=resnet18.")
    if config["pretrained"] is not True:
        raise SafetyError("The requested baseline requires pretrained=true.")
    if config["pretrained_weights"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise SafetyError(
            "pretrained_weights must be ResNet18_Weights.IMAGENET1K_V1."
        )
    if config["optimizer"] != "AdamW":
        raise SafetyError("The requested baseline requires optimizer=AdamW.")
    for field in ["image_size", "epochs", "batch_size", "early_stopping_patience"]:
        if int(config[field]) <= 0:
            raise SafetyError(f"{field} must be a positive integer.")
    if int(config["num_workers"]) < 0:
        raise SafetyError("num_workers must be nonnegative.")
    for field in ["learning_rate", "weight_decay"]:
        if float(config[field]) < 0:
            raise SafetyError(f"{field} must be nonnegative.")
    augmentation_required = {
        "rotation_degrees",
        "translate_fraction",
        "scale_min",
        "scale_max",
        "brightness",
        "contrast",
    }
    missing_augmentation = sorted(
        augmentation_required - set(config["augmentation"] or {})
    )
    if missing_augmentation:
        raise SafetyError(
            f"Augmentation config is missing: {missing_augmentation}"
        )
    if float(config["augmentation"]["scale_min"]) <= 0:
        raise SafetyError("augmentation.scale_min must be positive.")
    if float(config["augmentation"]["scale_max"]) < float(
        config["augmentation"]["scale_min"]
    ):
        raise SafetyError("augmentation.scale_max must be >= scale_min.")
    return config


def parse_bool_column(series: pd.Series, field: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = normalized.loc[~normalized.isin(["true", "false"])]
    if invalid.notna().any():
        values = sorted(invalid.dropna().astype(str).unique())
        raise SafetyError(f"Invalid boolean values in {field}: {values}")
    return normalized.eq("true").fillna(False)


def get_image_path(relative_value: object) -> Path:
    text = str(relative_value).strip()
    if not text:
        raise SafetyError("Blank ImageRelativePath encountered.")
    relative = Path(text)
    if relative.is_absolute():
        raise SafetyError(f"Absolute source image path is forbidden: {text}")
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
        "EyeLabel",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "DatasetSplit",
    ]
    for field in consistency_fields:
        inconsistent = frame.groupby("EyeExamID")[field].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise SafetyError(
                f"{name}: {int(inconsistent.sum())} EyeExamIDs have inconsistent {field}."
            )


def check_manifest(frame: pd.DataFrame, name: str, expected_split: str):
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SafetyError(f"{name}: missing required columns: {missing}")
    if frame.empty:
        raise SafetyError(f"{name}: manifest is empty.")
    labels = set(frame["EyeLabel"].dropna().astype(str).unique())
    if frame["EyeLabel"].isna().any() or not labels.issubset(CLASS_TO_INDEX):
        raise SafetyError(f"{name}: labels outside Normal/Abnormal: {sorted(labels)}")
    if frame["Laterality"].eq("UNKNOWN").any():
        raise SafetyError(f"{name}: UNKNOWN laterality is forbidden.")
    invalid_laterality = ~frame["Laterality"].isin(["OD", "OS"])
    if invalid_laterality.any():
        raise SafetyError(
            f"{name}: invalid laterality rows={int(invalid_laterality.sum())}."
        )
    split_values = set(frame["DatasetSplit"].dropna().astype(str).unique())
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
        raise SafetyError(f"{name}: duplicate SHA-256 within an EyeExamID.")
    check_eye_consistency(frame, name)


def run_preflight(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = read_allowed_manifest(TRAIN_INPUT_PATH)
    validation = read_allowed_manifest(VAL_INPUT_PATH)
    if LOADED_CSV_PATHS != ALLOWED_INPUT_PATHS:
        raise SafetyError(
            "Loaded CSV set differs from the exact train/validation allowlist: "
            f"{sorted(map(str, LOADED_CSV_PATHS))}"
        )
    check_manifest(train, "train", "train")
    check_manifest(validation, "validation", "validation")
    train_subjects = set(train["ResearchSubjectID"].astype(str))
    validation_subjects = set(validation["ResearchSubjectID"].astype(str))
    intersection = train_subjects & validation_subjects
    if intersection:
        raise SafetyError(
            f"Train/validation subject intersection is nonempty: {len(intersection)}"
        )
    train_eyes = set(train["EyeExamID"].astype(str))
    validation_eyes = set(validation["EyeExamID"].astype(str))
    if train_eyes & validation_eyes:
        raise SafetyError("Train/validation EyeExamID intersection is nonempty.")

    train = train.copy()
    validation = validation.copy()
    train["ResolvedImagePath"] = train["ImageRelativePath"].map(get_image_path)
    validation["ResolvedImagePath"] = validation["ImageRelativePath"].map(
        get_image_path
    )

    train_counts = train["EyeLabel"].value_counts()
    normal_count = int(train_counts.get("Normal", 0))
    abnormal_count = int(train_counts.get("Abnormal", 0))
    if normal_count == 0 or abnormal_count == 0:
        raise SafetyError("Training image rows must contain both classes.")
    total = normal_count + abnormal_count
    class_weights = [
        total / (2.0 * normal_count),
        total / (2.0 * abnormal_count),
    ]

    validation_eye_labels = validation.groupby("EyeExamID")["EyeLabel"].first()
    if set(validation_eye_labels.unique()) != set(CLASS_TO_INDEX):
        raise SafetyError("Validation eyes must contain both Normal and Abnormal.")

    report = {
        "train_images": len(train),
        "validation_images": len(validation),
        "train_eyes": len(train_eyes),
        "validation_eyes": len(validation_eyes),
        "train_subjects": len(train_subjects),
        "validation_subjects": len(validation_subjects),
        "subject_intersection": 0,
        "training_normal_images": normal_count,
        "training_abnormal_images": abnormal_count,
        "class_weights_normal_abnormal": class_weights,
        "batch_size": int(config["batch_size"]),
        "test_manifest_loaded": False,
        "loaded_csv_paths": sorted(map(str, LOADED_CSV_PATHS)),
    }
    return train, validation, report


def set_random_seeds(seed: int):
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


def make_loaders(train, validation, config, device):
    train_transform, validation_transform = make_transforms(config)
    train_dataset = BscanImageDataset(train, train_transform)
    validation_dataset = BscanImageDataset(validation, validation_transform)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_data_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, validation_loader


def combine_eye_predictions(image_predictions: pd.DataFrame) -> pd.DataFrame:
    consistency_fields = [
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "TrueEyeLabel",
    ]
    for field in consistency_fields:
        inconsistent = image_predictions.groupby("EyeExamID")[field].nunique(
            dropna=False
        )
        if inconsistent.gt(1).any():
            raise SafetyError(f"Validation prediction metadata inconsistent: {field}")
    metadata = image_predictions.groupby("EyeExamID", as_index=False).agg(
        ResearchSubjectID=("ResearchSubjectID", "first"),
        EncounterID=("EncounterID", "first"),
        Laterality=("Laterality", "first"),
        TrueLabel=("TrueEyeLabel", "first"),
        MaxAbnormalProbability=("AbnormalProbability", "max"),
    )
    return metadata.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)


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


def evaluate(model, loader, frame, criterion, device):
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    probabilities = []
    targets = []
    row_indices = []
    with torch.no_grad():
        for inputs, batch_targets, indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss_vector = criterion(logits, batch_targets)
            sample_weights = criterion.weight[batch_targets]
            loss_numerator += float(loss_vector.sum().item())
            loss_denominator += float(sample_weights.sum().item())
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().numpy().tolist())
            targets.extend(batch_targets.cpu().numpy().tolist())
            row_indices.extend(indices.numpy().tolist())
    if loss_denominator <= 0:
        raise SafetyError("Validation loss denominator is nonpositive.")
    predictions = frame.iloc[row_indices][
        [
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "ImageRelativePath",
            "EyeLabel",
        ]
    ].copy()
    predictions = predictions.rename(columns={"EyeLabel": "TrueEyeLabel"})
    predictions["AbnormalProbability"] = probabilities
    predictions = predictions.sort_values(
        ["EyeExamID", "ImageRelativePath"], kind="mergesort"
    ).reset_index(drop=True)

    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    image_auroc = compute_auroc(target_array, probability_array)
    image_accuracy = float(
        accuracy_score(target_array, (probability_array >= 0.5).astype(int))
    )
    eye_predictions = combine_eye_predictions(predictions)
    eye_targets = labels_to_targets(eye_predictions["TrueLabel"])
    eye_probabilities = eye_predictions["MaxAbnormalProbability"].to_numpy(
        dtype=np.float64
    )
    eye_auroc = compute_auroc(eye_targets, eye_probabilities)
    return {
        "loss": loss_numerator / loss_denominator,
        "image_auroc": image_auroc,
        "image_accuracy_at_0_5": image_accuracy,
        "eye_auroc": eye_auroc,
        "image_predictions": predictions,
        "eye_predictions": eye_predictions,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_numerator = 0.0
    loss_denominator = 0.0
    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss_vector = criterion(logits, targets)
        sample_weights = criterion.weight[targets]
        loss = loss_vector.sum() / sample_weights.sum()
        loss.backward()
        optimizer.step()
        loss_numerator += float(loss_vector.detach().sum().item())
        loss_denominator += float(sample_weights.detach().sum().item())
    if loss_denominator <= 0:
        raise SafetyError("Training loss denominator is nonpositive.")
    return loss_numerator / loss_denominator


def choose_classification_threshold(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict]:
    compute_auroc(targets, probabilities)
    candidates = sorted(
        set(float(value) for value in probabilities) | {0.0, 0.5, 1.0}
    )
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
        "BalancedAccuracy": float(balanced_accuracy_score(targets, predictions)),
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
        "ValidationImageAUROCDescriptive",
        "ValidationImageAccuracyAt0_5Descriptive",
        "ValidationEyeMaxAUROCPrimary",
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
    axis.set_title("ResNet-18 baseline loss")
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
        label=f"Eye-level MAX AUROC = {auroc:.3f}",
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


def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    set_random_seeds(int(config["seed"]))

    print("=" * 78)
    print("DAY-3 PER-IMAGE WEAKLY SUPERVISED RESNET-18 BASELINE")
    print("=" * 78)
    print(f"Configuration       : {config_path}")
    print(f"Train manifest      : {TRAIN_INPUT_PATH}")
    print(f"Validation manifest : {VAL_INPUT_PATH}")
    print("Test manifest       : NOT LOADED; TEST SET WILL NOT BE EVALUATED")
    print(
        "Label caveat        : eye-examination labels are inherited by images; "
        "individual image labels may be noisy"
    )
    print("Primary unit        : eye-examination, MAX image probability aggregation\n")

    train, validation, preflight_report = run_preflight(config)
    print("PREFLIGHT: PASS")
    print(
        f"Train images/eyes/subjects           : {len(train):,} / "
        f"{train['EyeExamID'].nunique():,} / {train['ResearchSubjectID'].nunique():,}"
    )
    print(
        f"Validation images/eyes/subjects      : {len(validation):,} / "
        f"{validation['EyeExamID'].nunique():,} / "
        f"{validation['ResearchSubjectID'].nunique():,}"
    )
    print(
        f"Training Normal / Abnormal images    : "
        f"{preflight_report['training_normal_images']:,} / "
        f"{preflight_report['training_abnormal_images']:,}"
    )
    print(
        "Class weights [Normal, Abnormal]     : "
        f"{preflight_report['class_weights_normal_abnormal']}"
    )
    print(f"Configured batch size                : {config['batch_size']} (unchanged)\n")

    if args.preflight_only:
        print("Preflight-only mode complete. No model was loaded or trained.")
        print("TEST SET WAS NOT EVALUATED")
        return

    device = get_device(config)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    model_dir = MODEL_OUTPUT_ROOT / timestamp
    figure_dir = FIGURE_OUTPUT_ROOT / timestamp
    model_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    best_model_path = model_dir / "best_model.pt"
    history_path = model_dir / "training_history.csv"
    config_used_path = model_dir / "config_used.yaml"
    val_image_path = model_dir / "val_image_predictions.csv"
    val_eye_path = model_dir / "val_eye_predictions.csv"
    metrics_path = model_dir / "val_metrics.csv"
    threshold_path = model_dir / "classification_threshold.json"
    metadata_path = model_dir / "run_metadata.json"

    with config_used_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(f"Device                               : {device}")
    if device.type == "cuda":
        print(f"GPU                                  : {torch.cuda.get_device_name(device)}")
    print(f"Model run directory                  : {model_dir}")
    print("Loading pretrained ResNet18_Weights.IMAGENET1K_V1 ...")

    # start from imagenet weights and replace the classifier with two outputs
    model = resnet18(weights=WEIGHTS_ENUM)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_INDEX))
    model = model.to(device)
    # keep the full network trainable instead of freezing the resnet backbone
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise SafetyError("All ResNet-18 parameters must remain trainable.")

    train_loader, validation_loader = make_loaders(
        train, validation, config, device
    )
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
    best_eye_auroc = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    started_at = datetime.now().astimezone()
    wall_start = time.perf_counter()

    # keep the checkpoint with the best validation eye-level auroc
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_result = evaluate(
            model,
            validation_loader,
            validation,
            criterion,
            device,
        )
        current_eye_auroc = validation_result["eye_auroc"]
        improved = current_eye_auroc > best_eye_auroc
        if improved:
            best_eye_auroc = current_eye_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_eye_max_auroc": best_eye_auroc,
                    "class_to_index": CLASS_TO_INDEX,
                    "architecture": "resnet18",
                    "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
                    "config": config,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "Epoch": epoch,
                "TrainingLoss": train_loss,
                "ValidationLoss": validation_result["loss"],
                "ValidationImageAUROCDescriptive": validation_result[
                    "image_auroc"
                ],
                "ValidationImageAccuracyAt0_5Descriptive": validation_result[
                    "image_accuracy_at_0_5"
                ],
                "ValidationEyeMaxAUROCPrimary": current_eye_auroc,
                "CheckpointImproved": improved,
            }
        )
        save_history(history, history_path)
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train loss {train_loss:.5f} | "
            f"val loss {validation_result['loss']:.5f} | "
            f"val image AUROC {validation_result['image_auroc']:.4f} | "
            f"val image accuracy@0.5 {validation_result['image_accuracy_at_0_5']:.4f} | "
            f"val eye MAX AUROC {current_eye_auroc:.4f} | "
            f"{'BEST' if improved else 'no improvement'} | "
            f"{time.perf_counter() - epoch_start:.1f}s"
        )
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            print(
                "Early stopping: validation eye-level MAX AUROC did not improve "
                f"for {config['early_stopping_patience']} consecutive epochs."
            )
            break

    completed_epochs = len(history)
    if best_epoch is None or not best_model_path.is_file():
        raise SafetyError("No best checkpoint was produced.")

    # reload the best checkpoint before producing the final validation outputs
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate(
        model, validation_loader, validation, criterion, device
    )
    if not np.isclose(
        final_validation["eye_auroc"], best_eye_auroc, rtol=0.0, atol=1e-12
    ):
        raise SafetyError(
            "Reloaded best checkpoint eye AUROC does not match recorded best AUROC."
        )

    val_image_predictions = final_validation["image_predictions"]
    val_eye_predictions = final_validation["eye_predictions"]
    eye_targets = labels_to_targets(val_eye_predictions["TrueLabel"])
    eye_probabilities = val_eye_predictions["MaxAbnormalProbability"].to_numpy(
        dtype=np.float64
    )
    # the classification threshold is tuned only on validation eye predictions
    threshold, threshold_details = choose_classification_threshold(
        eye_targets, eye_probabilities
    )
    val_eye_predictions["PredictedLabel"] = [
        INDEX_TO_CLASS[index]
        for index in (eye_probabilities >= threshold).astype(np.int64)
    ]
    val_eye_predictions["ClassificationThreshold"] = threshold
    val_eye_predictions["Correct"] = val_eye_predictions["PredictedLabel"].eq(
        val_eye_predictions["TrueLabel"]
    )

    eye_metrics = compute_metrics(eye_targets, eye_probabilities, threshold)
    image_targets = labels_to_targets(val_image_predictions["TrueEyeLabel"])
    image_probabilities = val_image_predictions["AbnormalProbability"].to_numpy(
        dtype=np.float64
    )
    image_metrics = compute_metrics(image_targets, image_probabilities, 0.5)

    val_image_predictions.to_csv(val_image_path, index=False)
    val_eye_predictions.to_csv(val_eye_path, index=False)
    metrics_rows = [
        {
            "EvaluationUnit": "eye_primary",
            "Aggregation": "MAX",
            "Threshold": threshold,
            "ThresholdSource": "validation balanced accuracy optimum",
            "PrimaryForModelSelection": True,
            **eye_metrics,
        },
        {
            "EvaluationUnit": "image_descriptive_weak_label",
            "Aggregation": "none",
            "Threshold": 0.5,
            "ThresholdSource": "fixed descriptive threshold",
            "PrimaryForModelSelection": False,
            **image_metrics,
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    threshold_payload = {
        "classification_threshold": threshold,
        "selected_on": "validation eye-level MAX probabilities only",
        "objective": "maximize balanced accuracy",
        "prediction_rule": "Abnormal if MaxAbnormalProbability >= threshold; otherwise Normal",
        "tie_breaking_rule": "closest to 0.5, then lower threshold",
        "candidate_rule": "sorted unique finite validation-eye probabilities plus 0.0, 0.5, 1.0",
        "candidate_count": threshold_details["candidate_count"],
        "balanced_accuracy": threshold_details["balanced_accuracy"],
        "tp": threshold_details["tp"],
        "tn": threshold_details["tn"],
        "fp": threshold_details["fp"],
        "fn": threshold_details["fn"],
        "distinct_from_selective_prediction_thresholds": True,
        "test_data_used": False,
    }
    save_json(threshold_path, threshold_payload)

    save_loss_plot(history, figure_dir / "training_loss_curve.png")
    save_eye_roc_plot(
        eye_targets,
        eye_probabilities,
        eye_metrics["AUROC"],
        figure_dir / "validation_eye_roc_curve.png",
    )
    save_confusion_plot(
        eye_metrics,
        figure_dir / "validation_confusion_matrix.png",
    )

    # save enough run information to reproduce the experiment later
    git_commit, git_status = get_git_info()
    completed_at = datetime.now().astimezone()
    metadata = {
        "run_timestamp": timestamp,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_time_seconds": time.perf_counter() - wall_start,
        "random_seed": int(config["seed"]),
        "git_commit_hash": git_commit,
        "git_worktree_status_at_completion": git_status,
        "training_input_csv": str(TRAIN_INPUT_PATH),
        "training_input_csv_sha256": file_sha256(TRAIN_INPUT_PATH),
        "validation_input_csv": str(VAL_INPUT_PATH),
        "validation_input_csv_sha256": file_sha256(VAL_INPUT_PATH),
        "loaded_csv_paths": sorted(map(str, LOADED_CSV_PATHS)),
        "test_manifest_loaded": False,
        "test_set_evaluated": False,
        "test_predictions_created": False,
        "task": "per-image weakly supervised Normal vs Abnormal",
        "label_provenance": (
            "Each image inherits its EyeExamID eye-examination label; individual "
            "training-image labels may therefore contain weak-label noise."
        ),
        "primary_evaluation_unit": "EyeExamID",
        "primary_aggregation": "MAX Abnormal probability across selected images",
        "model_architecture": "resnet18",
        "model_output_classes": ["Normal", "Abnormal"],
        "class_to_index": CLASS_TO_INDEX,
        "pretrained": True,
        "pretrained_weights_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "pretrained_weights_url": WEIGHTS_ENUM.url,
        "fine_tuned_backbone": True,
        "optimizer": "AdamW",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "batch_size_silently_changed": False,
        "configured_epochs": int(config["epochs"]),
        "completed_epochs": completed_epochs,
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "best_epoch": best_epoch,
        "best_validation_eye_max_auroc": best_eye_auroc,
        "selected_validation_classification_threshold": threshold,
        "training_image_class_counts": {
            "Normal": preflight_report["training_normal_images"],
            "Abnormal": preflight_report["training_abnormal_images"],
        },
        "class_weights_normal_abnormal": preflight_report[
            "class_weights_normal_abnormal"
        ],
        "class_weight_formula": "N / (2 * class_count), training image rows only",
        "loss": "class-weighted cross-entropy",
        "checkpoint_selection": (
            "strict improvement in validation eye-level AUROC after MAX aggregation; "
            "earliest epoch retained on equal AUROC"
        ),
        "preprocessing": {
            "load": "Pillow EXIF transpose then RGB conversion",
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
        "validation_eye_metrics": eye_metrics,
        "validation_image_metrics_descriptive": image_metrics,
        "artifacts": {
            "best_model": str(best_model_path),
            "config_used": str(config_used_path),
            "training_history": str(history_path),
            "val_image_predictions": str(val_image_path),
            "val_eye_predictions": str(val_eye_path),
            "val_metrics": str(metrics_path),
            "classification_threshold": str(threshold_path),
            "figure_directory": str(figure_dir),
        },
    }
    save_json(metadata_path, metadata)

    print("\n" + "=" * 78)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 78)
    print(f"Device                               : {device}")
    print(f"Completed epochs                     : {completed_epochs}")
    print(f"Best epoch                           : {best_epoch}")
    print(
        f"Training Normal / Abnormal images    : "
        f"{preflight_report['training_normal_images']} / "
        f"{preflight_report['training_abnormal_images']}"
    )
    print(
        "Class weights [Normal, Abnormal]     : "
        f"{preflight_report['class_weights_normal_abnormal']}"
    )
    print(f"Best validation eye MAX AUROC         : {best_eye_auroc:.6f}")
    print(f"Selected classification threshold    : {threshold:.10f}")
    print("\nValidation eye-level MAX metrics (PRIMARY):")
    for name in [
        "AUROC",
        "BalancedAccuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "Accuracy",
    ]:
        print(f"  {name:<18}: {eye_metrics[name]:.6f}")
    print(
        f"  TP / TN / FP / FN : {eye_metrics['TP']} / {eye_metrics['TN']} / "
        f"{eye_metrics['FP']} / {eye_metrics['FN']}"
    )
    print("\nValidation image-level metrics (DESCRIPTIVE WEAK LABELS, threshold 0.5):")
    for name in [
        "AUROC",
        "BalancedAccuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "Accuracy",
    ]:
        print(f"  {name:<18}: {image_metrics[name]:.6f}")
    print(
        f"  TP / TN / FP / FN : {image_metrics['TP']} / {image_metrics['TN']} / "
        f"{image_metrics['FP']} / {image_metrics['FN']}"
    )
    print(f"\nBest model                           : {best_model_path}")
    print(f"Validation eye predictions           : {val_eye_path}")
    print(f"Run metadata                         : {metadata_path}")
    print("\nTEST SET WAS NOT EVALUATED")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "CUDA out of memory. The configured batch size was not changed. "
            "Edit baseline_resnet18.yaml explicitly and start a new run if needed."
        ) from error
    except SafetyError as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
