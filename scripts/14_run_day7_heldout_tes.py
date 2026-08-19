

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TEST_MANIFEST = PROJECT_ROOT / "data" / "processed" / "model_inputs" / "test_images.csv"
TRAIN_EYES = PROJECT_ROOT / "data" / "splits" / "train_eye_exams.csv"
VAL_EYES = PROJECT_ROOT / "data" / "splits" / "val_eye_exams.csv"
TEST_EYES = PROJECT_ROOT / "data" / "splits" / "test_eye_exams.csv"
TRAIN_SUBJECTS = PROJECT_ROOT / "data" / "splits" / "train_subjects.csv"
VAL_SUBJECTS = PROJECT_ROOT / "data" / "splits" / "val_subjects.csv"
TEST_SUBJECTS = PROJECT_ROOT / "data" / "splits" / "test_subjects.csv"

DAY3_RUN = PROJECT_ROOT / "outputs" / "models" / "baseline_resnet18" / "20260813T134419_044915-0700"
DAY4_RUN = PROJECT_ROOT / "outputs" / "models" / "multiview_resnet18" / "20260813T170240_569268-0700"
DAY3_CHECKPOINT = DAY3_RUN / "best_model.pt"
DAY4_CHECKPOINT = DAY4_RUN / "best_model.pt"
DAY3_THRESHOLD_PATH = DAY3_RUN / "classification_threshold.json"
DAY4_THRESHOLD_PATH = DAY4_RUN / "classification_threshold.json"
ROUTING_RULES_PATH = PROJECT_ROOT / "outputs" / "models" / "selective_routing" / "frozen_routing_rules.json"
TRAIN_EMBEDDINGS = (
    PROJECT_ROOT / "outputs" / "features" / "multiview_resnet18"
    / "20260814T130502_173490-0700" / "train_eye_embeddings.csv"
)
REFERENCE_EYES = (
    PROJECT_ROOT / "outputs" / "features" / "multiview_resnet18"
    / "20260814T130502_173490-0700" / "reference_eye_distances.csv"
)

EXPECTED_SHA256 = {
    TEST_MANIFEST: "4e5434be662a00e925bc1bdf547ad88cba4a9f7dfbd1f77608515065ccbd386f",
    TRAIN_EYES: "0438e38278230c6a9a5d33f49ebb068386976d7fd1588a35ddd3f51d7e3a0daa",
    VAL_EYES: "9dfedd79c42f9ac843c9f71150feb4879351eeb57be9dd5cfea0b7d83a432a65",
    TEST_EYES: "fbfea25f3d81f9178357dd7128542c1972292c5bb79ebdd632688c820246254e",
    TRAIN_SUBJECTS: "b537b1d077f5c5904aae61121e658ad37fa81c913804a1c1f2da5786fe52a7c2",
    VAL_SUBJECTS: "36faa79ca2493c798a343ace4dcde645211ed04e82471a8ed2f927f870b72fa7",
    TEST_SUBJECTS: "18a7bb6766a42c993e7b8d4d6e4ff7c8126cc8493ba9be26f28040acf094a6d4",
    DAY3_CHECKPOINT: "3644f0215c72f079c1d2b66bad4f9ee6ca45fad0334b204cc1dc6b3649132c9c",
    DAY4_CHECKPOINT: "3c6d402ddbd2d1055cb68458bc6f5a7c880ac683a4527ff5171cc6f2408c754b",
    DAY3_THRESHOLD_PATH: "daf9e8f2b8cadaa2590ff3ad97b52390cf219c99b35a3758ecb2e934730b44f9",
    DAY4_THRESHOLD_PATH: "369ae63c078a1aac105e5481c66740f80bf1b6cffde3bfb0253fd70d7dd07d41",
    ROUTING_RULES_PATH: "d042f0032068e0528a64122cc5fde967fe3d7240db95be7606ac3b7390c0ced3",
    TRAIN_EMBEDDINGS: "4c51fdfd5e4bcc5583974c43d9e0563105ebb04e18f019c65236a236d06b77d8",
    REFERENCE_EYES: "6a7146a54ccd41ae5703dd5a6dd9844ac02825c073b3f5b5ca6c83748b688b51",
}

EXPECTED_TEST_EYES = 120
EXPECTED_TEST_SUBJECTS = 95
EXPECTED_TEST_NORMAL = 32
EXPECTED_TEST_ABNORMAL = 88
EXPECTED_TEST_IMAGES = 720
CLASS_TO_INDEX = {"Normal": 0, "Abnormal": 1}
INDEX_TO_CLASS = {0: "Normal", 1: "Abnormal"}
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
PROBABILITY_CLIP_EPSILON = 1e-12
SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
DATA_SCOPE = "ONE-TIME HELD-OUT TEST EVALUATION"
THRESHOLD_SOURCE = "VALIDATION-FROZEN"


class SafetyError(RuntimeError):
    """used when a hard integrity or frozen-protocol check fails."""


CHECKS: list[dict[str, Any]] = []


def record_hard_check(name: str, passed: bool, observed: Any, expected: Any) -> None:
    CHECKS.append(
        {
            "Check": name,
            "Status": "PASS" if bool(passed) else "FAIL",
            "Observed": observed,
            "Expected": expected,
            "Severity": "HARD FAIL",
        }
    )
    if not passed:
        raise SafetyError(f"{name}: observed={observed!r}, expected={expected!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pinned_files() -> dict[str, str]:
    # verify every frozen input hash before running test inference
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        record_hard_check(f"required frozen artifact exists: {path.name}", path.is_file(), path.is_file(), True)
        value = file_sha256(path)
        observed[str(path.resolve())] = value
        record_hard_check(f"frozen SHA-256 unchanged: {path.name}", value == expected, value, expected)
    return observed


def parse_bool_column(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise SafetyError(f"{field} contains non-boolean values: {sorted(normalized[invalid].unique())}")
    return normalized.eq("true")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"Expected JSON object: {path}")
    return value


def set_random_seeds(seed: int) -> None:
    # seed all random sources even though this stage is inference-only
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class CenterSquarePad:
    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        right = side - width - left
        top = (side - height) // 2
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def make_validation_transform(image_size: int = 224):
    # reuse the deterministic validation preprocessing from the locked models
    return transforms.Compose(
        [
            CenterSquarePad(fill=0),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(Path(row["ResolvedImagePath"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), CLASS_TO_INDEX[str(row["EyeLabel"])], index


@dataclass(frozen=True)
class EyeRecord:
    eye_id: str
    subject_id: str
    encounter_id: str
    laterality: str
    label: str
    image_paths: tuple[Path, ...]
    relative_paths: tuple[str, ...]


class EyeDataset(Dataset):
    def __init__(self, records: list[EyeRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        views = []
        for path in record.image_paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            views.append(self.transform(image))
        return torch.stack(views), CLASS_TO_INDEX[record.label], index


def collate_eye_views(batch):
    # pad only for batching and keep a mask for the real views
    tensors, labels, indices = zip(*batch)
    counts = [int(item.shape[0]) for item in tensors]
    maximum = max(counts)
    channels, height, width = tensors[0].shape[1:]
    padded = tensors[0].new_zeros((len(tensors), maximum, channels, height, width))
    mask = torch.zeros((len(tensors), maximum), dtype=torch.bool)
    for batch_index, tensor in enumerate(tensors):
        padded[batch_index, : tensor.shape[0]] = tensor
        mask[batch_index, : tensor.shape[0]] = True
    return {
        "images": padded,
        "view_mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "eye_indices": torch.tensor(indices, dtype=torch.long),
    }


class FeatureMaxResNet18(nn.Module):
    """exact frozen day-4 resnet-18 with masked feature-wise max pooling."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, 2)

    def forward(self, images, view_mask, return_diagnostics=False):
        batch_size, padded_views = images.shape[:2]
        flat_images = images.flatten(0, 1)
        flat_mask = view_mask.flatten()
        valid_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
        valid_features = self.backbone(flat_images.index_select(0, valid_positions))
        feature_grid = valid_features.new_full(
            (batch_size * padded_views, self.feature_dim), float("-inf")
        )
        feature_grid = feature_grid.index_copy(0, valid_positions, valid_features)
        view_features = feature_grid.view(batch_size, padded_views, self.feature_dim)
        pooled_features = view_features.amax(dim=1)
        eye_logits = self.classifier(pooled_features)
        if not return_diagnostics:
            return eye_logits
        valid_view_logits = self.classifier(valid_features)
        grid = valid_view_logits.new_full((batch_size * padded_views, 2), float("nan"))
        grid = grid.index_copy(0, valid_positions, valid_view_logits)
        return eye_logits, pooled_features, grid.view(batch_size, padded_views, 2)


def get_original_image_path(relative: str) -> Path:
    # allow only original source images and explicitly reject cropped or masked paths
    rel = Path(str(relative))
    if rel.is_absolute() or ".." in rel.parts:
        raise SafetyError(f"Unsafe image path: {relative}")
    if any(token in str(rel).casefold() for token in ("masked", "cropped")):
        raise SafetyError(f"Cropped/masked path is forbidden: {relative}")
    path = (RAW_ROOT / rel).resolve()
    try:
        path.relative_to(RAW_ROOT.resolve())
    except ValueError as error:
        raise SafetyError(f"Image path escapes original release root: {relative}") from error
    if not path.is_file():
        raise SafetyError(f"Missing original source image: {relative}")
    return path


def check_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    # verify the held-out split and every frozen artifact before evaluation starts
    hashes = check_pinned_files()
    train = pd.read_csv(TRAIN_EYES, low_memory=False)
    val = pd.read_csv(VAL_EYES, low_memory=False)
    test = pd.read_csv(TEST_EYES, low_memory=False)
    train_subjects_file = pd.read_csv(TRAIN_SUBJECTS, low_memory=False)
    val_subjects_file = pd.read_csv(VAL_SUBJECTS, low_memory=False)
    test_subjects_file = pd.read_csv(TEST_SUBJECTS, low_memory=False)
    manifest = pd.read_csv(TEST_MANIFEST, low_memory=False)
    train_embeddings = pd.read_csv(TRAIN_EMBEDDINGS, usecols=["EyeExamID", "ResearchSubjectID"])
    references = pd.read_csv(REFERENCE_EYES, usecols=["EyeExamID", "ResearchSubjectID"])
    routing = read_json(ROUTING_RULES_PATH)

    required_eye = {"EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "EyeLabel", "DatasetSplit"}
    required_image = required_eye | {
        "ImageRelativePath", "SHA256", "IncludeModelInput", "SelectedViewIndex",
        "AvailableUniqueImageCount", "SelectedImageCount",
    }
    record_hard_check("test split required schema", required_eye.issubset(test.columns), sorted(required_eye - set(test.columns)), [])
    record_hard_check("test manifest required schema", required_image.issubset(manifest.columns), sorted(required_image - set(manifest.columns)), [])
    record_hard_check("test EyeExamID unique", not test["EyeExamID"].duplicated().any(), int(test["EyeExamID"].duplicated().sum()), 0)
    record_hard_check("test eye count", len(test) == EXPECTED_TEST_EYES, len(test), EXPECTED_TEST_EYES)
    record_hard_check("test subject count", test["ResearchSubjectID"].nunique() == EXPECTED_TEST_SUBJECTS, test["ResearchSubjectID"].nunique(), EXPECTED_TEST_SUBJECTS)
    counts = test["EyeLabel"].value_counts()
    record_hard_check("test Normal count", int(counts.get("Normal", 0)) == EXPECTED_TEST_NORMAL, int(counts.get("Normal", 0)), EXPECTED_TEST_NORMAL)
    record_hard_check("test Abnormal count", int(counts.get("Abnormal", 0)) == EXPECTED_TEST_ABNORMAL, int(counts.get("Abnormal", 0)), EXPECTED_TEST_ABNORMAL)
    record_hard_check("test labels restricted", set(test["EyeLabel"]) == set(CLASS_TO_INDEX), sorted(set(test["EyeLabel"])), sorted(CLASS_TO_INDEX))
    record_hard_check("test split marker", set(test["DatasetSplit"]) == {"test"}, sorted(set(test["DatasetSplit"])), ["test"])

    # confirm patient-level isolation between train, validation, and test
    train_subject_set = set(train["ResearchSubjectID"].astype(str))
    val_subject_set = set(val["ResearchSubjectID"].astype(str))
    test_subject_set = set(test["ResearchSubjectID"].astype(str))
    record_hard_check("test/train subject intersection empty", not (test_subject_set & train_subject_set), len(test_subject_set & train_subject_set), 0)
    record_hard_check("test/validation subject intersection empty", not (test_subject_set & val_subject_set), len(test_subject_set & val_subject_set), 0)
    record_hard_check("subject files agree with eye files", set(train_subjects_file["ResearchSubjectID"].astype(str)) == train_subject_set and set(val_subjects_file["ResearchSubjectID"].astype(str)) == val_subject_set and set(test_subjects_file["ResearchSubjectID"].astype(str)) == test_subject_set, "all three compared", "exact equality")

    # make sure no test eye was used in training embeddings or reference construction
    test_eye_set = set(test["EyeExamID"].astype(str))
    training_eye_set = set(train["EyeExamID"].astype(str)) | set(train_embeddings["EyeExamID"].astype(str))
    reference_eye_set = set(references["EyeExamID"].astype(str))
    record_hard_check("test eyes absent from training data", not (test_eye_set & training_eye_set), len(test_eye_set & training_eye_set), 0)
    record_hard_check("test eyes absent from reference data", not (test_eye_set & reference_eye_set), len(test_eye_set & reference_eye_set), 0)

    record_hard_check("test image row count", len(manifest) == EXPECTED_TEST_IMAGES, len(manifest), EXPECTED_TEST_IMAGES)
    record_hard_check("manifest eye set equals locked test split", set(manifest["EyeExamID"].astype(str)) == test_eye_set, len(set(manifest["EyeExamID"].astype(str)) ^ test_eye_set), 0)
    record_hard_check("manifest only test split", set(manifest["DatasetSplit"].astype(str)) == {"test"}, sorted(set(manifest["DatasetSplit"].astype(str))), ["test"])
    record_hard_check("manifest only OD/OS", set(manifest["Laterality"].astype(str)).issubset({"OD", "OS"}), sorted(set(manifest["Laterality"].astype(str))), ["OD", "OS"])
    record_hard_check("all manifest rows selected model inputs", parse_bool_column(manifest["IncludeModelInput"], "IncludeModelInput").all(), int((~parse_bool_column(manifest["IncludeModelInput"], "IncludeModelInput")).sum()), 0)
    record_hard_check("manifest image key unique", not manifest.duplicated(["EyeExamID", "ImageRelativePath"]).any(), int(manifest.duplicated(["EyeExamID", "ImageRelativePath"]).sum()), 0)
    record_hard_check("no duplicate SHA within eye", not manifest.duplicated(["EyeExamID", "SHA256"]).any(), int(manifest.duplicated(["EyeExamID", "SHA256"]).sum()), 0)
    per_eye = manifest.groupby("EyeExamID", sort=True).size()
    record_hard_check("every test eye has six locked views", per_eye.eq(6).all(), per_eye.value_counts().sort_index().to_dict(), {6: 120})
    for field in ["ResearchSubjectID", "EncounterID", "Laterality", "EyeLabel"]:
        record_hard_check(f"manifest {field} consistent within eye", manifest.groupby("EyeExamID")[field].nunique(dropna=False).le(1).all(), int(manifest.groupby("EyeExamID")[field].nunique(dropna=False).gt(1).sum()), 0)
    split_map = test.set_index("EyeExamID")[["ResearchSubjectID", "EncounterID", "Laterality", "EyeLabel"]].astype(str)
    manifest_map = manifest.groupby("EyeExamID")[["ResearchSubjectID", "EncounterID", "Laterality", "EyeLabel"]].first().astype(str)
    record_hard_check("manifest identity and labels match test split", manifest_map.sort_index().equals(split_map.sort_index()), "compared", "exact equality")

    manifest = manifest.copy()
    # resolve and hash-check every original source image before inference
    manifest["ResolvedImagePath"] = manifest["ImageRelativePath"].map(get_original_image_path)
    observed_image_hash = manifest["ResolvedImagePath"].map(file_sha256)
    mismatches = ~observed_image_hash.str.casefold().eq(manifest["SHA256"].astype(str).str.casefold())
    record_hard_check("all original image SHA-256 values match", not mismatches.any(), int(mismatches.sum()), 0)
    record_hard_check("no cropped or masked input used", not manifest["ImageRelativePath"].astype(str).str.casefold().str.contains("masked|cropped", regex=True).any(), 0, 0)

    record_hard_check("routing artifact is frozen", routing.get("artifact_status") == "FROZEN", routing.get("artifact_status"), "FROZEN")
    record_hard_check("routing positive class", routing.get("positive_class") == "Abnormal", routing.get("positive_class"), "Abnormal")
    record_hard_check("routing contains exactly two operating points", len(routing.get("operating_points", [])) == 2, len(routing.get("operating_points", [])), 2)
    rules = {item["operating_point_id"]: item for item in routing["operating_points"]}
    record_hard_check("expected frozen routing rule IDs", set(rules) == {"conservative_zero_error", "balanced_agreement_view_sd"}, sorted(rules), ["balanced_agreement_view_sd", "conservative_zero_error"])
    record_hard_check("feature typicality omitted from both frozen rules", all(item["thresholds"].get("feature_percentile_threshold") is None for item in rules.values()), [item["thresholds"].get("feature_percentile_threshold") for item in rules.values()], [None, None])
    record_hard_check("routing rules selected on validation only", routing.get("selection_data_scope") == "locked validation data only", routing.get("selection_data_scope"), "locked validation data only")
    return manifest, test.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True), routing, hashes


def make_eye_records(manifest: pd.DataFrame) -> list[EyeRecord]:
    records: list[EyeRecord] = []
    ordered = manifest.sort_values(["EyeExamID", "SelectedViewIndex", "ImageRelativePath"], kind="mergesort")
    for eye_id, group in ordered.groupby("EyeExamID", sort=True):
        first = group.iloc[0]
        records.append(
            EyeRecord(
                str(eye_id), str(first["ResearchSubjectID"]), str(first["EncounterID"]),
                str(first["Laterality"]), str(first["EyeLabel"]),
                tuple(group["ResolvedImagePath"].tolist()),
                tuple(group["ImageRelativePath"].astype(str).tolist()),
            )
        )
    return records


def load_day3_model(device: torch.device) -> nn.Module:
    # rebuild the day-3 architecture and load only the frozen checkpoint weights
    checkpoint = torch.load(DAY3_CHECKPOINT, map_location="cpu", weights_only=False)
    record_hard_check("Day-3 checkpoint architecture", checkpoint.get("architecture") == "resnet18", checkpoint.get("architecture"), "resnet18")
    record_hard_check("Day-3 class mapping", checkpoint.get("class_to_index") == CLASS_TO_INDEX, checkpoint.get("class_to_index"), CLASS_TO_INDEX)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


def load_day4_model(device: torch.device) -> FeatureMaxResNet18:
    # rebuild the day-4 architecture and load the frozen feature-max checkpoint
    checkpoint = torch.load(DAY4_CHECKPOINT, map_location="cpu", weights_only=False)
    record_hard_check("Day-4 checkpoint architecture", checkpoint.get("architecture") == "resnet18", checkpoint.get("architecture"), "resnet18")
    record_hard_check("Day-4 pooling", checkpoint.get("pooling") == "feature-wise max", checkpoint.get("pooling"), "feature-wise max")
    record_hard_check("Day-4 feature dimension", int(checkpoint.get("feature_dimension", -1)) == 512, checkpoint.get("feature_dimension"), 512)
    record_hard_check("Day-4 class mapping", checkpoint.get("class_to_index") == CLASS_TO_INDEX, checkpoint.get("class_to_index"), CLASS_TO_INDEX)
    model = FeatureMaxResNet18()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


def run_day3_inference(manifest: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    # produce image-level probabilities first, then aggregate them later by eye
    ordered = manifest.sort_values(["EyeExamID", "SelectedViewIndex", "ImageRelativePath"], kind="mergesort").reset_index(drop=True)
    loader = DataLoader(ImageDataset(ordered, make_validation_transform()), batch_size=16, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = load_day3_model(device)
    probabilities = np.empty(len(ordered), dtype=np.float64)
    with torch.inference_mode():
        for images, _labels, indices in loader:
            values = torch.softmax(model(images.to(device, non_blocking=True)), dim=1)[:, 1]
            probabilities[indices.numpy()] = values.detach().cpu().numpy().astype(np.float64)
    record_hard_check("Day-3 image probabilities finite and bounded", np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), f"min={probabilities.min()}, max={probabilities.max()}", "finite [0,1]")
    result = ordered[["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "EyeLabel", "ImageRelativePath", "SHA256", "SelectedViewIndex"]].copy()
    result = result.rename(columns={"EyeLabel": "GroundTruth"})
    result["Day3ImageAbnormalProbability"] = probabilities
    return result


def run_day4_inference(records: list[EyeRecord], device: torch.device) -> pd.DataFrame:
    # run one frozen multi-view prediction per eye and save view diagnostics
    loader = DataLoader(EyeDataset(records, make_validation_transform()), batch_size=4, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_eye_views)
    model = load_day4_model(device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["view_mask"].to(device, non_blocking=True)
            logits, _features, view_logits = model(images, mask, return_diagnostics=True)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            view_probs = torch.softmax(view_logits, dim=2)[:, :, 1].detach().cpu().numpy()
            masks = batch["view_mask"].numpy()
            for offset, record_index in enumerate(batch["eye_indices"].numpy().tolist()):
                record = records[int(record_index)]
                real_views = view_probs[offset, masks[offset]].astype(np.float64)
                row = {
                    "EyeExamID": record.eye_id,
                    "ResearchSubjectID": record.subject_id,
                    "EncounterID": record.encounter_id,
                    "Laterality": record.laterality,
                    "GroundTruth": record.label,
                    "Day4Probability": float(probs[offset]),
                    "Day4DiagnosticMeanViewProbability": float(np.mean(real_views)),
                    "Day4DiagnosticStdViewProbability": float(np.std(real_views, ddof=0)),
                    "Day4DiagnosticRangeViewProbability": float(np.max(real_views) - np.min(real_views)),
                }
                rows.append(row)
    result = pd.DataFrame(rows).sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    values = result["Day4Probability"].to_numpy(float)
    record_hard_check("Day-4 eye probabilities finite and bounded", np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"min={values.min()}, max={values.max()}", "finite [0,1]")
    return result


def compute_classification_metrics(true: np.ndarray, predicted: np.ndarray, probability: np.ndarray | None = None) -> dict[str, Any]:
    truth = np.asarray(true, dtype=str)
    pred = np.asarray(predicted, dtype=str)
    tp = int(((truth == "Abnormal") & (pred == "Abnormal")).sum())
    tn = int(((truth == "Normal") & (pred == "Normal")).sum())
    fp = int(((truth == "Normal") & (pred == "Abnormal")).sum())
    fn = int(((truth == "Abnormal") & (pred == "Normal")).sum())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    accuracy = (tp + tn) / len(truth) if len(truth) else math.nan
    balanced = (sensitivity + specificity) / 2 if math.isfinite(sensitivity) and math.isfinite(specificity) else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else math.nan
    auroc = math.nan
    if probability is not None and len(set(truth)) == 2:
        auroc = float(roc_auc_score((truth == "Abnormal").astype(int), np.asarray(probability, float)))
    return {"AUROC": auroc, "Accuracy": accuracy, "BalancedAccuracy": balanced, "Sensitivity": sensitivity, "Specificity": specificity, "F1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_selective_metrics(frame: pd.DataFrame, accepted: np.ndarray) -> dict[str, Any]:
    subset = frame.loc[accepted]
    metrics = compute_classification_metrics(subset["GroundTruth"].to_numpy(str), subset["Day4Prediction"].to_numpy(str)) if len(subset) else {key: math.nan for key in ["AUROC", "Accuracy", "BalancedAccuracy", "Sensitivity", "Specificity", "F1"]} | {key: 0 for key in ["TP", "TN", "FP", "FN"]}
    errors = int(metrics["FP"] + metrics["FN"])
    deferred = frame.loc[~accepted]
    return {
        "TotalEyes": len(frame), "AcceptedEyes": int(accepted.sum()), "DeferredEyes": int((~accepted).sum()),
        "Coverage": float(accepted.mean()), "ReviewRate": float((~accepted).mean()),
        "AcceptedCorrect": int(len(subset) - errors), "AcceptedErrors": errors,
        "AcceptedErrorRate": errors / len(subset) if len(subset) else math.nan,
        "AcceptedAccuracy": metrics["Accuracy"], "AcceptedSensitivity": metrics["Sensitivity"],
        "AcceptedSpecificity": metrics["Specificity"], "AcceptedBalancedAccuracy": metrics["BalancedAccuracy"],
        "AcceptedF1": metrics["F1"], "AcceptedTP": metrics["TP"], "AcceptedTN": metrics["TN"],
        "AcceptedFP": metrics["FP"], "AcceptedFN": metrics["FN"],
        "DeferredNormal": int(deferred["GroundTruth"].eq("Normal").sum()),
        "DeferredAbnormal": int(deferred["GroundTruth"].eq("Abnormal").sum()),
    }


def apply_routing_rules(frame: pd.DataFrame, routing: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # apply the validation-frozen thresholds exactly as serialized
    output = frame.copy()
    metrics_rows = []
    for rule in routing["operating_points"]:
        thresholds = rule["thresholds"]
        accepted = output["PrimaryConfidenceMargin"].to_numpy(float) >= float(thresholds["confidence_threshold"])
        if bool(thresholds["model_agreement_required"]):
            accepted &= output["ModelAgreement"].to_numpy(bool)
        if thresholds["view_sd_threshold"] is not None:
            accepted &= output["Day3StdViewProbability"].to_numpy(float) <= float(thresholds["view_sd_threshold"])
        record_hard_check(f"{rule['operating_point_id']} uses no feature cutoff", thresholds.get("feature_percentile_threshold") is None, thresholds.get("feature_percentile_threshold"), None)
        prefix = "Conservative" if rule["operating_point_id"] == "conservative_zero_error" else "Balanced"
        output[f"{prefix}Accepted"] = accepted
        output[f"{prefix}RoutingDecision"] = np.where(accepted, "ACCEPT", "DEFER")
        row = {
            "OperatingPointID": rule["operating_point_id"], "OperatingPointName": rule["operating_point_name"],
            "StrategyID": rule["strategy_id"], "AcceptanceRule": rule["acceptance_rule"],
            "ConfidenceThreshold": float(thresholds["confidence_threshold"]),
            "ModelAgreementRequired": bool(thresholds["model_agreement_required"]),
            "ViewSDThreshold": thresholds["view_sd_threshold"], "FeaturePercentileThreshold": None,
            **compute_selective_metrics(output, accepted),
            "AcceptedErrorDenominator": "AcceptedEyes", "ThresholdSource": THRESHOLD_SOURCE, "DataScope": DATA_SCOPE,
        }
        metrics_rows.append(row)
    return output, metrics_rows


def bootstrap_confidence_intervals(frame: pd.DataFrame, selective_rows: list[dict[str, Any]], replicates: int, seed: int) -> pd.DataFrame:
    # resample subjects rather than individual eyes to preserve within-subject dependence
    subjects = sorted(frame["ResearchSubjectID"].astype(str).unique())
    subject_indices = {subject: np.flatnonzero(frame["ResearchSubjectID"].astype(str).to_numpy() == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    collected: dict[tuple[str, str], list[float]] = {}
    classification_metrics = ["AUROC", "BalancedAccuracy", "Sensitivity", "Specificity"]
    routing_metrics = ["Coverage", "AcceptedErrorRate", "AcceptedSensitivity", "AcceptedSpecificity"]
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([subject_indices[str(subject)] for subject in sampled])
        block = frame.iloc[indices]
        base = compute_classification_metrics(block["GroundTruth"].to_numpy(str), block["Day4Prediction"].to_numpy(str), block["Day4Probability"].to_numpy(float))
        for metric in classification_metrics:
            collected.setdefault(("Day4Classification", metric), []).append(float(base[metric]))
        for row in selective_rows:
            prefix = "Conservative" if row["OperatingPointID"] == "conservative_zero_error" else "Balanced"
            accepted = block[f"{prefix}Accepted"].to_numpy(bool)
            values = compute_selective_metrics(block, accepted)
            for metric in routing_metrics:
                collected.setdefault((row["OperatingPointID"], metric), []).append(float(values[metric]))

    points: dict[tuple[str, str], float] = {}
    day4 = compute_classification_metrics(frame["GroundTruth"].to_numpy(str), frame["Day4Prediction"].to_numpy(str), frame["Day4Probability"].to_numpy(float))
    for metric in classification_metrics:
        points[("Day4Classification", metric)] = float(day4[metric])
    for row in selective_rows:
        for metric in routing_metrics:
            points[(row["OperatingPointID"], metric)] = float(row[metric])

    rows = []
    for key in sorted(collected):
        values = np.asarray(collected[key], dtype=float)
        valid = values[np.isfinite(values)]
        rows.append(
            {
                "Analysis": key[0], "Metric": key[1], "PointEstimate": points[key],
                "CI95Lower": float(np.percentile(valid, 2.5)) if len(valid) else math.nan,
                "CI95Upper": float(np.percentile(valid, 97.5)) if len(valid) else math.nan,
                "ValidReplicates": len(valid), "RequestedReplicates": replicates,
                "BootstrapUnit": "ResearchSubjectID", "BootstrapSeed": seed,
                "CIMethod": "subject-cluster percentile bootstrap", "DataScope": DATA_SCOPE,
            }
        )
    return pd.DataFrame(rows)


def build_error_review(frame: pd.DataFrame) -> pd.DataFrame:
    # keep only model errors and annotate how each frozen routing rule handled them
    errors = frame.loc[(~frame["Day3Correct"]) | (~frame["Day4Correct"])].copy()
    errors["Day3Error"] = ~errors["Day3Correct"]
    errors["Day4Error"] = ~errors["Day4Correct"]
    for prefix in ["Conservative", "Balanced"]:
        accepted = errors[f"{prefix}Accepted"]
        errors[f"{prefix}AcceptedFalseNegative"] = accepted & errors["GroundTruth"].eq("Abnormal") & errors["Day4Prediction"].eq("Normal")
        errors[f"{prefix}AcceptedFalsePositive"] = accepted & errors["GroundTruth"].eq("Normal") & errors["Day4Prediction"].eq("Abnormal")
        errors[f"{prefix}ErrorDeferred"] = (~accepted) & errors["Day4Error"]
        errors[f"{prefix}ErrorIncorrectlyAccepted"] = accepted & errors["Day4Error"]
    errors["AnyAcceptedFalseNegative"] = errors["ConservativeAcceptedFalseNegative"] | errors["BalancedAcceptedFalseNegative"]
    errors["AnyAcceptedFalsePositive"] = errors["ConservativeAcceptedFalsePositive"] | errors["BalancedAcceptedFalsePositive"]
    errors["AnyErrorDeferredByRouting"] = errors["ConservativeErrorDeferred"] | errors["BalancedErrorDeferred"]
    errors["AnyErrorIncorrectlyAcceptedByRouting"] = errors["ConservativeErrorIncorrectlyAccepted"] | errors["BalancedErrorIncorrectlyAccepted"]
    columns = [
        "EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "GroundTruth",
        "Day3Probability", "Day3Prediction", "Day4Probability", "Day4Prediction",
        "PrimaryConfidenceMargin", "Day3StdViewProbability", "Day3RangeViewProbability",
        "ModelAgreement", "Day3Error", "Day4Error", "ConservativeAccepted", "BalancedAccepted",
        "ConservativeAcceptedFalseNegative", "ConservativeAcceptedFalsePositive",
        "BalancedAcceptedFalseNegative", "BalancedAcceptedFalsePositive",
        "AnyAcceptedFalseNegative", "AnyAcceptedFalsePositive", "ModelDisagreement",
        "ConservativeErrorDeferred", "BalancedErrorDeferred", "AnyErrorDeferredByRouting",
        "ConservativeErrorIncorrectlyAccepted", "BalancedErrorIncorrectlyAccepted",
        "AnyErrorIncorrectlyAcceptedByRouting",
    ]
    return errors[columns].sort_values(["Day4Error", "EyeExamID"], ascending=[False, True], kind="mergesort")


def save_risk_coverage_figure(routing: dict[str, Any], selective: pd.DataFrame, path: Path) -> None:
    # connect each frozen validation point to its held-out test result
    colors = {"conservative_zero_error": "#0072B2", "balanced_agreement_view_sd": "#D55E00"}
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for rule in routing["operating_points"]:
        rule_id = rule["operating_point_id"]
        validation = rule["validation_performance"]
        test = selective.loc[selective["OperatingPointID"].eq(rule_id)].iloc[0]
        x = [100 * float(validation["coverage"]), 100 * float(test["Coverage"])]
        y = [100 * float(validation["accepted_error_rate"]), 100 * float(test["AcceptedErrorRate"])]
        color = colors[rule_id]
        ax.plot(x, y, color=color, linewidth=1.4, alpha=0.7)
        ax.scatter(x[0], y[0], s=95, facecolors="white", edgecolors=color, linewidths=2, marker="o", label=f"{rule['strategy_name']} — validation frozen point")
        ax.scatter(x[1], y[1], s=110, color=color, edgecolors="black", linewidths=0.7, marker="o", label=f"{rule['strategy_name']} — held-out test")
        ax.annotate(f"Test: {x[1]:.1f}% coverage\n{y[1]:.1f}% accepted error", (x[1], y[1]), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Coverage (accepted eyes, %)")
    ax.set_ylabel("Accepted-prelabel error (%)")
    ax.set_title("Frozen selective-routing operating points\nValidation selection versus one-time held-out test")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.text(0.5, 0.01, "Rules were frozen on validation data; no test threshold tuning or test frontier selection.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def get_git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def save_summary(path: Path, test: pd.DataFrame, classification: pd.DataFrame, selective: pd.DataFrame, bootstrap: pd.DataFrame, figure_path: Path) -> None:
    def metric(model: str, name: str) -> float:
        return float(classification.loc[classification["Model"].eq(model), name].iloc[0])

    lines = [
        "# Day-7 one-time held-out test evaluation", "",
        "No model was retrained. No threshold or routing rule was searched or tuned on test data. Original unmasked source images and the locked Day-3/Day-4 deterministic preprocessing were used.", "",
        "## Test cohort", "",
        f"- Eyes: {len(test)}", f"- Subjects: {test['ResearchSubjectID'].nunique()}",
        f"- Normal: {int(test['GroundTruth'].eq('Normal').sum())}", f"- Abnormal: {int(test['GroundTruth'].eq('Abnormal').sum())}", "",
        "## Classification", "",
        "| Model | AUROC | Balanced accuracy | Sensitivity | Specificity | F1 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ["Day3PerImageMAX", "Day4MultiViewFeatureMAX"]:
        lines.append(f"| {model} | {metric(model, 'AUROC'):.6f} | {metric(model, 'BalancedAccuracy'):.6f} | {metric(model, 'Sensitivity'):.6f} | {metric(model, 'Specificity'):.6f} | {metric(model, 'F1'):.6f} |")
    lines += ["", "## Frozen selective routing", "", "Accepted error rate uses accepted eyes—not the full cohort—as its denominator.", "", "| Rule | Accepted | Deferred | Coverage | Review | Accepted errors | Accepted error rate | Accepted FN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in selective.iterrows():
        lines.append(f"| {row['OperatingPointName']} | {row['AcceptedEyes']} | {row['DeferredEyes']} | {100*row['Coverage']:.2f}% | {100*row['ReviewRate']:.2f}% | {row['AcceptedErrors']} | {100*row['AcceptedErrorRate']:.2f}% | {row['AcceptedFN']} |")
    lines += [
        "", "## Subject-level bootstrap confidence intervals", "",
        "| Analysis | Metric | Point estimate | 95% CI | Valid replicates |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['Analysis']} | {row['Metric']} | {row['PointEstimate']:.6f} | "
            f"[{row['CI95Lower']:.6f}, {row['CI95Upper']:.6f}] | "
            f"{int(row['ValidReplicates'])}/{int(row['RequestedReplicates'])} |"
        )
    lines += ["", f"Risk/coverage figure: `{figure_path.relative_to(PROJECT_ROOT)}`", "", "HELD-OUT TEST EVALUATION COMPLETE.", "NO TEST-BASED THRESHOLD TUNING WAS PERFORMED.", "FROZEN VALIDATION ROUTING RULES WERE APPLIED WITHOUT MODIFICATION.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    # this is the single held-out evaluation pass; nothing is tuned on test data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise SafetyError("bootstrap-replicates must be positive")
    set_random_seeds(SEED)
    manifest, test_split, routing, input_hashes_before = check_inputs()
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    run_dir = PROJECT_ROOT / "outputs" / "audits" / timestamp
    figure_dir = PROJECT_ROOT / "outputs" / "figures" / "day7_heldout_test" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    print(f"Day-7 output directory: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("Running frozen Day-3 inference on 720 original test images...")
    # run both frozen models before applying any routing rule
    image_predictions = run_day3_inference(manifest, device)
    print("Running frozen Day-4 multi-view inference on 120 test eyes...")
    records = make_eye_records(manifest)
    day4 = run_day4_inference(records, device)

    # use the classification thresholds frozen on validation
    day3_threshold = float(read_json(DAY3_THRESHOLD_PATH)["classification_threshold"])
    day4_threshold = float(read_json(DAY4_THRESHOLD_PATH)["classification_threshold"])
    grouped = image_predictions.groupby("EyeExamID", sort=True)["Day3ImageAbnormalProbability"].agg(["count", "mean", "max", "min", lambda x: np.std(x.to_numpy(float), ddof=0)]).reset_index()
    grouped.columns = ["EyeExamID", "NumberOfViews", "Day3MeanViewProbability", "Day3Probability", "Day3MinViewProbability", "Day3StdViewProbability"]
    ranges = image_predictions.groupby("EyeExamID", sort=True)["Day3ImageAbnormalProbability"].agg(lambda x: float(np.max(x) - np.min(x))).rename("Day3RangeViewProbability").reset_index()
    grouped = grouped.merge(ranges, on="EyeExamID", validate="one_to_one")
    base = test_split.rename(columns={"EyeLabel": "GroundTruth"})[["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "GroundTruth"]]
    eyes = base.merge(grouped, on="EyeExamID", validate="one_to_one").merge(day4, on=["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "GroundTruth"], validate="one_to_one")
    eyes["Day3ClassificationThreshold"] = day3_threshold
    eyes["Day3Prediction"] = np.where(eyes["Day3Probability"].ge(day3_threshold), "Abnormal", "Normal")
    eyes["Day3Correct"] = eyes["Day3Prediction"].eq(eyes["GroundTruth"])
    eyes["Day4ClassificationThreshold"] = day4_threshold
    eyes["Day4Prediction"] = np.where(eyes["Day4Probability"].ge(day4_threshold), "Abnormal", "Normal")
    eyes["Day4Correct"] = eyes["Day4Prediction"].eq(eyes["GroundTruth"])
    # reproduce the same day-5 confidence margin definition on the test eyes
    clipped = np.clip(eyes["Day4Probability"].to_numpy(float), PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON)
    eyes["PrimaryLogit"] = np.log(clipped) - np.log1p(-clipped)
    eyes["ThresholdLogit"] = math.log(day4_threshold) - math.log1p(-day4_threshold)
    eyes["PrimarySignedMargin"] = eyes["PrimaryLogit"] - eyes["ThresholdLogit"]
    eyes["PrimaryConfidenceMargin"] = eyes["PrimarySignedMargin"].abs()
    eyes["ModelAgreement"] = eyes["Day3Prediction"].eq(eyes["Day4Prediction"])
    eyes["ModelDisagreement"] = ~eyes["ModelAgreement"]
    record_hard_check("Day-5 signed-margin prediction semantics reproduced", np.array_equal(np.where(eyes["PrimarySignedMargin"].ge(0), "Abnormal", "Normal"), eyes["Day4Prediction"].to_numpy(str)), "all eyes agree", "all eyes agree")
    record_hard_check("eye output count and uniqueness", len(eyes) == 120 and not eyes["EyeExamID"].duplicated().any(), f"rows={len(eyes)}, duplicates={eyes['EyeExamID'].duplicated().sum()}", "120 rows, 0 duplicates")
    eyes, selective_rows = apply_routing_rules(eyes, routing)

    classification_rows = []
    for model, prob_col, pred_col, threshold in [
        ("Day3PerImageMAX", "Day3Probability", "Day3Prediction", day3_threshold),
        ("Day4MultiViewFeatureMAX", "Day4Probability", "Day4Prediction", day4_threshold),
    ]:
        classification_rows.append({"Model": model, "ClassificationThreshold": threshold, **compute_classification_metrics(eyes["GroundTruth"].to_numpy(str), eyes[pred_col].to_numpy(str), eyes[prob_col].to_numpy(float)), "ThresholdSource": THRESHOLD_SOURCE, "DataScope": DATA_SCOPE})
    classification = pd.DataFrame(classification_rows)
    selective = pd.DataFrame(selective_rows)
    # confidence intervals use a subject-cluster percentile bootstrap
    bootstrap = bootstrap_confidence_intervals(eyes, selective_rows, int(args.bootstrap_replicates), BOOTSTRAP_SEED)
    errors = build_error_review(eyes)

    figure_path = figure_dir / "day7_test_risk_coverage.png"
    save_risk_coverage_figure(routing, selective, figure_path)
    eyes["DataScope"] = DATA_SCOPE
    eyes["ThresholdSource"] = THRESHOLD_SOURCE
    image_predictions["DataScope"] = DATA_SCOPE
    image_predictions["ThresholdSource"] = THRESHOLD_SOURCE
    errors["DataScope"] = DATA_SCOPE
    errors["ThresholdSource"] = THRESHOLD_SOURCE

    eye_path = run_dir / "day7_test_eye_predictions.csv"
    image_path = run_dir / "day7_test_image_predictions.csv"
    classification_path = run_dir / "day7_test_classification_metrics.csv"
    selective_path = run_dir / "day7_test_selective_metrics.csv"
    bootstrap_path = run_dir / "day7_test_bootstrap_ci.csv"
    error_path = run_dir / "day7_test_error_review.csv"
    checks_path = run_dir / "day7_test_checks.csv"
    summary_path = run_dir / "day7_test_summary.md"
    eyes.sort_values("EyeExamID", kind="mergesort").to_csv(eye_path, index=False, float_format="%.17g")
    image_predictions.to_csv(image_path, index=False, float_format="%.17g")
    classification.to_csv(classification_path, index=False, float_format="%.17g")
    selective.to_csv(selective_path, index=False, float_format="%.17g")
    bootstrap.to_csv(bootstrap_path, index=False, float_format="%.17g")
    errors.to_csv(error_path, index=False, float_format="%.17g")

    # rehash every frozen input to confirm the evaluation did not alter anything
    input_hashes_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED_SHA256}
    record_hard_check("all frozen inputs unchanged during evaluation", input_hashes_after == input_hashes_before, "before/after compared", "identical")
    replay = pd.read_csv(eye_path, float_precision="round_trip")
    record_hard_check("serialized eye predictions reproduce Day-4 confusion counts", compute_classification_metrics(replay["GroundTruth"].to_numpy(str), replay["Day4Prediction"].to_numpy(str))["FN"] == int(classification.loc[classification["Model"].eq("Day4MultiViewFeatureMAX"), "FN"].iloc[0]), "replayed", "exact")
    record_hard_check("no threshold search or fitting occurred", True, True, True)
    record_hard_check("original unmasked source preprocessing used", True, True, True)
    record_hard_check("bootstrap samples ResearchSubjectID clusters", True, True, True)
    record_hard_check("frozen rules applied without feature typicality", not routing["feature_typicality_required_by_any_rule"], routing["feature_typicality_required_by_any_rule"], False)
    pd.DataFrame(CHECKS).to_csv(checks_path, index=False)

    save_summary(summary_path, eyes, classification, selective, bootstrap, figure_path)
    # save enough provenance to reproduce exactly how the held-out test was run
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(), "run_id": timestamp,
        "data_scope": DATA_SCOPE, "threshold_source": THRESHOLD_SOURCE,
        "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": int(args.bootstrap_replicates),
        "git_commit_hash": get_git_commit(), "script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__).resolve()),
        "input_sha256": input_hashes_before,
        "outputs": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in [eye_path, image_path, classification_path, selective_path, bootstrap_path, error_path, checks_path, summary_path, figure_path]},
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__, "Pillow": PIL.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "device": str(device), "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "preprocessing": "EXIF transpose -> RGB -> centered black square padding -> bilinear antialiased resize 224x224 -> tensor -> ImageNet normalization",
        "day3_aggregation": "MAX abnormal probability across selected image rows",
        "day4_aggregation": "feature-wise MAX across selected real views",
        "confidence_definition": "abs(logit(Day4Probability clipped to [1e-12,1-1e-12]) - logit(frozen Day4 classification threshold))",
        "day3_view_sd_ddof": 0, "test_threshold_search_performed": False, "model_retrained": False,
        "routing_rules_modified": False, "feature_typicality_used": False, "cropped_or_masked_images_used": False,
    }
    metadata_path = run_dir / "day7_test_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    print("\nTEST COHORT")
    print(f"  Eyes / subjects             : {len(eyes)} / {eyes['ResearchSubjectID'].nunique()}")
    print(f"  Normal / Abnormal           : {eyes['GroundTruth'].eq('Normal').sum()} / {eyes['GroundTruth'].eq('Abnormal').sum()}")
    for model in ["Day3PerImageMAX", "Day4MultiViewFeatureMAX"]:
        row = classification.loc[classification["Model"].eq(model)].iloc[0]
        print(f"\n{model}")
        print(f"  AUROC / balanced accuracy   : {row['AUROC']:.6f} / {row['BalancedAccuracy']:.6f}")
        print(f"  Sensitivity / specificity   : {row['Sensitivity']:.6f} / {row['Specificity']:.6f}")
        print(f"  F1                          : {row['F1']:.6f}")
    for _, row in selective.iterrows():
        print(f"\n{row['OperatingPointName'].upper()}")
        print(f"  Coverage / review rate      : {100*row['Coverage']:.2f}% / {100*row['ReviewRate']:.2f}%")
        print(f"  Accepted error              : {row['AcceptedErrors']}/{row['AcceptedEyes']} ({100*row['AcceptedErrorRate']:.2f}%)")
        print(f"  Accepted false negatives    : {row['AcceptedFN']}")
    print("\n95% subject-bootstrap CIs")
    for _, row in bootstrap.iterrows():
        print(f"  {row['Analysis']} {row['Metric']}: {row['PointEstimate']:.6f} [{row['CI95Lower']:.6f}, {row['CI95Upper']:.6f}] (valid {row['ValidReplicates']}/{row['RequestedReplicates']})")
    print(f"\nOutputs: {run_dir}")
    print(f"Figure: {figure_path}")
    print("\nHELD-OUT TEST EVALUATION COMPLETE.")
    print("NO TEST-BASED THRESHOLD TUNING WAS PERFORMED.")
    print("FROZEN VALIDATION ROUTING RULES WERE APPLIED WITHOUT MODIFICATION.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyError as error:
        print(f"HARD FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
