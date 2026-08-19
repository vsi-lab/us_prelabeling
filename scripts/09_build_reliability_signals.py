

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reliability_metrics import calculate_selective_metrics


DAY3_RUN_ID = "20260813T134419_044915-0700"
DAY4_RUN_ID = "20260813T170240_569268-0700"
TYPICALITY_RUN_ID = "20260814T130502_173490-0700"

DAY3_RUN_DIR = (
    PROJECT_ROOT / "outputs" / "models" / "baseline_resnet18" / DAY3_RUN_ID
).resolve()
DAY4_RUN_DIR = (
    PROJECT_ROOT / "outputs" / "models" / "multiview_resnet18" / DAY4_RUN_ID
).resolve()
TYPICALITY_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "multiview_resnet18"
    / TYPICALITY_RUN_ID
).resolve()

DAY3_IMAGE_PREDICTIONS = (DAY3_RUN_DIR / "val_image_predictions.csv").resolve()
DAY3_EYE_PREDICTIONS = (DAY3_RUN_DIR / "val_eye_predictions.csv").resolve()
DAY3_THRESHOLD = (DAY3_RUN_DIR / "classification_threshold.json").resolve()
DAY3_METADATA = (DAY3_RUN_DIR / "run_metadata.json").resolve()
DAY4_EYE_PREDICTIONS = (DAY4_RUN_DIR / "val_eye_predictions.csv").resolve()
DAY4_THRESHOLD = (DAY4_RUN_DIR / "classification_threshold.json").resolve()
DAY4_METADATA = (DAY4_RUN_DIR / "run_metadata.json").resolve()
TYPICALITY_EMBEDDINGS = (TYPICALITY_RUN_DIR / "val_eye_embeddings.csv").resolve()
TYPICALITY_METADATA = (TYPICALITY_RUN_DIR / "reference_metadata.json").resolve()

PREDICTION_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "predictions" / "reliability"
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "reliability"
AUDIT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"

EXPECTED_HASHES = {
    DAY3_IMAGE_PREDICTIONS: "78f9762aa1dd823bc5a57333d2826e5ccf622940d6e59b3dfa9f828f04beba90",
    DAY3_EYE_PREDICTIONS: "1b6e8a27d6bf8f985bf2a436dcb7ccc93c606c799a140de50e82cbdee1b4d1d9",
    DAY3_THRESHOLD: "daf9e8f2b8cadaa2590ff3ad97b52390cf219c99b35a3758ecb2e934730b44f9",
    DAY3_METADATA: "f96b1a10330f8daae4317dab733c9f04da2a1e3c38dc9b85dd8982087d7a9946",
    DAY4_EYE_PREDICTIONS: "194257a476648e630870e1a343b4dfcbe82897a949b1774c07df6ffdddac1234",
    DAY4_THRESHOLD: "369ae63c078a1aac105e5481c66740f80bf1b6cffde3bfb0253fd70d7dd07d41",
    DAY4_METADATA: "ed6edf2cb4ce415e6d965fadd5ad9fb413a79600aeffe47a0d6791edec1f5d10",
    TYPICALITY_EMBEDDINGS: "22563615dd5e2d61ad0cdbdc8225084abd2a5a7a878333eec5d946f05bd22045",
    TYPICALITY_METADATA: "0b16d43864228a527bc16067ded2e7425f12f249f894e95b41fb3550615a51bd",
}
CSV_INPUTS = {
    DAY3_IMAGE_PREDICTIONS,
    DAY3_EYE_PREDICTIONS,
    DAY4_EYE_PREDICTIONS,
    TYPICALITY_EMBEDDINGS,
}
JSON_INPUTS = {
    DAY3_THRESHOLD,
    DAY3_METADATA,
    DAY4_THRESHOLD,
    DAY4_METADATA,
    TYPICALITY_METADATA,
}
LOADED_CSV_PATHS: set[Path] = set()
LOADED_JSON_PATHS: set[Path] = set()

EXPECTED_VALIDATION_EYES = 120
EXPECTED_DAY4_CORRECT = 110
EXPECTED_DAY4_INCORRECT = 10
EXPECTED_NORMAL_REFERENCES = 61
EXPECTED_ABNORMAL_REFERENCES = 194
PROBABILITY_CLIP_EPSILON = 1e-12
PROBABILITY_RTOL = 1e-12
PROBABILITY_ATOL = 1e-12
SUMMARY_STANDARD_DEVIATION_DDOF = 0
VIEW_STANDARD_DEVIATION_DDOF = 0
DESCRIPTIVE_NEAR_REDUNDANCY_ABS_RHO = 0.90
RISK_BAND_PERCENTAGES = [10, 20, 30, 40, 50]
RISK_SPECS = [
    ("Lowest confidence", "PrimaryConfidenceMargin", True),
    ("Highest Day-3 view SD", "Day3StdViewProbability", False),
    ("Highest Day-3 view range", "Day3ViewProbabilityRange", False),
    (
        "Highest feature atypicality percentile",
        "PredictedClassReferencePercentile",
        False,
    ),
]
CORRELATION_SIGNALS = [
    "PrimaryConfidenceMargin",
    "Day3StdViewProbability",
    "Day3ViewProbabilityRange",
    "PredictedClassReferencePercentile",
]
FORBIDDEN_TEST_NAMES = {
    "test_images.csv",
    "test_eye_predictions.csv",
    "test_image_predictions.csv",
}


class SafetyError(RuntimeError):
    """used when a locked input, integrity, or test-protection check fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_test_path(path: Path) -> None:
    if path.name.casefold() in FORBIDDEN_TEST_NAMES or path.name.casefold().startswith(
        "test_"
    ):
        raise SafetyError(f"Test artifact access was attempted: {path}")


def check_locked_file(path: Path) -> str:
    # every input is hash-pinned so a changed artifact stops the analysis
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in EXPECTED_HASHES:
        raise SafetyError(f"Input is outside the exact locked allowlist: {resolved}")
    if not resolved.is_file():
        raise SafetyError(f"Locked input is missing: {resolved}")
    observed = file_sha256(resolved)
    expected = EXPECTED_HASHES[resolved]
    if observed != expected:
        raise SafetyError(
            f"Locked input hash mismatch for {resolved}: observed={observed}, "
            f"expected={expected}."
        )
    return observed


def read_allowed_csv(path: Path) -> pd.DataFrame:
    # only the listed validation csv files are allowed here
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in CSV_INPUTS:
        raise SafetyError(f"CSV input is outside the exact validation allowlist: {resolved}")
    LOADED_CSV_PATHS.add(resolved)
    return pd.read_csv(
        resolved,
        dtype={
            "EyeExamID": "string",
            "ResearchSubjectID": "string",
            "EncounterID": "string",
            "Laterality": "string",
            "ImageRelativePath": "string",
            "TrueLabel": "string",
            "TrueEyeLabel": "string",
            "PredictedLabel": "string",
        },
        float_precision="round_trip",
        low_memory=False,
    )


def read_allowed_json(path: Path) -> dict[str, Any]:
    # supporting json files follow the same exact allowlist rule
    resolved = path.resolve()
    check_not_test_path(resolved)
    if resolved not in JSON_INPUTS:
        raise SafetyError(f"JSON input is outside the exact validation allowlist: {resolved}")
    LOADED_JSON_PATHS.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SafetyError(f"Expected a JSON object: {resolved}")
    return value


def check_required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SafetyError(f"{name} is missing required columns: {missing}")


def check_nonblank_fields(frame: pd.DataFrame, fields: list[str], name: str) -> None:
    for field in fields:
        values = frame[field].astype("string")
        blank = values.isna() | values.str.strip().eq("")
        if blank.any():
            raise SafetyError(f"{name} has {int(blank.sum())} blank {field} values.")


def parse_bool_column(series: pd.Series, field: str) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = normalized.isna() | ~normalized.isin(["true", "false"])
    if invalid.any():
        raise SafetyError(f"Invalid or blank boolean values in {field}.")
    return normalized.eq("true")


def finite_numeric_values(series: pd.Series, field: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SafetyError(f"{field} contains NaN or infinity.")
    return values


def checked_probabilities(series: pd.Series, field: str) -> np.ndarray:
    values = finite_numeric_values(series, field)
    if ((values < 0.0) | (values > 1.0)).any():
        raise SafetyError(f"{field} contains values outside [0,1].")
    return values


def check_unique_eye_rows(frame: pd.DataFrame, name: str) -> None:
    if frame.empty or frame["EyeExamID"].duplicated().any():
        raise SafetyError(f"{name} must have one unique row per EyeExamID.")


def check_eye_mapping(
    canonical: pd.DataFrame,
    other: pd.DataFrame,
    fields: list[str],
    name: str,
) -> None:
    left = canonical.copy()
    right = other.copy()
    left["EyeExamID"] = left["EyeExamID"].astype(str)
    right["EyeExamID"] = right["EyeExamID"].astype(str)
    left = left.set_index("EyeExamID").sort_index()
    right = right.set_index("EyeExamID").sort_index()
    if set(left.index.astype(str)) != set(right.index.astype(str)):
        raise SafetyError(f"{name} EyeExamID set differs from authoritative Day 4.")
    right = right.loc[left.index]
    for field in fields:
        if not np.array_equal(
            left[field].astype(str).to_numpy(),
            right[field].astype(str).to_numpy(),
        ):
            raise SafetyError(f"{name} {field} mapping differs from authoritative Day 4.")


def get_locked_threshold(document: dict[str, Any], name: str) -> float:
    threshold = float(document.get("classification_threshold", math.nan))
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise SafetyError(f"{name} classification threshold is invalid.")
    if document.get("test_data_used") is not False:
        raise SafetyError(f"{name} threshold metadata does not confirm test_data_used=false.")
    return threshold


def summarize_view_probabilities(probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    # these summaries describe how much the day-3 image predictions disagree within an eye
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise SafetyError("View probabilities must be a nonempty finite vector.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise SafetyError("View probabilities lie outside [0,1].")
    return {
        "NumberOfViews": int(len(values)),
        "Day3MeanViewProbability": float(np.mean(values)),
        "Day3MaxViewProbability": float(np.max(values)),
        "Day3MinViewProbability": float(np.min(values)),
        "Day3StdViewProbability": float(np.std(values, ddof=VIEW_STANDARD_DEVIATION_DDOF)),
        "Day3ViewProbabilityRange": float(np.max(values) - np.min(values)),
        "Day3PositiveViewFraction": float(np.mean(values >= threshold)),
    }


def summarize_day3_views(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # rebuild one eye-level row from the day-3 per-image predictions
    rows: list[dict[str, Any]] = []
    ordered = frame.sort_values(
        ["EyeExamID", "ImageRelativePath"], kind="mergesort"
    )
    for eye_id, group in ordered.groupby("EyeExamID", sort=True):
        for field in [
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "TrueEyeLabel",
        ]:
            if group[field].nunique(dropna=False) != 1:
                raise SafetyError(f"Day-3 images have inconsistent {field} for {eye_id}.")
        first = group.iloc[0]
        probabilities = checked_probabilities(
            group["AbnormalProbability"], "Day-3 image AbnormalProbability"
        )
        rows.append(
            {
                "EyeExamID": str(eye_id),
                "ResearchSubjectID": str(first["ResearchSubjectID"]),
                "EncounterID": str(first["EncounterID"]),
                "Laterality": str(first["Laterality"]),
                "TrueLabel": str(first["TrueEyeLabel"]),
                **summarize_view_probabilities(probabilities, threshold),
            }
        )
    return pd.DataFrame(rows)


def check_day4_view_diagnostics(frame: pd.DataFrame) -> float:
    # recompute the saved day-4 view summaries before using them downstream
    view_columns = [f"View{index}AbnormalProbability" for index in range(1, 7)]
    maximum_difference = 0.0
    for _, row in frame.iterrows():
        view_values = pd.to_numeric(row[view_columns], errors="coerce").dropna().to_numpy(float)
        expected_count = int(row["NumberOfViews"])
        if len(view_values) != expected_count:
            raise SafetyError(
                f"Day-4 diagnostic view count mismatch for {row['EyeExamID']}."
            )
        calculated = [
            float(np.mean(view_values)),
            float(np.std(view_values, ddof=0)),
            float(np.max(view_values) - np.min(view_values)),
        ]
        saved = [
            float(row["MeanViewProbability"]),
            float(row["StdViewProbability"]),
            float(row["ViewProbabilityRange"]),
        ]
        if (
            not np.isfinite(view_values).all()
            or ((view_values < 0.0) | (view_values > 1.0)).any()
            or not np.isfinite(saved).all()
            or not all(0.0 <= value <= 1.0 for value in saved)
        ):
            raise SafetyError(
                f"Day-4 diagnostic values are nonfinite or outside [0,1] for "
                f"{row['EyeExamID']}."
            )
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(np.asarray(calculated) - np.asarray(saved)))),
        )
    if maximum_difference > 1e-12:
        raise SafetyError(
            "Day-4 diagnostic view summaries do not reproduce saved view values; "
            f"max_abs_difference={maximum_difference}."
        )
    return maximum_difference


def summarize_group(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "PredictionGroup": name,
        "Count": int(len(frame)),
    }
    for signal in CORRELATION_SIGNALS:
        values = frame[signal].to_numpy(dtype=np.float64)
        result[f"{signal}Mean"] = float(np.mean(values))
        result[f"{signal}Median"] = float(np.median(values))
        result[f"{signal}PopulationStandardDeviation"] = float(
            np.std(values, ddof=SUMMARY_STANDARD_DEVIATION_DDOF)
        )
    disagreement_count = int(frame["ModelDisagreement"].sum())
    result["ModelDisagreementCount"] = disagreement_count
    result["ModelDisagreementPercent"] = 100.0 * disagreement_count / len(frame)
    return result


def select_risk_band(
    frame: pd.DataFrame,
    signal: str,
    ascending: bool,
    percentage: float,
) -> tuple[pd.Index, dict[str, Any]]:
    # use an exact eye count and eye id as the deterministic tie-break
    count = int(math.ceil(len(frame) * percentage / 100.0))
    ranked = frame.sort_values(
        [signal, "EyeExamID"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    selected = ranked.iloc[:count]
    boundary_value = float(selected.iloc[-1][signal])
    all_boundary = int(np.isclose(frame[signal], boundary_value, rtol=0.0, atol=0.0).sum())
    selected_boundary = int(
        np.isclose(selected[signal], boundary_value, rtol=0.0, atol=0.0).sum()
    )
    details = {
        "BandEyeCount": count,
        "BoundarySignalValue": boundary_value,
        "BoundaryTieTotal": all_boundary,
        "BoundaryTieSelected": selected_boundary,
    }
    return selected.index, details


def make_error_capture_table(frame: pd.DataFrame) -> pd.DataFrame:
    # compare how many day-4 errors each signal captures at fixed review bands
    rows = []
    total_errors = int((~frame["Day4Correct"]).sum())
    for display_name, signal, ascending in RISK_SPECS:
        for percentage in RISK_BAND_PERCENTAGES:
            indices, details = select_risk_band(frame, signal, ascending, percentage)
            selected = frame.loc[indices]
            errors = selected.loc[~selected["Day4Correct"], "EyeExamID"].astype(str).tolist()
            rows.append(
                {
                    "Signal": display_name,
                    "SignalColumn": signal,
                    "RiskDirection": "lowest values" if ascending else "highest values",
                    "RiskBandPercent": percentage,
                    **details,
                    "CapturedErrors": int(len(errors)),
                    "TotalDay4Errors": total_errors,
                    "ErrorCapturePercent": 100.0 * len(errors) / total_errors,
                    "CapturedErrorEyeExamIDs": "|".join(errors),
                    "TieRule": "signal risk order, then EyeExamID ascending; exact band size",
                }
            )
    return pd.DataFrame(rows)


def make_overlap_rows(
    frame: pd.DataFrame,
    framework: str,
    confidence_mask: pd.Series,
    view_sd_mask: pd.Series,
    feature_mask: pd.Series,
) -> list[dict[str, Any]]:
    masks = {
        "Lowest-confidence 25%": confidence_mask,
        "Highest-Day3-view-SD 25%": view_sd_mask,
        "Feature atypicality band": feature_mask,
        "Confidence AND view SD": confidence_mask & view_sd_mask,
        "Confidence AND feature": confidence_mask & feature_mask,
        "View SD AND feature": view_sd_mask & feature_mask,
        "All three": confidence_mask & view_sd_mask & feature_mask,
        "Union of three": confidence_mask | view_sd_mask | feature_mask,
        "Feature unique beyond confidence OR view SD": (
            feature_mask & ~(confidence_mask | view_sd_mask)
        ),
    }
    rows = []
    for component, mask in masks.items():
        selected = frame.loc[mask]
        errors = selected.loc[~selected["Day4Correct"], "EyeExamID"].astype(str).tolist()
        rows.append(
            {
                "Framework": framework,
                "Component": component,
                "SelectedValidationEyes": int(mask.sum()),
                "CapturedErrors": int(len(errors)),
                "TotalDay4Errors": int((~frame["Day4Correct"]).sum()),
                "ErrorCapturePercent": 100.0 * len(errors) / int((~frame["Day4Correct"]).sum()),
                "CapturedErrorEyeExamIDs": "|".join(errors),
            }
        )
    return rows


def make_signal_overlap_table(frame: pd.DataFrame) -> pd.DataFrame:
    # compare which validation eyes are flagged by the different reliability signals
    confidence_indices, _ = select_risk_band(
        frame, "PrimaryConfidenceMargin", True, 25.0
    )
    view_indices, _ = select_risk_band(frame, "Day3StdViewProbability", False, 25.0)
    feature_indices, _ = select_risk_band(
        frame, "PredictedClassReferencePercentile", False, 25.0
    )
    confidence_mask = frame.index.isin(confidence_indices)
    view_mask = frame.index.isin(view_indices)
    feature_rank_mask = frame.index.isin(feature_indices)
    feature_literal_mask = frame["PredictedClassReferencePercentile"].ge(75.0)

    rows = make_overlap_rows(
        frame,
        "Exact validation-ranked quartiles (30 eyes per signal)",
        pd.Series(confidence_mask, index=frame.index),
        pd.Series(view_mask, index=frame.index),
        pd.Series(feature_rank_mask, index=frame.index),
    )
    rows.extend(
        make_overlap_rows(
            frame,
            "Literal feature reference percentile >=75 (not a 30-eye quartile)",
            pd.Series(confidence_mask, index=frame.index),
            pd.Series(view_mask, index=frame.index),
            feature_literal_mask,
        )
    )
    return pd.DataFrame(rows)


def run_metric_self_checks() -> int:
    # small synthetic cases protect the selective-metric helper from silent changes
    tests_run = 0
    truth = ["Normal", "Normal", "Abnormal", "Abnormal"]
    predicted = ["Normal", "Abnormal", "Abnormal", "Normal"]
    all_accepted = calculate_selective_metrics(predicted, truth, [True] * 4)
    expected_all = {
        "Coverage": 1.0,
        "ReviewRate": 0.0,
        "AcceptedErrorRate": 0.5,
        "AcceptedAccuracy": 0.5,
        "AcceptedSensitivity": 0.5,
        "AcceptedSpecificity": 0.5,
        "AcceptedBalancedAccuracy": 0.5,
        "FalseNegativesAmongAccepted": 1,
    }
    if any(not np.isclose(all_accepted[key], value) for key, value in expected_all.items()):
        raise SafetyError("Synthetic all-accepted selective-metric test failed.")
    tests_run += 1

    partial = calculate_selective_metrics(
        predicted, truth, [True, False, True, False]
    )
    if not (
        partial["Coverage"] == 0.5
        and partial["ReviewRate"] == 0.5
        and partial["AcceptedAccuracy"] == 1.0
        and partial["AcceptedBalancedAccuracy"] == 1.0
    ):
        raise SafetyError("Synthetic partial-acceptance selective-metric test failed.")
    tests_run += 1

    zero = calculate_selective_metrics(predicted, truth, [False] * 4)
    nan_fields = [
        "AcceptedErrorRate",
        "AcceptedAccuracy",
        "AcceptedSensitivity",
        "AcceptedSpecificity",
        "AcceptedBalancedAccuracy",
    ]
    if not (
        zero["Coverage"] == 0.0
        and zero["ReviewRate"] == 1.0
        and zero["FalseNegativesAmongAccepted"] == 0
        and all(math.isnan(zero[field]) for field in nan_fields)
    ):
        raise SafetyError("Synthetic zero-acceptance selective-metric test failed.")
    tests_run += 1

    normal_only = calculate_selective_metrics(predicted, truth, [True, True, False, False])
    if not (
        math.isnan(normal_only["AcceptedSensitivity"])
        and math.isnan(normal_only["AcceptedBalancedAccuracy"])
        and math.isfinite(normal_only["AcceptedSpecificity"])
    ):
        raise SafetyError("Synthetic missing-class selective-metric test failed.")
    tests_run += 1

    invalid_cases = [
        (["Normal"], ["Normal"], [True, False]),
        (["Other"], ["Normal"], [True]),
        (["Normal"], ["Normal"], [1]),
        ([], [], []),
    ]
    for predicted_values, truth_values, mask_values in invalid_cases:
        try:
            calculate_selective_metrics(predicted_values, truth_values, mask_values)
        except ValueError:
            tests_run += 1
        else:
            raise SafetyError("Synthetic invalid-input selective-metric test failed.")
    return tests_run


def save_group_boxplot(
    frame: pd.DataFrame,
    signal: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    correct = frame.loc[frame["Day4Correct"], signal].to_numpy(float)
    incorrect = frame.loc[~frame["Day4Correct"], signal].to_numpy(float)
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
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_correlation_plot(correlation: pd.DataFrame, path: Path) -> None:
    labels = [
        "Confidence",
        "View SD",
        "View range",
        "Feature percentile",
    ]
    values = correlation.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8.0, 6.8))
    image = axis.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            color = "white" if abs(values[row, column]) > 0.55 else "black"
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
            )
    axis.set_title("Validation reliability-signal Spearman correlations")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    # leave extra space for the longer axis labels on some matplotlib renderers
    fig.subplots_adjust(left=0.22, bottom=0.22, right=0.90, top=0.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confidence_typicality_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    for correct_value, label, color, marker in [
        (True, "Correct", "#4C78A8", "o"),
        (False, "Incorrect", "#E45756", "X"),
    ]:
        group = frame.loc[frame["Day4Correct"].eq(correct_value)]
        axis.scatter(
            group["PrimaryConfidenceMargin"],
            group["PredictedClassReferencePercentile"],
            color=color,
            marker=marker,
            alpha=0.8,
            edgecolors="none",
            label=f"{label} (n={len(group)})",
        )
    axis.set_xlabel("Primary confidence margin (log-odds from Day-4 boundary)")
    axis.set_ylabel("Predicted-class reference percentile")
    axis.set_title("Validation confidence versus feature atypicality")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def make_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")


def save_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def main() -> None:
    # this stage reads frozen outputs only and never retrains either model
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat()
    checks: list[dict[str, str]] = []

    def hard_check(name: str, condition: bool, details: str) -> None:
        checks.append(
            {
                "Check": name,
                "Severity": "HARD FAIL",
                "Status": "PASS" if condition else "FAIL",
                "Details": details,
            }
        )
        if not condition:
            raise SafetyError(f"{name}: {details}")

    print("DAY-5 COMBINED RELIABILITY SIGNALS")
    print(f"Locked Day-3 run       : {DAY3_RUN_ID}")
    print(f"Authoritative Day-4 run: {DAY4_RUN_ID}")
    print(f"Typicality run         : {TYPICALITY_RUN_ID}")
    print("Mode: validation-only descriptive analysis; no model training")

    # verify every frozen input before loading any of the analysis tables
    input_hashes: dict[str, str] = {}
    for path in EXPECTED_HASHES:
        observed = check_locked_file(path)
        input_hashes[str(path)] = observed
        hard_check(
            f"locked input hash: {path.name} ({path.parent.name})",
            True,
            f"sha256={observed}",
        )

    # load the three locked sources that will be joined at the eye level
    day3_images = read_allowed_csv(DAY3_IMAGE_PREDICTIONS)
    day3_eyes = read_allowed_csv(DAY3_EYE_PREDICTIONS)
    day4_eyes = read_allowed_csv(DAY4_EYE_PREDICTIONS)
    typicality = read_allowed_csv(TYPICALITY_EMBEDDINGS)
    day3_threshold_document = read_allowed_json(DAY3_THRESHOLD)
    day3_metadata = read_allowed_json(DAY3_METADATA)
    day4_threshold_document = read_allowed_json(DAY4_THRESHOLD)
    day4_metadata = read_allowed_json(DAY4_METADATA)
    typicality_metadata = read_allowed_json(TYPICALITY_METADATA)

    expected_csv_paths = CSV_INPUTS
    expected_json_paths = JSON_INPUTS
    hard_check(
        "only exact validation CSV artifacts loaded",
        LOADED_CSV_PATHS == expected_csv_paths,
        f"loaded={sorted(map(str, LOADED_CSV_PATHS))}",
    )
    hard_check(
        "only exact validation/supporting JSON artifacts loaded",
        LOADED_JSON_PATHS == expected_json_paths,
        f"loaded={sorted(map(str, LOADED_JSON_PATHS))}",
    )
    hard_check(
        "no test artifact path loaded",
        all(path.name.casefold() not in FORBIDDEN_TEST_NAMES for path in LOADED_CSV_PATHS | LOADED_JSON_PATHS),
        "test_data_loaded=False",
    )

    day3_metrics_path = Path(
        day4_metadata.get("day3_baseline_comparison", {}).get("metrics_path", "")
    ).resolve()
    hard_check(
        "Day-4 provenance resolves the locked Day-3 run",
        day3_metrics_path.parent == DAY3_RUN_DIR and day3_metrics_path.name == "val_metrics.csv",
        f"recorded_path={day3_metrics_path}",
    )
    metadata_test_fields = [
        (
            "Day-3",
            day3_metadata,
            [
                "test_manifest_loaded",
                "test_set_evaluated",
                "test_predictions_created",
            ],
        ),
        (
            "Day-4",
            day4_metadata,
            [
                "test_manifest_loaded",
                "test_set_evaluated",
                "test_predictions_created",
            ],
        ),
        (
            "feature typicality",
            typicality_metadata,
            [
                "test_manifest_loaded",
                "test_set_evaluated",
                "test_embeddings_extracted",
                "test_predictions_created",
            ],
        ),
    ]
    for name, metadata, required_flags in metadata_test_fields:
        test_flags = {field: metadata.get(field) for field in required_flags}
        hard_check(
            f"{name} provenance confirms test isolation",
            all(value is False for value in test_flags.values()),
            f"test_flags={test_flags}",
        )

    check_required_columns(
        day3_images,
        {
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "ImageRelativePath",
            "TrueEyeLabel",
            "AbnormalProbability",
        },
        "Day-3 image predictions",
    )
    check_required_columns(
        day3_eyes,
        {
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "TrueLabel",
            "MaxAbnormalProbability",
            "PredictedLabel",
            "ClassificationThreshold",
            "Correct",
        },
        "Day-3 eye predictions",
    )
    check_required_columns(
        day4_eyes,
        {
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
            "MeanViewProbability",
            "StdViewProbability",
            "ViewProbabilityRange",
            *{f"View{index}AbnormalProbability" for index in range(1, 7)},
        },
        "Day-4 eye predictions",
    )
    check_required_columns(
        typicality,
        {
            "EyeExamID",
            "ResearchSubjectID",
            "EncounterID",
            "Laterality",
            "TrueLabel",
            "PredictedLabel",
            "EyeAbnormalProbability",
            "Correct",
            "PrimaryConfidenceMargin",
            "NumberOfViews",
            "FeatureTypicalityPredictedClass",
            "PredictedClassReferencePercentile",
        },
        "feature-typicality validation embeddings",
    )
    identifier_fields = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
    ]
    check_nonblank_fields(day3_images, identifier_fields + ["TrueEyeLabel"], "Day-3 images")
    for name, frame in [
        ("Day-3 eyes", day3_eyes),
        ("Day-4 eyes", day4_eyes),
        ("feature typicality", typicality),
    ]:
        check_nonblank_fields(frame, identifier_fields + ["TrueLabel"], name)
        check_unique_eye_rows(frame, name)
        hard_check(
            f"{name} has no duplicate EyeExamID",
            not frame["EyeExamID"].duplicated().any(),
            f"rows={len(frame)}, unique={frame['EyeExamID'].nunique()}",
        )
    hard_check(
        "Day-3 image rows unique within eye",
        not day3_images.duplicated(["EyeExamID", "ImageRelativePath"]).any(),
        f"image_rows={len(day3_images)}",
    )

    # use the already frozen classification thresholds from the original runs
    day3_threshold = get_locked_threshold(day3_threshold_document, "Day-3")
    day4_threshold = get_locked_threshold(day4_threshold_document, "Day-4")
    day3_view_summary = summarize_day3_views(day3_images, day3_threshold)
    check_unique_eye_rows(day3_view_summary, "Day-3 grouped image predictions")

    # day-4 is the authoritative eye list for this combined validation table
    canonical = day4_eyes[
        ["EyeExamID", "ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel"]
    ].copy()
    canonical = canonical.sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)
    hard_check(
        "validation eye count == 120",
        len(canonical) == EXPECTED_VALIDATION_EYES,
        f"observed={len(canonical)}, expected={EXPECTED_VALIDATION_EYES}",
    )
    canonical_ids = set(canonical["EyeExamID"].astype(str))
    day3_eye_ids = set(day3_eyes["EyeExamID"].astype(str))
    day3_image_eye_ids = set(day3_view_summary["EyeExamID"].astype(str))
    typicality_ids = set(typicality["EyeExamID"].astype(str))
    hard_check(
        "Day-3 and Day-4 EyeExamID sets identical",
        day3_eye_ids == canonical_ids and day3_image_eye_ids == canonical_ids,
        f"Day3_eye={len(day3_eye_ids)}, Day3_image_groups={len(day3_image_eye_ids)}, Day4={len(canonical_ids)}",
    )
    hard_check(
        "feature-typicality EyeExamID set identical",
        typicality_ids == canonical_ids,
        f"typicality={len(typicality_ids)}, Day4={len(canonical_ids)}",
    )
    hard_check(
        "no non-authoritative validation EyeExamID appears",
        day3_eye_ids == day3_image_eye_ids == typicality_ids == canonical_ids,
        (
            "test-safe operational check: all IDs equal the hash-pinned authoritative "
            "validation allowlist; test IDs were not loaded for comparison"
        ),
    )
    hard_check(
        "all true and predicted labels are Normal/Abnormal",
        set(canonical["TrueLabel"].astype(str)) == {"Normal", "Abnormal"}
        and set(day3_eyes["PredictedLabel"].astype(str)).issubset({"Normal", "Abnormal"})
        and set(day4_eyes["PredictedLabel"].astype(str)).issubset({"Normal", "Abnormal"})
        and set(typicality["PredictedLabel"].astype(str)).issubset({"Normal", "Abnormal"}),
        "Normal and Abnormal are the only permitted labels",
    )
    for name, frame in [
        ("Day-3 eyes", day3_eyes),
        ("Day-3 grouped images", day3_view_summary),
        ("feature typicality", typicality),
    ]:
        check_eye_mapping(
            canonical,
            frame,
            ["ResearchSubjectID", "EncounterID", "Laterality", "TrueLabel"],
            name,
        )
        hard_check(
            f"{name} identifier and true-label mappings agree",
            True,
            "ResearchSubjectID, EncounterID, Laterality, and TrueLabel exact",
        )

    day3_image_probabilities = checked_probabilities(
        day3_images["AbnormalProbability"], "Day-3 image probability"
    )
    day3_eye_probabilities = checked_probabilities(
        day3_eyes["MaxAbnormalProbability"], "Day-3 eye probability"
    )
    day4_probabilities = checked_probabilities(
        day4_eyes["EyeAbnormalProbability"], "Day-4 eye probability"
    )
    hard_check(
        "all required probabilities are in [0,1]",
        len(day3_image_probabilities) == len(day3_images)
        and len(day3_eye_probabilities) == len(day3_eyes)
        and len(day4_probabilities) == len(day4_eyes),
        "Day-3 image, Day-3 eye, and Day-4 eye probabilities checked",
    )

    day3_eyes = day3_eyes.copy()
    day4_eyes = day4_eyes.copy()
    typicality = typicality.copy()
    day3_eyes["Correct"] = parse_bool_column(day3_eyes["Correct"], "Day-3 Correct")
    day4_eyes["Correct"] = parse_bool_column(day4_eyes["Correct"], "Day-4 Correct")
    typicality["Correct"] = parse_bool_column(typicality["Correct"], "typicality Correct")
    if not np.allclose(
        finite_numeric_values(day3_eyes["ClassificationThreshold"], "Day-3 row threshold"),
        day3_threshold,
        rtol=0.0,
        atol=0.0,
    ):
        raise SafetyError("Day-3 row thresholds differ from locked threshold JSON.")
    if not np.allclose(
        finite_numeric_values(day4_eyes["ClassificationThreshold"], "Day-4 row threshold"),
        day4_threshold,
        rtol=0.0,
        atol=0.0,
    ):
        raise SafetyError("Day-4 row thresholds differ from locked threshold JSON.")

    expected_day3_label = np.where(
        day3_eyes["MaxAbnormalProbability"].to_numpy(float) >= day3_threshold,
        "Abnormal",
        "Normal",
    )
    expected_day4_label = np.where(
        day4_eyes["EyeAbnormalProbability"].to_numpy(float) >= day4_threshold,
        "Abnormal",
        "Normal",
    )
    hard_check(
        "locked Day-3 predictions and correctness reproduce",
        np.array_equal(expected_day3_label, day3_eyes["PredictedLabel"].astype(str))
        and np.array_equal(
            expected_day3_label == day3_eyes["TrueLabel"].astype(str),
            day3_eyes["Correct"].to_numpy(bool),
        ),
        f"classification_threshold={day3_threshold:.17g}",
    )
    hard_check(
        "locked Day-4 predictions and correctness reproduce",
        np.array_equal(expected_day4_label, day4_eyes["PredictedLabel"].astype(str))
        and np.array_equal(
            expected_day4_label == day4_eyes["TrueLabel"].astype(str),
            day4_eyes["Correct"].to_numpy(bool),
        ),
        f"classification_threshold={day4_threshold:.17g}",
    )

    grouped_indexed = day3_view_summary.set_index("EyeExamID").sort_index()
    day3_eye_indexed = day3_eyes.set_index("EyeExamID").sort_index()
    max_difference = float(
        np.max(
            np.abs(
                grouped_indexed["Day3MaxViewProbability"].to_numpy(float)
                - day3_eye_indexed["MaxAbnormalProbability"].to_numpy(float)
            )
        )
    )
    hard_check(
        "Day-3 image MAX reproduces locked Day-3 eye probability",
        np.allclose(
            grouped_indexed["Day3MaxViewProbability"],
            day3_eye_indexed["MaxAbnormalProbability"],
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        ),
        f"max_abs_difference={max_difference:.12g}",
    )
    singleton = summarize_view_probabilities(np.asarray([0.4]), day3_threshold)
    hard_check(
        "single-view disagreement behavior is defined",
        singleton["Day3StdViewProbability"] == 0.0
        and singleton["Day3ViewProbabilityRange"] == 0.0,
        "one view yields SD=0 and range=0; disagreement is not meaningfully assessable",
    )
    day4_diagnostic_difference = check_day4_view_diagnostics(day4_eyes)
    hard_check(
        "Day-4 diagnostic view summaries reproduce",
        day4_diagnostic_difference <= 1e-12,
        f"max_abs_difference={day4_diagnostic_difference:.12g}",
    )

    reference_counts = typicality_metadata.get("reference_selection", {}).get("counts", {})
    normal_references = int(reference_counts.get("Normal", {}).get("reference_eyes", -1))
    abnormal_references = int(reference_counts.get("Abnormal", {}).get("reference_eyes", -1))
    hard_check(
        "feature reference sets remain locked at 61 Normal / 194 Abnormal",
        normal_references == EXPECTED_NORMAL_REFERENCES
        and abnormal_references == EXPECTED_ABNORMAL_REFERENCES,
        f"Normal={normal_references}, Abnormal={abnormal_references}",
    )
    feature_distances = finite_numeric_values(
        typicality["FeatureTypicalityPredictedClass"], "feature typicality distance"
    )
    feature_percentiles = finite_numeric_values(
        typicality["PredictedClassReferencePercentile"], "feature reference percentile"
    )
    hard_check(
        "feature percentiles present and in [0,100]",
        ((feature_percentiles >= 0.0) & (feature_percentiles <= 100.0)).all(),
        f"minimum={feature_percentiles.min():.6f}, maximum={feature_percentiles.max():.6f}",
    )

    day4_indexed = day4_eyes.set_index("EyeExamID").sort_index()
    typicality_indexed = typicality.set_index("EyeExamID").sort_index()
    hard_check(
        "typicality artifact reproduces locked Day-4 predictions",
        np.allclose(
            typicality_indexed["EyeAbnormalProbability"],
            day4_indexed["EyeAbnormalProbability"],
            rtol=0.0,
            atol=0.0,
        )
        and typicality_indexed["PredictedLabel"].astype(str).equals(
            day4_indexed["PredictedLabel"].astype(str)
        )
        and typicality_indexed["Correct"].astype(bool).equals(
            day4_indexed["Correct"].astype(bool)
        ),
        "probabilities, labels, and correctness exact",
    )

    # start from the day-4 eye table and attach the other reliability signals by eye id
    combined = canonical.copy().set_index("EyeExamID")
    combined["Day4EyeProbability"] = day4_indexed["EyeAbnormalProbability"].astype(float)
    combined["Day4ClassificationThreshold"] = day4_threshold
    combined["Day4PredictedLabel"] = day4_indexed["PredictedLabel"].astype(str)
    combined["Day4Correct"] = day4_indexed["Correct"].astype(bool)
    clipped = np.clip(
        combined["Day4EyeProbability"].to_numpy(float),
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    clipped_count = int(
        np.not_equal(clipped, combined["Day4EyeProbability"].to_numpy(float)).sum()
    )
    # confidence is measured as distance from the frozen day-4 decision boundary in logit space
    primary_logit = np.log(clipped) - np.log1p(-clipped)
    threshold_logit = math.log(day4_threshold) - math.log1p(-day4_threshold)
    combined["PrimaryLogit"] = primary_logit
    combined["ThresholdLogit"] = threshold_logit
    combined["PrimarySignedMargin"] = primary_logit - threshold_logit
    combined["PrimaryConfidenceMargin"] = np.abs(combined["PrimarySignedMargin"])
    hard_check(
        "confidence margins finite",
        np.isfinite(
            combined[
                [
                    "PrimaryLogit",
                    "ThresholdLogit",
                    "PrimarySignedMargin",
                    "PrimaryConfidenceMargin",
                ]
            ].to_numpy(float)
        ).all(),
        f"probability_clip_epsilon={PROBABILITY_CLIP_EPSILON}; clipped_probabilities={clipped_count}",
    )
    signed_predictions = np.where(combined["PrimarySignedMargin"].ge(0), "Abnormal", "Normal")
    hard_check(
        "signed confidence margin agrees with locked Day-4 label",
        np.array_equal(signed_predictions, combined["Day4PredictedLabel"].astype(str)),
        "PrimarySignedMargin >= 0 corresponds to Abnormal",
    )

    for column in [
        "NumberOfViews",
        "Day3MeanViewProbability",
        "Day3MaxViewProbability",
        "Day3MinViewProbability",
        "Day3StdViewProbability",
        "Day3ViewProbabilityRange",
        "Day3PositiveViewFraction",
    ]:
        combined[column] = grouped_indexed[column]
    combined["Day3EyeProbability"] = day3_eye_indexed["MaxAbnormalProbability"].astype(float)
    combined["Day3ClassificationThreshold"] = day3_threshold
    combined["Day3PredictedLabel"] = day3_eye_indexed["PredictedLabel"].astype(str)
    combined["Day3Correct"] = day3_eye_indexed["Correct"].astype(bool)
    combined["ModelAgreement"] = combined["Day3PredictedLabel"].eq(
        combined["Day4PredictedLabel"]
    )
    combined["ModelDisagreement"] = ~combined["ModelAgreement"]
    combined["FeatureTypicalityDistance"] = typicality_indexed[
        "FeatureTypicalityPredictedClass"
    ].astype(float)
    combined["PredictedClassReferencePercentile"] = typicality_indexed[
        "PredictedClassReferencePercentile"
    ].astype(float)
    combined["Day4DiagnosticMeanViewProbability"] = day4_indexed[
        "MeanViewProbability"
    ].astype(float)
    combined["Day4DiagnosticStdViewProbability"] = day4_indexed[
        "StdViewProbability"
    ].astype(float)
    combined["Day4DiagnosticViewProbabilityRange"] = day4_indexed[
        "ViewProbabilityRange"
    ].astype(float)
    combined = combined.reset_index().sort_values("EyeExamID", kind="mergesort").reset_index(drop=True)

    hard_check(
        "combined table contains 120 unique validation eyes",
        len(combined) == EXPECTED_VALIDATION_EYES
        and not combined["EyeExamID"].duplicated().any()
        and set(combined["EyeExamID"].astype(str)) == canonical_ids,
        f"rows={len(combined)}, unique={combined['EyeExamID'].nunique()}",
    )
    view_count_match = combined.set_index("EyeExamID")["NumberOfViews"].astype(int).equals(
        day4_indexed["NumberOfViews"].astype(int)
    )
    hard_check(
        "Day-3 and Day-4 view counts agree",
        view_count_match,
        f"view_count_distribution={combined['NumberOfViews'].value_counts().sort_index().to_dict()}",
    )
    correct_count = int(combined["Day4Correct"].sum())
    incorrect_count = len(combined) - correct_count
    hard_check(
        "Day-4 correct count == 110",
        correct_count == EXPECTED_DAY4_CORRECT,
        f"observed={correct_count}, expected={EXPECTED_DAY4_CORRECT}",
    )
    hard_check(
        "Day-4 incorrect count == 10",
        incorrect_count == EXPECTED_DAY4_INCORRECT,
        f"observed={incorrect_count}, expected={EXPECTED_DAY4_INCORRECT}",
    )
    hard_check(
        "combined table has no routing decision",
        not any(
            token in column.casefold()
            for column in combined.columns
            for token in ["accept", "defer", "routingdecision"]
        ),
        "only reliability signals and exploratory diagnostics are present",
    )

    # everything below is descriptive validation analysis, not a routing rule
    summary = pd.DataFrame(
        [
            summarize_group(combined, "Overall"),
            summarize_group(combined.loc[combined["Day4Correct"]], "Correct"),
            summarize_group(combined.loc[~combined["Day4Correct"]], "Incorrect"),
        ]
    )
    error_capture = make_error_capture_table(combined)
    overlap = make_signal_overlap_table(combined)
    # use spearman correlation because the signals do not need to be on the same scale
    correlation = combined[CORRELATION_SIGNALS].corr(method="spearman")
    hard_check(
        "Spearman correlation matrix finite, symmetric, and unit diagonal",
        np.isfinite(correlation.to_numpy(float)).all()
        and np.allclose(correlation, correlation.T, rtol=0.0, atol=1e-12)
        and np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12),
        "all 120 finite validation rows; average ranks for ties; no imputation",
    )
    for signal, group in error_capture.groupby("Signal", sort=False):
        hard_check(
            f"error capture monotonic across bands: {signal}",
            group.sort_values("RiskBandPercent")["CapturedErrors"].is_monotonic_increasing,
            "captured errors must not decrease as the descriptive risk band expands",
        )

    embedded_tests = run_metric_self_checks()
    hard_check(
        "risk-coverage utility synthetic tests pass",
        embedded_tests == 8,
        f"synthetic_cases_passed={embedded_tests}; zero-accepted NaN behavior included",
    )

    # rank signals by mean captured errors across the fixed bands
    # break ties using the 20 percent band, then 10 percent, then signal name
    usefulness_rows = []
    for signal, group in error_capture.groupby("Signal", sort=False):
        indexed = group.set_index("RiskBandPercent")
        usefulness_rows.append(
            {
                "Signal": signal,
                "MeanCapturedErrorsAcrossBands": float(group["CapturedErrors"].mean()),
                "CapturedAt20Percent": int(indexed.loc[20, "CapturedErrors"]),
                "CapturedAt10Percent": int(indexed.loc[10, "CapturedErrors"]),
            }
        )
    usefulness = pd.DataFrame(usefulness_rows).sort_values(
        [
            "MeanCapturedErrorsAcrossBands",
            "CapturedAt20Percent",
            "CapturedAt10Percent",
            "Signal",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    most_useful_signal = str(usefulness.iloc[0]["Signal"])

    # check whether feature atypicality adds information beyond confidence and view disagreement
    feature_correlations = {
        signal: float(
            correlation.loc["PredictedClassReferencePercentile", signal]
        )
        for signal in [
            "PrimaryConfidenceMargin",
            "Day3StdViewProbability",
            "Day3ViewProbabilityRange",
        ]
    }
    exact_framework = "Exact validation-ranked quartiles (30 eyes per signal)"
    feature_unique_errors = int(
        overlap.loc[
            overlap["Framework"].eq(exact_framework)
            & overlap["Component"].eq("Feature unique beyond confidence OR view SD"),
            "CapturedErrors",
        ].iloc[0]
    )
    feature_absolute_correlations = [abs(value) for value in feature_correlations.values()]
    maximum_feature_absolute_correlation = max(feature_absolute_correlations)
    if maximum_feature_absolute_correlation >= DESCRIPTIVE_NEAR_REDUNDANCY_ABS_RHO:
        correlation_interpretation = (
            "meets the prespecified descriptive near-redundancy criterion and "
            "therefore does not appear strongly nonredundant"
        )
    else:
        correlation_interpretation = (
            "does not meet the prespecified descriptive near-redundancy criterion "
            "and therefore appears to retain some nonredundant continuous information"
        )
    if feature_unique_errors == 0:
        overlap_interpretation = (
            "it adds no uniquely captured error beyond confidence or view SD in "
            "the exploratory exact top-quarter bands"
        )
    else:
        overlap_interpretation = (
            f"it uniquely captures {feature_unique_errors} error(s) beyond confidence "
            "or view SD in the exploratory exact top-quarter bands"
        )
    feature_interpretation = (
        "Feature typicality "
        f"{correlation_interpretation} (observed absolute Spearman range "
        f"{min(feature_absolute_correlations):.3f}-"
        f"{maximum_feature_absolute_correlation:.3f}); however, {overlap_interpretation}."
    )

    # rehash every locked input to confirm the analysis did not modify anything
    hashes_after = {str(path): file_sha256(path) for path in EXPECTED_HASHES}
    hard_check(
        "all locked inputs unchanged during analysis",
        hashes_after == input_hashes,
        "all before/after SHA-256 values match",
    )

    # create outputs only after the validation and integrity checks have passed
    run_id = make_run_id()
    prediction_dir = PREDICTION_OUTPUT_ROOT / run_id
    figure_dir = FIGURE_OUTPUT_ROOT / run_id
    audit_dir = AUDIT_OUTPUT_ROOT / run_id
    for directory in [prediction_dir, figure_dir, audit_dir]:
        directory.mkdir(parents=True, exist_ok=False)

    reliability_path = prediction_dir / "val_reliability_signals.csv"
    metadata_path = prediction_dir / "reliability_metadata.json"
    checks_path = audit_dir / "day5_reliability_checks.csv"
    summary_path = audit_dir / "day5_reliability_summary.csv"
    capture_path = audit_dir / "day5_error_capture.csv"
    correlation_path = audit_dir / "day5_signal_correlations.csv"
    overlap_path = audit_dir / "day5_signal_overlap.csv"
    usefulness_path = audit_dir / "day5_signal_usefulness.csv"

    combined.to_csv(reliability_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    error_capture.to_csv(capture_path, index=False, float_format="%.17g")
    correlation.rename_axis("Signal").reset_index().to_csv(
        correlation_path, index=False, float_format="%.17g"
    )
    overlap.to_csv(overlap_path, index=False, float_format="%.17g")
    usefulness.to_csv(usefulness_path, index=False, float_format="%.17g")

    save_group_boxplot(
        combined,
        "PrimaryConfidenceMargin",
        "Confidence margin (log-odds from Day-4 boundary)",
        "Day-4 confidence by validation correctness",
        figure_dir / "confidence_correct_vs_incorrect.png",
    )
    save_group_boxplot(
        combined,
        "Day3StdViewProbability",
        "Day-3 view probability SD (population)",
        "Day-3 cross-view disagreement by Day-4 correctness",
        figure_dir / "view_sd_correct_vs_incorrect.png",
    )
    save_group_boxplot(
        combined,
        "PredictedClassReferencePercentile",
        "Predicted-class reference percentile",
        "Feature atypicality by Day-4 correctness",
        figure_dir / "feature_percentile_correct_vs_incorrect.png",
    )
    save_correlation_plot(
        correlation, figure_dir / "reliability_signal_correlation.png"
    )
    save_confidence_typicality_plot(
        combined,
        figure_dir / "confidence_vs_feature_percentile.png",
    )

    # save the exact input hashes and environment with the final analysis artifacts
    git_commit, git_status = get_git_info()
    metadata = {
        "run_timestamp": run_id,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": float(time.monotonic() - started),
        "objective": "combined validation reliability signals; descriptive analysis only",
        "validation_eyes": len(combined),
        "day4_correct": correct_count,
        "day4_incorrect": incorrect_count,
        "model_retrained": False,
        "day6_implemented": False,
        "routing_thresholds_selected": False,
        "accept_defer_decisions_created": False,
        "test_data_loaded": False,
        "test_data_evaluated": False,
        "test_predictions_loaded": False,
        "locked_runs": {
            "day3_baseline": DAY3_RUN_ID,
            "day4_multiview": DAY4_RUN_ID,
            "feature_typicality": TYPICALITY_RUN_ID,
        },
        "inputs": {
            str(path): {"sha256": input_hashes[str(path)]}
            for path in EXPECTED_HASHES
        },
        "loaded_csv_paths": sorted(map(str, LOADED_CSV_PATHS)),
        "loaded_json_paths": sorted(map(str, LOADED_JSON_PATHS)),
        "confidence": {
            "probability_source": "locked Day-4 EyeAbnormalProbability",
            "probability_clip_epsilon": PROBABILITY_CLIP_EPSILON,
            "clipped_probability_count": clipped_count,
            "primary_logit": "log(clipped p / (1 - clipped p))",
            "threshold_logit": threshold_logit,
            "primary_signed_margin": "PrimaryLogit - ThresholdLogit",
            "primary_confidence_margin": "absolute PrimarySignedMargin",
            "relative_to_probability_0_5": False,
        },
        "cross_view_disagreement": {
            "model": "locked Day-3 per-image ResNet-18",
            "positive_view_threshold": day3_threshold,
            "standard_deviation_ddof": VIEW_STANDARD_DEVIATION_DDOF,
            "single_view_behavior": (
                "SD=0 and range=0; disagreement cannot be meaningfully assessed "
                "from one view"
            ),
        },
        "feature_typicality": {
            "field_source": "FeatureTypicalityPredictedClass",
            "output_alias": "FeatureTypicalityDistance",
            "reference_sets_recomputed": False,
            "normal_reference_eyes": normal_references,
            "abnormal_reference_eyes": abnormal_references,
        },
        "group_summary": {
            "standard_deviation_ddof": SUMMARY_STANDARD_DEVIATION_DDOF,
            "statistical_significance_claimed": False,
        },
        "error_capture": {
            "risk_bands_percent": RISK_BAND_PERCENTAGES,
            "band_size_rule": "ceil(validation eyes * percent / 100)",
            "tie_rule": "risk direction, then EyeExamID ascending; no boundary expansion",
            "validation_outcomes_used_only_to_count_captured_errors": True,
            "routing_threshold_created": False,
            "single_signal_usefulness_rule": (
                "highest mean captured-error count across fixed bands; tie-break "
                "20%, then 10%, then signal name"
            ),
            "most_useful_signal_descriptively": most_useful_signal,
        },
        "overlap": {
            "exact_ranked_quartile_reported": True,
            "literal_reference_percentile_at_least_75_reported": True,
            "feature_unique_errors_in_exact_quartile": feature_unique_errors,
        },
        "correlations": {
            "method": "Spearman rank correlation over all 120 finite validation eyes",
            "tie_ranking": "average ranks",
            "imputation": False,
            "p_values_calculated": False,
            "causal_interpretation": False,
            "descriptive_near_redundancy_absolute_rho": (
                DESCRIPTIVE_NEAR_REDUNDANCY_ABS_RHO
            ),
            "rubric_is_not_a_routing_threshold": True,
            "feature_correlations": feature_correlations,
            "interpretation": feature_interpretation,
        },
        "risk_coverage_utility": {
            "module": str(
                PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
            ),
            "abnormal_is_positive": True,
            "zero_accepted_conditional_rates": "NaN",
            "zero_accepted_false_negative_count": 0,
            "embedded_synthetic_cases_passed": embedded_tests,
            "thresholds_selected": False,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "utility_module_sha256": file_sha256(
            PROJECT_ROOT / "src" / "evaluation" / "reliability_metrics.py"
        ),
        "unit_test_file_sha256": file_sha256(
            PROJECT_ROOT / "tests" / "test_reliability_metrics.py"
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
            "scipy_version": scipy.__version__,
        },
        "artifacts": {
            "val_reliability_signals": str(reliability_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "error_capture": str(capture_path),
            "signal_correlations": str(correlation_path),
            "signal_overlap": str(overlap_path),
            "signal_usefulness": str(usefulness_path),
            "figure_directory": str(figure_dir),
        },
    }
    save_json(metadata_path, metadata)
    pd.DataFrame(checks).to_csv(checks_path, index=False, quoting=csv.QUOTE_MINIMAL)

    overall_summary = summary.loc[summary["PredictionGroup"].eq("Overall")].iloc[0]
    correct_summary = summary.loc[summary["PredictionGroup"].eq("Correct")].iloc[0]
    incorrect_summary = summary.loc[summary["PredictionGroup"].eq("Incorrect")].iloc[0]
    capture_pivot = error_capture.pivot(
        index="Signal", columns="RiskBandPercent", values="CapturedErrors"
    ).loc[[spec[0] for spec in RISK_SPECS]]
    literal_framework = "Literal feature reference percentile >=75 (not a 30-eye quartile)"
    literal_overlap = overlap.loc[overlap["Framework"].eq(literal_framework)]
    literal_triple = literal_overlap.loc[
        literal_overlap["Component"].eq("All three"), "CapturedErrors"
    ].iloc[0]
    literal_union = literal_overlap.loc[
        literal_overlap["Component"].eq("Union of three"), "CapturedErrors"
    ].iloc[0]

    print("\n" + "=" * 92)
    print("DAY-5 COMBINED RELIABILITY SIGNALS COMPLETE")
    print("=" * 92)
    print(f"Validation eyes : {len(combined)}")
    print(f"Correct         : {correct_count}")
    print(f"Incorrect       : {incorrect_count}")
    print("\nCONFIDENCE (PrimaryConfidenceMargin)")
    print(
        "  Correct mean / median   : "
        f"{correct_summary['PrimaryConfidenceMarginMean']:.6f} / "
        f"{correct_summary['PrimaryConfidenceMarginMedian']:.6f}"
    )
    print(
        "  Incorrect mean / median : "
        f"{incorrect_summary['PrimaryConfidenceMarginMean']:.6f} / "
        f"{incorrect_summary['PrimaryConfidenceMarginMedian']:.6f}"
    )
    print("\nVIEW DISAGREEMENT (Day-3 probability SD, ddof=0)")
    print(
        "  Correct mean / median   : "
        f"{correct_summary['Day3StdViewProbabilityMean']:.6f} / "
        f"{correct_summary['Day3StdViewProbabilityMedian']:.6f}"
    )
    print(
        "  Incorrect mean / median : "
        f"{incorrect_summary['Day3StdViewProbabilityMean']:.6f} / "
        f"{incorrect_summary['Day3StdViewProbabilityMedian']:.6f}"
    )
    print("\nMODEL DISAGREEMENT")
    print(
        f"  Overall   : {int(overall_summary['ModelDisagreementCount'])}/{len(combined)} "
        f"({overall_summary['ModelDisagreementPercent']:.2f}%)"
    )
    print(
        f"  Correct   : {int(correct_summary['ModelDisagreementCount'])}/{correct_count} "
        f"({correct_summary['ModelDisagreementPercent']:.2f}%)"
    )
    print(
        f"  Incorrect : {int(incorrect_summary['ModelDisagreementCount'])}/{incorrect_count} "
        f"({incorrect_summary['ModelDisagreementPercent']:.2f}%)"
    )
    print("\nFEATURE TYPICALITY (predicted-class reference percentile)")
    print(
        "  Correct mean / median   : "
        f"{correct_summary['PredictedClassReferencePercentileMean']:.3f} / "
        f"{correct_summary['PredictedClassReferencePercentileMedian']:.3f}"
    )
    print(
        "  Incorrect mean / median : "
        f"{incorrect_summary['PredictedClassReferencePercentileMean']:.3f} / "
        f"{incorrect_summary['PredictedClassReferencePercentileMedian']:.3f}"
    )
    print("\nERROR CAPTURE (errors among highest-risk validation eyes; total errors=10)")
    print(capture_pivot.to_string())
    print(f"\nMost useful single signal descriptively: {most_useful_signal}")
    print(
        "Exploratory literal overlap (low-confidence 25%, high-view-SD 25%, "
        "feature reference percentile >=75): "
        f"all-three={int(literal_triple)}/10, union={int(literal_union)}/10"
    )
    print("\nFEATURE NONREDUNDANCY")
    for signal, coefficient in feature_correlations.items():
        print(f"  Spearman(feature percentile, {signal}) = {coefficient:.6f}")
    print(f"  {feature_interpretation}")
    print(f"\nReliability table : {reliability_path}")
    print(f"Summary           : {summary_path}")
    print(f"Error capture     : {capture_path}")
    print(f"Correlations      : {correlation_path}")
    print(f"Validation checks : {checks_path}")
    print("\nDescriptive validation analysis only; no statistical-significance claim.")
    print("NO ROUTING THRESHOLDS WERE SELECTED.")
    print("TEST SET WAS NOT LOADED OR EVALUATED.")


if __name__ == "__main__":
    try:
        main()
    except (SafetyError, ValueError, KeyError) as error:
        raise SystemExit(f"HARD FAIL: {error}") from error
