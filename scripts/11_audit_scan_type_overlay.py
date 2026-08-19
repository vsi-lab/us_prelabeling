
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as PIL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_ROOT / "metadata" / "Bscan_full_image_manifest_260811.csv"
SCAN_AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits" / "scan_type"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "scan_type"

EXPECTED_FULL_IMAGES = 16_206
EXPECTED_UNKNOWN_IMAGES = 398
EXPECTED_OD_OS_IMAGES = 15_808

ALLOWED_LATERALITY = {"OD", "OS", "UNKNOWN"}
ALLOWED_AUTO_SCAN_TYPES = {
    "A_SCAN_ONLY_CANDIDATE",
    "COMBINED_A_B_CANDIDATE",
    "B_SCAN_ONLY_CANDIDATE",
    "UNCERTAIN",
}

REQUIRED_MANIFEST_COLUMNS = {
    "ResearchSubjectID",
    "EncounterID",
    "Laterality",
    "ImageID",
    "ImageFileName",
    "ImageRelativePath",
    "SHA256",
    "Bytes",
    "Width",
    "Height",
}

DETECTOR_VERSION = "scan_type_overlay_qa_v1"


@dataclass(frozen=True)
class DetectorConfig:

    yellow_hue_min: int = 28
    yellow_hue_max: int = 60
    yellow_saturation_min: int = 64
    yellow_value_min: int = 90
    any_yellow_min_pixels: int = 100
    any_yellow_min_fraction: float = 0.0002

    horizontal_row_min_width_fraction: float = 0.25
    fitted_guide_y_min_fraction: float = 0.20
    fitted_guide_y_max_fraction: float = 0.80
    fitted_guide_x_min_fraction: float = 0.08
    fitted_guide_x_max_fraction: float = 0.92
    fitted_guide_initial_residual_px: float = 4.0
    fitted_guide_final_residual_px: float = 3.0
    fitted_guide_min_initial_columns_fraction: float = 0.25
    fitted_guide_min_support_columns_fraction: float = 0.40
    fitted_guide_min_x_span_fraction: float = 0.50
    fitted_guide_max_abs_slope: float = 0.25
    guide_suppression_radius_px: int = 3

    waveform_roi_y_min_fraction: float = 0.58
    waveform_min_columns_fraction: float = 0.05
    waveform_peak_min_column_span_fraction: float = 0.013
    waveform_min_peak_columns_fraction: float = 0.025
    waveform_min_vertical_extent_fraction: float = 0.067
    waveform_min_pixel_fraction: float = 0.0005

    bscan_roi_y_min_fraction: float = 0.08
    bscan_roi_y_max_fraction: float = 0.65
    bscan_roi_x_min_fraction: float = 0.05
    bscan_roi_x_max_fraction: float = 0.95
    achromatic_channel_spread_max: int = 25
    bscan_midtone_min: int = 12
    bscan_midtone_max: int = 242
    bscan_edge_delta_min: int = 12
    bscan_tile_rows: int = 6
    bscan_tile_columns: int = 8
    bscan_tile_midtone_fraction_min: float = 0.08
    bscan_tile_edge_fraction_min: float = 0.02
    bscan_present_score_min: float = 0.70
    bscan_absent_score_max: float = 0.30

    standalone_white_fraction_min: float = 0.65
    standalone_red_fraction_min: float = 0.001
    standalone_dark_ink_fraction_min: float = 0.01


CONFIG = DetectorConfig()


class AuditFailure(RuntimeError):
    "used when a hard scan-type audit check fails."


def parse_args() -> argparse.Namespace:
    # worker count only affects read-only image analysis speed
    parser = argparse.ArgumentParser(
        description="Audit A-scan overlays and B-scan content in the raw release."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Read-only image-analysis worker threads (default: up to 8).",
    )
    parser.add_argument(
        "--images-per-sheet",
        type=int,
        default=16,
        choices=range(12, 21),
        metavar="12..20",
        help="Number of thumbnails per contact sheet (default: 16).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    expected: Any,
    observed: Any,
    details: str = "",
) -> None:
    checks.append(
        {
            "Category": "HARD CHECK",
            "Check": name,
            "Status": "PASS" if passed else "FAIL",
            "Expected": expected,
            "Observed": observed,
            "Details": details,
        }
    )


def save_checks(checks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(path, index=False)


def stop_on_failed_checks(checks: list[dict[str, Any]], checks_path: Path) -> None:
    failed = [row for row in checks if row["Status"] == "FAIL"]
    if not failed:
        return
    save_checks(checks, checks_path)
    print("\nSCAN-TYPE AUDIT: HARD FAIL")
    for row in failed:
        print(f"[FAIL] {row['Check']}: {row['Details'] or row['Observed']}")
    print(f"Validation checks: {checks_path}")
    raise AuditFailure(f"{len(failed)} hard validation check(s) failed")


def clean_relative_path(value: str) -> PurePosixPath:
    # reject absolute paths and parent traversal before touching the filesystem
    normalized = str(value).strip().replace("\\", "/")
    rel = PurePosixPath(normalized)
    if not normalized or rel.is_absolute() or ".." in rel.parts:
        raise AuditFailure(f"Unsafe ImageRelativePath: {value!r}")
    return rel


def get_source_path(value: str, raw_root_resolved: Path) -> Path:
    rel = clean_relative_path(value)
    candidate = (RAW_ROOT / Path(*rel.parts)).resolve(strict=False)
    if not candidate.is_relative_to(raw_root_resolved):
        raise AuditFailure(f"Image path escapes data/raw: {value!r}")
    return candidate


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    values = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )
    return np.rint(values).astype(np.uint8)


def make_yellow_mask(hsv: np.ndarray) -> np.ndarray:
    return (
        (hsv[:, :, 0] >= CONFIG.yellow_hue_min)
        & (hsv[:, :, 0] <= CONFIG.yellow_hue_max)
        & (hsv[:, :, 1] >= CONFIG.yellow_saturation_min)
        & (hsv[:, :, 2] >= CONFIG.yellow_value_min)
    )


def longest_true_streak(values: np.ndarray) -> int:
    positions = np.flatnonzero(values)
    if positions.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, positions.size - 1]
    return int(np.max(positions[ends] - positions[starts] + 1))


def fit_yellow_guide(
    mask: np.ndarray,
) -> tuple[bool, float, float, int, int, int]:
    """fit a long shallow yellow guide from the topmost yellow pixel in each column."""
    # use only the middle image region where the long yellow guide normally appears
    height, width = mask.shape
    y0 = int(math.floor(CONFIG.fitted_guide_y_min_fraction * height))
    y1 = int(math.ceil(CONFIG.fitted_guide_y_max_fraction * height))
    x0 = int(math.floor(CONFIG.fitted_guide_x_min_fraction * width))
    x1 = int(math.ceil(CONFIG.fitted_guide_x_max_fraction * width))
    region = mask[y0:y1, x0:x1]
    present = region.any(axis=0)
    local_x = np.flatnonzero(present)
    minimum_initial = math.ceil(
        CONFIG.fitted_guide_min_initial_columns_fraction * width
    )
    if local_x.size < minimum_initial:
        return False, 0.0, 0.0, 0, -1, 0

    local_y = np.argmax(region[:, present], axis=0).astype(np.float64)
    x = (local_x + x0).astype(np.float64)
    y = local_y + y0

    def ordinary_least_squares(x_values: np.ndarray, y_values: np.ndarray):
        x_mean = float(x_values.mean())
        y_mean = float(y_values.mean())
        denominator = float(np.sum((x_values - x_mean) ** 2))
        if denominator <= 0.0:
            return 0.0, y_mean
        slope = float(
            np.sum((x_values - x_mean) * (y_values - y_mean)) / denominator
        )
        return slope, y_mean - slope * x_mean

    slope, intercept = ordinary_least_squares(x, y)
    residual = np.abs(y - (slope * x + intercept))
    retained = residual <= CONFIG.fitted_guide_initial_residual_px
    if int(retained.sum()) < minimum_initial:
        return False, 0.0, 0.0, 0, -1, 0

    slope, intercept = ordinary_least_squares(x[retained], y[retained])
    residual = np.abs(y - (slope * x + intercept))
    supported = residual <= CONFIG.fitted_guide_final_residual_px
    supported_x = x[supported]
    support_count = int(supported.sum())
    if support_count == 0:
        return False, 0.0, 0.0, 0, -1, 0
    support_min_x = int(round(float(supported_x.min())))
    support_max_x = int(round(float(supported_x.max())))
    support_span = support_max_x - support_min_x + 1
    detected = (
        support_count
        >= math.ceil(CONFIG.fitted_guide_min_support_columns_fraction * width)
        and support_span
        >= math.ceil(CONFIG.fitted_guide_min_x_span_fraction * width)
        and abs(slope) <= CONFIG.fitted_guide_max_abs_slope
    )
    if not detected:
        return False, slope, intercept, support_min_x, support_max_x, support_count
    return True, slope, intercept, support_min_x, support_max_x, support_count


def measure_overlay_features(yellow: np.ndarray) -> dict[str, Any]:
    # separate long guide lines from the irregular lower waveform evidence
    height, width = yellow.shape
    area = height * width
    count = int(yellow.sum())
    bottom = yellow[height // 2 :, :]
    bottom_count = int(bottom.sum())
    covered_columns = int(yellow.any(axis=0).sum())
    covered_rows = int(yellow.any(axis=1).sum())
    yellow_y = np.flatnonzero(yellow.any(axis=1))
    vertical_extent = (
        int(yellow_y[-1] - yellow_y[0] + 1) if yellow_y.size else 0
    )

    row_line_min = math.ceil(
        CONFIG.horizontal_row_min_width_fraction * width
    )
    row_line_rows: list[int] = []
    for row_index in np.flatnonzero(yellow.sum(axis=1) >= row_line_min):
        if longest_true_streak(yellow[row_index, :]) >= row_line_min:
            row_line_rows.append(int(row_index))

    fitted = fit_yellow_guide(yellow)
    fitted_detected, slope, intercept, fitted_x0, fitted_x1, support_count = fitted
    long_horizontal = bool(row_line_rows or fitted_detected)

    # remove detected guide pixels before measuring the irregular waveform
    residual_mask = yellow.copy()
    radius = CONFIG.guide_suppression_radius_px
    for row_index in row_line_rows:
        residual_mask[
            max(0, row_index - radius) : min(height, row_index + radius + 1), :
        ] = False
    if fitted_detected:
        for x in range(max(0, fitted_x0), min(width - 1, fitted_x1) + 1):
            y = int(round(slope * x + intercept))
            residual_mask[max(0, y - radius) : min(height, y + radius + 1), x] = (
                False
            )

    waveform_y0 = int(math.floor(CONFIG.waveform_roi_y_min_fraction * height))
    waveform = residual_mask[waveform_y0:, :]
    waveform_pixel_count = int(waveform.sum())
    column_counts = waveform.sum(axis=0)
    waveform_columns = int((column_counts > 0).sum())
    if waveform_columns:
        column_min = np.argmax(waveform, axis=0)
        column_max = waveform.shape[0] - 1 - np.argmax(waveform[::-1, :], axis=0)
        present = column_counts > 0
        column_span = column_max - column_min + 1
        peak_columns = int(
            (
                present
                & (
                    column_span
                    >= math.ceil(
                        CONFIG.waveform_peak_min_column_span_fraction * height
                    )
                )
            ).sum()
        )
        waveform_rows = np.flatnonzero(waveform.any(axis=1))
        waveform_extent = int(waveform_rows[-1] - waveform_rows[0] + 1)
    else:
        peak_columns = 0
        waveform_extent = 0

    minimum_waveform_pixels = math.ceil(CONFIG.waveform_min_pixel_fraction * area)
    minimum_waveform_columns = math.ceil(
        CONFIG.waveform_min_columns_fraction * width
    )
    minimum_peak_columns = math.ceil(
        CONFIG.waveform_min_peak_columns_fraction * width
    )
    minimum_vertical_extent = math.ceil(
        CONFIG.waveform_min_vertical_extent_fraction * height
    )
    irregular_waveform = bool(
        waveform_pixel_count >= minimum_waveform_pixels
        and waveform_columns >= minimum_waveform_columns
        and peak_columns >= minimum_peak_columns
        and waveform_extent >= minimum_vertical_extent
    )

    evidence_ratios = [
        waveform_pixel_count / max(1, minimum_waveform_pixels),
        waveform_columns / max(1, minimum_waveform_columns),
        peak_columns / max(1, minimum_peak_columns),
        waveform_extent / max(1, minimum_vertical_extent),
    ]
    waveform_score = float(np.mean(np.clip(evidence_ratios, 0.0, 1.0)))
    any_yellow_threshold = max(
        CONFIG.any_yellow_min_pixels,
        math.ceil(CONFIG.any_yellow_min_fraction * area),
    )

    return {
        "YellowPixelCount": count,
        "YellowPixelFraction": count / area,
        "BottomHalfYellowPixelCount": bottom_count,
        "BottomHalfYellowFraction": bottom_count / bottom.size,
        "YellowColumnsCovered": covered_columns,
        "YellowRowsCovered": covered_rows,
        "YellowVerticalExtent": vertical_extent,
        "LongHorizontalYellowLineDetected": long_horizontal,
        "RowHorizontalYellowLineDetected": bool(row_line_rows),
        "FittedShallowYellowGuideDetected": bool(fitted_detected),
        "FittedYellowGuideSlope": float(slope),
        "FittedYellowGuideSupportColumns": int(support_count),
        "ResidualWaveformPixelCount": waveform_pixel_count,
        "ResidualWaveformColumnsCovered": waveform_columns,
        "ResidualWaveformPeakColumns": peak_columns,
        "ResidualWaveformVerticalExtent": waveform_extent,
        "WaveformEvidenceScore": waveform_score,
        "IrregularYellowWaveformDetected": irregular_waveform,
        "HasAnyYellowOverlay": bool(count >= any_yellow_threshold),
    }


def measure_bscan_features(rgb: np.ndarray, hsv: np.ndarray) -> dict[str, Any]:
    # b-scan evidence comes from grayscale texture spread across the main scan region
    """score whether the image contains substantial b-scan-like texture."""
    height, width = rgb.shape[:2]
    y0 = int(math.floor(CONFIG.bscan_roi_y_min_fraction * height))
    y1 = int(math.ceil(CONFIG.bscan_roi_y_max_fraction * height))
    x0 = int(math.floor(CONFIG.bscan_roi_x_min_fraction * width))
    x1 = int(math.ceil(CONFIG.bscan_roi_x_max_fraction * width))
    roi = rgb[y0:y1, x0:x1]
    gray = rgb_to_luma(roi)
    spread = roi.max(axis=2).astype(np.int16) - roi.min(axis=2).astype(np.int16)
    achromatic = spread <= CONFIG.achromatic_channel_spread_max
    midtone = (
        achromatic
        & (gray >= CONFIG.bscan_midtone_min)
        & (gray <= CONFIG.bscan_midtone_max)
    )

    gray_i16 = gray.astype(np.int16)
    delta_x = np.zeros_like(gray_i16)
    delta_y = np.zeros_like(gray_i16)
    delta_x[:, 1:] = np.abs(np.diff(gray_i16, axis=1))
    delta_y[1:, :] = np.abs(np.diff(gray_i16, axis=0))
    edge = midtone & (
        np.maximum(delta_x, delta_y) >= CONFIG.bscan_edge_delta_min
    )

    midtone_fraction = float(midtone.mean())
    edge_fraction = float(edge.mean())
    dark_fraction = float((achromatic & (gray < 18)).mean())
    white_fraction_roi = float((achromatic & (gray > 242)).mean())

    # require texture in multiple tiles instead of one small bright or noisy patch
    occupied_tiles = 0
    total_tiles = CONFIG.bscan_tile_rows * CONFIG.bscan_tile_columns
    roi_height, roi_width = gray.shape
    for tile_y in range(CONFIG.bscan_tile_rows):
        tile_y0 = tile_y * roi_height // CONFIG.bscan_tile_rows
        tile_y1 = (tile_y + 1) * roi_height // CONFIG.bscan_tile_rows
        for tile_x in range(CONFIG.bscan_tile_columns):
            tile_x0 = tile_x * roi_width // CONFIG.bscan_tile_columns
            tile_x1 = (tile_x + 1) * roi_width // CONFIG.bscan_tile_columns
            if (
                float(midtone[tile_y0:tile_y1, tile_x0:tile_x1].mean())
                >= CONFIG.bscan_tile_midtone_fraction_min
                and float(edge[tile_y0:tile_y1, tile_x0:tile_x1].mean())
                >= CONFIG.bscan_tile_edge_fraction_min
            ):
                occupied_tiles += 1
    tile_fraction = occupied_tiles / total_tiles

    edge_y, edge_x = np.nonzero(edge)
    if edge_y.size:
        bbox_area = int(
            (edge_y.max() - edge_y.min() + 1)
            * (edge_x.max() - edge_x.min() + 1)
        )
        spatial_extent = bbox_area / edge.size
    else:
        spatial_extent = 0.0

    # combine several independent b-scan clues
    # clipping each component keeps one strong feature from dominating the score
    component_midtone = float(np.clip(midtone_fraction / 0.12, 0.0, 1.0))
    component_edge = float(np.clip(edge_fraction / 0.06, 0.0, 1.0))
    component_tiles = float(np.clip(tile_fraction / 0.25, 0.0, 1.0))
    component_extent = float(np.clip(spatial_extent / 0.30, 0.0, 1.0))
    component_dark = float(np.clip((dark_fraction - 0.15) / 0.35, 0.0, 1.0))
    score = (
        0.28 * component_midtone
        + 0.27 * component_edge
        + 0.25 * component_tiles
        + 0.10 * component_extent
        + 0.10 * component_dark
    )
    has_bscan = bool(score >= CONFIG.bscan_present_score_min)
    uncertain = bool(
        CONFIG.bscan_absent_score_max < score < CONFIG.bscan_present_score_min
    )

    full_gray = rgb_to_luma(rgb)
    full_spread = (
        rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    )
    full_achromatic = full_spread <= 20
    white_background_fraction = float(
        (full_achromatic & (full_gray >= 242)).mean()
    )
    dark_ink_fraction = float((full_achromatic & (full_gray <= 30)).mean())
    red = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 244))
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 80)
    )
    red_fraction = float(red.mean())
    standalone_ascan = bool(
        white_background_fraction >= CONFIG.standalone_white_fraction_min
        and red_fraction >= CONFIG.standalone_red_fraction_min
        and dark_ink_fraction >= CONFIG.standalone_dark_ink_fraction_min
    )

    return {
        "BscanContentScore": float(score),
        "HasBscanContent": has_bscan,
        "BscanContentUncertain": uncertain,
        "BscanMidtoneFraction": midtone_fraction,
        "BscanEdgeDensity": edge_fraction,
        "BscanTexturedTileFraction": float(tile_fraction),
        "BscanTextureSpatialExtent": float(spatial_extent),
        "BscanDarkBackgroundFraction": dark_fraction,
        "BscanWhiteBackgroundFractionROI": white_fraction_roi,
        "FullImageWhiteBackgroundFraction": white_background_fraction,
        "FullImageDarkInkFraction": dark_ink_fraction,
        "RedPlotOverlayFraction": red_fraction,
        "WhiteCanvasAscanPlotDetected": standalone_ascan,
    }


def classify_scan_candidate(row: dict[str, Any] | pd.Series) -> str:
    # automatic classes are screening candidates and always remain reviewable
    """assign a screening class from image features only."""
    has_bscan = bool(row["HasBscanContent"])
    bscan_uncertain = bool(row["BscanContentUncertain"])
    yellow_waveform = bool(row["IrregularYellowWaveformDetected"])
    standalone_ascan = bool(row["WhiteCanvasAscanPlotDetected"])

    if bscan_uncertain or (has_bscan and standalone_ascan):
        return "UNCERTAIN"
    if has_bscan and yellow_waveform:
        return "COMBINED_A_B_CANDIDATE"
    if has_bscan and not yellow_waveform:
        return "B_SCAN_ONLY_CANDIDATE"
    if not has_bscan and (yellow_waveform or standalone_ascan):
        return "A_SCAN_ONLY_CANDIDATE"
    return "UNCERTAIN"


def analyze_source_image(task: dict[str, Any]) -> dict[str, Any]:
    # laterality and other labels are intentionally absent from this task payload
    """read one source image and extract pixel-based scan features."""
    result: dict[str, Any] = {
        "ManifestRowNumber": task["ManifestRowNumber"],
        "ImageRelativePath": task["ImageRelativePath"],
        "AnalysisInvocationCount": 1,
        "Readable": False,
        "ProcessingError": "",
    }
    try:
        path = Path(task["ResolvedPath"])
        image_bytes = path.read_bytes()
        observed_sha = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        result.update(
            {
                "Readable": True,
                "ObservedSHA256": observed_sha,
                "ObservedBytes": len(image_bytes),
                "ObservedWidth": width,
                "ObservedHeight": height,
            }
        )
        result.update(measure_overlay_features(make_yellow_mask(hsv)))
        result.update(measure_bscan_features(rgb, hsv))
    except Exception as exc:  # keep the per-image error so the failed source can be reviewed later
        result["ProcessingError"] = f"{type(exc).__name__}: {exc}"
    return result


def make_classification_signature(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("ImageRelativePath", kind="mergesort")
    text = "".join(
        f"{row.ImageRelativePath}\t{row.AutoScanType}\n"
        for row in ordered.itertuples(index=False)
    )
    return text_sha256(text)


def find_previous_matching_run(
    current_run: Path, manifest_sha256: str
) -> tuple[Path | None, pd.DataFrame | None]:
    # only compare against a run that used the same detector version, manifest, and config
    if not SCAN_AUDIT_ROOT.exists():
        return None, None
    for directory in sorted(SCAN_AUDIT_ROOT.iterdir(), reverse=True):
        if directory == current_run or not directory.is_dir():
            continue
        metadata_path = directory / "scan_type_method.json"
        diagnostics_path = directory / "all_scan_type_diagnostics.csv"
        if not metadata_path.is_file() or not diagnostics_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if (
                metadata.get("detector_version") != DETECTOR_VERSION
                or metadata.get("input_manifest", {}).get("sha256")
                != manifest_sha256
                or metadata.get("configuration") != asdict(CONFIG)
            ):
                continue
            previous = pd.read_csv(
                diagnostics_path,
                dtype=str,
                keep_default_na=False,
                usecols=["ImageRelativePath", "AutoScanType"],
            )
            return directory, previous
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return None, None


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_contact_sheets(
    frame: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    images_per_sheet: int,
    run_id: str,
) -> tuple[dict[str, tuple[str, int, int]], list[Path]]:
    """create deterministic contact sheets for manual review."""
    # contact sheets are generated in a fixed order so manual review is reproducible
    output_dir.mkdir(parents=True, exist_ok=False)
    ordered = frame.sort_values("ImageRelativePath", kind="mergesort").reset_index(
        drop=True
    )
    if ordered.empty:
        return {}, []

    columns = 4
    rows = math.ceil(images_per_sheet / columns)
    thumb_width, thumb_height = 320, 240
    label_height, header_height = 78, 38
    tile_height = thumb_height + label_height
    sheet_width = columns * thumb_width
    sheet_height = header_height + rows * tile_height
    font = get_font(13)
    header_font = get_font(18, bold=True)
    mapping: dict[str, tuple[str, int, int]] = {}
    generated: list[Path] = []
    page_count = math.ceil(len(ordered) / images_per_sheet)

    for page_zero in range(page_count):
        page_number = page_zero + 1
        page = Image.new("RGB", (sheet_width, sheet_height), color=(24, 24, 24))
        draw = ImageDraw.Draw(page)
        draw.text(
            (10, 8),
            f"Scan-type QA | {prefix} | page {page_number}/{page_count} | {run_id}",
            fill=(255, 255, 255),
            font=header_font,
        )
        start = page_zero * images_per_sheet
        stop = min(len(ordered), start + images_per_sheet)
        for page_slot, row in enumerate(
            ordered.iloc[start:stop].itertuples(index=False), start=1
        ):
            zero_slot = page_slot - 1
            tile_row, tile_column = divmod(zero_slot, columns)
            left = tile_column * thumb_width
            top = header_height + tile_row * tile_height
            source_path = get_source_path(
                row.ImageRelativePath, RAW_ROOT.resolve()
            )
            with Image.open(source_path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
            thumb = ImageOps.contain(
                image,
                (thumb_width, thumb_height),
                method=Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGB", (thumb_width, thumb_height), (0, 0, 0))
            canvas.paste(
                thumb,
                ((thumb_width - thumb.width) // 2, (thumb_height - thumb.height) // 2),
            )
            page.paste(canvas, (left, top))
            label_top = top + thumb_height
            draw.rectangle(
                (left, label_top, left + thumb_width - 1, label_top + label_height - 1),
                fill=(8, 8, 8),
                outline=(90, 90, 90),
            )
            lines = [
                f"Image: {row.ImageID}",
                f"Subject: {row.ResearchSubjectID}",
                f"Encounter: {row.EncounterID}",
                f"AUTO: {row.AutoScanType}",
            ]
            for line_number, line in enumerate(lines):
                color = (255, 218, 80) if line_number == 3 else (245, 245, 245)
                draw.text(
                    (left + 6, label_top + 3 + 18 * line_number),
                    line,
                    fill=color,
                    font=font,
                )
            mapping[row.ImageRelativePath] = (
                f"{prefix}_{page_number:03d}.png",
                page_number,
                page_slot,
            )
        output_path = output_dir / f"{prefix}_{page_number:03d}.png"
        page.save(output_path, format="PNG", optimize=False)
        generated.append(output_path)
    return mapping, generated


def pick_debug_examples(frame: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    # choose representative examples instead of random images
    """pick a few deterministic examples for checking the detector masks."""
    selections: list[pd.DataFrame] = []

    waveform = frame.loc[frame["IrregularYellowWaveformDetected"]].copy()
    if not waveform.empty:
        median_score = float(waveform["WaveformEvidenceScore"].median())
        waveform["_rank"] = (
            waveform["WaveformEvidenceScore"] - median_score
        ).abs()
        chosen = waveform.sort_values(
            ["_rank", "ImageRelativePath"], kind="mergesort"
        ).head(limit)
        chosen = chosen.assign(DebugCategory="waveform_detected")
        selections.append(chosen)

    no_waveform = frame.loc[
        (~frame["IrregularYellowWaveformDetected"])
        & frame["HasBscanContent"]
    ].copy()
    if no_waveform.empty:
        no_waveform = frame.loc[~frame["IrregularYellowWaveformDetected"]].copy()
    if not no_waveform.empty:
        # include straight-guide examples and one strong b-scan with almost no yellow
        # this makes it easier to inspect both sides of the waveform decision
        guide_examples = no_waveform.loc[
            no_waveform["LongHorizontalYellowLineDetected"]
            | no_waveform["HasAnyYellowOverlay"]
        ].sort_values(
            ["YellowPixelCount", "WaveformEvidenceScore", "ImageRelativePath"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(limit)
        zero_yellow_examples = no_waveform.loc[
            ~no_waveform["HasAnyYellowOverlay"]
        ].sort_values(
            ["BscanContentScore", "ImageRelativePath"],
            ascending=[False, True],
            kind="mergesort",
        ).head(1)
        chosen = (
            pd.concat([guide_examples, zero_yellow_examples], ignore_index=False)
            .drop_duplicates("ImageRelativePath", keep="first")
            .reset_index(drop=True)
        )
        chosen = chosen.assign(DebugCategory="no_waveform_detected")
        selections.append(chosen)

    uncertain = frame.loc[frame["AutoScanType"].eq("UNCERTAIN")].copy()
    if not uncertain.empty:
        uncertain["_rank"] = (
            uncertain["BscanContentScore"]
            - (
                CONFIG.bscan_present_score_min
                + CONFIG.bscan_absent_score_max
            )
            / 2.0
        ).abs()
        chosen = uncertain.sort_values(
            ["_rank", "ImageRelativePath"], kind="mergesort"
        ).head(limit)
        chosen = chosen.assign(DebugCategory="uncertain")
        selections.append(chosen)

    if not selections:
        return frame.head(0).copy()
    return pd.concat(selections, ignore_index=True)


def save_debug_examples(
    frame: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = pick_debug_examples(frame)
    index_rows: list[dict[str, Any]] = []
    generated: list[Path] = []
    category_counts = {
        "waveform_detected": int(frame["IrregularYellowWaveformDetected"].sum()),
        "no_waveform_detected": int((~frame["IrregularYellowWaveformDetected"]).sum()),
        "uncertain": int(frame["AutoScanType"].eq("UNCERTAIN").sum()),
    }
    for category, count in category_counts.items():
        if count == 0:
            index_rows.append(
                {
                    "DebugCategory": category,
                    "ImageRelativePath": "",
                    "AutoScanType": "",
                    "OriginalDebugFile": "",
                    "YellowMaskDebugFile": "",
                    "Note": "No images in this automatic category.",
                }
            )

    category_sequence: dict[str, int] = {}
    for row in selected.itertuples(index=False):
        category = row.DebugCategory
        sequence = category_sequence.get(category, 0) + 1
        category_sequence[category] = sequence
        stem = f"{category}_{sequence:03d}"
        original_name = f"{stem}_original.png"
        mask_name = f"{stem}_yellow_mask.png"
        path = get_source_path(row.ImageRelativePath, RAW_ROOT.resolve())
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
        mask = make_yellow_mask(hsv)
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[mask] = np.array([255, 230, 0], dtype=np.uint8)
        image.save(output_dir / original_name, format="PNG", optimize=False)
        Image.fromarray(mask_rgb, mode="RGB").save(
            output_dir / mask_name, format="PNG", optimize=False
        )
        generated.extend([output_dir / original_name, output_dir / mask_name])
        index_rows.append(
            {
                "DebugCategory": category,
                "ImageRelativePath": row.ImageRelativePath,
                "AutoScanType": row.AutoScanType,
                "OriginalDebugFile": original_name,
                "YellowMaskDebugFile": mask_name,
                "Note": "AUTO screening example; not manually confirmed.",
            }
        )

    index_frame = pd.DataFrame(index_rows)
    index_path = output_dir / "debug_mask_index.csv"
    index_frame.to_csv(index_path, index=False)
    generated.append(index_path)
    return index_frame, generated


def attach_contact_sheet_locations(
    frame: pd.DataFrame, mapping: dict[str, tuple[str, int, int]]
) -> pd.DataFrame:
    result = frame.copy()
    result["ContactSheetFile"] = result["ImageRelativePath"].map(
        lambda value: mapping.get(value, ("", "", ""))[0]
    )
    result["ContactSheetPage"] = result["ImageRelativePath"].map(
        lambda value: mapping.get(value, ("", "", ""))[1]
    )
    result["ContactSheetSlot"] = result["ImageRelativePath"].map(
        lambda value: mapping.get(value, ("", "", ""))[2]
    )
    return result


def snapshot_source_files(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    # size and modification time are enough to catch accidental source writes here
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    }


def to_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    # this is a read-only qa pass over the raw release
    args = parse_args()
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    scan_audit_dir = SCAN_AUDIT_ROOT / run_id
    validation_dir = AUDIT_ROOT / run_id
    figure_dir = FIGURE_ROOT / run_id
    checks_path = validation_dir / "scan_type_audit_checks.csv"
    scan_audit_dir.mkdir(parents=True, exist_ok=False)
    validation_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)

    checks: list[dict[str, Any]] = []
    print("=" * 78)
    print("SCAN-TYPE / OVERLAY DATASET QA")
    print("=" * 78)
    print(f"Run ID             : {run_id}")
    print(f"Input manifest     : {MANIFEST_PATH}")
    print(f"Scan audit output  : {scan_audit_dir}")
    print(f"Figure output      : {figure_dir}")
    print(f"Worker threads     : {args.workers}\n")

    record_check(
        checks,
        "full manifest exists",
        MANIFEST_PATH.is_file(),
        True,
        MANIFEST_PATH.is_file(),
        str(MANIFEST_PATH),
    )
    stop_on_failed_checks(checks, checks_path)

    # hash the manifest before analysis so we can prove it was not modified
    manifest_sha_before = file_sha256(MANIFEST_PATH)
    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    record_check(
        checks,
        "required manifest columns are present",
        not missing_columns,
        "none missing",
        missing_columns,
        f"missing={missing_columns}",
    )
    stop_on_failed_checks(checks, checks_path)

    manifest["Laterality"] = manifest["Laterality"].str.strip().str.upper()
    full_count = len(manifest)
    unknown_count = int(manifest["Laterality"].eq("UNKNOWN").sum())
    od_os_count = int(manifest["Laterality"].isin(["OD", "OS"]).sum())
    record_check(
        checks,
        "full manifest rows == 16,206",
        full_count == EXPECTED_FULL_IMAGES,
        EXPECTED_FULL_IMAGES,
        full_count,
    )
    record_check(
        checks,
        "UNKNOWN rows == 398",
        unknown_count == EXPECTED_UNKNOWN_IMAGES,
        EXPECTED_UNKNOWN_IMAGES,
        unknown_count,
    )
    record_check(
        checks,
        "OD/OS rows == 15,808",
        od_os_count == EXPECTED_OD_OS_IMAGES,
        EXPECTED_OD_OS_IMAGES,
        od_os_count,
    )
    observed_laterality = sorted(manifest["Laterality"].unique().tolist())
    record_check(
        checks,
        "manifest Laterality values are allowed",
        set(observed_laterality) <= ALLOWED_LATERALITY,
        sorted(ALLOWED_LATERALITY),
        observed_laterality,
    )
    blank_core = manifest[
        [
            "ResearchSubjectID",
            "EncounterID",
            "ImageID",
            "ImageRelativePath",
            "SHA256",
        ]
    ].apply(lambda column: column.str.strip().eq(""))
    record_check(
        checks,
        "manifest identifiers, paths, and hashes are nonblank",
        not bool(blank_core.any(axis=None)),
        0,
        int(blank_core.any(axis=1).sum()),
    )
    duplicate_paths = int(manifest["ImageRelativePath"].duplicated(keep=False).sum())
    record_check(
        checks,
        "ImageRelativePath is unique",
        duplicate_paths == 0,
        0,
        duplicate_paths,
        "Each manifest row must invoke image analysis exactly once.",
    )
    stop_on_failed_checks(checks, checks_path)

    raw_root_resolved = RAW_ROOT.resolve()
    resolved_paths: list[Path] = []
    path_errors: list[str] = []
    for value in manifest["ImageRelativePath"]:
        try:
            resolved_paths.append(get_source_path(value, raw_root_resolved))
        except AuditFailure as exc:
            path_errors.append(str(exc))
            resolved_paths.append(Path("__INVALID__"))
    missing_paths = [
        str(path) for path in resolved_paths if not path.is_file()
    ]
    record_check(
        checks,
        "all ImageRelativePath values remain under data/raw",
        not path_errors,
        0,
        len(path_errors),
        "; ".join(path_errors[:5]),
    )
    record_check(
        checks,
        "every manifest image exists",
        not missing_paths,
        0,
        len(missing_paths),
        "; ".join(missing_paths[:5]),
    )
    stop_on_failed_checks(checks, checks_path)

    # snapshot every source image before the threaded pixel analysis starts
    before_snapshot = snapshot_source_files(resolved_paths)
    # pass only image location and row identity into the pixel detector
    tasks = [
        {
            "ManifestRowNumber": int(index),
            "ImageRelativePath": manifest.iloc[index]["ImageRelativePath"],
            "ResolvedPath": str(resolved_paths[index]),
        }
        for index in range(full_count)
    ]

    print("Analyzing all source images from pixels (Laterality is not supplied to detector)...")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(
            executor.map(analyze_source_image, tasks, chunksize=16), start=1
        ):
            results.append(result)
            if completed % 1_000 == 0 or completed == full_count:
                print(f"  processed {completed:,}/{full_count:,}")

    # restore manifest row order after threaded processing
    features = pd.DataFrame(results).sort_values(
        "ManifestRowNumber", kind="mergesort"
    ).reset_index(drop=True)
    unreadable = features.loc[~features["Readable"].fillna(False)].copy()
    unreadable.to_csv(scan_audit_dir / "unreadable_images.csv", index=False)
    record_check(
        checks,
        "every image is readable",
        unreadable.empty,
        0,
        len(unreadable),
        "; ".join(unreadable["ProcessingError"].head(5).tolist()),
    )
    invocation_ok = (
        len(features) == full_count
        and features["ManifestRowNumber"].nunique() == full_count
        and features["AnalysisInvocationCount"].eq(1).all()
    )
    record_check(
        checks,
        "every image is processed exactly once",
        bool(invocation_ok),
        full_count,
        int(features["ManifestRowNumber"].nunique()),
    )
    stop_on_failed_checks(checks, checks_path)

    expected_sha = manifest["SHA256"].str.strip().str.lower()
    observed_sha = features["ObservedSHA256"].str.strip().str.lower()
    sha_match = expected_sha.eq(observed_sha)
    record_check(
        checks,
        "decoded source bytes match manifest SHA-256",
        bool(sha_match.all()),
        full_count,
        int(sha_match.sum()),
        f"mismatches={int((~sha_match).sum())}",
    )
    expected_width = pd.to_numeric(manifest["Width"], errors="coerce")
    expected_height = pd.to_numeric(manifest["Height"], errors="coerce")
    expected_bytes = pd.to_numeric(manifest["Bytes"], errors="coerce")
    dimension_match = expected_width.eq(features["ObservedWidth"]) & expected_height.eq(
        features["ObservedHeight"]
    )
    bytes_match = expected_bytes.eq(features["ObservedBytes"])
    record_check(
        checks,
        "decoded image dimensions match manifest",
        bool(dimension_match.all()),
        full_count,
        int(dimension_match.sum()),
        f"mismatches={int((~dimension_match).sum())}",
    )
    record_check(
        checks,
        "source byte sizes match manifest",
        bool(bytes_match.all()),
        full_count,
        int(bytes_match.sum()),
        f"mismatches={int((~bytes_match).sum())}",
    )
    stop_on_failed_checks(checks, checks_path)

    # classify the same extracted features twice to catch any hidden randomness
    # a later check also compares the result with the most recent compatible run
    replay_one = [classify_scan_candidate(row) for _, row in features.iterrows()]
    replay_two = [classify_scan_candidate(row) for _, row in features.iterrows()]
    record_check(
        checks,
        "deterministic in-run automatic classification replay",
        replay_one == replay_two,
        True,
        replay_one == replay_two,
        "The same features were classified twice with no random state.",
    )
    features["AutoScanType"] = replay_one
    allowed_classes_ok = set(features["AutoScanType"].unique()) <= ALLOWED_AUTO_SCAN_TYPES
    record_check(
        checks,
        "classifications only use documented allowed values",
        allowed_classes_ok,
        sorted(ALLOWED_AUTO_SCAN_TYPES),
        sorted(features["AutoScanType"].unique().tolist()),
    )
    stop_on_failed_checks(checks, checks_path)

    # join manifest metadata only after pixel features and classes are finished
    # this keeps laterality out of the detector itself
    identity_columns = [
        "ResearchSubjectID",
        "EncounterID",
        "ImageID",
        "ImageFileName",
        "ImageRelativePath",
        "Laterality",
        "SHA256",
        "Bytes",
        "Width",
        "Height",
    ]
    combined = pd.concat(
        [
            manifest[identity_columns].reset_index(drop=True),
            features.drop(columns=["ImageRelativePath"]).reset_index(drop=True),
        ],
        axis=1,
    )
    combined["HasAscanWaveformEvidence"] = (
        combined["IrregularYellowWaveformDetected"]
        | combined["WhiteCanvasAscanPlotDetected"]
    )
    combined["ManualReviewRequired"] = (
        combined["Laterality"].eq("UNKNOWN")
        | combined["AutoScanType"].isin(["B_SCAN_ONLY_CANDIDATE", "UNCERTAIN"])
    )
    combined = combined.sort_values("ImageRelativePath", kind="mergesort").reset_index(
        drop=True
    )

    signature = make_classification_signature(combined)
    previous_dir, previous = find_previous_matching_run(
        scan_audit_dir, manifest_sha_before
    )
    if previous is not None:
        current_mapping = combined[["ImageRelativePath", "AutoScanType"]].sort_values(
            "ImageRelativePath", kind="mergesort"
        ).reset_index(drop=True)
        previous_mapping = previous.sort_values(
            "ImageRelativePath", kind="mergesort"
        ).reset_index(drop=True)
        prior_equal = current_mapping.equals(previous_mapping)
        record_check(
            checks,
            "repeat run reproduces image-level automatic classifications",
            prior_equal,
            True,
            prior_equal,
            f"previous_compatible_run={previous_dir}",
        )
    else:
        record_check(
            checks,
            "repeat run reproducibility signature is recorded",
            True,
            "deterministic signature",
            signature,
            "No earlier compatible run was available; in-run replay passed. Re-run will compare exact mappings.",
        )
    stop_on_failed_checks(checks, checks_path)

    # laterality is used only now, after all automatic image classes are fixed
    unknown = combined.loc[combined["Laterality"].eq("UNKNOWN")].copy()
    od_os = combined.loc[combined["Laterality"].isin(["OD", "OS"])].copy()
    # queue od/os images that look like b-scans but lack irregular waveform evidence
    candidates = od_os.loc[
        od_os["HasBscanContent"]
        & (~od_os["IrregularYellowWaveformDetected"])
    ].copy()

    unknown_required_columns = [
        "ResearchSubjectID",
        "EncounterID",
        "ImageID",
        "ImageRelativePath",
        "Laterality",
        "HasAnyYellowOverlay",
        "LongHorizontalYellowLineDetected",
        "IrregularYellowWaveformDetected",
        "WhiteCanvasAscanPlotDetected",
        "HasAscanWaveformEvidence",
        "HasBscanContent",
        "BscanContentUncertain",
        "BscanContentScore",
        "YellowPixelCount",
        "YellowPixelFraction",
        "BottomHalfYellowPixelCount",
        "BottomHalfYellowFraction",
        "YellowColumnsCovered",
        "YellowRowsCovered",
        "YellowVerticalExtent",
        "AutoScanType",
        "ManualReviewRequired",
    ]
    diagnostic_columns = [
        "RowHorizontalYellowLineDetected",
        "FittedShallowYellowGuideDetected",
        "FittedYellowGuideSlope",
        "FittedYellowGuideSupportColumns",
        "ResidualWaveformPixelCount",
        "ResidualWaveformColumnsCovered",
        "ResidualWaveformPeakColumns",
        "ResidualWaveformVerticalExtent",
        "WaveformEvidenceScore",
        "BscanMidtoneFraction",
        "BscanEdgeDensity",
        "BscanTexturedTileFraction",
        "BscanTextureSpatialExtent",
        "BscanDarkBackgroundFraction",
        "BscanWhiteBackgroundFractionROI",
        "FullImageWhiteBackgroundFraction",
        "FullImageDarkInkFraction",
        "RedPlotOverlayFraction",
    ]
    unknown_output = unknown[unknown_required_columns + diagnostic_columns].copy()
    unknown_path = scan_audit_dir / "unknown_scan_type_audit.csv"
    unknown_output.to_csv(unknown_path, index=False, float_format="%.17g")

    candidate_columns = [
        "ResearchSubjectID",
        "EncounterID",
        "ImageID",
        "ImageRelativePath",
        "Laterality",
        "HasAnyYellowOverlay",
        "LongHorizontalYellowLineDetected",
        "IrregularYellowWaveformDetected",
        "HasBscanContent",
        "BscanContentScore",
        "YellowPixelCount",
        "YellowPixelFraction",
        "BottomHalfYellowFraction",
        "YellowVerticalExtent",
        "AutoScanType",
        "ManualReviewRequired",
    ] + diagnostic_columns
    candidate_output = candidates[candidate_columns].copy()
    candidate_path = scan_audit_dir / "bscan_without_waveform_candidates.csv"
    candidate_output.to_csv(candidate_path, index=False, float_format="%.17g")

    all_diagnostics_path = scan_audit_dir / "all_scan_type_diagnostics.csv"
    combined.to_csv(all_diagnostics_path, index=False, float_format="%.17g")

    # generate review material separately for unknown scans and b-scan-only candidates
    unknown_contact_dir = figure_dir / "unknown_contact_sheets"
    bscan_contact_dir = figure_dir / "bscan_only_candidates"
    debug_dir = figure_dir / "debug_masks"
    print("Generating UNKNOWN contact sheets...")
    unknown_mapping, unknown_sheets = create_contact_sheets(
        unknown,
        unknown_contact_dir,
        "unknown",
        args.images_per_sheet,
        run_id,
    )
    print("Generating B-scan-only candidate contact sheets...")
    candidate_mapping, candidate_sheets = create_contact_sheets(
        candidates,
        bscan_contact_dir,
        "bscan_only",
        args.images_per_sheet,
        run_id,
    )

    unknown_manual = attach_contact_sheet_locations(
        unknown[
            [
                "ImageRelativePath",
                "ImageID",
                "ResearchSubjectID",
                "EncounterID",
                "Laterality",
                "AutoScanType",
            ]
        ],
        unknown_mapping,
    )
    unknown_manual["ManualScanType"] = ""
    unknown_manual["ReviewerNotes"] = ""
    unknown_manual_path = scan_audit_dir / "unknown_manual_review.csv"
    unknown_manual.to_csv(unknown_manual_path, index=False)

    candidate_manual = attach_contact_sheet_locations(
        candidates[
            [
                "ImageRelativePath",
                "ImageID",
                "ResearchSubjectID",
                "EncounterID",
                "Laterality",
                "AutoScanType",
            ]
        ],
        candidate_mapping,
    )
    candidate_manual["ManualConfirmedBscanOnly"] = ""
    candidate_manual["ReviewerNotes"] = ""
    candidate_manual_path = scan_audit_dir / "bscan_only_manual_review.csv"
    candidate_manual.to_csv(candidate_manual_path, index=False)

    _, debug_files = save_debug_examples(combined, debug_dir)

    unknown_set_ok = set(unknown_output["ImageRelativePath"]) == set(
        manifest.loc[manifest["Laterality"].eq("UNKNOWN"), "ImageRelativePath"]
    )
    record_check(
        checks,
        "every UNKNOWN image appears in unknown_scan_type_audit.csv",
        unknown_set_ok and len(unknown_output) == EXPECTED_UNKNOWN_IMAGES,
        EXPECTED_UNKNOWN_IMAGES,
        len(unknown_output),
    )
    expected_candidate_paths = set(
        od_os.loc[
            od_os["HasBscanContent"]
            & (~od_os["IrregularYellowWaveformDetected"]),
            "ImageRelativePath",
        ]
    )
    candidate_set_ok = set(candidate_output["ImageRelativePath"]) == expected_candidate_paths
    record_check(
        checks,
        "every B-scan-only screen candidate appears in candidate CSV",
        candidate_set_ok,
        len(expected_candidate_paths),
        len(candidate_output),
    )
    record_check(
        checks,
        "UNKNOWN contact sheets cover all 398 images",
        len(unknown_mapping) == EXPECTED_UNKNOWN_IMAGES,
        EXPECTED_UNKNOWN_IMAGES,
        len(unknown_mapping),
        f"sheets={len(unknown_sheets)}",
    )
    record_check(
        checks,
        "B-scan-only contact sheets cover every candidate",
        len(candidate_mapping) == len(candidates),
        len(candidates),
        len(candidate_mapping),
        f"sheets={len(candidate_sheets)}",
    )
    manual_blank_ok = (
        unknown_manual["ManualScanType"].eq("").all()
        and unknown_manual["ReviewerNotes"].eq("").all()
        and candidate_manual["ManualConfirmedBscanOnly"].eq("").all()
        and candidate_manual["ReviewerNotes"].eq("").all()
    )
    record_check(
        checks,
        "manual-review decision fields are blank",
        bool(manual_blank_ok),
        True,
        bool(manual_blank_ok),
    )

    # verify that neither source images nor the manifest changed during the audit
    after_snapshot = snapshot_source_files(resolved_paths)
    manifest_sha_after = file_sha256(MANIFEST_PATH)
    source_unchanged = before_snapshot == after_snapshot
    record_check(
        checks,
        "no source image is modified",
        source_unchanged,
        True,
        source_unchanged,
        "Compared byte size and nanosecond mtime for every source image before and after processing.",
    )
    record_check(
        checks,
        "raw manifest is unchanged",
        manifest_sha_before == manifest_sha_after,
        manifest_sha_before,
        manifest_sha_after,
    )

    # keep the summary tables flat so they are easy to filter and review
    unknown_class_counts = unknown["AutoScanType"].value_counts().to_dict()
    summary_rows = [
        {"Group": "FULL DATASET", "Metric": "TotalImages", "Value": full_count},
        {"Group": "UNKNOWN", "Metric": "TotalImages", "Value": len(unknown)},
    ]
    for class_name in sorted(ALLOWED_AUTO_SCAN_TYPES):
        summary_rows.append(
            {
                "Group": "UNKNOWN",
                "Metric": class_name,
                "Value": int(unknown_class_counts.get(class_name, 0)),
            }
        )
    summary_rows.extend(
        [
            {"Group": "OD_OS", "Metric": "TotalImages", "Value": len(od_os)},
            {
                "Group": "OD_OS",
                "Metric": "WaveformDetected",
                "Value": int(od_os["IrregularYellowWaveformDetected"].sum()),
            },
            {
                "Group": "OD_OS",
                "Metric": "WaveformNotDetected",
                "Value": int((~od_os["IrregularYellowWaveformDetected"]).sum()),
            },
            {
                "Group": "OD_OS",
                "Metric": "BscanOnlyCandidates",
                "Value": len(candidates),
            },
            {
                "Group": "OD_OS",
                "Metric": "Uncertain",
                "Value": int(od_os["AutoScanType"].eq("UNCERTAIN").sum()),
            },
        ]
    )
    for laterality in ["OD", "OS"]:
        laterality_rows = od_os.loc[od_os["Laterality"].eq(laterality)]
        laterality_candidates = laterality_rows.loc[
            laterality_rows["HasBscanContent"]
            & (~laterality_rows["IrregularYellowWaveformDetected"])
        ]
        summary_rows.extend(
            [
                {
                    "Group": laterality,
                    "Metric": "TotalImages",
                    "Value": len(laterality_rows),
                },
                {
                    "Group": laterality,
                    "Metric": "WaveformDetected",
                    "Value": int(
                        laterality_rows["IrregularYellowWaveformDetected"].sum()
                    ),
                },
                {
                    "Group": laterality,
                    "Metric": "WaveformNotDetected",
                    "Value": int(
                        (~laterality_rows["IrregularYellowWaveformDetected"]).sum()
                    ),
                },
                {
                    "Group": laterality,
                    "Metric": "BscanOnlyCandidates",
                    "Value": len(laterality_candidates),
                },
                {
                    "Group": laterality,
                    "Metric": "Uncertain",
                    "Value": int(
                        laterality_rows["AutoScanType"].eq("UNCERTAIN").sum()
                    ),
                },
            ]
        )
    candidate_subject_counts = candidates.groupby("ResearchSubjectID").size()
    candidate_encounters = candidates[
        ["ResearchSubjectID", "EncounterID"]
    ].drop_duplicates()
    largest_subject_count = (
        int(candidate_subject_counts.max()) if not candidate_subject_counts.empty else 0
    )
    summary_rows.extend(
        [
            {
                "Group": "OD_OS",
                "Metric": "BscanOnlyCandidateSubjects",
                "Value": int(candidates["ResearchSubjectID"].nunique()),
            },
            {
                "Group": "OD_OS",
                "Metric": "BscanOnlyCandidateEncounters",
                "Value": len(candidate_encounters),
            },
            {
                "Group": "OD_OS",
                "Metric": "LargestSubjectCandidateImages",
                "Value": largest_subject_count,
            },
            {
                "Group": "OD_OS",
                "Metric": "LargestSubjectCandidateSharePercent",
                "Value": (
                    100.0 * largest_subject_count / len(candidates)
                    if len(candidates)
                    else 0.0
                ),
            },
        ]
    )
    summary_path = scan_audit_dir / "scan_type_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    method = {
        "run_id": run_id,
        "detector_version": DETECTOR_VERSION,
        "purpose": "Automated image-layout screening; not clinical ground truth.",
        "automatic_classification_status": "CANDIDATE_ONLY",
        "manual_confirmation_required": True,
        "laterality_used_for_detector_or_threshold_tuning": False,
        "laterality_use": "Post-classification reporting and queue selection only.",
        "input_manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": manifest_sha_before,
            "rows": full_count,
        },
        "configuration": asdict(CONFIG),
        "yellow_mask": {
            "color_space": "Pillow HSV, integer channels 0..255",
            "rule": (
                f"H in [{CONFIG.yellow_hue_min}, {CONFIG.yellow_hue_max}], "
                f"S >= {CONFIG.yellow_saturation_min}, V >= {CONFIG.yellow_value_min}"
            ),
        },
        "long_horizontal_yellow_line_rule": (
            "True for a contiguous yellow row run covering >=25% of image width OR "
            "a shallow OLS-fitted yellow guide with <=3 px final residual, >=40% "
            "column support, >=50% x-span, and |slope|<=0.25 px/column."
        ),
        "irregular_yellow_waveform_rule": (
            "Suppress +/-3 px around detected straight guides; in the lower 42% "
            "require residual yellow coverage in >=5% of columns, vertical peaks "
            "in >=2.5% of columns (each >=1.3% image height), total vertical extent "
            ">=6.7% height, and residual pixels >=0.05% of image area."
        ),
        "standalone_ascan_layout_rule": (
            "White-background plot screen: >=65% achromatic white canvas, >=0.1% "
            "red graph overlay, and >=1% achromatic dark ink. This permits factual "
            "A-scan candidate screening when the waveform is black rather than yellow."
        ),
        "bscan_content_rule": (
            "Weighted score of achromatic midtones, grayscale edge density, textured "
            "6x8 tile occupancy, texture spatial extent, and dark-background support "
            "within a fixed central/upper ROI. Score >=0.70 is present; <=0.30 is "
            "absent; intermediate scores are UNCERTAIN."
        ),
        "numeric_score_interpretation": (
            "BscanContentScore and WaveformEvidenceScore are clipped rule-evidence "
            "diagnostics, not probabilities or calibrated confidence values."
        ),
        "classification_rule": {
            "A_SCAN_ONLY_CANDIDATE": "A-scan waveform/layout evidence and no convincing B-scan content",
            "COMBINED_A_B_CANDIDATE": "irregular yellow waveform and convincing B-scan content",
            "B_SCAN_ONLY_CANDIDATE": "convincing B-scan content and no irregular yellow waveform",
            "UNCERTAIN": "borderline/conflicting B-scan evidence or neither content pattern",
        },
        "known_screening_limitation": (
            "A white/gray A-scan trace over a dark B-scan can be a no-yellow false "
            "candidate; candidate contact sheets are therefore mandatory."
        ),
        "classification_signature_sha256": signature,
        "previous_compatible_run": str(previous_dir) if previous_dir else None,
        "outputs": {
            "scan_audit_directory": str(scan_audit_dir),
            "figure_directory": str(figure_dir),
            "validation_checks": str(checks_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL_VERSION,
            "platform": platform.platform(),
            "workers": args.workers,
            "images_per_sheet": args.images_per_sheet,
        },
    }
    method_path = scan_audit_dir / "scan_type_method.json"
    with method_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(method, stream, indent=2, sort_keys=True, allow_nan=False, default=to_json_value)
        stream.write("\n")

    record_check(
        checks,
        "all required output files exist",
        all(
            path.is_file()
            for path in [
                unknown_path,
                unknown_manual_path,
                candidate_path,
                candidate_manual_path,
                all_diagnostics_path,
                summary_path,
                method_path,
            ]
        ),
        True,
        True,
    )
    record_check(
        checks,
        "debug mask artifacts were generated",
        bool(debug_files) and all(path.is_file() for path in debug_files),
        True,
        bool(debug_files) and all(path.is_file() for path in debug_files),
    )
    stop_on_failed_checks(checks, checks_path)
    save_checks(checks, checks_path)

    unknown_a = int(unknown["AutoScanType"].eq("A_SCAN_ONLY_CANDIDATE").sum())
    unknown_ab = int(unknown["AutoScanType"].eq("COMBINED_A_B_CANDIDATE").sum())
    unknown_b = int(unknown["AutoScanType"].eq("B_SCAN_ONLY_CANDIDATE").sum())
    unknown_uncertain = int(unknown["AutoScanType"].eq("UNCERTAIN").sum())
    od_os_waveform = int(od_os["IrregularYellowWaveformDetected"].sum())
    od_os_no_waveform = int((~od_os["IrregularYellowWaveformDetected"]).sum())
    od_os_uncertain = int(od_os["AutoScanType"].eq("UNCERTAIN").sum())

    print("\n" + "=" * 78)
    print("SCAN-TYPE QA COMPLETE — ALL HARD CHECKS PASS")
    print("=" * 78)
    print("FULL DATASET")
    print(f"  Total images                    : {full_count:,}")
    print("\nUNKNOWN")
    print(f"  Total                           : {len(unknown):,}")
    print(f"  A-scan-only candidates          : {unknown_a:,}")
    print(f"  Combined A+B candidates         : {unknown_ab:,}")
    print(f"  B-scan-only candidates          : {unknown_b:,}")
    print(f"  Uncertain                       : {unknown_uncertain:,}")
    print("\nOD/OS")
    print(f"  Total                           : {len(od_os):,}")
    print(f"  Irregular yellow waveform found : {od_os_waveform:,}")
    print(f"  Irregular yellow waveform absent: {od_os_no_waveform:,}")
    print(f"  B-scan-only candidates          : {len(candidates):,}")
    print(f"  Uncertain                       : {od_os_uncertain:,}")
    for laterality in ["OD", "OS"]:
        laterality_rows = od_os.loc[od_os["Laterality"].eq(laterality)]
        laterality_candidates = laterality_rows.loc[
            laterality_rows["HasBscanContent"]
            & (~laterality_rows["IrregularYellowWaveformDetected"])
        ]
        print(
            f"  {laterality}: total={len(laterality_rows):,}, "
            f"waveform={int(laterality_rows['IrregularYellowWaveformDetected'].sum()):,}, "
            f"no-waveform={int((~laterality_rows['IrregularYellowWaveformDetected']).sum()):,}, "
            f"candidates={len(laterality_candidates):,}, "
            f"uncertain={int(laterality_rows['AutoScanType'].eq('UNCERTAIN').sum()):,}"
        )
    print(
        f"  Candidate concentration         : "
        f"{candidates['ResearchSubjectID'].nunique():,} subjects, "
        f"{len(candidate_encounters):,} encounters; largest subject="
        f"{largest_subject_count:,}/{len(candidates):,} images "
        f"({(100.0 * largest_subject_count / len(candidates) if len(candidates) else 0.0):.1f}%)"
    )
    print("\nOUTPUTS")
    print(f"  UNKNOWN audit                   : {unknown_path}")
    print(f"  UNKNOWN manual review           : {unknown_manual_path}")
    print(f"  B-scan/no-waveform candidates   : {candidate_path}")
    print(f"  B-scan candidate manual review  : {candidate_manual_path}")
    print(f"  UNKNOWN contact sheets          : {unknown_contact_dir}")
    print(f"  B-scan candidate contact sheets : {bscan_contact_dir}")
    print(f"  Debug masks                     : {debug_dir}")
    print(f"  Validation checks               : {checks_path}")
    print(f"  Method/provenance               : {method_path}")
    print("\nSCIENTIFIC CAUTION")
    print("  AUTO classifications are screening candidates, not clinical ground truth.")
    print("  MANUALLY CONFIRMED scan type remains blank pending contact-sheet review.")
    print("  No Laterality value or analytic/model dataset was changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
