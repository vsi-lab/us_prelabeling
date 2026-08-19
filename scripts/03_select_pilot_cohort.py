from datetime import datetime
from pathlib import Path
import hashlib
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "interim" / "bscan_eye_level_master.csv"
PILOT_PATH = PROJECT_ROOT / "data" / "processed" / "pilot_cohort.csv"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"

RANDOM_SEED = 42
TARGET_PILOT_SIZE = 800
ALLOWED_LABELS = {"Normal", "Abnormal"}
EXPECTED_SOURCE_COUNTS = {
    "total": 1510,
    "Normal": 402,
    "Abnormal": 1108,
}


def record_check(rows, check, passed, details):
    rows.append(
        {
            "Check": check,
            "Status": "PASS" if passed else "FAIL",
            "Details": details,
        }
    )


def record_metric(rows, metric, value):
    rows.append({"Metric": metric, "Value": value})


def parse_bool_column(series: pd.Series) -> tuple[pd.Series, list[str]]:
    "Convert true/false text from the CSV and keep track of anything unexpected."
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = sorted(
        normalized.loc[~normalized.isin(["true", "false"])]
        .dropna()
        .astype(str)
        .unique()
    )
    return normalized.eq("true").fillna(False), invalid


def allocate_by_label(label_counts: pd.Series, target: int) -> dict[str, int]:
    "Split the target sample across labels while keeping the source proportions."
    total = int(label_counts.sum())
    if total == 0 or target == 0:
        return {str(label): 0 for label in label_counts.index}

    quotas = {str(label): target * int(count) / total for label, count in label_counts.items()}
    allocation = {label: math.floor(quota) for label, quota in quotas.items()}
    remaining = target - sum(allocation.values())
    remainder_order = sorted(
        quotas,
        key=lambda label: (-(quotas[label] - allocation[label]), label),
    )
    for label in remainder_order[:remaining]:
        allocation[label] += 1
    return allocation


def sample_hash(eye_exam_id: str, label: str) -> str:
    "Build a repeatable hash used to rank eyes within each class."
    token = f"{RANDOM_SEED}|{label}|{eye_exam_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def choose_stratified_sample(
    source_pool: pd.DataFrame, allocation: dict[str, int]
) -> pd.DataFrame:
    "Pick the requested number of eyes from each label using the stable hash rank."
    selected = []
    for label in sorted(allocation):
        stratum = source_pool.loc[source_pool["EyeLabel"] == label].copy()
        stratum["_PilotSampleRank"] = stratum["EyeExamID"].map(
            lambda eye_exam_id: sample_hash(str(eye_exam_id), label)
        )
        stratum = stratum.sort_values(
            ["_PilotSampleRank", "EyeExamID"], kind="mergesort"
        )
        selected.append(stratum.head(allocation[label]))
    return (
        pd.concat(selected, ignore_index=True)
        .drop(columns="_PilotSampleRank")
        .sort_values("EyeExamID", kind="mergesort")
        .reset_index(drop=True)
    )


def classify_subject_labels(labels: pd.Series) -> str:
    label_set = set(labels)
    if label_set == {"Normal"}:
        return "Normal-only"
    if label_set == {"Abnormal"}:
        return "Abnormal-only"
    return "Mixed"


def percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def stop_on_validation_failure(
    checks: list[dict], checks_path: Path, message: str
) -> None:
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    print("STEP-3 VALIDATION: HARD FAIL\n")
    failed = [row for row in checks if row["Status"] == "FAIL"]
    for row in failed:
        print(f"[FAIL] {row['Check']}: {row['Details']}")
    print(f"\n{message}")
    print(f"Validation checks written to:\n{checks_path}")
    raise SystemExit(1)


def main():
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_ROOT / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    checks_path = audit_dir / "step3_pilot_checks.csv"
    summary_path = audit_dir / "step3_pilot_summary.csv"
    postoperative_queue_path = audit_dir / "pilot_postoperative_review_queue.csv"

    print("=" * 72)
    print("STEP 3: SELECT AND LOCK PILOT COHORT")
    print(f"Run ID             : {run_id}")
    print(f"Canonical source   : {SOURCE_PATH}")
    print(f"Pilot output       : {PILOT_PATH}")
    print(f"Random seed        : {RANDOM_SEED}\n")

    checks = []
    source_exists = SOURCE_PATH.is_file()
    record_check(
        checks,
        "canonical_source_file_exists",
        source_exists,
        str(SOURCE_PATH),
    )
    if not source_exists:
        stop_on_validation_failure(
            checks,
            checks_path,
            "Canonical source is missing; no pilot cohort was produced.",
        )

    # Load the eye-level table produced in Step 2.
    master = pd.read_csv(SOURCE_PATH, low_memory=False)
    required_columns = {
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "EyeFindings",
        "ImageQuality",
        "OriginalImageCount",
        "UniqueImageCount",
        "DuplicateImageCount",
        "PostoperativeReviewFlag",
        "AuditHold",
        "AuditHoldReason",
        "IncludePrimary",
    }
    missing_columns = sorted(required_columns - set(master.columns))
    record_check(
        checks,
        "canonical_source_required_columns",
        not missing_columns,
        f"missing={missing_columns}",
    )
    if missing_columns:
        stop_on_validation_failure(
            checks,
            checks_path,
            "Canonical source schema is incomplete; no pilot cohort was produced.",
        )

    include_primary, invalid_include_primary = parse_bool_column(master["IncludePrimary"])
    audit_hold, invalid_audit_hold = parse_bool_column(master["AuditHold"])
    postoperative_flag, invalid_postoperative_flag = parse_bool_column(
        master["PostoperativeReviewFlag"]
    )
    boolean_values_valid = not (
        invalid_include_primary or invalid_audit_hold or invalid_postoperative_flag
    )
    record_check(
        checks,
        "canonical_boolean_fields_valid",
        boolean_values_valid,
        "invalid IncludePrimary="
        f"{invalid_include_primary}; AuditHold={invalid_audit_hold}; "
        f"PostoperativeReviewFlag={invalid_postoperative_flag}",
    )
    master["IncludePrimary"] = include_primary
    master["AuditHold"] = audit_hold
    master["PostoperativeReviewFlag"] = postoperative_flag

    source_eye_ids_unique = master["EyeExamID"].notna().all() and master[
        "EyeExamID"
    ].is_unique
    record_check(
        checks,
        "source_eye_exam_id_unique",
        source_eye_ids_unique,
        f"rows={len(master)}; unique={master['EyeExamID'].nunique()}",
    )

    # Only eyes that survived the Step-2 audit are allowed into the pilot pool.
    source_pool = master.loc[master["IncludePrimary"]].copy()
    held_source_rows = int(source_pool["AuditHold"].sum())
    record_check(
        checks,
        "no_include_primary_source_rows_are_held",
        held_source_rows == 0,
        f"held_rows={held_source_rows}",
    )

    source_label_counts = source_pool["EyeLabel"].value_counts()
    source_total = len(source_pool)
    source_normal = int(source_label_counts.get("Normal", 0))
    source_abnormal = int(source_label_counts.get("Abnormal", 0))
    record_check(
        checks,
        "expected_source_eligible_total",
        source_total == EXPECTED_SOURCE_COUNTS["total"],
        f"expected={EXPECTED_SOURCE_COUNTS['total']}; observed={source_total}",
    )
    record_check(
        checks,
        "expected_source_normal_count",
        source_normal == EXPECTED_SOURCE_COUNTS["Normal"],
        f"expected={EXPECTED_SOURCE_COUNTS['Normal']}; observed={source_normal}",
    )
    record_check(
        checks,
        "expected_source_abnormal_count",
        source_abnormal == EXPECTED_SOURCE_COUNTS["Abnormal"],
        f"expected={EXPECTED_SOURCE_COUNTS['Abnormal']}; observed={source_abnormal}",
    )
    source_labels_valid = set(source_pool["EyeLabel"].dropna().unique()).issubset(
        ALLOWED_LABELS
    ) and source_pool["EyeLabel"].notna().all()
    record_check(
        checks,
        "source_pool_only_normal_abnormal",
        source_labels_valid,
        f"labels={sorted(source_pool['EyeLabel'].dropna().astype(str).unique())}",
    )

    # Match the class balance of the eligible source pool as closely as possible.
    target_size = min(TARGET_PILOT_SIZE, source_total)
    allocation = allocate_by_label(
        source_label_counts.reindex(sorted(ALLOWED_LABELS), fill_value=0),
        target_size,
    )
    pilot = choose_stratified_sample(source_pool, allocation)
    repeated_pilot = choose_stratified_sample(source_pool, allocation)

    pilot_output_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "EyeFindings",
        "ImageQuality",
        "OriginalImageCount",
        "UniqueImageCount",
        "DuplicateImageCount",
        "PostoperativeReviewFlag",
        "AuditHold",
        "AuditHoldReason",
        "IncludePrimary",
    ]
    pilot = pilot[pilot_output_columns].copy()
    repeated_pilot = repeated_pilot[pilot_output_columns].copy()
    pilot["PilotSelected"] = True
    pilot["PilotSamplingSeed"] = RANDOM_SEED

    pilot_total = len(pilot)
    expected_pilot_total = min(TARGET_PILOT_SIZE, source_total)
    record_check(
        checks,
        "expected_pilot_total",
        pilot_total == expected_pilot_total,
        f"expected={expected_pilot_total}; observed={pilot_total}",
    )
    source_eye_ids = set(master["EyeExamID"])
    missing_pilot_ids = sorted(set(pilot["EyeExamID"]) - source_eye_ids)
    record_check(
        checks,
        "every_pilot_eye_exists_in_canonical_master",
        not missing_pilot_ids,
        f"missing_ids={len(missing_pilot_ids)}",
    )
    master_primary_by_id = master.set_index("EyeExamID")["IncludePrimary"]
    pilot_primary_valid = pilot["EyeExamID"].map(master_primary_by_id).fillna(False).all()
    record_check(
        checks,
        "every_pilot_row_was_include_primary",
        pilot_primary_valid,
        f"invalid_rows={int((~pilot['EyeExamID'].map(master_primary_by_id).fillna(False)).sum())}",
    )
    record_check(
        checks,
        "pilot_eye_exam_id_unique",
        pilot["EyeExamID"].is_unique,
        f"duplicate_rows={int(pilot.duplicated('EyeExamID', keep=False).sum())}",
    )
    record_check(
        checks,
        "no_pilot_rows_are_held",
        not pilot["AuditHold"].any(),
        f"held_rows={int(pilot['AuditHold'].sum())}",
    )
    pilot_labels = set(pilot["EyeLabel"].dropna().unique())
    pilot_labels_valid = pilot["EyeLabel"].notna().all() and pilot_labels.issubset(
        ALLOWED_LABELS
    )
    record_check(
        checks,
        "pilot_only_normal_abnormal_labels",
        pilot_labels_valid,
        f"labels={sorted(str(label) for label in pilot_labels)}",
    )
    deterministic_selection = pilot["EyeExamID"].tolist() == repeated_pilot[
        "EyeExamID"
    ].tolist()
    record_check(
        checks,
        "repeated_seed_42_selection_is_identical",
        deterministic_selection,
        f"selected_eye_ids={pilot_total}",
    )

    pilot_label_counts = pilot["EyeLabel"].value_counts()
    allocation_matches = all(
        int(pilot_label_counts.get(label, 0)) == count
        for label, count in allocation.items()
    )
    record_check(
        checks,
        "pilot_matches_proportional_largest_remainder_allocation",
        allocation_matches,
        f"allocation={allocation}; observed={pilot_label_counts.to_dict()}",
    )

    # If a pilot file already exists, treat it as locked. A rerun must reproduce the same EyeExamIDs.
    lock_status = "CREATED"
    existing_ids_identical = True
    if PILOT_PATH.exists():
        existing = pd.read_csv(PILOT_PATH, low_memory=False)
        if "EyeExamID" not in existing.columns:
            existing_ids_identical = False
        else:
            existing_ids_identical = sorted(existing["EyeExamID"].astype(str).tolist()) == sorted(
                pilot["EyeExamID"].astype(str).tolist()
            )
        lock_status = (
            "REPRODUCIBLE_EXISTING" if existing_ids_identical else "LOCK_CONFLICT"
        )
    record_check(
        checks,
        "locked_cohort_matches_new_selection",
        existing_ids_identical,
        f"pilot_exists={PILOT_PATH.exists()}; lock_status={lock_status}",
    )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    if (checks_df["Status"] == "FAIL").any():
        stop_on_validation_failure(
            checks,
            checks_path,
            "Pilot validation failed; the locked cohort was not created or overwritten.",
        )

    if not PILOT_PATH.exists():
        PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pilot.to_csv(PILOT_PATH, index=False)
    else:
        print("Existing locked pilot has identical EyeExamIDs; leaving it unchanged.\n")

    # These subject-level numbers are only for reporting. The pilot is still selected at the eye level.
    subject_eye_counts = pilot.groupby("ResearchSubjectID").size()
    unique_subjects = len(subject_eye_counts)
    subjects_one_eye = int(subject_eye_counts.eq(1).sum())
    subjects_multiple_eyes = int(subject_eye_counts.gt(1).sum())
    max_eyes_per_subject = int(subject_eye_counts.max())
    subject_categories = (
        pilot.groupby("ResearchSubjectID")["EyeLabel"]
        .apply(classify_subject_labels)
        .value_counts()
    )
    normal_only_subjects = int(subject_categories.get("Normal-only", 0))
    abnormal_only_subjects = int(subject_categories.get("Abnormal-only", 0))
    mixed_subjects = int(subject_categories.get("Mixed", 0))

    pilot_normal = int(pilot_label_counts.get("Normal", 0))
    pilot_abnormal = int(pilot_label_counts.get("Abnormal", 0))
    source_normal_pct = percent(source_normal, source_total)
    source_abnormal_pct = percent(source_abnormal, source_total)
    pilot_normal_pct = percent(pilot_normal, pilot_total)
    pilot_abnormal_pct = percent(pilot_abnormal, pilot_total)

    # Postoperative eyes stay in the pilot; this file is just a review queue.
    postoperative_queue = pilot.loc[pilot["PostoperativeReviewFlag"]].copy()
    postoperative_queue.to_csv(postoperative_queue_path, index=False)
    postoperative_count = len(postoperative_queue)
    postoperative_pct = percent(postoperative_count, pilot_total)
    postoperative_labels = postoperative_queue["EyeLabel"].value_counts()

    unique_image_distribution = pilot["UniqueImageCount"].value_counts().sort_index()
    median_unique_images = float(pilot["UniqueImageCount"].median())
    mean_unique_images = float(pilot["UniqueImageCount"].mean())
    fewer_than_six = int(pilot["UniqueImageCount"].lt(6).sum())
    exactly_six = int(pilot["UniqueImageCount"].eq(6).sum())
    more_than_six = int(pilot["UniqueImageCount"].gt(6).sum())

    summary = []
    record_metric(summary, "run_id", run_id)
    record_metric(summary, "pilot_lock_status", lock_status)
    record_metric(summary, "random_seed", RANDOM_SEED)
    record_metric(summary, "target_pilot_size", TARGET_PILOT_SIZE)
    record_metric(summary, "source_pool_total", source_total)
    record_metric(summary, "source_normal_count", source_normal)
    record_metric(summary, "source_normal_percent", source_normal_pct)
    record_metric(summary, "source_abnormal_count", source_abnormal)
    record_metric(summary, "source_abnormal_percent", source_abnormal_pct)
    record_metric(summary, "pilot_total", pilot_total)
    record_metric(summary, "pilot_normal_count", pilot_normal)
    record_metric(summary, "pilot_normal_percent", pilot_normal_pct)
    record_metric(summary, "pilot_abnormal_count", pilot_abnormal)
    record_metric(summary, "pilot_abnormal_percent", pilot_abnormal_pct)
    record_metric(
        summary,
        "normal_absolute_percentage_point_deviation",
        abs(pilot_normal_pct - source_normal_pct),
    )
    record_metric(
        summary,
        "abnormal_absolute_percentage_point_deviation",
        abs(pilot_abnormal_pct - source_abnormal_pct),
    )
    record_metric(summary, "unique_subjects", unique_subjects)
    record_metric(summary, "subjects_with_exactly_one_selected_eye", subjects_one_eye)
    record_metric(summary, "subjects_with_multiple_selected_eyes", subjects_multiple_eyes)
    record_metric(summary, "maximum_selected_eyes_per_subject", max_eyes_per_subject)
    record_metric(summary, "normal_only_subjects", normal_only_subjects)
    record_metric(summary, "abnormal_only_subjects", abnormal_only_subjects)
    record_metric(summary, "mixed_subjects", mixed_subjects)
    record_metric(summary, "postoperative_flagged_pilot_eyes", postoperative_count)
    record_metric(summary, "postoperative_flagged_pilot_percent", postoperative_pct)
    record_metric(
        summary,
        "postoperative_flagged_normal_eyes",
        int(postoperative_labels.get("Normal", 0)),
    )
    record_metric(
        summary,
        "postoperative_flagged_abnormal_eyes",
        int(postoperative_labels.get("Abnormal", 0)),
    )
    record_metric(summary, "median_unique_images_per_eye", median_unique_images)
    record_metric(summary, "mean_unique_images_per_eye", mean_unique_images)
    record_metric(summary, "pilot_eyes_with_fewer_than_6_unique_images", fewer_than_six)
    record_metric(summary, "pilot_eyes_with_exactly_6_unique_images", exactly_six)
    record_metric(summary, "pilot_eyes_with_more_than_6_unique_images", more_than_six)
    for image_count, eye_count in unique_image_distribution.items():
        record_metric(
            summary,
            f"pilot_eyes_with_{int(image_count)}_unique_images",
            int(eye_count),
        )
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("STEP-3 VALIDATION: PASS\n")
    print("Source pool distribution:")
    print(f"  Total                              : {source_total:,}")
    print(f"  Normal                             : {source_normal:,} ({source_normal_pct:.3f}%)")
    print(f"  Abnormal                           : {source_abnormal:,} ({source_abnormal_pct:.3f}%)")
    print("Pilot distribution:")
    print(f"  Total                              : {pilot_total:,}")
    print(f"  Normal                             : {pilot_normal:,} ({pilot_normal_pct:.3f}%)")
    print(f"  Abnormal                           : {pilot_abnormal:,} ({pilot_abnormal_pct:.3f}%)")
    print(f"  Normal absolute deviation          : {abs(pilot_normal_pct - source_normal_pct):.3f} percentage points")
    print(f"  Abnormal absolute deviation        : {abs(pilot_abnormal_pct - source_abnormal_pct):.3f} percentage points")
    print()
    print(f"Unique subjects                     : {unique_subjects:,}")
    print(f"  Normal-only                       : {normal_only_subjects:,}")
    print(f"  Abnormal-only                     : {abnormal_only_subjects:,}")
    print(f"  Mixed                             : {mixed_subjects:,}")
    print(f"Subjects with exactly 1 eye         : {subjects_one_eye:,}")
    print(f"Subjects with >1 selected eye       : {subjects_multiple_eyes:,}")
    print(f"Maximum selected eyes per subject   : {max_eyes_per_subject:,}")
    print()
    print(f"Postoperative-flagged pilot eyes    : {postoperative_count:,} ({postoperative_pct:.3f}%)")
    print(f"  Normal                            : {int(postoperative_labels.get('Normal', 0)):,}")
    print(f"  Abnormal                          : {int(postoperative_labels.get('Abnormal', 0)):,}")
    print()
    print("Unique image-count distribution:")
    for image_count, eye_count in unique_image_distribution.items():
        print(f"  {int(image_count):>2} image(s): {int(eye_count):,} pilot eyes")
    print(f"Median unique images per eye        : {median_unique_images:.2f}")
    print(f"Mean unique images per eye          : {mean_unique_images:.2f}")
    print(f"Eyes with <6 unique images          : {fewer_than_six:,}")
    print(f"Eyes with exactly 6 unique images   : {exactly_six:,}")
    print(f"Eyes with >6 unique images          : {more_than_six:,}")
    print()
    print(f"Random seed                         : {RANDOM_SEED}")
    print(f"Pilot lock status                   : {lock_status}")
    print(f"Pilot cohort                        : {PILOT_PATH}")
    print(f"Validation report                   : {checks_path}")
    print(f"Audit summary                       : {summary_path}")
    print(f"Postoperative review queue          : {postoperative_queue_path}")


if __name__ == "__main__":
    main()
