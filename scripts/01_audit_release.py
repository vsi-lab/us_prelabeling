from datetime import datetime
from pathlib import Path
import json
import re

import pandas as pd


#everything is resolved from the repository root, so the script works the same way no matter which folder it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"

ENCOUNTER_KEY = ["ResearchSubjectID", "EncounterID"]
EYE_KEY = [*ENCOUNTER_KEY, "Laterality"]
ALLOWED_LATERALITY = {"OD", "OS", "UNKNOWN"}

#expected row counts for the locked v2 release.
EXPECTED_RELEASE_V2 = {
    "release_version": "v2",
    "encounter_count": 2296,
    "subject_count": 1129,
    "full_image_count": 16206,
    "reviewed_encounter_count": 1378,
    "unreviewed_encounter_count": 918,
    "unknown_image_count": 398,
    "bscan_candidate_image_count": 15808,
    "bscan_candidate_encounter_count": 2245,
    "bscan_candidate_subject_count": 1106,
}

#terms used only to flag possible postoperative cases for manual review.
POSTOPERATIVE_PATTERN = re.compile(
    r"(?:silicone[ -]?oil|gas(?:[ -]?tamponade)?|ppv|vitrectom|"
    r"oil[ -]?removal|tamponade|emulsified[ -]?oil)",
    flags=re.IGNORECASE,
)


def find_release_file(filename: str) -> Path:
    """Look for one release file under data/raw and make sure it is unique."""
    matches = list(RAW_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under:\n{RAW_DIR}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple copies of '{filename}' under data/raw. "
            f"Expected exactly one release copy. Matches: {matches}"
        )
    return matches[0]


def add_summary_metric(rows, metric, value):
    rows.append({"Metric": metric, "Value": value})


def add_check(rows, category, check, passed, details):
    rows.append(
        {
            "Category": category,
            "Check": check,
            "Status": "PASS" if passed else category,
            "Details": details,
        }
    )


def rows_with_blank_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Mark rows where at least one key field is missing or only whitespace."""
    result = pd.Series(False, index=frame.index)
    for column in columns:
        result |= frame[column].isna()
        result |= frame[column].astype("string").str.strip().eq("").fillna(True)
    return result


def print_check_section(checks_df: pd.DataFrame, category: str):
    print(category)
    print("-" * len(category))
    section = checks_df.loc[checks_df["Category"] == category]
    for row in section.itertuples(index=False):
        print(f"[{row.Status}] {row.Check}: {row.Details}")
    print()


def stop_after_preflight_failure(audit_dir: Path, checks, summary, message: str):
    """Write what we have so far, then stop because the schema is unusable."""
    add_summary_metric(summary, "overall_audit_status", "HARD FAIL")
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(audit_dir / "audit_checks.csv", index=False)
    pd.DataFrame(summary).to_csv(audit_dir / "dataset_summary.csv", index=False)
    print("AUDIT STATUS: HARD FAIL\n")
    print_check_section(checks_df, "HARD FAIL")
    print(message)
    print(f"\nAudit files written to:\n{audit_dir}")
    raise SystemExit(1)


def main():
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_ROOT / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 72)
    print("B-SCAN RELEASE AUDIT")
    print("=" * 72)
    print(f"Run ID             : {run_id}")
    print(f"Output directory   : {audit_dir}\n")

    encounter_path = find_release_file("Bscan_all_encounters_metadata_260811.csv")
    manifest_path = find_release_file("Bscan_full_image_manifest_260811.csv")
    dictionary_path = find_release_file("Bscan_data_dictionary_260811.csv")
    release_path = find_release_file("release_manifest.json")

    print(f"Encounter metadata : {encounter_path}")
    print(f"Image manifest     : {manifest_path}")
    print(f"Data dictionary    : {dictionary_path}")
    print(f"Release manifest   : {release_path}\n")

    encounters = pd.read_csv(encounter_path, low_memory=False)
    images = pd.read_csv(manifest_path, low_memory=False)
    data_dictionary = pd.read_csv(dictionary_path, low_memory=False)
    with open(release_path, "r", encoding="utf-8") as f:
        release = json.load(f)

    summary = []
    checks = []

    #start with the schema. There is no point running the rest of the audit
    #if one of the columns used below is missing.
    required_encounter_cols = {
        "ResearchSubjectID",
        "EncounterID",
        "TotalImageCount",
        "ODImageCount",
        "OSImageCount",
        "UnknownLateralityImageCount",
        "ReviewStatus",
        "OD_Status",
        "OD_Findings",
        "OD_ImageQuality",
        "OS_Status",
        "OS_Findings",
        "OS_ImageQuality",
        "EncounterClassification",
        "SurgicalContextTags",
        "ReviewerNotes",
    }

    required_image_cols = {
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "ImageID",
        "ImageFileName",
        "ImageRelativePath",
        "SHA256",
        "Width",
        "Height",
        "ReviewStatus",
        "EncounterClassification",
    }

    missing_enc = sorted(required_encounter_cols - set(encounters.columns))
    missing_img = sorted(required_image_cols - set(images.columns))
    add_check(
        checks,
        "HARD FAIL",
        "encounter_required_columns",
        not missing_enc,
        f"missing={missing_enc}",
    )
    add_check(
        checks,
        "HARD FAIL",
        "manifest_required_columns",
        not missing_img,
        f"missing={missing_img}",
    )
    if missing_enc or missing_img:
        stop_after_preflight_failure(
            audit_dir,
            checks,
            summary,
            "Required columns are missing; dependent checks were not run.",
        )

    #check the IDs before doing any joins. Duplicate or blank keys can make
    #later counts look valid even when the underlying rows are wrong.
    invalid_encounter_keys = encounters.loc[
        rows_with_blank_keys(encounters, ENCOUNTER_KEY)
        | encounters.duplicated(ENCOUNTER_KEY, keep=False)
    ].copy()
    invalid_encounter_keys.to_csv(
        audit_dir / "invalid_encounter_composite_keys.csv", index=False
    )
    add_check(
        checks,
        "HARD FAIL",
        "encounter_composite_keys",
        invalid_encounter_keys.empty,
        f"invalid_or_duplicate_rows={len(invalid_encounter_keys)}",
    )

    image_composite_key = [*EYE_KEY, "ImageID"]
    invalid_image_keys = images.loc[
        rows_with_blank_keys(images, image_composite_key)
        | images.duplicated(image_composite_key, keep=False)
    ].copy()
    invalid_image_keys.to_csv(
        audit_dir / "invalid_image_composite_keys.csv", index=False
    )
    add_check(
        checks,
        "HARD FAIL",
        "image_composite_keys",
        invalid_image_keys.empty,
        f"invalid_or_duplicate_rows={len(invalid_image_keys)}",
    )

    manifest_encounter_keys = images[ENCOUNTER_KEY].drop_duplicates()
    encounter_keys = encounters[ENCOUNTER_KEY].drop_duplicates()
    orphan_manifest_keys = manifest_encounter_keys.merge(
        encounter_keys, on=ENCOUNTER_KEY, how="left", indicator=True
    ).loc[lambda frame: frame["_merge"] == "left_only", ENCOUNTER_KEY]
    orphan_manifest_keys.to_csv(
        audit_dir / "orphan_manifest_encounter_keys.csv", index=False
    )
    add_check(
        checks,
        "HARD FAIL",
        "manifest_encounter_foreign_keys",
        orphan_manifest_keys.empty,
        f"orphan_encounter_keys={len(orphan_manifest_keys)}",
    )

    invalid_laterality = images.loc[
        images["Laterality"].isna()
        | ~images["Laterality"].isin(ALLOWED_LATERALITY)
    ].copy()
    invalid_laterality.to_csv(
        audit_dir / "invalid_manifest_laterality.csv", index=False
    )
    invalid_values = sorted(
        images.loc[
            ~images["Laterality"].isin(ALLOWED_LATERALITY), "Laterality"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    add_check(
        checks,
        "HARD FAIL",
        "manifest_laterality_allowed_values",
        invalid_laterality.empty,
        f"invalid_rows={len(invalid_laterality)}; invalid_values={invalid_values}",
    )

    #these counts are fixed for release v2. A mismatch usually means the wrong
    # release was loaded or one of the source files changed.
    bscan_images = images.loc[images["Laterality"].isin(["OD", "OS"])].copy()
    unknown_images = images.loc[images["Laterality"] == "UNKNOWN"].copy()

    observed_release_v2 = {
        "release_version": release.get("release_version"),
        "encounter_count": len(encounters),
        "subject_count": encounters["ResearchSubjectID"].nunique(),
        "full_image_count": len(images),
        "reviewed_encounter_count": int(
            (encounters["ReviewStatus"] == "Reviewed").sum()
        ),
        "unreviewed_encounter_count": int(
            (encounters["ReviewStatus"] == "Unreviewed").sum()
        ),
        "unknown_image_count": len(unknown_images),
        "bscan_candidate_image_count": len(bscan_images),
        "bscan_candidate_encounter_count": len(
            bscan_images[ENCOUNTER_KEY].drop_duplicates()
        ),
        "bscan_candidate_subject_count": bscan_images[
            "ResearchSubjectID"
        ].nunique(),
    }

    release_v2_mismatches = []
    for field, expected in EXPECTED_RELEASE_V2.items():
        observed = observed_release_v2[field]
        passed = observed == expected
        add_check(
            checks,
            "HARD FAIL",
            f"expected_release_v2_{field}",
            passed,
            f"expected={expected}; observed={observed}",
        )
        if not passed:
            release_v2_mismatches.append(
                {"Field": field, "Expected": expected, "Observed": observed}
            )
    pd.DataFrame(
        release_v2_mismatches,
        columns=["Field", "Expected", "Observed"],
    ).to_csv(audit_dir / "expected_release_v2_mismatches.csv", index=False)

    release_manifest_fields = {
        "encounter_count": len(encounters),
        "subject_count": encounters["ResearchSubjectID"].nunique(),
        "full_image_count": len(images),
        "reviewed_encounter_count": int(
            (encounters["ReviewStatus"] == "Reviewed").sum()
        ),
        "unreviewed_encounter_count": int(
            (encounters["ReviewStatus"] == "Unreviewed").sum()
        ),
    }
    release_manifest_mismatches = [
        {"Field": field, "Expected": release.get(field), "Observed": observed}
        for field, observed in release_manifest_fields.items()
        if release.get(field) != observed
    ]
    pd.DataFrame(
        release_manifest_mismatches,
        columns=["Field", "Expected", "Observed"],
    ).to_csv(audit_dir / "release_manifest_mismatches.csv", index=False)
    add_check(
        checks,
        "HARD FAIL",
        "release_manifest_counts_match_tables",
        not release_manifest_mismatches,
        f"mismatched_fields={len(release_manifest_mismatches)}",
    )

    # recount images directly from the manifest and compare those numbers with
    # the counts stored in the encounter metadata.
    manifest_side_counts = (
        images.groupby([*ENCOUNTER_KEY, "Laterality"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["OD", "OS", "UNKNOWN"], fill_value=0)
        .rename(
            columns={
                "OD": "ManifestODImageCount",
                "OS": "ManifestOSImageCount",
                "UNKNOWN": "ManifestUnknownLateralityImageCount",
            }
        )
        .reset_index()
    )

    count_check = encounters[
        ENCOUNTER_KEY
        + [
            "ODImageCount",
            "OSImageCount",
            "UnknownLateralityImageCount",
            "TotalImageCount",
        ]
    ].merge(manifest_side_counts, on=ENCOUNTER_KEY, how="left")

    manifest_count_cols = [
        "ManifestODImageCount",
        "ManifestOSImageCount",
        "ManifestUnknownLateralityImageCount",
    ]
    count_check[manifest_count_cols] = count_check[manifest_count_cols].fillna(0).astype(int)
    count_check["ManifestTotalImageCount"] = count_check[manifest_count_cols].sum(axis=1)

    count_pairs = {
        "ODImageCount": "ManifestODImageCount",
        "OSImageCount": "ManifestOSImageCount",
        "UnknownLateralityImageCount": "ManifestUnknownLateralityImageCount",
        "TotalImageCount": "ManifestTotalImageCount",
    }
    for metadata_col, manifest_col in count_pairs.items():
        check_col = f"{metadata_col}Matches"
        count_check[check_col] = count_check[metadata_col] == count_check[manifest_col]
        mismatch_count = int((~count_check[check_col]).sum())
        add_check(
            checks,
            "HARD FAIL",
            f"{metadata_col}_matches_manifest",
            mismatch_count == 0,
            f"mismatched_encounters={mismatch_count}",
        )

    match_columns = [f"{column}Matches" for column in count_pairs]
    image_count_mismatches = count_check.loc[~count_check[match_columns].all(axis=1)].copy()
    image_count_mismatches.to_csv(
        audit_dir / "image_count_mismatches.csv", index=False
    )

    # keep the UNKNOWN rows as a separate audit file. They are not used as
    # B-scan candidates because only OD and OS rows are treated as eye images.
    laterality_counts = images["Laterality"].value_counts(dropna=False)
    unknown_images.to_csv(audit_dir / "unknown_laterality_images.csv", index=False)

    # Save a quick breakdown of the labels assigned to reviewed encounters.
    reviewed = encounters.loc[encounters["ReviewStatus"] == "Reviewed"].copy()
    reviewed_class_counts = (
        reviewed["EncounterClassification"]
        .value_counts(dropna=False)
        .rename_axis("EncounterClassification")
        .reset_index(name="Count")
    )
    reviewed_class_counts.to_csv(
        audit_dir / "reviewed_encounter_class_counts.csv", index=False
    )

    # convert the encounter table into an eye-level table. This is only an
    # audit view at this stage, not the final model-training dataset.
    eye_rows = []
    for laterality in ["OD", "OS"]:
        side = encounters[
            ENCOUNTER_KEY
            + [
                "ReviewStatus",
                "EncounterClassification",
                f"{laterality}_Status",
                f"{laterality}_Findings",
                f"{laterality}_ImageQuality",
                "SurgicalContextTags",
                "ReviewerNotes",
            ]
        ].copy()
        side["Laterality"] = laterality
        side = side.rename(
            columns={
                f"{laterality}_Status": "EyeStatus",
                f"{laterality}_Findings": "EyeFindings",
                f"{laterality}_ImageQuality": "EyeImageQuality",
            }
        )
        eye_rows.append(side)

    eyes = pd.concat(eye_rows, ignore_index=True)
    side_manifest_counts = (
        bscan_images.groupby(EYE_KEY)
        .size()
        .rename("ManifestImageCount")
        .reset_index()
    )
    eyes = eyes.merge(side_manifest_counts, on=EYE_KEY, how="left")
    eyes["ManifestImageCount"] = eyes["ManifestImageCount"].fillna(0).astype(int)

    # flag cases where metadata says "Not imaged" but the manifest still contains OD/OS images.
    laterality_conflicts = eyes.loc[
        (eyes["ReviewStatus"] == "Reviewed")
        & (eyes["EyeStatus"] == "Not imaged")
        & (eyes["ManifestImageCount"] > 0)
    ].copy()
    laterality_conflicts["AuditHold"] = True
    laterality_conflicts["AuditHoldReason"] = "Laterality conflict"
    laterality_conflicts["IncludePrimary"] = False
    laterality_conflicts.to_csv(
        audit_dir / "laterality_conflicts.csv", index=False
    )

    conflict_encounter_keys = laterality_conflicts[ENCOUNTER_KEY].drop_duplicates()

    # keep the same provisional eligibility rule used in the original audit.
    eligible_eyes = eyes.loc[
        (eyes["ReviewStatus"] == "Reviewed")
        & (eyes["EncounterClassification"].isin(["Normal", "Abnormal"]))
        & (eyes["EyeStatus"] == "Imaged")
        & (eyes["ManifestImageCount"] > 0)
    ].copy()

    # findings are the eye-level label source: blank stays unlabeled, "Normal" stays Normal, everything else is Abnormal.
    findings_clean = eligible_eyes["EyeFindings"].fillna("").astype(str).str.strip()
    eligible_eyes["EyeLabel"] = pd.Series(pd.NA, index=eligible_eyes.index, dtype="string")
    eligible_eyes.loc[findings_clean.eq("Normal"), "EyeLabel"] = "Normal"
    eligible_eyes.loc[findings_clean.ne("") & findings_clean.ne("Normal"), "EyeLabel"] = "Abnormal"

    empty_findings = eligible_eyes.loc[findings_clean.eq("")].copy()
    empty_findings.to_csv(
        audit_dir / "eligible_eyes_with_blank_findings.csv", index=False
    )
    add_check(
        checks,
        "HARD FAIL",
        "eligible_normal_abnormal_eyes_have_findings",
        empty_findings.empty,
        f"blank_finding_eyes={len(empty_findings)}",
    )

    conflict_key_index = pd.MultiIndex.from_frame(conflict_encounter_keys)
    eligible_key_index = pd.MultiIndex.from_frame(eligible_eyes[ENCOUNTER_KEY])
    eligible_eyes["AuditHold"] = eligible_key_index.isin(conflict_key_index)
    eligible_eyes["AuditHoldReason"] = ""
    eligible_eyes.loc[eligible_eyes["AuditHold"], "AuditHoldReason"] = "Laterality conflict"
    eligible_eyes["IncludePrimary"] = (
        ~eligible_eyes["AuditHold"] & eligible_eyes["EyeLabel"].notna()
    )

    eligible_eyes.to_csv(
        audit_dir / "provisional_eligible_eyes.csv", index=False
    )
    primary_candidates = eligible_eyes.loc[eligible_eyes["IncludePrimary"]].copy()
    primary_candidates.to_csv(
        audit_dir / "primary_candidate_eyes.csv", index=False
    )

    # rebuild the encounter label from its eyes and make sure it agrees with the reviewed label.
    derived_encounter = (
        eligible_eyes.loc[eligible_eyes["EyeLabel"].notna()]
        .groupby(ENCOUNTER_KEY)["EyeLabel"]
        .apply(lambda values: "Abnormal" if (values == "Abnormal").any() else "Normal")
        .rename("DerivedEncounterClassification")
        .reset_index()
    )
    expected_encounter = reviewed.loc[
        reviewed["EncounterClassification"].isin(["Normal", "Abnormal"]),
        ENCOUNTER_KEY + ["EncounterClassification"],
    ]
    label_check = expected_encounter.merge(
        derived_encounter, on=ENCOUNTER_KEY, how="left"
    )
    label_check["Matches"] = (
        label_check["EncounterClassification"]
        == label_check["DerivedEncounterClassification"]
    )
    label_mismatches = label_check.loc[~label_check["Matches"]].copy()
    label_mismatches.to_csv(
        audit_dir / "eye_to_encounter_label_mismatches.csv", index=False
    )
    add_check(
        checks,
        "HARD FAIL",
        "eye_labels_match_encounter_classification",
        label_mismatches.empty,
        f"mismatched_encounters={len(label_mismatches)}",
    )

    # postoperative wording is only used to create a review queue. These rows
    # stay in the data unless another rule excludes them.
    postoperative_source_columns = [
        "SurgicalContextTags",
        "EyeFindings",
        "ReviewerNotes",
    ]
    postoperative_matches = pd.DataFrame(
        {
            column: eligible_eyes[column]
            .fillna("")
            .astype(str)
            .str.contains(POSTOPERATIVE_PATTERN, na=False)
            for column in postoperative_source_columns
        },
        index=eligible_eyes.index,
    )
    postoperative_queue = eligible_eyes.loc[postoperative_matches.any(axis=1)].copy()
    postoperative_queue["PostoperativeFlagSources"] = postoperative_matches.loc[
        postoperative_queue.index
    ].apply(
        lambda row: " | ".join(row.index[row].tolist()),
        axis=1,
    )
    postoperative_queue["ReviewFlagOnly"] = True
    postoperative_queue["PostoperativeAutoExclude"] = False
    postoperative_queue.to_csv(
        audit_dir / "postoperative_review_queue.csv", index=False
    )

    # SHA-256 gives us an exact duplicate check. First check the whole release,
    # then check duplicates that occur within the same eligible eye.
    hashes = images["SHA256"].fillna("").astype(str).str.strip()
    valid_hashes = hashes.str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    hash_counts = images.loc[valid_hashes, "SHA256"].value_counts()
    duplicate_hashes = hash_counts.loc[hash_counts > 1].index

    duplicate_images = images.loc[images["SHA256"].isin(duplicate_hashes)].copy()
    duplicate_images = duplicate_images.sort_values(
        ["SHA256", *EYE_KEY, "ImageFileName"]
    )
    duplicate_images.to_csv(audit_dir / "duplicate_images.csv", index=False)

    redundant_global = int((hash_counts.loc[hash_counts > 1] - 1).sum())
    eligible_keys = eligible_eyes[EYE_KEY].drop_duplicates()
    eligible_images = bscan_images.merge(eligible_keys, on=EYE_KEY, how="inner")
    eligible_image_hashes = eligible_images["SHA256"].fillna("").astype(str).str.strip()
    eligible_images_with_valid_hash = eligible_images.loc[
        eligible_image_hashes.str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    ].copy()
    within_eye_counts = (
        eligible_images_with_valid_hash.groupby([*EYE_KEY, "SHA256"])
        .size()
        .rename("Copies")
        .reset_index()
    )
    within_eye_duplicates = within_eye_counts.loc[
        within_eye_counts["Copies"] > 1
    ].copy()
    within_eye_duplicates.to_csv(
        audit_dir / "duplicate_hashes_within_eligible_eyes.csv", index=False
    )

    redundant_within_eyes = int((within_eye_duplicates["Copies"] - 1).sum())
    affected_eyes = within_eye_duplicates[EYE_KEY].drop_duplicates()

    # these are review items, not reasons to fail the whole release.
    add_check(
        checks,
        "WARNING / HOLD",
        "reviewed_laterality_conflicts",
        laterality_conflicts.empty,
        f"conflict_rows={len(laterality_conflicts)}; affected provisional eyes held={int(eligible_eyes['AuditHold'].sum())}",
    )
    add_check(
        checks,
        "WARNING / HOLD",
        "global_exact_duplicate_images",
        duplicate_images.empty,
        f"duplicate_hash_groups={len(duplicate_hashes)}; redundant_copies={redundant_global}",
    )
    add_check(
        checks,
        "WARNING / HOLD",
        "exact_duplicates_within_eligible_eyes",
        within_eye_duplicates.empty,
        f"affected_eyes={len(affected_eyes)}; redundant_copies={redundant_within_eyes}",
    )
    add_check(
        checks,
        "WARNING / HOLD",
        "postoperative_context_review_queue",
        postoperative_queue.empty,
        f"flagged_provisional_eyes={len(postoperative_queue)}; review flag only",
    )

    # work out the final status and collect the numbers that are useful for
    # comparing this run with earlier audit runs.
    checks_df = pd.DataFrame(checks)
    hard_fail_count = int((checks_df["Status"] == "HARD FAIL").sum())
    warning_count = int((checks_df["Status"] == "WARNING / HOLD").sum())
    if hard_fail_count:
        overall_status = "HARD FAIL"
    elif warning_count:
        overall_status = "WARNING / HOLD"
    else:
        overall_status = "PASS"

    add_summary_metric(summary, "run_id", run_id)
    add_summary_metric(summary, "overall_audit_status", overall_status)
    add_summary_metric(summary, "hard_fail_check_count", hard_fail_count)
    add_summary_metric(summary, "warning_hold_check_count", warning_count)
    add_summary_metric(summary, "release_version", release.get("release_version"))
    add_summary_metric(summary, "encounters_metadata_rows", len(encounters))
    add_summary_metric(summary, "subjects_metadata", encounters["ResearchSubjectID"].nunique())
    add_summary_metric(summary, "images_manifest_rows", len(images))
    add_summary_metric(summary, "reviewed_encounters", int((encounters["ReviewStatus"] == "Reviewed").sum()))
    add_summary_metric(summary, "unreviewed_encounters", int((encounters["ReviewStatus"] == "Unreviewed").sum()))
    for value, count in laterality_counts.items():
        add_summary_metric(summary, f"manifest_laterality_{value}", int(count))
    add_summary_metric(summary, "a_scan_unknown_images_removed", len(unknown_images))
    add_summary_metric(summary, "bscan_candidate_images", len(bscan_images))
    add_summary_metric(summary, "bscan_candidate_encounters", len(bscan_images[ENCOUNTER_KEY].drop_duplicates()))
    add_summary_metric(summary, "bscan_candidate_subjects", bscan_images["ResearchSubjectID"].nunique())
    add_summary_metric(summary, "encounter_image_count_mismatches", len(image_count_mismatches))
    for _, row in reviewed_class_counts.iterrows():
        add_summary_metric(summary, f"reviewed_class_{row['EncounterClassification']}", int(row["Count"]))
    add_summary_metric(summary, "provisional_eligible_eye_exams", len(eligible_eyes))
    add_summary_metric(summary, "audit_held_eye_exams", int(eligible_eyes["AuditHold"].sum()))
    add_summary_metric(summary, "primary_candidates_after_audit_holds", len(primary_candidates))
    add_summary_metric(summary, "postoperative_review_queue_count", len(postoperative_queue))
    add_summary_metric(summary, "eligible_reviewed_subjects", eligible_eyes["ResearchSubjectID"].nunique())
    add_summary_metric(summary, "eligible_eye_normal", int((eligible_eyes["EyeLabel"] == "Normal").sum()))
    add_summary_metric(summary, "eligible_eye_abnormal", int((eligible_eyes["EyeLabel"] == "Abnormal").sum()))
    add_summary_metric(summary, "eligible_eyes_blank_findings", len(empty_findings))
    add_summary_metric(summary, "eye_to_encounter_label_mismatches", len(label_mismatches))
    add_summary_metric(summary, "reviewed_laterality_conflicts", len(laterality_conflicts))
    add_summary_metric(summary, "duplicate_sha256_groups_global", len(duplicate_hashes))
    add_summary_metric(summary, "redundant_image_copies_global", redundant_global)
    add_summary_metric(summary, "eligible_eyes_with_exact_duplicates", len(affected_eyes))
    add_summary_metric(summary, "redundant_image_copies_within_eligible_eyes", redundant_within_eyes)
    add_summary_metric(summary, "data_dictionary_rows", len(data_dictionary))

    checks_df.to_csv(audit_dir / "audit_checks.csv", index=False)
    pd.DataFrame(summary).to_csv(audit_dir / "dataset_summary.csv", index=False)





    print(f"AUDIT STATUS: {overall_status}\n")
    print(f"Subjects                              : {encounters['ResearchSubjectID'].nunique():,}")
    print(f"Encounters                            : {len(encounters):,}")
    print(f"Images                                : {len(images):,}")
    print(f"Reviewed encounters                   : {(encounters['ReviewStatus'] == 'Reviewed').sum():,}")
    print(f"Unreviewed encounters                 : {(encounters['ReviewStatus'] == 'Unreviewed').sum():,}")
    print()
    print(f"UNKNOWN / A-scan images               : {len(unknown_images):,}")
    print(f"B-scan candidate images               : {len(bscan_images):,}")
    print(f"B-scan candidate encounters           : {len(bscan_images[ENCOUNTER_KEY].drop_duplicates()):,}")
    print(f"B-scan candidate subjects             : {bscan_images['ResearchSubjectID'].nunique():,}")
    print()
    print(f"Provisional eligible eye-exams        : {len(eligible_eyes):,}")
    print(f"  Normal                              : {(eligible_eyes['EyeLabel'] == 'Normal').sum():,}")
    print(f"  Abnormal                            : {(eligible_eyes['EyeLabel'] == 'Abnormal').sum():,}")
    print(f"Audit-held eye-exams                  : {eligible_eyes['AuditHold'].sum():,}")
    print(f"Primary candidates after audit holds : {len(primary_candidates):,}")
    print(f"Postoperative review queue            : {len(postoperative_queue):,}")
    print()
    print_check_section(checks_df, "HARD FAIL")
    print_check_section(checks_df, "WARNING / HOLD")
    print(f"Audit files written to:\n{audit_dir}")

    if hard_fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
