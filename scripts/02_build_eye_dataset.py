# Build the canonical B-scan eye dataset used by the later training steps.
# The script also handles duplicate images and runs consistency checks before
# anything is written to data/interim.

from datetime import datetime
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"

ENCOUNTER_KEY = ["ResearchSubjectID", "EncounterID"]
EYE_KEY = [*ENCOUNTER_KEY, "Laterality"]
ALLOWED_BSCAN_LATERALITY = {"OD", "OS"}

EXPECTED_COUNTS = {
    "bscan_images": 15808,
    "provisional_eligible_eyes": 1514,
    "provisional_normal_eyes": 402,
    "provisional_abnormal_eyes": 1112,
    "audit_held_provisional_eyes": 4,
    "include_primary_eyes": 1510,
}

POSTOPERATIVE_PATTERN = re.compile(
    r"(?:silicone[ -]?oil|gas(?:[ -]?tamponade)?|ppv|vitrectom|"
    r"oil[ -]?removal|tamponade|emulsified[ -]?oil)",
    flags=re.IGNORECASE,
)


def locate_release_file(filename: str) -> Path:
    """Return the single matching release file from data/raw."""
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


def build_eye_exam_id(frame: pd.DataFrame) -> pd.Series:
    """Create one ID for each subject / encounter / eye combination."""
    return (
        frame["ResearchSubjectID"].astype(str)
        + "__"
        + frame["EncounterID"].astype(str)
        + "__"
        + frame["Laterality"].astype(str)
    )


def record_check(rows, check, passed, details):
    rows.append(
        {
            "Check": check,
            "Status": "PASS" if passed else "FAIL",
            "Details": details,
        }
    )


def build_exclusion_reasons(eye_master: pd.DataFrame, provisional_eligible: pd.Series) -> pd.Series:
    """Collect the reasons an eye is not part of the primary dataset."""
    reasons = pd.Series("", index=eye_master.index, dtype="string")

    def add_reason(mask: pd.Series, reason: str):
        existing = reasons.loc[mask]
        reasons.loc[mask] = existing.where(existing.eq(""), existing + "; ") + reason

    add_reason(eye_master["ReviewStatus"] != "Reviewed", "Not reviewed")
    add_reason(~eye_master["EyeStatusNormalized"].eq("imaged"), "Eye not imaged/interpretable")
    add_reason(eye_master["EyeLabel"].isna(), "Missing eye label")
    add_reason(eye_master["AuditHold"], "Laterality conflict")

    reasons.loc[provisional_eligible & ~eye_master["AuditHold"]] = ""
    return reasons


def main():
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_ROOT / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    validation_path = audit_dir / "step2_eye_dataset_checks.csv"

    encounter_path = locate_release_file("Bscan_all_encounters_metadata_260811.csv")
    manifest_path = locate_release_file("Bscan_full_image_manifest_260811.csv")

    print("=" * 72)
    print("STEP 2: BUILD CANONICAL B-SCAN EYE DATASET")
    print("=" * 72)
    print(f"Run ID             : {run_id}")
    print(f"Encounter metadata : {encounter_path}")
    print(f"Image manifest     : {manifest_path}")
    print(f"Validation output  : {validation_path}\n")

    encounters = pd.read_csv(encounter_path, low_memory=False)
    images = pd.read_csv(manifest_path, low_memory=False)
    validation_checks = []

    required_encounter_columns = {
        "ResearchSubjectID",
        "EncounterID",
        "ReviewStatus",
        "EncounterClassification",
        "OD_Status",
        "OD_Findings",
        "OD_ImageQuality",
        "OS_Status",
        "OS_Findings",
        "OS_ImageQuality",
        "SurgicalContextTags",
        "ReviewerNotes",
    }
    required_image_columns = {
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "ImageID",
        "ImageFileName",
        "ImageRelativePath",
        "AssignmentStatus",
        "SHA256",
        "Bytes",
        "Width",
        "Height",
        "ReviewStatus",
        "EncounterClassification",
    }
    missing_encounter_columns = sorted(
        required_encounter_columns - set(encounters.columns)
    )
    missing_image_columns = sorted(required_image_columns - set(images.columns))
    record_check(
        validation_checks,
        "required_source_columns",
        not missing_encounter_columns and not missing_image_columns,
        f"encounter_missing={missing_encounter_columns}; manifest_missing={missing_image_columns}",
    )
    if missing_encounter_columns or missing_image_columns:
        pd.DataFrame(validation_checks).to_csv(validation_path, index=False)
        print("STEP-2 VALIDATION: FAIL")
        print("Required source columns are missing; no canonical files were written.")
        print(f"Validation checks written to:\n{validation_path}")
        raise SystemExit(1)

    # Each encounter should appear only once in the metadata.
    duplicate_encounter_rows = encounters.duplicated(ENCOUNTER_KEY, keep=False)
    blank_encounter_key = encounters[ENCOUNTER_KEY].isna().any(axis=1)
    encounter_keys_valid = not (duplicate_encounter_rows | blank_encounter_key).any()
    record_check(
        validation_checks,
        "source_encounter_composite_key_valid",
        encounter_keys_valid,
        f"invalid_or_duplicate_rows={int((duplicate_encounter_rows | blank_encounter_key).sum())}",
    )

    # UNKNOWN rows stay in the raw release, but they are not usable B-scan views.
    bscan_images = images.loc[
        images["Laterality"].isin(ALLOWED_BSCAN_LATERALITY)
    ].copy()
    source_invalid_laterality = images.loc[
        ~images["Laterality"].isin({"OD", "OS", "UNKNOWN"})
        | images["Laterality"].isna()
    ]
    record_check(
        validation_checks,
        "source_manifest_laterality_valid",
        source_invalid_laterality.empty,
        f"invalid_rows={len(source_invalid_laterality)}",
    )
    record_check(
        validation_checks,
        "expected_bscan_image_count",
        len(bscan_images) == EXPECTED_COUNTS["bscan_images"],
        f"expected={EXPECTED_COUNTS['bscan_images']}; observed={len(bscan_images)}",
    )

    # Build the image-level table first. Exact duplicates within the same eye
    # are kept for traceability, but only one copy is marked for model input.
    bscan_images["EyeExamID"] = build_eye_exam_id(bscan_images)
    image_output_columns = [
        "EyeExamID",
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
        "AssignmentStatus",
        "ReviewStatus",
        "EncounterClassification",
    ]
    eye_image_manifest = bscan_images[image_output_columns].copy()

    clean_hash = eye_image_manifest["SHA256"].fillna("").astype(str).str.strip()
    valid_sha = clean_hash.str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    valid_paths = (
        eye_image_manifest["ImageRelativePath"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    record_check(
        validation_checks,
        "bscan_images_have_valid_sha256",
        valid_sha.all(),
        f"invalid_rows={int((~valid_sha).sum())}",
    )
    record_check(
        validation_checks,
        "bscan_images_have_paths",
        valid_paths.all(),
        f"blank_path_rows={int((~valid_paths).sum())}",
    )

    # Use a lowercase hash only while comparing duplicates. Keep the original hash in the CSV.
    eye_image_manifest["_SHA256Normalized"] = clean_hash.str.lower()
    eye_image_manifest = eye_image_manifest.sort_values(
        ["EyeExamID", "_SHA256Normalized", "ImageRelativePath"],
        kind="mergesort",
    ).reset_index(drop=True)
    duplicate_group = ["EyeExamID", "_SHA256Normalized"]
    duplicate_rank = eye_image_manifest.groupby(duplicate_group).cumcount()
    canonical_path = eye_image_manifest.groupby(duplicate_group)[
        "ImageRelativePath"
    ].transform("first")
    eye_image_manifest["IsExactDuplicateWithinEye"] = duplicate_rank.gt(0)
    eye_image_manifest["DuplicateOf"] = pd.Series(
        pd.NA, index=eye_image_manifest.index, dtype="string"
    )
    duplicate_mask = eye_image_manifest["IsExactDuplicateWithinEye"]
    eye_image_manifest.loc[duplicate_mask, "DuplicateOf"] = canonical_path.loc[
        duplicate_mask
    ]
    eye_image_manifest["IncludeModelInput"] = ~duplicate_mask
    eye_image_manifest = eye_image_manifest.drop(columns="_SHA256Normalized")
    eye_image_manifest = eye_image_manifest.sort_values(
        ["EyeExamID", "ImageRelativePath"], kind="mergesort"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Canonical eye-level master: every manifest-represented OD/OS eye
    # -----------------------------------------------------
    manifest_eye_keys = eye_image_manifest[EYE_KEY].drop_duplicates()
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
                f"{laterality}_ImageQuality": "ImageQuality",
            }
        )
        eye_rows.append(side)

    metadata_eyes = pd.concat(eye_rows, ignore_index=True)
    eye_master = manifest_eye_keys.merge(
        metadata_eyes,
        on=EYE_KEY,
        how="left",
        validate="one_to_one",
    )
    eye_master["EyeExamID"] = build_eye_exam_id(eye_master)

    # Eye findings are the source of the Normal / Abnormal label. Blank findings
    # stay unlabeled instead of being treated as abnormal.
    findings_clean = eye_master["EyeFindings"].fillna("").astype(str).str.strip()
    findings_normal = findings_clean.str.casefold().eq("normal")
    eye_master["EyeLabel"] = pd.Series(pd.NA, index=eye_master.index, dtype="string")
    eye_master.loc[findings_normal, "EyeLabel"] = "Normal"
    eye_master.loc[findings_clean.ne("") & ~findings_normal, "EyeLabel"] = "Abnormal"
    record_check(
        validation_checks,
        "blank_findings_never_become_abnormal",
        eye_master.loc[findings_clean.eq(""), "EyeLabel"].isna().all(),
        f"violating_rows={int(eye_master.loc[findings_clean.eq(''), 'EyeLabel'].notna().sum())}",
    )

    image_counts = (
        eye_image_manifest.groupby("EyeExamID")
        .agg(
            OriginalImageCount=("ImageRelativePath", "size"),
            UniqueImageCount=("IncludeModelInput", "sum"),
        )
        .reset_index()
    )
    image_counts["UniqueImageCount"] = image_counts["UniqueImageCount"].astype(int)
    image_counts["DuplicateImageCount"] = (
        image_counts["OriginalImageCount"] - image_counts["UniqueImageCount"]
    )
    eye_master = eye_master.merge(image_counts, on="EyeExamID", how="left", validate="one_to_one")

    eye_master["EyeStatusNormalized"] = (
        eye_master["EyeStatus"].fillna("").astype(str).str.strip().str.casefold()
    )
    # A reviewed eye marked "Not imaged" should not also have OD/OS image files.
    # When that happens, hold the whole encounter so it can be reviewed manually.
    direct_laterality_conflicts = eye_master.loc[
        (eye_master["ReviewStatus"] == "Reviewed")
        & eye_master["EyeStatusNormalized"].eq("not imaged")
        & eye_master["OriginalImageCount"].gt(0)
    ].copy()
    conflict_encounter_keys = direct_laterality_conflicts[
        ENCOUNTER_KEY
    ].drop_duplicates()
    conflict_index = pd.MultiIndex.from_frame(conflict_encounter_keys)
    master_index = pd.MultiIndex.from_frame(eye_master[ENCOUNTER_KEY])
    eye_master["AuditHold"] = master_index.isin(conflict_index)
    eye_master["AuditHoldReason"] = ""
    eye_master.loc[eye_master["AuditHold"], "AuditHoldReason"] = "Laterality conflict"

    # Postoperative terms are only a review flag. They do not remove an eye from
    # the primary dataset by themselves.
    postop_matches = pd.DataFrame(
        {
            "SurgicalContextTags": eye_master["SurgicalContextTags"]
            .fillna("")
            .astype(str)
            .str.contains(POSTOPERATIVE_PATTERN, na=False),
            "EyeFindings": eye_master["EyeFindings"]
            .fillna("")
            .astype(str)
            .str.contains(POSTOPERATIVE_PATTERN, na=False),
            "ReviewerNotes": eye_master["ReviewerNotes"]
            .fillna("")
            .astype(str)
            .str.contains(POSTOPERATIVE_PATTERN, na=False),
        },
        index=eye_master.index,
    )
    eye_master["PostoperativeReviewFlag"] = postop_matches.any(axis=1)

    # This reproduces the eligibility rule established during the release audit.
    provisional_eligible = (
        eye_master["ReviewStatus"].eq("Reviewed")
        & eye_master["Laterality"].isin(ALLOWED_BSCAN_LATERALITY)
        & eye_master["EyeStatusNormalized"].eq("imaged")
        & eye_master["EyeLabel"].isin(["Normal", "Abnormal"])
    )
    eye_master["IncludePrimary"] = provisional_eligible & ~eye_master["AuditHold"]
    eye_master["ExclusionReason"] = build_exclusion_reasons(eye_master, provisional_eligible)

    provisional_count = int(provisional_eligible.sum())
    provisional_normal_count = int(
        (provisional_eligible & eye_master["EyeLabel"].eq("Normal")).sum()
    )
    provisional_abnormal_count = int(
        (provisional_eligible & eye_master["EyeLabel"].eq("Abnormal")).sum()
    )
    held_provisional_count = int((provisional_eligible & eye_master["AuditHold"]).sum())
    include_primary_count = int(eye_master["IncludePrimary"].sum())

    observed_counts = {
        "provisional_eligible_eyes": provisional_count,
        "provisional_normal_eyes": provisional_normal_count,
        "provisional_abnormal_eyes": provisional_abnormal_count,
        "audit_held_provisional_eyes": held_provisional_count,
        "include_primary_eyes": include_primary_count,
    }
    for name, observed in observed_counts.items():
        expected = EXPECTED_COUNTS[name]
        record_check(
            validation_checks,
            f"expected_{name}",
            observed == expected,
            f"expected={expected}; observed={observed}",
        )

    expected_inclusion = provisional_eligible & ~eye_master["AuditHold"]
    record_check(
        validation_checks,
        "postoperative_flag_does_not_change_primary_eligibility",
        eye_master["IncludePrimary"].equals(expected_inclusion),
        "IncludePrimary is determined without using PostoperativeReviewFlag",
    )

    master_output_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "ReviewStatus",
        "EncounterClassification",
        "EyeStatus",
        "EyeFindings",
        "EyeLabel",
        "ImageQuality",
        "OriginalImageCount",
        "UniqueImageCount",
        "DuplicateImageCount",
        "AuditHold",
        "AuditHoldReason",
        "PostoperativeReviewFlag",
        "IncludePrimary",
        "ExclusionReason",
    ]
    eye_master = eye_master[master_output_columns].sort_values(
        EYE_KEY, kind="mergesort"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Required Step-2 validation checks
    # -----------------------------------------------------
    record_check(
        validation_checks,
        "master_eye_exam_id_unique",
        eye_master["EyeExamID"].notna().all() and eye_master["EyeExamID"].is_unique,
        f"rows={len(eye_master)}; unique_eye_exam_ids={eye_master['EyeExamID'].nunique()}",
    )
    record_check(
        validation_checks,
        "master_composite_eye_key_unique",
        not eye_master.duplicated(EYE_KEY, keep=False).any(),
        f"duplicate_rows={int(eye_master.duplicated(EYE_KEY, keep=False).sum())}",
    )
    record_check(
        validation_checks,
        "image_manifest_has_no_unknown_laterality",
        not eye_image_manifest["Laterality"].eq("UNKNOWN").any(),
        f"unknown_rows={int(eye_image_manifest['Laterality'].eq('UNKNOWN').sum())}",
    )
    invalid_output_laterality = ~eye_image_manifest["Laterality"].isin(
        ALLOWED_BSCAN_LATERALITY
    )
    record_check(
        validation_checks,
        "image_manifest_only_od_os",
        not invalid_output_laterality.any(),
        f"invalid_rows={int(invalid_output_laterality.sum())}",
    )

    # Every image row should point back to exactly one eye-level record, and the
    # subject / encounter / laterality pieces must agree on both sides.
    image_to_eye = eye_image_manifest.merge(
        eye_master[["EyeExamID", *EYE_KEY]],
        on="EyeExamID",
        how="left",
        suffixes=("_image", "_master"),
        validate="many_to_one",
        indicator=True,
    )
    all_images_have_master = image_to_eye["_merge"].eq("both").all()
    matching_components = all(
        image_to_eye[f"{column}_image"].equals(
            image_to_eye[f"{column}_master"]
        )
        for column in EYE_KEY
    )
    record_check(
        validation_checks,
        "every_image_maps_to_one_master_eye",
        all_images_have_master and matching_components,
        f"unmapped_rows={int((image_to_eye['_merge'] != 'both').sum())}; key_mismatch={not matching_components}",
    )
    image_id_eye_counts = eye_image_manifest.groupby("ImageID")["EyeExamID"].nunique()
    record_check(
        validation_checks,
        "every_image_id_belongs_to_one_eye",
        image_id_eye_counts.le(1).all(),
        f"image_ids_with_multiple_eyes={int(image_id_eye_counts.gt(1).sum())}",
    )

    validation_counts = (
        eye_image_manifest.groupby("EyeExamID")
        .agg(
            ObservedOriginal=("ImageRelativePath", "size"),
            ObservedUnique=("IncludeModelInput", "sum"),
        )
        .reset_index()
    )
    count_validation = eye_master[
        [
            "EyeExamID",
            "OriginalImageCount",
            "UniqueImageCount",
            "DuplicateImageCount",
        ]
    ].merge(validation_counts, on="EyeExamID", how="left", validate="one_to_one")
    original_matches = count_validation["OriginalImageCount"].eq(
        count_validation["ObservedOriginal"]
    )
    unique_matches = count_validation["UniqueImageCount"].eq(
        count_validation["ObservedUnique"]
    )
    duplicate_matches = count_validation["DuplicateImageCount"].eq(
        count_validation["OriginalImageCount"]
        - count_validation["UniqueImageCount"]
    )
    record_check(
        validation_checks,
        "original_image_counts_match_manifest",
        original_matches.all(),
        f"mismatched_eyes={int((~original_matches).sum())}",
    )
    record_check(
        validation_checks,
        "unique_image_counts_match_model_input_rows",
        unique_matches.all(),
        f"mismatched_eyes={int((~unique_matches).sum())}",
    )
    record_check(
        validation_checks,
        "duplicate_image_count_calculation",
        duplicate_matches.all(),
        f"mismatched_eyes={int((~duplicate_matches).sum())}",
    )

    # Re-check the duplicate bookkeeping so the chosen canonical image is stable
    # across runs and duplicate rows can never become model inputs.
    duplicate_rows = eye_image_manifest["IsExactDuplicateWithinEye"]
    canonical_rows = ~duplicate_rows
    normalized_hash = eye_image_manifest["SHA256"].astype(str).str.strip().str.lower()
    deterministic_canonical_path = (
        eye_image_manifest.assign(_SHA256Normalized=normalized_hash)
        .groupby(["EyeExamID", "_SHA256Normalized"])["ImageRelativePath"]
        .transform("min")
    )
    duplicate_flags_valid = (
        eye_image_manifest.loc[duplicate_rows, "DuplicateOf"].notna().all()
        and ~eye_image_manifest.loc[duplicate_rows, "IncludeModelInput"].any()
        and eye_image_manifest.loc[canonical_rows, "DuplicateOf"].isna().all()
        and eye_image_manifest.loc[canonical_rows, "IncludeModelInput"].all()
    )
    record_check(
        validation_checks,
        "within_eye_duplicate_flags_valid",
        duplicate_flags_valid,
        f"duplicate_rows={int(duplicate_rows.sum())}",
    )
    canonical_path_valid = (
        eye_image_manifest.loc[canonical_rows, "ImageRelativePath"]
        .eq(deterministic_canonical_path.loc[canonical_rows])
        .all()
        and eye_image_manifest.loc[duplicate_rows, "DuplicateOf"]
        .eq(deterministic_canonical_path.loc[duplicate_rows])
        .all()
    )
    record_check(
        validation_checks,
        "duplicate_canonical_path_is_deterministic",
        canonical_path_valid,
        "canonical path is the first ImageRelativePath after deterministic sorting",
    )

    checks_df = pd.DataFrame(validation_checks)
    checks_df.to_csv(validation_path, index=False)
    failed_checks = checks_df.loc[checks_df["Status"] == "FAIL"]
    if not failed_checks.empty:
        print("STEP-2 VALIDATION: FAIL\n")
        for row in failed_checks.itertuples(index=False):
            print(f"[FAIL] {row.Check}: {row.Details}")
        print("\nCanonical interim files were not written.")
        print(f"Validation checks written to:\n{validation_path}")
        raise SystemExit(1)

    # Write canonical outputs only after all validation validation_checks pass.
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    master_path = INTERIM_DIR / "bscan_eye_level_master.csv"
    image_manifest_path = INTERIM_DIR / "bscan_eye_image_manifest.csv"
    eye_master.to_csv(master_path, index=False)
    eye_image_manifest.to_csv(image_manifest_path, index=False)

    reviewed_counts = eye_master["ReviewStatus"].value_counts(dropna=False)
    image_distribution = eye_master["OriginalImageCount"].value_counts().sort_index()

    print("STEP-2 VALIDATION: PASS\n")
    print(f"Canonical eye-exams                  : {len(eye_master):,}")
    print(f"B-scan image rows                    : {len(eye_image_manifest):,}")
    print(f"Unique model-input images            : {eye_image_manifest['IncludeModelInput'].sum():,}")
    print(f"Reviewed eye-exams                   : {reviewed_counts.get('Reviewed', 0):,}")
    print(f"Unreviewed eye-exams                 : {reviewed_counts.get('Unreviewed', 0):,}")
    print(f"Provisional eligible eye-exams       : {provisional_count:,}")
    print(f"  Normal                             : {provisional_normal_count:,}")
    print(f"  Abnormal                           : {provisional_abnormal_count:,}")
    print(f"Audit-held provisional eye-exams     : {held_provisional_count:,}")
    print(f"IncludePrimary eye-exams             : {include_primary_count:,}")
    print(f"All master eye-exams on audit hold   : {eye_master['AuditHold'].sum():,}")
    print(f"Postoperative-review-flagged eyes    : {eye_master['PostoperativeReviewFlag'].sum():,}")
    print("\nOriginal image-count distribution per eye:")
    for image_count, eye_count in image_distribution.items():
        print(f"  {int(image_count):>2} image(s): {int(eye_count):,} eye-exams")
    print("\nFiles created:")
    print(master_path)
    print(image_manifest_path)
    print(validation_path)


if __name__ == "__main__":
    main()
