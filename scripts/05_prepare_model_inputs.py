from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "interim" / "bscan_eye_image_manifest.csv"
)
SPLIT_PATHS = {
    "train": PROJECT_ROOT / "data" / "splits" / "train_eye_exams.csv",
    "validation": PROJECT_ROOT / "data" / "splits" / "val_eye_exams.csv",
    "test": PROJECT_ROOT / "data" / "splits" / "test_eye_exams.csv",
}
OUTPUT_PATHS = {
    "train": PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "train_images.csv",
    "validation": PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "val_images.csv",
    "test": PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "test_images.csv",
}
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"

MAX_VIEWS = 6
EXPECTED_EYES = 800
ALLOWED_LATERALITY = {"OD", "OS"}

# there is no image-level acquisition order in the release
# image paths are used so the ordering stays deterministic across runs
ORDERING_RULE = "ImageRelativePath ascending (no acquisition-order field available)"
RULE_ALL_AVAILABLE = "PATH_ASC_ALL_AVAILABLE_LE6"
RULE_EVENLY_SPACED = "PATH_ASC_EVEN6_ROUND_NEAREST_INCLUDE_ENDPOINTS"


def record_check(rows, check, passed, details, split="overall"):
    rows.append(
        {
            "Scope": split,
            "Check": check,
            "Status": "PASS" if passed else "FAIL",
            "Details": details,
        }
    )


def parse_bool_column(series: pd.Series) -> tuple[pd.Series, list[str]]:
    "parse a boolean csv column and keep track of any unexpected values."
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = sorted(
        normalized.loc[~normalized.isin(["true", "false"])]
        .dropna()
        .astype(str)
        .unique()
    )
    return normalized.eq("true").fillna(False), invalid


def pick_evenly_spaced_indices(available_count: int) -> list[int]:
    "pick up to six evenly spaced positions, including the first and last."
    if available_count <= MAX_VIEWS:
        return list(range(available_count))
    denominator = MAX_VIEWS - 1
    return [
        (index * (available_count - 1) + denominator // 2) // denominator
        for index in range(MAX_VIEWS)
    ]


def select_views(usable_images: pd.DataFrame) -> pd.DataFrame:
    "choose model views using only the available image paths."
    selected_groups = []
    for eye_exam_id, group in usable_images.groupby("EyeExamID", sort=True):
        ordered = group.sort_values("ImageRelativePath", kind="mergesort").reset_index(
            drop=True
        )
        available_count = len(ordered)
        indices = pick_evenly_spaced_indices(available_count)
        selected = ordered.iloc[indices].copy().reset_index(drop=True)
        selected["SelectedViewIndex"] = range(1, len(selected) + 1)
        selected["AvailableUniqueImageCount"] = available_count
        selected["SelectedImageCount"] = len(selected)
        selected["ViewSelectionRule"] = (
            RULE_ALL_AVAILABLE
            if available_count <= MAX_VIEWS
            else RULE_EVENLY_SPACED
        )
        selected_groups.append(selected)

    if not selected_groups:
        return usable_images.iloc[0:0].copy()
    return (
        pd.concat(selected_groups, ignore_index=True)
        .sort_values(["EyeExamID", "SelectedViewIndex"], kind="mergesort")
        .reset_index(drop=True)
    )


def selected_path_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(
        zip(
            table["EyeExamID"].astype(str),
            table["ImageRelativePath"].astype(str),
        )
    )


def stop_on_validation_failure(checks, checks_path: Path, message: str):
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    print("STEP-5 VALIDATION: HARD FAIL\n")
    for row in checks:
        if row["Status"] == "FAIL":
            print(
                f"[FAIL] {row['Scope']} :: {row['Check']}: "
                f"{row['Details']}"
            )
    print(f"\n{message}")
    print(f"Validation report:\n{checks_path}")
    raise SystemExit(1)


def main():
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_ROOT / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    checks_path = audit_dir / "step5_model_input_checks.csv"
    summary_path = audit_dir / "step5_model_input_summary.csv"
    checks = []

    print("=" * 72)
    print("STEP 5: PREPARE DETERMINISTIC MODEL-INPUT VIEWS")
    print("=" * 72)
    print(f"Run ID             : {run_id}")
    print(f"Image manifest     : {IMAGE_MANIFEST_PATH}")
    print(f"Maximum views      : {MAX_VIEWS}")
    print(f"Ordering rule      : {ORDERING_RULE}")
    print(
        "Selection rule     : <=6 retain all; >6 choose nearest indices to "
        "i*(n-1)/5 for i=0..5"
    )
    print("                     (first and last ordered paths always included)\n")

    # make sure all locked inputs are available before doing any work
    manifest_exists = IMAGE_MANIFEST_PATH.is_file()
    record_check(
        checks,
        "image_manifest_exists",
        manifest_exists,
        str(IMAGE_MANIFEST_PATH),
    )
    for split, path in SPLIT_PATHS.items():
        record_check(
            checks,
            "locked_eye_split_exists",
            path.is_file(),
            str(path),
            split,
        )
    if not manifest_exists or not all(path.is_file() for path in SPLIT_PATHS.values()):
        stop_on_validation_failure(
            checks,
            checks_path,
            "A required locked input is missing; model-input files were not written.",
        )

    # load the image manifest and the patient-level splits from step 4
    image_manifest = pd.read_csv(IMAGE_MANIFEST_PATH, low_memory=False)
    split_tables = {
        split: pd.read_csv(path, low_memory=False)
        for split, path in SPLIT_PATHS.items()
    }

    # fail early if a previous step produced an unexpected schema
    required_manifest_columns = {
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
        "IncludeModelInput",
    }
    required_split_columns = {
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "DatasetSplit",
        "SplitSeed",
        "UniqueImageCount",
    }
    missing_manifest = sorted(required_manifest_columns - set(image_manifest.columns))
    record_check(
        checks,
        "image_manifest_required_columns",
        not missing_manifest,
        f"missing={missing_manifest}",
    )
    missing_split_columns = {
        split: sorted(required_split_columns - set(table.columns))
        for split, table in split_tables.items()
    }
    for split, missing in missing_split_columns.items():
        record_check(
            checks,
            "locked_eye_split_required_columns",
            not missing,
            f"missing={missing}",
            split,
        )
    if missing_manifest or any(missing_split_columns.values()):
        stop_on_validation_failure(
            checks,
            checks_path,
            "A required source column is missing; model-input files were not written.",
        )

    include_model_input, invalid_model_input_values = parse_bool_column(
        image_manifest["IncludeModelInput"]
    )
    record_check(
        checks,
        "include_model_input_boolean_valid",
        not invalid_model_input_values,
        f"invalid_values={invalid_model_input_values}",
    )
    image_manifest["IncludeModelInput"] = include_model_input

    expected_split_values = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    for split, table in split_tables.items():
        split_values = set(table["DatasetSplit"].dropna().astype(str).unique())
        record_check(
            checks,
            "locked_dataset_split_value",
            split_values == {expected_split_values[split]},
            f"expected={expected_split_values[split]}; observed={sorted(split_values)}",
            split,
        )
        record_check(
            checks,
            "locked_eye_exam_id_unique",
            table["EyeExamID"].notna().all() and table["EyeExamID"].is_unique,
            f"rows={len(table)}; unique={table['EyeExamID'].nunique()}",
            split,
        )

    # combine the split files temporarily so we can check coverage and leakage
    locked_eyes = pd.concat(split_tables.values(), ignore_index=True)
    locked_eye_ids = set(locked_eyes["EyeExamID"].astype(str))
    record_check(
        checks,
        "locked_split_union_is_800_unique_eyes",
        len(locked_eyes) == EXPECTED_EYES
        and len(locked_eye_ids) == EXPECTED_EYES,
        f"rows={len(locked_eyes)}; unique_eye_ids={len(locked_eye_ids)}",
    )
    duplicated_locked_eyes = locked_eyes.duplicated("EyeExamID", keep=False)
    record_check(
        checks,
        "each_locked_eye_occurs_in_exactly_one_split",
        not duplicated_locked_eyes.any(),
        f"duplicate_rows={int(duplicated_locked_eyes.sum())}",
    )

    subject_sets = {
        split: set(table["ResearchSubjectID"].astype(str))
        for split, table in split_tables.items()
    }
    subject_intersections = {
        "train_intersection_validation": subject_sets["train"]
        & subject_sets["validation"],
        "train_intersection_test": subject_sets["train"] & subject_sets["test"],
        "validation_intersection_test": subject_sets["validation"]
        & subject_sets["test"],
    }
    for name, intersection in subject_intersections.items():
        record_check(
            checks,
            name,
            not intersection,
            f"intersection_subjects={len(intersection)}",
        )
    record_check(
        checks,
        "research_subject_separation_remains_intact",
        all(not intersection for intersection in subject_intersections.values()),
        "; ".join(
            f"{name}={len(intersection)}"
            for name, intersection in subject_intersections.items()
        ),
    )








    # selection is performed using identifiers and paths only. eyelabel is joined
    # after the deterministic paths are fixed, preventing labels (including test
    # labels) from influencing view selection.
    usable_images = image_manifest.loc[
        image_manifest["IncludeModelInput"]
        & image_manifest["EyeExamID"].astype(str).isin(locked_eye_ids)
    ].copy()
    # only images marked usable by step 2 are allowed into the model inputs
    source_eye_counts = usable_images.groupby("EyeExamID").size()
    missing_usable_eyes = sorted(locked_eye_ids - set(source_eye_counts.index.astype(str)))
    record_check(
        checks,
        "every_locked_eye_has_usable_image",
        not missing_usable_eyes,
        f"eyes_without_usable_images={len(missing_usable_eyes)}",
    )

    count_comparison = locked_eyes[["EyeExamID", "UniqueImageCount"]].copy()
    count_comparison["ManifestUsableImageCount"] = count_comparison["EyeExamID"].map(
        source_eye_counts
    )
    usable_count_matches = count_comparison["UniqueImageCount"].eq(
        count_comparison["ManifestUsableImageCount"]
    )
    record_check(
        checks,
        "locked_unique_image_counts_match_manifest",
        usable_count_matches.all(),
        f"mismatched_eyes={int((~usable_count_matches).sum())}",
    )

    # run the selection twice in different row orders to confirm it is deterministic
    selected_images = select_views(usable_images)
    repeated_images = select_views(usable_images.iloc[::-1].copy())
    repeated_selection_matches = selected_path_pairs(selected_images) == selected_path_pairs(
        repeated_images
    )
    record_check(
        checks,
        "repeated_execution_selects_identical_paths",
        repeated_selection_matches,
        f"selected_paths={len(selected_images)}",
    )

    eye_metadata_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "DatasetSplit",
        "SplitSeed",
    ]









    # attach labels and split information only after the image paths are fixed
    eye_metadata = locked_eyes[eye_metadata_columns].copy()
    selected_images = selected_images.merge(
        eye_metadata,
        on="EyeExamID",
        how="left",
        validate="many_to_one",
        suffixes=("_image", "_split"),
        indicator=True,
    )

    component_matches = pd.Series(True, index=selected_images.index)
    for column in ["ResearchSubjectID", "EncounterID", "Laterality"]:
        component_matches &= selected_images[f"{column}_image"].eq(
            selected_images[f"{column}_split"]
        )
    all_selected_have_eye = selected_images["_merge"].eq("both").all()
    record_check(
        checks,
        "selected_images_belong_to_stated_eye_exam_id",
        all_selected_have_eye and component_matches.all(),
        f"unmapped_rows={int((selected_images['_merge'] != 'both').sum())}; "
        f"component_mismatches={int((~component_matches).sum())}",
    )

    selected_images = selected_images.rename(
        columns={
            "ResearchSubjectID_split": "ResearchSubjectID",
            "EncounterID_split": "EncounterID",
            "Laterality_split": "Laterality",
        }
    ).drop(
        columns=[
            "ResearchSubjectID_image",
            "EncounterID_image",
            "Laterality_image",
            "_merge",
        ]
    )

    output_columns = [
        "EyeExamID",
        "ResearchSubjectID",
        "EncounterID",
        "Laterality",
        "EyeLabel",
        "DatasetSplit",
        "SplitSeed",
        "ImageID",
        "ImageFileName",
        "ImageRelativePath",
        "SHA256",
        "Bytes",
        "Width",
        "Height",
        "IncludeModelInput",
        "SelectedViewIndex",
        "AvailableUniqueImageCount",
        "SelectedImageCount",
        "ViewSelectionRule",
    ]
    selected_images = selected_images[output_columns].sort_values(
        ["DatasetSplit", "EyeExamID", "SelectedViewIndex"], kind="mergesort"
    ).reset_index(drop=True)
    output_tables = {
        split: selected_images.loc[
            selected_images["DatasetSplit"] == expected_split_values[split]
        ]
        .copy()
        .sort_values(["EyeExamID", "SelectedViewIndex"], kind="mergesort")
        .reset_index(drop=True)
        for split in SPLIT_PATHS
    }

    # final checks make sure every locked eye is represented exactly as intended
    selected_eye_ids = set(selected_images["EyeExamID"].astype(str))
    record_check(
        checks,
        "all_800_eyes_occur_in_one_selected_input_split",
        len(selected_eye_ids) == EXPECTED_EYES
        and selected_eye_ids == locked_eye_ids,
        f"selected_eye_ids={len(selected_eye_ids)}; "
        f"missing={len(locked_eye_ids - selected_eye_ids)}; "
        f"added={len(selected_eye_ids - locked_eye_ids)}",
    )
    record_check(
        checks,
        "selected_laterality_only_od_os",
        selected_images["Laterality"].isin(ALLOWED_LATERALITY).all(),
        f"invalid_rows={int((~selected_images['Laterality'].isin(ALLOWED_LATERALITY)).sum())}",
    )
    record_check(
        checks,
        "no_unknown_laterality_selected",
        not selected_images["Laterality"].eq("UNKNOWN").any(),
        f"unknown_rows={int(selected_images['Laterality'].eq('UNKNOWN').sum())}",
    )
    record_check(
        checks,
        "every_selected_image_was_include_model_input",
        selected_images["IncludeModelInput"].all(),
        f"invalid_rows={int((~selected_images['IncludeModelInput']).sum())}",
    )
    selected_sha_duplicates = selected_images.duplicated(
        ["EyeExamID", "SHA256"], keep=False
    )
    record_check(
        checks,
        "no_sha_duplicate_selected_within_eye",
        not selected_sha_duplicates.any(),
        f"duplicate_rows={int(selected_sha_duplicates.sum())}",
    )
    selected_path_duplicates = selected_images.duplicated(
        ["EyeExamID", "ImageRelativePath"], keep=False
    )
    record_check(
        checks,
        "no_image_path_selected_twice_within_eye",
        not selected_path_duplicates.any(),
        f"duplicate_rows={int(selected_path_duplicates.sum())}",
    )

    selected_counts = selected_images.groupby("EyeExamID").size()
    record_check(
        checks,
        "every_eye_has_at_least_one_selected_image",
        selected_counts.ge(1).all() and len(selected_counts) == EXPECTED_EYES,
        f"eyes={len(selected_counts)}; zero_image_eyes={EXPECTED_EYES - len(selected_counts)}",
    )
    record_check(
        checks,
        "no_eye_has_more_than_six_selected_images",
        selected_counts.le(MAX_VIEWS).all(),
        f"maximum_selected={int(selected_counts.max())}",
    )
    all_small_eyes_retained = all(
        selected_counts.get(eye_exam_id, 0) == available_count
        for eye_exam_id, available_count in source_eye_counts.items()
        if available_count <= MAX_VIEWS
    )
    record_check(
        checks,
        "eyes_with_le6_retain_all_usable_images",
        all_small_eyes_retained,
        f"eligible_eyes={int(source_eye_counts.le(MAX_VIEWS).sum())}",
    )
    all_large_eyes_have_six = all(
        selected_counts.get(eye_exam_id, 0) == MAX_VIEWS
        for eye_exam_id, available_count in source_eye_counts.items()
        if available_count > MAX_VIEWS
    )
    record_check(
        checks,
        "eyes_with_gt6_have_exactly_six_selected_images",
        all_large_eyes_have_six,
        f"reduced_eyes={int(source_eye_counts.gt(MAX_VIEWS).sum())}",
    )
    dataset_split_agrees = all(
        set(table["DatasetSplit"].unique()) == {expected_split_values[split]}
        and set(table["EyeExamID"].astype(str))
        == set(split_tables[split]["EyeExamID"].astype(str))
        for split, table in output_tables.items()
    )
    record_check(
        checks,
        "dataset_split_agrees_with_locked_step4_split",
        dataset_split_agrees,
        "selected EyeExamID sets and DatasetSplit values match locked eye files",
    )



    for split, table in output_tables.items():
        locked_ids = set(split_tables[split]["EyeExamID"].astype(str))
        table_ids = set(table["EyeExamID"].astype(str))
        split_counts = table.groupby("EyeExamID").size()
        record_check(
            checks,
            "selected_eye_set_matches_locked_split",
            table_ids == locked_ids,
            f"selected_eyes={len(table_ids)}; locked_eyes={len(locked_ids)}",
            split,
        )
        record_check(
            checks,
            "selected_image_count_range_1_to_6",
            split_counts.between(1, MAX_VIEWS).all(),
            f"minimum={int(split_counts.min())}; maximum={int(split_counts.max())}",
            split,
        )

    # do not overwrite an existing lock unless the recomputed selection matches it
    existing_paths = [path for path in OUTPUT_PATHS.values() if path.exists()]
    lock_status = "CREATED"
    lock_matches = True
    if existing_paths and len(existing_paths) != len(OUTPUT_PATHS):
        lock_matches = False
        lock_status = "INCOMPLETE_EXISTING_LOCK"
    elif len(existing_paths) == len(OUTPUT_PATHS):
        existing_tables = {
            split: pd.read_csv(path, low_memory=False)
            for split, path in OUTPUT_PATHS.items()
        }
        lock_matches = all(
            selected_path_pairs(existing_tables[split]) == selected_path_pairs(output_tables[split])
            for split in OUTPUT_PATHS
        )
        lock_status = "REPRODUCIBLE_EXISTING" if lock_matches else "LOCK_CONFLICT"
    record_check(
        checks,
        "locked_model_input_selection_matches_recomputed_paths",
        lock_matches,
        f"existing_files={len(existing_paths)}/3; lock_status={lock_status}",
    )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    if (checks_df["Status"] == "FAIL").any():
        stop_on_validation_failure(
            checks,
            checks_path,
            "Model-input validation failed; existing manifests were not overwritten.",
        )

    if not existing_paths:
        next(iter(OUTPUT_PATHS.values())).parent.mkdir(parents=True, exist_ok=True)
        for split, path in OUTPUT_PATHS.items():
            output_tables[split].to_csv(path, index=False)
    else:
        print("Existing model-input manifests reproduce exactly; leaving them unchanged.\n")



    # save a compact summary of how many views were selected in each split
    summary_rows = []
    for split, table in output_tables.items():
        counts = table.groupby("EyeExamID").size()
        available_by_eye = table.groupby("EyeExamID")[
            "AvailableUniqueImageCount"
        ].first()
        row = {
            "DatasetSplit": expected_split_values[split],
            "LockStatus": lock_status,
            "OrderingRule": ORDERING_RULE,
            "MaximumViews": MAX_VIEWS,
            "Eyes": len(counts),
            "SelectedImages": len(table),
            "MeanSelectedImagesPerEye": float(counts.mean()),
            "MedianSelectedImagesPerEye": float(counts.median()),
            "SourceEyesReducedFromMoreThan6": int(
                available_by_eye.gt(MAX_VIEWS).sum()
            ),
        }
        for count in range(1, MAX_VIEWS + 1):
            row[f"EyesWith{count}SelectedImages"] = int(counts.eq(count).sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("STEP-5 VALIDATION: PASS\n")
    for row in summary_df.itertuples(index=False):
        print(row.DatasetSplit.upper())
        print(f"  Eyes                              : {row.Eyes:,}")
        print(f"  Selected images                   : {row.SelectedImages:,}")
        print(f"  Mean selected images per eye      : {row.MeanSelectedImagesPerEye:.3f}")
        print(f"  Median selected images per eye    : {row.MedianSelectedImagesPerEye:.3f}")
        for count in range(1, MAX_VIEWS + 1):
            value = getattr(row, f"EyesWith{count}SelectedImages")
            print(f"  Eyes with {count} selected image(s)      : {value:,}")
        print(f"  Source eyes reduced from >6       : {row.SourceEyesReducedFromMoreThan6:,}")
        print()

    total_reduced = int(source_eye_counts.gt(MAX_VIEWS).sum())
    print(f"Total selected images                : {len(selected_images):,}")
    print(f"Total source eyes reduced from >6    : {total_reduced:,}")
    print(f"Ordering rule                        : {ORDERING_RULE}")
    print(
        "Exact >6 rule                       : sort paths ascending; select "
        "zero-based indices nearest to i*(n-1)/5 for i=0..5"
    )
    print("                                      (always includes indices 0 and n-1)")
    print(f"Model-input lock status              : {lock_status}")
    print(f"Validation report                    : {checks_path}")
    print(f"Audit summary                        : {summary_path}")
    print("Model-input files:")
    for path in OUTPUT_PATHS.values():
        print(f"  {path}")

    print("\nVALIDATION CHECKS")
    for row in checks_df.itertuples(index=False):
        print(f"[{row.Status}] {row.Scope} :: {row.Check}: {row.Details}")









if __name__ == "__main__":
    main()
