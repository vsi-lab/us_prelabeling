from datetime import datetime
from pathlib import Path
import hashlib
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = PROJECT_ROOT / "data" / "processed" / "pilot_cohort.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"

SPLIT_SEED = 42
SPLIT_PROPORTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
SPLIT_FILE_STEMS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
EXPECTED_PILOT_COUNTS = {
    "total": 800,
    "Normal": 213,
    "Abnormal": 587,
    "subjects": 369,
}
ALLOWED_LABELS = {"Normal", "Abnormal"}
DOMINANCE_WARNING_PERCENT = 20.0


def record_check(rows, check, passed, details):
    rows.append(
        {
            "Check": check,
            "Status": "PASS" if passed else "FAIL",
            "Details": details,
        }
    )


def seeded_hash(*parts: object) -> str:
    token = "|".join(str(part) for part in (SPLIT_SEED, *parts))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def allocate_by_proportion(
    total: int, proportions: dict[str, float]
) -> dict[str, int]:
    "Turn percentage targets into whole-number counts."
    quotas = {split: total * proportion for split, proportion in proportions.items()}
    allocation = {split: math.floor(quota) for split, quota in quotas.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        proportions,
        key=lambda split: (
            -(quotas[split] - allocation[split]),
            list(proportions).index(split),
        ),
    )
    for split in order[:remaining]:
        allocation[split] += 1
    return allocation


def summarize_subjects(pilot: pd.DataFrame) -> pd.DataFrame:
    subjects = (
        pilot.groupby("ResearchSubjectID", sort=True)
        .agg(
            TotalEyes=("EyeExamID", "size"),
            NormalEyes=("EyeLabel", lambda values: int((values == "Normal").sum())),
            AbnormalEyes=(
                "EyeLabel", lambda values: int((values == "Abnormal").sum())
            ),
        )
        .reset_index()
    )
    subjects["SubjectComposition"] = "Mixed"
    subjects.loc[subjects["AbnormalEyes"] == 0, "SubjectComposition"] = (
        "Normal-only"
    )
    subjects.loc[subjects["NormalEyes"] == 0, "SubjectComposition"] = (
        "Abnormal-only"
    )
    return subjects


def split_error(
    counts: dict[str, list[int]], targets: dict[str, list[int]]
) -> int:
    "Score how far the current split counts are from the targets."
    return sum(
        2 * (counts[split][0] - targets[split][0]) ** 2
        + (counts[split][1] - targets[split][1]) ** 2
        + (counts[split][2] - targets[split][2]) ** 2
        for split in SPLIT_PROPORTIONS
    )


def copy_split_counts(counts: dict[str, list[int]]) -> dict[str, list[int]]:
    return {split: values.copy() for split, values in counts.items()}


def get_subject_vectors(subjects: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    return {
        str(row.ResearchSubjectID): (
            int(row.TotalEyes),
            int(row.NormalEyes),
            int(row.AbnormalEyes),
        )
        for row in subjects.itertuples(index=False)
    }


def assign_subjects_to_splits(
    subjects: pd.DataFrame, targets: dict[str, list[int]]
) -> tuple[dict[str, str], dict[str, list[int]], int]:
    # Put the subjects with the most eyes first. They are harder to place later.
    vectors = get_subject_vectors(subjects)
    subject_order = sorted(
        vectors,
        key=lambda subject_id: (
            -vectors[subject_id][0],
            -max(vectors[subject_id][1], vectors[subject_id][2]),
            seeded_hash("subject-order", subject_id),
            subject_id,
        ),
    )
    counts = {split: [0, 0, 0] for split in SPLIT_PROPORTIONS}
    assignment = {}

    # First pass: try the subject in each split and keep the best placement.
    for subject_id in subject_order:
        vector = vectors[subject_id]
        candidates = []
        for split in SPLIT_PROPORTIONS:
            candidate_counts = copy_split_counts(counts)
            candidate_counts[split] = [
                candidate_counts[split][index] + vector[index]
                for index in range(3)
            ]
            candidates.append(
                (
                    split_error(candidate_counts, targets),
                    seeded_hash("greedy-tie", subject_id, split),
                    split,
                    candidate_counts,
                )
            )
        _, _, selected_split, counts = min(candidates)
        assignment[subject_id] = selected_split

    # The first pass is usually close enough, but a move or swap can sometimes
    # get the class counts closer to the requested split.
    improvement_steps = 0
    while True:
        current_score = split_error(counts, targets)
        best_operation = None
        best_key = None
        split_subject_counts = {
            split: sum(value == split for value in assignment.values())
            for split in SPLIT_PROPORTIONS
        }
        deterministic_subjects = sorted(
            assignment, key=lambda subject_id: (seeded_hash("local", subject_id), subject_id)
        )

        # Try moving one complete subject to another split.
        for subject_id in deterministic_subjects:
            source_split = assignment[subject_id]
            if split_subject_counts[source_split] <= 1:
                continue
            vector = vectors[subject_id]
            for destination_split in SPLIT_PROPORTIONS:
                if destination_split == source_split:
                    continue
                candidate_counts = copy_split_counts(counts)
                for index in range(3):
                    candidate_counts[source_split][index] -= vector[index]
                    candidate_counts[destination_split][index] += vector[index]
                score = split_error(candidate_counts, targets)
                key = (
                    score,
                    seeded_hash(
                        "move", subject_id, source_split, destination_split
                    ),
                )
                if score < current_score and (best_key is None or key < best_key):
                    best_key = key
                    best_operation = (
                        "move",
                        subject_id,
                        source_split,
                        destination_split,
                        candidate_counts,
                    )

        # If a move does not help enough, also test swapping two subjects.
        for first_index, first_subject in enumerate(deterministic_subjects):
            first_split = assignment[first_subject]
            first_vector = vectors[first_subject]
            for second_subject in deterministic_subjects[first_index + 1 :]:
                second_split = assignment[second_subject]
                if first_split == second_split:
                    continue
                second_vector = vectors[second_subject]
                candidate_counts = copy_split_counts(counts)
                for index in range(3):
                    candidate_counts[first_split][index] += (
                        second_vector[index] - first_vector[index]
                    )
                    candidate_counts[second_split][index] += (
                        first_vector[index] - second_vector[index]
                    )
                score = split_error(candidate_counts, targets)
                key = (
                    score,
                    seeded_hash("swap", first_subject, second_subject),
                )
                if score < current_score and (best_key is None or key < best_key):
                    best_key = key
                    best_operation = (
                        "swap",
                        first_subject,
                        second_subject,
                        first_split,
                        second_split,
                        candidate_counts,
                    )

        if best_operation is None:
            break
        if best_operation[0] == "move":
            _, subject_id, _, destination_split, counts = best_operation
            assignment[subject_id] = destination_split
        else:
            (
                _,
                first_subject,
                second_subject,
                first_split,
                second_split,
                counts,
            ) = best_operation
            assignment[first_subject] = second_split
            assignment[second_subject] = first_split
        improvement_steps += 1

    return assignment, counts, improvement_steps


def create_split_tables(
    pilot: pd.DataFrame,
    subjects: pd.DataFrame,
    assignment: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    eye_tables = {}
    subject_tables = {}
    # Add the subject-level assignment first, then use it to collect eye rows.
    subject_assignment = subjects.copy()
    subject_assignment["DatasetSplit"] = subject_assignment[
        "ResearchSubjectID"
    ].map(assignment)
    subject_assignment["SplitSeed"] = SPLIT_SEED

    for split in SPLIT_PROPORTIONS:
        subject_table = subject_assignment.loc[
            subject_assignment["DatasetSplit"] == split,
            [
                "ResearchSubjectID",
                "TotalEyes",
                "NormalEyes",
                "AbnormalEyes",
                "SubjectComposition",
                "DatasetSplit",
                "SplitSeed",
            ],
        ].copy()
        subject_table = subject_table.sort_values(
            "ResearchSubjectID", kind="mergesort"
        ).reset_index(drop=True)
        subject_tables[split] = subject_table

        eye_table = pilot.loc[
            pilot["ResearchSubjectID"].isin(subject_table["ResearchSubjectID"])
        ].copy()
        eye_table["DatasetSplit"] = split
        eye_table["SplitSeed"] = SPLIT_SEED
        eye_tables[split] = eye_table.sort_values(
            "EyeExamID", kind="mergesort"
        ).reset_index(drop=True)
    return eye_tables, subject_tables


def get_split_paths() -> tuple[dict[str, Path], dict[str, Path]]:
    eye_paths = {
        split: SPLIT_DIR / f"{stem}_eye_exams.csv"
        for split, stem in SPLIT_FILE_STEMS.items()
    }
    subject_paths = {
        split: SPLIT_DIR / f"{stem}_subjects.csv"
        for split, stem in SPLIT_FILE_STEMS.items()
    }
    return eye_paths, subject_paths


def get_eye_assignment(eye_tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        str(eye_exam_id): split
        for split, table in eye_tables.items()
        for eye_exam_id in table["EyeExamID"]
    }


def subject_table_matches_eyes(
    eye_table: pd.DataFrame, subject_table: pd.DataFrame, split: str
) -> bool:
    expected = summarize_subjects(eye_table)
    expected["DatasetSplit"] = split
    expected["SplitSeed"] = SPLIT_SEED
    columns = [
        "ResearchSubjectID",
        "TotalEyes",
        "NormalEyes",
        "AbnormalEyes",
        "SubjectComposition",
        "DatasetSplit",
        "SplitSeed",
    ]
    expected = expected[columns].sort_values("ResearchSubjectID").reset_index(drop=True)
    observed = subject_table[columns].sort_values("ResearchSubjectID").reset_index(drop=True)
    return expected.equals(observed)


def stop_on_validation_failure(checks, checks_path: Path, message: str):
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    print("STEP-4 VALIDATION: HARD FAIL\n")
    for row in checks:
        if row["Status"] == "FAIL":
            print(f"[FAIL] {row['Check']}: {row['Details']}")
    print(f"\n{message}")
    print(f"Validation report:\n{checks_path}")
    raise SystemExit(1)


def main():
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    audit_dir = AUDIT_ROOT / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    checks_path = audit_dir / "step4_split_checks.csv"
    summary_path = audit_dir / "step4_split_summary.csv"
    checks = []

    print("=" * 72)
    print("STEP 4: CREATE AND LOCK PATIENT-LEVEL SPLITS")
    print("=" * 72)
    print(f"Run ID             : {run_id}")
    print(f"Pilot source       : {PILOT_PATH}")
    print(f"Split seed         : {SPLIT_SEED}")
    print("Assignment method  : large-subject-first deterministic greedy")
    print("                     + whole-subject move/swap refinement")
    print("Objective           : squared deviation in total, Normal, and")
    print("                     Abnormal eye counts; no subject is divided\n")

    # Do not continue if Step 3 has not produced the locked pilot cohort.
    pilot_exists = PILOT_PATH.is_file()
    record_check(checks, "pilot_cohort_exists", pilot_exists, str(PILOT_PATH))
    if not pilot_exists:
        stop_on_validation_failure(
            checks, checks_path, "Pilot cohort is missing; no splits were written."
        )

    # Check the pilot before building any split files.
    pilot = pd.read_csv(PILOT_PATH, low_memory=False)
    required_columns = {
        "EyeExamID",
        "ResearchSubjectID",
        "EyeLabel",
        "PostoperativeReviewFlag",
    }
    missing_columns = sorted(required_columns - set(pilot.columns))
    record_check(
        checks,
        "pilot_required_columns",
        not missing_columns,
        f"missing={missing_columns}",
    )
    if missing_columns:
        stop_on_validation_failure(
            checks,
            checks_path,
            "Pilot schema is incomplete; no splits were written.",
        )

    pilot_total = len(pilot)
    label_counts = pilot["EyeLabel"].value_counts()
    normal_count = int(label_counts.get("Normal", 0))
    abnormal_count = int(label_counts.get("Abnormal", 0))
    record_check(
        checks,
        "pilot_total_is_800",
        pilot_total == EXPECTED_PILOT_COUNTS["total"],
        f"expected=800; observed={pilot_total}",
    )
    record_check(
        checks,
        "pilot_normal_is_213",
        normal_count == EXPECTED_PILOT_COUNTS["Normal"],
        f"expected=213; observed={normal_count}",
    )
    record_check(
        checks,
        "pilot_abnormal_is_587",
        abnormal_count == EXPECTED_PILOT_COUNTS["Abnormal"],
        f"expected=587; observed={abnormal_count}",
    )
    record_check(
        checks,
        "pilot_eye_exam_id_unique",
        pilot["EyeExamID"].notna().all() and pilot["EyeExamID"].is_unique,
        f"duplicate_rows={int(pilot.duplicated('EyeExamID', keep=False).sum())}",
    )
    labels_valid = pilot["EyeLabel"].notna().all() and set(
        pilot["EyeLabel"].unique()
    ).issubset(ALLOWED_LABELS)
    record_check(
        checks,
        "pilot_only_normal_abnormal_labels",
        labels_valid,
        f"labels={sorted(pilot['EyeLabel'].dropna().astype(str).unique())}",
    )

    subjects = summarize_subjects(pilot)
    record_check(
        checks,
        "pilot_subject_count_is_369",
        len(subjects) == EXPECTED_PILOT_COUNTS["subjects"],
        f"expected=369; observed={len(subjects)}",
    )

    # Work out the desired total and class counts for each split.
    total_targets = allocate_by_proportion(
        pilot_total, SPLIT_PROPORTIONS
    )
    normal_targets = allocate_by_proportion(
        normal_count, SPLIT_PROPORTIONS
    )
    abnormal_targets = allocate_by_proportion(
        abnormal_count, SPLIT_PROPORTIONS
    )
    targets = {
        split: [
            total_targets[split],
            normal_targets[split],
            abnormal_targets[split],
        ]
        for split in SPLIT_PROPORTIONS
    }

    # Subjects, not individual eyes, are the unit of assignment.
    assignment, achieved_counts, improvement_steps = assign_subjects_to_splits(
        subjects, targets
    )
    repeated_assignment, _, _ = assign_subjects_to_splits(subjects, targets)
    record_check(
        checks,
        "repeated_seed_42_assignment_is_identical",
        assignment == repeated_assignment,
        f"assigned_subjects={len(assignment)}",
    )

    # Build both eye-level and subject-level versions of each split.
    eye_tables, subject_tables = create_split_tables(pilot, subjects, assignment)
    combined_eyes = pd.concat(eye_tables.values(), ignore_index=True)
    pilot_ids = set(pilot["EyeExamID"].astype(str))
    split_ids = set(combined_eyes["EyeExamID"].astype(str))

    record_check(
        checks,
        "split_union_contains_800_eye_exam_ids",
        len(combined_eyes) == 800 and len(split_ids) == 800,
        f"rows={len(combined_eyes)}; unique_eye_ids={len(split_ids)}",
    )
    duplicate_split_eye_ids = combined_eyes.duplicated("EyeExamID", keep=False)
    record_check(
        checks,
        "no_eye_exam_id_in_multiple_splits",
        not duplicate_split_eye_ids.any(),
        f"duplicate_rows={int(duplicate_split_eye_ids.sum())}",
    )
    record_check(
        checks,
        "no_pilot_eye_exam_id_omitted",
        not (pilot_ids - split_ids),
        f"omitted_ids={len(pilot_ids - split_ids)}",
    )
    record_check(
        checks,
        "no_nonpilot_eye_exam_id_added",
        not (split_ids - pilot_ids),
        f"added_ids={len(split_ids - pilot_ids)}",
    )

    # This is the main leakage check: a subject must appear in only one split.
    subject_sets = {
        split: set(table["ResearchSubjectID"].astype(str))
        for split, table in subject_tables.items()
    }
    intersections = {
        "train_intersection_validation": subject_sets["train"]
        & subject_sets["validation"],
        "train_intersection_test": subject_sets["train"] & subject_sets["test"],
        "validation_intersection_test": subject_sets["validation"]
        & subject_sets["test"],
    }
    for name, intersection in intersections.items():
        record_check(
            checks,
            name,
            not intersection,
            f"intersection_subjects={len(intersection)}",
        )
    record_check(
        checks,
        "no_research_subject_id_in_multiple_splits",
        all(not values for values in intersections.values()),
        "; ".join(
            f"{name}={len(values)}" for name, values in intersections.items()
        ),
    )

    for split, table in eye_tables.items():
        split_labels = set(table["EyeLabel"].dropna().unique())
        record_check(
            checks,
            f"{split}_contains_normal_and_abnormal",
            split_labels == ALLOWED_LABELS,
            f"labels={sorted(split_labels)}",
        )
        record_check(
            checks,
            f"{split}_subject_file_matches_eye_file",
            subject_table_matches_eyes(table, subject_tables[split], split),
            f"eyes={len(table)}; subjects={len(subject_tables[split])}",
        )

    seed_valid = all(
        table["SplitSeed"].eq(SPLIT_SEED).all()
        for table in [*eye_tables.values(), *subject_tables.values()]
    )
    record_check(
        checks,
        "split_seed_is_42_everywhere",
        seed_valid,
        f"expected_seed={SPLIT_SEED}",
    )

    # Existing split files are treated as locked. We only accept an exact match.
    eye_paths, subject_paths = get_split_paths()
    all_paths = [*eye_paths.values(), *subject_paths.values()]
    existing_paths = [path for path in all_paths if path.exists()]
    lock_status = "CREATED"
    lock_matches = True
    if existing_paths and len(existing_paths) != len(all_paths):
        lock_matches = False
        lock_status = "INCOMPLETE_EXISTING_LOCK"
    elif len(existing_paths) == len(all_paths):
        existing_eye_tables = {
            split: pd.read_csv(path, low_memory=False)
            for split, path in eye_paths.items()
        }
        existing_subject_tables = {
            split: pd.read_csv(path, low_memory=False)
            for split, path in subject_paths.items()
        }
        existing_assignment = get_eye_assignment(existing_eye_tables)
        new_assignment = get_eye_assignment(eye_tables)
        existing_subject_files_valid = all(
            subject_table_matches_eyes(
                existing_eye_tables[split], existing_subject_tables[split], split
            )
            for split in SPLIT_PROPORTIONS
        )
        lock_matches = (
            existing_assignment == new_assignment and existing_subject_files_valid
        )
        lock_status = "REPRODUCIBLE_EXISTING" if lock_matches else "LOCK_CONFLICT"
    record_check(
        checks,
        "locked_split_assignment_matches_recomputed_assignment",
        lock_matches,
        f"existing_files={len(existing_paths)}/6; lock_status={lock_status}",
    )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    if (checks_df["Status"] == "FAIL").any():
        stop_on_validation_failure(
            checks,
            checks_path,
            "Split validation failed; existing locks were not overwritten.",
        )

    # Only write a new lock when no previous split files exist.
    if not existing_paths:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_PROPORTIONS:
            eye_tables[split].to_csv(eye_paths[split], index=False)
            subject_tables[split].to_csv(subject_paths[split], index=False)
    else:
        print("Existing locked splits reproduce exactly; leaving them unchanged.\n")

    # The rest of the script is reporting only; it does not change the split.
    overall_normal_percent = 100.0 * normal_count / pilot_total
    overall_abnormal_percent = 100.0 * abnormal_count / pilot_total
    summary_rows = []
    warnings = []
    for split in SPLIT_PROPORTIONS:
        eyes = eye_tables[split]
        split_subjects = subject_tables[split]
        eyes_count = len(eyes)
        normals = int((eyes["EyeLabel"] == "Normal").sum())
        abnormals = int((eyes["EyeLabel"] == "Abnormal").sum())
        normal_percent = 100.0 * normals / eyes_count
        abnormal_percent = 100.0 * abnormals / eyes_count
        postoperative = eyes.loc[eyes["PostoperativeReviewFlag"] == True]
        largest_subject_eye_count = int(split_subjects["TotalEyes"].max())
        largest_subject_share = 100.0 * largest_subject_eye_count / eyes_count
        if split in {"validation", "test"} and largest_subject_share >= DOMINANCE_WARNING_PERCENT:
            warnings.append(
                f"{split}: largest subject contributes {largest_subject_share:.3f}% "
                f"({largest_subject_eye_count}/{eyes_count} eyes)"
            )
        composition_counts = split_subjects["SubjectComposition"].value_counts()
        summary_rows.append(
            {
                "DatasetSplit": split,
                "SplitSeed": SPLIT_SEED,
                "LockStatus": lock_status,
                "Eyes": eyes_count,
                "TargetEyes": targets[split][0],
                "RequestedAllocationPercent": 100.0
                * SPLIT_PROPORTIONS[split],
                "ActualAllocationPercent": 100.0 * eyes_count / pilot_total,
                "AllocationDeviationPercentagePoints": abs(
                    100.0 * eyes_count / pilot_total
                    - 100.0 * SPLIT_PROPORTIONS[split]
                ),
                "Subjects": len(split_subjects),
                "Normal": normals,
                "TargetNormal": targets[split][1],
                "Abnormal": abnormals,
                "TargetAbnormal": targets[split][2],
                "NormalPercent": normal_percent,
                "AbnormalPercent": abnormal_percent,
                "NormalDeviationFromPilotPercentagePoints": abs(
                    normal_percent - overall_normal_percent
                ),
                "AbnormalDeviationFromPilotPercentagePoints": abs(
                    abnormal_percent - overall_abnormal_percent
                ),
                "PostoperativeFlaggedEyes": len(postoperative),
                "PostoperativeFlaggedPercent": 100.0
                * len(postoperative)
                / eyes_count,
                "PostoperativeFlaggedNormal": int(
                    (postoperative["EyeLabel"] == "Normal").sum()
                ),
                "PostoperativeFlaggedAbnormal": int(
                    (postoperative["EyeLabel"] == "Abnormal").sum()
                ),
                "LargestSubjectEyeCount": largest_subject_eye_count,
                "LargestSubjectSharePercent": largest_subject_share,
                "MultiEyeSubjects": int((split_subjects["TotalEyes"] > 1).sum()),
                "NormalOnlySubjects": int(
                    composition_counts.get("Normal-only", 0)
                ),
                "AbnormalOnlySubjects": int(
                    composition_counts.get("Abnormal-only", 0)
                ),
                "MixedSubjects": int(composition_counts.get("Mixed", 0)),
                "DominanceWarning": largest_subject_share
                >= DOMINANCE_WARNING_PERCENT
                if split in {"validation", "test"}
                else False,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("STEP-4 VALIDATION: PASS\n")
    print("Target vectors (Total, Normal, Abnormal):")
    for split in SPLIT_PROPORTIONS:
        print(f"  {split:<10}: {tuple(targets[split])}")
    print(f"Local improvement steps              : {improvement_steps}")
    print(f"Final assignment objective           : {split_error(achieved_counts, targets)}\n")

    for row in summary_df.itertuples(index=False):
        print(row.DatasetSplit.upper())
        print(f"  Eyes / subjects                    : {row.Eyes:,} / {row.Subjects:,}")
        print(f"  Normal                             : {row.Normal:,} ({row.NormalPercent:.3f}%)")
        print(f"  Abnormal                           : {row.Abnormal:,} ({row.AbnormalPercent:.3f}%)")
        print(f"  Class deviation from pilot         : {row.NormalDeviationFromPilotPercentagePoints:.3f} pp")
        print(f"  Allocation deviation               : {row.AllocationDeviationPercentagePoints:.3f} pp")
        print(f"  Postoperative flagged              : {row.PostoperativeFlaggedEyes:,} ({row.PostoperativeFlaggedPercent:.3f}%)")
        print(f"    Normal / Abnormal                : {row.PostoperativeFlaggedNormal:,} / {row.PostoperativeFlaggedAbnormal:,}")
        print(f"  Largest subject                    : {row.LargestSubjectEyeCount:,} eyes ({row.LargestSubjectSharePercent:.3f}%)")
        print(f"  Subjects contributing >1 eye       : {row.MultiEyeSubjects:,}")
        print(f"  Normal-only / Abnormal-only / Mixed: {row.NormalOnlySubjects:,} / {row.AbnormalOnlySubjects:,} / {row.MixedSubjects:,}")
        print()

    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"  {warning}")
        print()
    else:
        print("Large-subject concentration warnings : none\n")

    print("Subject intersections:")
    for name, intersection in intersections.items():
        print(f"  {name:<30}: {len(intersection)}")
    print()
    print(f"Split seed                           : {SPLIT_SEED}")
    print(f"Split lock status                    : {lock_status}")
    print(f"Validation report                    : {checks_path}")
    print(f"Audit summary                        : {summary_path}")
    print("Split files:")
    for split in SPLIT_PROPORTIONS:
        print(f"  {eye_paths[split]}")
        print(f"  {subject_paths[split]}")

    print("\nVALIDATION CHECKS")
    for row in checks_df.itertuples(index=False):
        print(f"[{row.Status}] {row.Check}: {row.Details}")


if __name__ == "__main__":
    main()
