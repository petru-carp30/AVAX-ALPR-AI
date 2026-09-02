from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool"
METADATA_ROOT = CANONICAL_ROOT / "metadata"

SAMPLES_PATH = METADATA_ROOT / "canonical_samples.csv"
BOXES_PATH = METADATA_ROOT / "canonical_boxes.csv"
ADJUDICATION_PATH = METADATA_ROOT / "similarity" / "exact_duplicate_adjudication.csv"

OUTPUT_SAMPLES_PATH = METADATA_ROOT / "baseline_candidate_manifest.csv"
OUTPUT_BOXES_PATH = METADATA_ROOT / "baseline_candidate_boxes.csv"

EXPECTED_SOURCE_IMAGES = 9299
EXPECTED_EXCLUDED_IMAGES = 2
EXPECTED_ELIGIBLE_IMAGES = 9297
EXPECTED_ELIGIBLE_POSITIVES = 8793
EXPECTED_ELIGIBLE_NEGATIVES = 504
EXPECTED_ELIGIBLE_INSTANCES = 12081


def main() -> None:
    samples = pd.read_csv(SAMPLES_PATH)
    boxes = pd.read_csv(BOXES_PATH)
    adjudication = pd.read_csv(ADJUDICATION_PATH, dtype=str)

    if len(samples) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_IMAGES} canonical samples, found {len(samples)}.")

    unresolved = adjudication[adjudication["review_status"] != "resolved"]
    if not unresolved.empty:
        raise RuntimeError(f"Unresolved exact duplicate groups: {len(unresolved)}")

    excluded_ids = set(adjudication["exclude_canonical_id"].dropna())
    kept_ids = set(adjudication["keep_canonical_id"].dropna())

    if len(excluded_ids) != EXPECTED_EXCLUDED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_EXCLUDED_IMAGES} excluded duplicate images, found {len(excluded_ids)}.")

    if excluded_ids & kept_ids:
        raise RuntimeError(f"Canonical IDs appear in both keep and exclude sets: {sorted(excluded_ids & kept_ids)}")

    missing_excluded = excluded_ids - set(samples["canonical_id"])
    if missing_excluded:
        raise RuntimeError(f"Excluded IDs missing from canonical samples: {sorted(missing_excluded)}")

    samples["baseline_status"] = "eligible_candidate"
    samples["baseline_exclusion_reason"] = ""
    samples.loc[samples["canonical_id"].isin(excluded_ids), "baseline_status"] = "excluded"
    samples.loc[samples["canonical_id"].isin(excluded_ids), "baseline_exclusion_reason"] = "exact_duplicate_annotation_conflict"

    eligible = samples[samples["baseline_status"] == "eligible_candidate"].copy()
    eligible_ids = set(eligible["canonical_id"])
    eligible_boxes = boxes[boxes["canonical_id"].isin(eligible_ids)].copy()

    positive_count = int((~eligible["is_negative"]).sum())
    negative_count = int(eligible["is_negative"].sum())
    instance_count = int(eligible["plate_instance_count"].sum())

    if len(eligible) != EXPECTED_ELIGIBLE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE_IMAGES} eligible images, found {len(eligible)}.")
    if positive_count != EXPECTED_ELIGIBLE_POSITIVES:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE_POSITIVES} eligible positives, found {positive_count}.")
    if negative_count != EXPECTED_ELIGIBLE_NEGATIVES:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE_NEGATIVES} eligible negatives, found {negative_count}.")
    if instance_count != EXPECTED_ELIGIBLE_INSTANCES or len(eligible_boxes) != EXPECTED_ELIGIBLE_INSTANCES:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE_INSTANCES} eligible instances, found samples={instance_count}, boxes={len(eligible_boxes)}.")

    samples.to_csv(OUTPUT_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    eligible_boxes.to_csv(OUTPUT_BOXES_PATH, index=False, encoding="utf-8-sig")

    print("\n=== CANONICAL EXACT DUPLICATE FILTER FINALIZATION ===")
    print(f"Canonical pool images: {len(samples)}")
    print(f"Excluded exact duplicates: {len(excluded_ids)}")
    print(f"Eligible images: {len(eligible)}")
    print(f"Eligible positive images: {positive_count}")
    print(f"Eligible negative images: {negative_count}")
    print(f"Eligible plate instances: {instance_count}")

    print("\nExcluded canonical IDs:")
    for canonical_id in sorted(excluded_ids):
        print(f"  {canonical_id}")

    print("\nEligible source composition:")
    for source, count in eligible["source_dataset"].value_counts().items():
        source_instances = int(eligible.loc[eligible["source_dataset"] == source, "plate_instance_count"].sum())
        print(f"  {source}: {count} images / {source_instances} plate instances")

    print("\nPolicy:")
    print("  Canonical image files deleted: NO")
    print("  Canonical label files deleted: NO")
    print("  Audit trail preserved: YES")
    print("  AVAX split assigned: NO")

    print(f"\nBaseline candidate manifest: {OUTPUT_SAMPLES_PATH}")
    print(f"Baseline candidate boxes: {OUTPUT_BOXES_PATH}")
    print("RESULT: EXACT DUPLICATE FILTER FINALIZED")


if __name__ == "__main__":
    main()