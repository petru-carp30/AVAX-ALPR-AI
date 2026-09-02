from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"

BATCH1_PATH = (
    DATASETS_ROOT
    / "audits"
    / "open_images_negative_pool"
    / "accepted_negatives_metadata.csv"
)

BATCH2_PATH = (
    DATASETS_ROOT
    / "audits"
    / "open_images_negative_pool"
    / "batch2"
    / "accepted_negatives_batch2_metadata.csv"
)

OUTPUT_ROOT = (
    DATASETS_ROOT
    / "manifests"
    / "open_images_negative_pool"
)

OUTPUT_PATH = OUTPUT_ROOT / "open_images_negative_pool_manifest.csv"

EXPECTED_BATCH1 = 456
EXPECTED_BATCH2 = 48
EXPECTED_TOTAL = 504


def validate_required_columns(frame: pd.DataFrame, source_name: str) -> None:
    required_columns = {
        "ImageID",
        "SourceSplit",
        "CandidateCategory",
        "MatchedClasses",
        "License",
        "OriginalURL",
        "OriginalLandingURL",
        "Author",
        "AuthorProfileURL",
        "SourceDataset",
        "RealSynthetic",
        "ReviewStatus",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{source_name} is missing required columns: {sorted(missing)}"
        )


def validate_batch(
    frame: pd.DataFrame,
    source_name: str,
    expected_count: int,
) -> None:
    validate_required_columns(frame, source_name)

    if len(frame) != expected_count:
        raise RuntimeError(
            f"{source_name} expected {expected_count} rows, found {len(frame)}."
        )

    if frame["ImageID"].duplicated().any():
        duplicates = frame.loc[
            frame["ImageID"].duplicated(keep=False),
            "ImageID",
        ].tolist()

        raise RuntimeError(
            f"{source_name} contains duplicate ImageIDs: {duplicates[:10]}"
        )

    invalid_review = frame[
        frame["ReviewStatus"] != "accepted_negative"
    ]

    if not invalid_review.empty:
        raise RuntimeError(
            f"{source_name} contains non-accepted review statuses."
        )

    invalid_real = frame[
        frame["RealSynthetic"] != "REAL"
    ]

    if not invalid_real.empty:
        raise RuntimeError(
            f"{source_name} contains non-REAL samples."
        )

    invalid_source = frame[
        frame["SourceDataset"] != "open_images_v7"
    ]

    if not invalid_source.empty:
        raise RuntimeError(
            f"{source_name} contains unexpected source datasets."
        )

    required_provenance_columns = [
        "ImageID",
        "License",
        "OriginalURL",
        "OriginalLandingURL",
        "Author",
        "AuthorProfileURL",
    ]

    for column in required_provenance_columns:
        if frame[column].isna().any():
            raise RuntimeError(
                f"{source_name} has missing provenance values in {column}."
            )


def print_distribution(frame: pd.DataFrame, column: str) -> None:
    print(f"\n{column}:")

    for value, count in frame[column].value_counts().items():
        print(f"  {value}: {count}")


def main() -> None:
    if not BATCH1_PATH.exists():
        raise FileNotFoundError(f"Batch 1 manifest not found: {BATCH1_PATH}")

    if not BATCH2_PATH.exists():
        raise FileNotFoundError(f"Batch 2 manifest not found: {BATCH2_PATH}")

    batch1 = pd.read_csv(BATCH1_PATH, dtype={"ImageID": str})
    batch2 = pd.read_csv(BATCH2_PATH, dtype={"ImageID": str})

    validate_batch(batch1, "Batch 1", EXPECTED_BATCH1)
    validate_batch(batch2, "Batch 2", EXPECTED_BATCH2)

    batch1["NegativeBatch"] = 1
    batch2["NegativeBatch"] = 2

    combined = pd.concat(
        [batch1, batch2],
        ignore_index=True,
        sort=False,
    )

    if len(combined) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} combined negatives, found {len(combined)}."
        )

    duplicate_mask = combined["ImageID"].duplicated(keep=False)

    if duplicate_mask.any():
        duplicates = combined.loc[
            duplicate_mask,
            "ImageID",
        ].tolist()

        raise RuntimeError(
            f"Cross-batch ImageID duplicates detected: {duplicates[:10]}"
        )

    combined["IsNegative"] = True
    combined["CanonicalClass"] = "license_plate"
    combined["CanonicalPlateInstanceCount"] = 0

    combined = combined.sort_values(
        by="ImageID",
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== FINAL OPEN IMAGES REAL NEGATIVE POOL ===")
    print(f"Batch 1: {len(batch1)}")
    print(f"Batch 2: {len(batch2)}")
    print(f"Total accepted negatives: {len(combined)}")
    print(f"Unique ImageIDs: {combined['ImageID'].nunique()}")
    print(f"Duplicate ImageIDs: {combined['ImageID'].duplicated().sum()}")

    print_distribution(combined, "CandidateCategory")
    print_distribution(combined, "SourceSplit")
    print_distribution(combined, "NegativeBatch")
    print_distribution(combined, "License")

    print("\nProvenance validation:")
    print("  ImageID: COMPLETE")
    print("  License: COMPLETE")
    print("  OriginalURL: COMPLETE")
    print("  OriginalLandingURL: COMPLETE")
    print("  Author: COMPLETE")
    print("  AuthorProfileURL: COMPLETE")

    print("\nNegative annotation policy:")
    print("  Canonical class: license_plate")
    print("  Plate instances per negative image: 0")
    print("  All samples visually audited: YES")

    print("\nRESULT: OPEN IMAGES REAL NEGATIVE POOL READY")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()