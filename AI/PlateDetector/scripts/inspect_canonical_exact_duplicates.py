from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool"

SAMPLES_PATH = CANONICAL_ROOT / "metadata" / "canonical_samples.csv"
BOXES_PATH = CANONICAL_ROOT / "metadata" / "canonical_boxes.csv"
EXACT_PATH = CANONICAL_ROOT / "metadata" / "similarity" / "exact_duplicate_groups.csv"


def get_box_signature(boxes: pd.DataFrame, canonical_id: str) -> tuple:
    frame = boxes[boxes["canonical_id"] == canonical_id].sort_values("box_index")
    return tuple(
        (
            round(float(row.x_center), 8),
            round(float(row.y_center), 8),
            round(float(row.width), 8),
            round(float(row.height), 8),
        )
        for row in frame.itertuples(index=False)
    )


def main() -> None:
    samples = pd.read_csv(SAMPLES_PATH)
    boxes = pd.read_csv(BOXES_PATH)
    exact = pd.read_csv(EXACT_PATH)

    if exact.empty:
        print("No exact duplicate groups found.")
        return

    print("\n=== CANONICAL EXACT DUPLICATE INSPECTION ===")

    for group_id, group in exact.groupby("exact_group_id"):
        print(f"\n{group_id}")
        print(f"Pixel hash: {group.iloc[0]['pixel_sha256']}")

        signatures = []

        for row in group.itertuples(index=False):
            sample = samples[samples["canonical_id"] == row.canonical_id].iloc[0]
            signature = get_box_signature(boxes, row.canonical_id)
            signatures.append(signature)

            print(f"\n  Canonical ID: {row.canonical_id}")
            print(f"  Dataset: {sample['source_dataset']}")
            print(f"  Source image: {sample['source_image']}")
            print(f"  Source group: {sample['source_group']}")
            print(f"  Allowed split: {sample['allowed_split']}")
            print(f"  Plate instances: {sample['plate_instance_count']}")
            print(f"  Boxes: {signature}")

        annotation_match = all(signature == signatures[0] for signature in signatures[1:])

        print(f"\n  Same source group: {group['leakage_source_group'].nunique() == 1}")
        print(f"  Annotation match: {annotation_match}")

    print("\nRESULT: EXACT DUPLICATE INSPECTION COMPLETE")


if __name__ == "__main__":
    main()