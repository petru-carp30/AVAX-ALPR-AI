from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "kaggle_plate_license_recognition"
AUDIT_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "audits" / "kaggle_plate_license_recognition" / "kaggle_variant_review.csv"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "manifests" / "kaggle_plate_license_recognition"

FULL_MANIFEST_PATH = OUTPUT_ROOT / "kaggle_detector_filter_manifest.csv"
ACCEPTED_MANIFEST_PATH = OUTPUT_ROOT / "kaggle_positive_manifest.csv"
BOX_MANIFEST_PATH = OUTPUT_ROOT / "kaggle_positive_boxes.csv"

SPLITS = ("train", "valid", "test")
TARGET_CLASS = "LicensePlate"
ROBOFLOW_SUFFIX_PATTERN = re.compile(r"\.rf\.[0-9a-fA-F]{32}\.[^.]+$")

EXPECTED_DETECTOR_VARIANTS = 1539
EXPECTED_ACCEPTED_VARIANTS = 609
EXPECTED_ACCEPTED_GROUPS = 348
EXPECTED_ACCEPTED_INSTANCES = 635


def get_source_key(filename: str) -> str:
    return ROBOFLOW_SUFFIX_PATTERN.sub("", filename)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_detector_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_records = []
    box_records = []

    for split in SPLITS:
        split_root = DATASET_ROOT / split
        coco = load_coco(split_root / "_annotations.coco.json")
        category_by_id = {category["id"]: category["name"] for category in coco["categories"]}
        annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            if category_by_id.get(annotation["category_id"]) == TARGET_CLASS:
                annotations_by_image[annotation["image_id"]].append(annotation)

        for image in coco["images"]:
            annotations = annotations_by_image.get(image["id"], [])
            if not annotations:
                continue

            filename = image["file_name"]
            source_group = get_source_key(filename)

            image_records.append({
                "source_dataset": "kaggle_plate_license_recognition",
                "source_group": source_group,
                "source_split": split,
                "filename": filename,
                "source_image_path": (split_root / filename).relative_to(PROJECT_ROOT).as_posix(),
                "image_width": image["width"],
                "image_height": image["height"],
                "plate_instance_count": len(annotations),
                "real_synthetic": "REAL",
                "allowed_split": "TRAIN_ONLY",
                "canonical_class": "license_plate",
            })

            for box_index, annotation in enumerate(annotations, start=1):
                x, y, width, height = annotation["bbox"]

                box_records.append({
                    "source_dataset": "kaggle_plate_license_recognition",
                    "source_group": source_group,
                    "source_split": split,
                    "filename": filename,
                    "box_index": box_index,
                    "class_name": TARGET_CLASS,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "image_width": image["width"],
                    "image_height": image["height"],
                    "canonical_class": "license_plate",
                })

    return pd.DataFrame(image_records), pd.DataFrame(box_records)


def main() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Variant review CSV not found: {AUDIT_PATH}")

    review = pd.read_csv(AUDIT_PATH, dtype=str)
    image_frame, box_frame = collect_detector_data()

    if len(image_frame) != EXPECTED_DETECTOR_VARIANTS:
        raise RuntimeError(f"Expected {EXPECTED_DETECTOR_VARIANTS} detector variants, found {len(image_frame)}.")

    review_key_duplicates = review.duplicated(subset=["source_group", "filename"]).sum()
    if review_key_duplicates:
        raise RuntimeError(f"Variant review contains {review_key_duplicates} duplicate keys.")

    image_frame = image_frame.merge(
        review[["source_group", "filename", "review_status"]],
        on=["source_group", "filename"],
        how="left",
        validate="one_to_one",
    )

    missing_reviews = int(image_frame["review_status"].isna().sum())
    if missing_reviews:
        raise RuntimeError(f"{missing_reviews} detector variants are missing visual review status.")

    unsure_count = int((image_frame["review_status"] == "unsure").sum())
    if unsure_count:
        raise RuntimeError(f"{unsure_count} variants are still marked unsure.")

    accepted = image_frame[image_frame["review_status"] == "accepted_training_positive"].copy()
    accepted_keys = accepted[["source_group", "filename"]].copy()

    accepted_boxes = box_frame.merge(
        accepted_keys,
        on=["source_group", "filename"],
        how="inner",
        validate="many_to_one",
    )

    if len(accepted) != EXPECTED_ACCEPTED_VARIANTS:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED_VARIANTS} accepted variants, found {len(accepted)}.")

    accepted_groups = accepted["source_group"].nunique()
    if accepted_groups != EXPECTED_ACCEPTED_GROUPS:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED_GROUPS} accepted source groups, found {accepted_groups}.")

    if len(accepted_boxes) != EXPECTED_ACCEPTED_INSTANCES:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED_INSTANCES} accepted plate instances, found {len(accepted_boxes)}.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_frame = image_frame.sort_values(["source_group", "filename"], kind="stable").reset_index(drop=True)
    accepted = accepted.sort_values(["source_group", "filename"], kind="stable").reset_index(drop=True)
    accepted_boxes = accepted_boxes.sort_values(["source_group", "filename", "box_index"], kind="stable").reset_index(drop=True)

    image_frame.to_csv(FULL_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    accepted.to_csv(ACCEPTED_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    accepted_boxes.to_csv(BOX_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print("\n=== KAGGLE FILTERED DETECTOR MANIFEST ===")
    print(f"Detector-positive variants: {len(image_frame)}")
    print(f"Reviewed variants: {image_frame['review_status'].notna().sum()}")
    print(f"Unsure variants: {unsure_count}")

    print("\nFiltering results:")
    for status, count in image_frame["review_status"].value_counts().items():
        print(f"  {status}: {count}")

    print("\nAccepted training subset:")
    print(f"  Images: {len(accepted)}")
    print(f"  LicensePlate instances: {len(accepted_boxes)}")
    print(f"  Source groups represented: {accepted_groups}")
    print("  RealSynthetic: REAL")
    print("  Allowed split: TRAIN_ONLY")

    print("\nUpstream split composition of accepted samples:")
    for split, count in accepted["source_split"].value_counts().items():
        print(f"  {split}: {count}")

    cross_split_groups = accepted.groupby("source_group")["source_split"].nunique()
    cross_split_group_count = int((cross_split_groups > 1).sum())

    print("\nGrouping:")
    print(f"  Accepted source groups spanning multiple upstream splits: {cross_split_group_count}")
    print("  Upstream splits will NOT be reused: YES")
    print("  Source group identity preserved: YES")
    print("  Raw source remains untouched: YES")

    print("\nRESULT: KAGGLE FILTERED POSITIVE SOURCE MANIFEST READY")
    print(f"Full filter manifest: {FULL_MANIFEST_PATH}")
    print(f"Accepted manifest: {ACCEPTED_MANIFEST_PATH}")
    print(f"Accepted boxes: {BOX_MANIFEST_PATH}")


if __name__ == "__main__":
    main()