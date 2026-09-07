from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "baseline_v1"
MANIFEST_PATH = DATASET_ROOT / "metadata" / "baseline_split_manifest.csv"

ADAPTER_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "datasets" / "baseline_v1_coco"
OUTPUT_PATH = ADAPTER_ROOT / "annotations" / "instances_test.json"

EXPECTED_IMAGES = 669
EXPECTED_INSTANCES = 914
EXPECTED_NEGATIVES = 50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORY_ID = 1
CATEGORY_NAME = "license_plate"


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def load_test_manifest() -> dict[str, dict]:
    samples = {}

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "canonical_id",
            "split_status",
            "avax_split",
            "is_negative",
            "plate_instance_count",
        }

        missing_columns = required_columns.difference(reader.fieldnames or [])

        if missing_columns:
            raise RuntimeError(f"Missing manifest columns: {sorted(missing_columns)}")

        for row in reader:
            if row["split_status"].strip().lower() != "included":
                continue

            if row["avax_split"].strip().lower() != "test":
                continue

            canonical_id = row["canonical_id"].strip()
            is_negative = parse_bool(row["is_negative"])
            instance_count = int(float(row["plate_instance_count"]))

            if not canonical_id:
                raise RuntimeError("Empty canonical_id in TEST manifest")

            if canonical_id in samples:
                raise RuntimeError(f"Duplicate TEST canonical_id: {canonical_id}")

            if is_negative and instance_count != 0:
                raise RuntimeError(f"{canonical_id}: negative sample has {instance_count} instances")

            if not is_negative and instance_count <= 0:
                raise RuntimeError(f"{canonical_id}: positive sample has invalid instance count")

            samples[canonical_id] = {
                "is_negative": is_negative,
                "plate_instance_count": instance_count,
            }

    if len(samples) != EXPECTED_IMAGES:
        raise RuntimeError(f"TEST image count mismatch: expected={EXPECTED_IMAGES}, actual={len(samples)}")

    total_instances = sum(sample["plate_instance_count"] for sample in samples.values())
    total_negatives = sum(sample["is_negative"] for sample in samples.values())

    if total_instances != EXPECTED_INSTANCES:
        raise RuntimeError(f"TEST instance count mismatch: expected={EXPECTED_INSTANCES}, actual={total_instances}")

    if total_negatives != EXPECTED_NEGATIVES:
        raise RuntimeError(f"TEST negative count mismatch: expected={EXPECTED_NEGATIVES}, actual={total_negatives}")

    return samples


def scan_directory(directory: Path, extensions: set[str]) -> dict[str, Path]:
    result = {}

    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        if path.stem in result:
            raise RuntimeError(f"Duplicate file stem: {path.stem}")

        result[path.stem] = path

    return result


def parse_label(path: Path, expected_instances: int, is_negative: bool) -> list[tuple[float, float, float, float]]:
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines() if text else []

    if is_negative:
        if lines:
            raise RuntimeError(f"Negative TEST sample has non-empty label: {path}")
        return []

    if len(lines) != expected_instances:
        raise RuntimeError(
            f"Label count mismatch for {path}: expected={expected_instances}, actual={len(lines)}"
        )

    boxes = []

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(f"Invalid YOLO label: {path}:{line_number}")

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        if class_id != 0:
            raise RuntimeError(f"Unexpected class {class_id}: {path}:{line_number}")

        values = (x_center, y_center, width, height)

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite coordinates: {path}:{line_number}")

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise RuntimeError(f"Out-of-range coordinates: {path}:{line_number}")

        boxes.append(values)

    return boxes


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Final TEST adapter already exists: {OUTPUT_PATH}")

    samples = load_test_manifest()

    image_dir = DATASET_ROOT / "images" / "test"
    label_dir = DATASET_ROOT / "labels" / "test"

    images = scan_directory(image_dir, IMAGE_EXTENSIONS)
    labels = scan_directory(label_dir, {".txt"})

    expected_ids = set(samples)

    if set(images) != expected_ids:
        raise RuntimeError("TEST image IDs do not exactly match the canonical manifest")

    if set(labels) != expected_ids:
        raise RuntimeError("TEST label IDs do not exactly match the canonical manifest")

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for image_id, canonical_id in enumerate(sorted(samples), start=1):
        image_path = images[canonical_id]
        label_path = labels[canonical_id]
        sample = samples[canonical_id]

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = parse_label(
            label_path,
            sample["plate_instance_count"],
            sample["is_negative"],
        )

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        for x_center, y_center, width, height in boxes:
            box_width = width * image_width
            box_height = height * image_height
            x_min = (x_center - width / 2.0) * image_width
            y_min = (y_center - height / 2.0) * image_height

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

            annotation_id += 1

    if len(coco_images) != EXPECTED_IMAGES:
        raise RuntimeError("Generated TEST image count mismatch")

    if len(coco_annotations) != EXPECTED_INSTANCES:
        raise RuntimeError("Generated TEST annotation count mismatch")

    annotated_ids = {annotation["image_id"] for annotation in coco_annotations}
    negative_count = len(coco_images) - len(annotated_ids)

    if negative_count != EXPECTED_NEGATIVES:
        raise RuntimeError("Generated TEST negative count mismatch")

    coco = {
        "info": {
            "description": "AVAX ALPR baseline_v1 FINAL TEST adapter",
            "version": "baseline_v1",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {
                "id": CATEGORY_ID,
                "name": CATEGORY_NAME,
                "supercategory": CATEGORY_NAME,
            }
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, OUTPUT_PATH)

    print("=== FINAL TEST ADAPTER CREATED ===")
    print(f"Images: {len(coco_images)}")
    print(f"Plate instances: {len(coco_annotations)}")
    print(f"Negative images: {negative_count}")
    print(f"Output: {OUTPUT_PATH}")
    print("Canonical dataset: UNCHANGED")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()