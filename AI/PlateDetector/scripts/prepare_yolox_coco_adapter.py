import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "baseline_v1"
MANIFEST_PATH = DATASET_ROOT / "metadata" / "baseline_split_manifest.csv"

OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "datasets" / "baseline_v1_coco"
ANNOTATION_OUTPUT_DIR = OUTPUT_ROOT / "annotations"

SPLITS = ("train", "val")
EXPECTED_IMAGE_COUNTS = {"train": 7622, "val": 625}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCE_CLASS_ID = 0
CATEGORY_ID = 1
CATEGORY_NAME = "license_plate"


@dataclass(frozen=True)
class ManifestSample:
    canonical_id: str
    split: str
    is_negative: bool
    plate_instance_count: int


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value in manifest: {value!r}")


def load_development_manifest() -> dict[str, dict[str, ManifestSample]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing canonical manifest: {MANIFEST_PATH}")

    samples = {split: {} for split in SPLITS}

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
            raise RuntimeError(
                f"Canonical manifest is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            if row["split_status"].strip().lower() != "included":
                continue

            split = row["avax_split"].strip().lower()

            if split not in SPLITS:
                continue

            canonical_id = row["canonical_id"].strip()

            if not canonical_id:
                raise RuntimeError("Canonical manifest contains an empty canonical_id.")

            if canonical_id in samples[split]:
                raise RuntimeError(
                    f"Duplicate canonical_id in {split}: {canonical_id}"
                )

            is_negative = parse_bool(row["is_negative"])
            instance_count = int(float(row["plate_instance_count"]))

            if is_negative and instance_count != 0:
                raise RuntimeError(
                    f"{canonical_id}: negative sample has plate_instance_count={instance_count}"
                )

            if not is_negative and instance_count <= 0:
                raise RuntimeError(
                    f"{canonical_id}: positive sample has invalid plate_instance_count={instance_count}"
                )

            samples[split][canonical_id] = ManifestSample(
                canonical_id=canonical_id,
                split=split,
                is_negative=is_negative,
                plate_instance_count=instance_count,
            )

    for split in SPLITS:
        expected = EXPECTED_IMAGE_COUNTS[split]
        actual = len(samples[split])

        if actual != expected:
            raise RuntimeError(
                f"{split.upper()} manifest count mismatch: expected={expected}, actual={actual}"
            )

    return samples


def map_unique_by_stem(paths: list[Path], description: str) -> dict[str, Path]:
    result = {}

    for path in paths:
        if path.stem in result:
            raise RuntimeError(
                f"Duplicate {description} stem {path.stem}: {result[path.stem]} and {path}"
            )

        result[path.stem] = path

    return result


def scan_canonical_split(split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical image directory: {image_dir}")

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical label directory: {label_dir}")

    image_entries = list(image_dir.iterdir())
    label_entries = list(label_dir.iterdir())

    nested_image_dirs = [path for path in image_entries if path.is_dir()]
    nested_label_dirs = [path for path in label_entries if path.is_dir()]

    if nested_image_dirs:
        raise RuntimeError(
            f"{split.upper()}: unexpected nested image directories: {nested_image_dirs[:5]}"
        )

    if nested_label_dirs:
        raise RuntimeError(
            f"{split.upper()}: unexpected nested label directories: {nested_label_dirs[:5]}"
        )

    unsupported_images = [
        path
        for path in image_entries
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
    ]

    unsupported_labels = [
        path
        for path in label_entries
        if path.is_file() and path.suffix.lower() != ".txt"
    ]

    if unsupported_images:
        raise RuntimeError(
            f"{split.upper()}: unexpected files in image directory: {unsupported_images[:5]}"
        )

    if unsupported_labels:
        raise RuntimeError(
            f"{split.upper()}: unexpected files in label directory: {unsupported_labels[:5]}"
        )

    images = map_unique_by_stem(
        [
            path
            for path in image_entries
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        f"{split} image",
    )

    labels = map_unique_by_stem(
        [
            path
            for path in label_entries
            if path.is_file() and path.suffix.lower() == ".txt"
        ],
        f"{split} label",
    )

    return images, labels


def preflight_canonical_sources(
    manifest: dict[str, dict[str, ManifestSample]],
) -> dict[str, tuple[dict[str, Path], dict[str, Path]]]:
    sources = {}
    errors = []

    for split in SPLITS:
        try:
            images, labels = scan_canonical_split(split)
        except Exception as exc:
            errors.append(f"{split.upper()}: {exc}")
            continue

        expected_ids = set(manifest[split])
        image_ids = set(images)
        label_ids = set(labels)
        expected_count = EXPECTED_IMAGE_COUNTS[split]

        if len(images) != expected_count:
            errors.append(
                f"{split.upper()}: image count mismatch: expected={expected_count}, actual={len(images)}"
            )

        if len(labels) != expected_count:
            errors.append(
                f"{split.upper()}: label count mismatch: expected={expected_count}, actual={len(labels)}"
            )

        missing_images = sorted(expected_ids - image_ids)
        extra_images = sorted(image_ids - expected_ids)
        missing_labels = sorted(expected_ids - label_ids)
        extra_labels = sorted(label_ids - expected_ids)

        if missing_images:
            errors.append(
                f"{split.upper()}: missing canonical images: count={len(missing_images)}, first={missing_images[:5]}"
            )

        if extra_images:
            errors.append(
                f"{split.upper()}: unexpected canonical images: count={len(extra_images)}, first={extra_images[:5]}"
            )

        if missing_labels:
            errors.append(
                f"{split.upper()}: missing canonical labels: count={len(missing_labels)}, first={missing_labels[:5]}"
            )

        if extra_labels:
            errors.append(
                f"{split.upper()}: unexpected canonical labels: count={len(extra_labels)}, first={extra_labels[:5]}"
            )

        sources[split] = (images, labels)

    if errors:
        raise RuntimeError(
            "Canonical baseline_v1 integrity check failed.\n"
            + "\n".join(f"  - {error}" for error in errors)
            + "\nNo COCO adapter files were written."
        )

    return sources


def parse_yolo_label(
    label_path: Path,
    expected_instances: int,
    is_negative: bool,
) -> list[tuple[float, float, float, float]]:
    text = label_path.read_text(encoding="utf-8").strip()
    lines = text.splitlines() if text else []

    if is_negative:
        if expected_instances != 0:
            raise RuntimeError(
                f"Negative sample has non-zero expected instance count: {label_path}"
            )

        if lines:
            raise RuntimeError(
                f"Canonical negative has non-empty label: {label_path}"
            )

        return []

    if len(lines) != expected_instances:
        raise RuntimeError(
            f"Label count mismatch for {label_path}: "
            f"expected={expected_instances}, actual={len(lines)}"
        )

    boxes = []

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Invalid YOLO label format: {label_path}:{line_number}"
            )

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid numeric YOLO label: {label_path}:{line_number}"
            ) from exc

        if class_id != SOURCE_CLASS_ID:
            raise RuntimeError(
                f"Unexpected class ID {class_id} in {label_path}:{line_number}"
            )

        values = (x_center, y_center, width, height)

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Non-finite YOLO coordinates in {label_path}:{line_number}"
            )

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise RuntimeError(
                f"Out-of-range YOLO box in {label_path}:{line_number}"
            )

        boxes.append(values)

    return boxes


def yolo_to_coco_bbox(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> list[float]:
    x_center, y_center, width, height = box

    box_width = width * image_width
    box_height = height * image_height
    x_min = (x_center - width / 2.0) * image_width
    y_min = (y_center - height / 2.0) * image_height

    return [x_min, y_min, box_width, box_height]


def build_split(
    split: str,
    manifest: dict[str, ManifestSample],
    images: dict[str, Path],
    labels: dict[str, Path],
) -> tuple[dict, dict]:
    coco_images = []
    coco_annotations = []

    annotation_id = 1

    for image_id, canonical_id in enumerate(sorted(manifest), start=1):
        sample = manifest[canonical_id]
        image_path = images[canonical_id]
        label_path = labels[canonical_id]

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        if image_width <= 0 or image_height <= 0:
            raise RuntimeError(f"Invalid image dimensions: {image_path}")

        boxes = parse_yolo_label(
            label_path,
            sample.plate_instance_count,
            sample.is_negative,
        )

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        for box in boxes:
            bbox = yolo_to_coco_bbox(box, image_width, image_height)

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

            annotation_id += 1

    coco = {
        "info": {
            "description": f"AVAX ALPR baseline_v1 {split} adapter for YOLOX",
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

    expected_annotations = sum(
        sample.plate_instance_count for sample in manifest.values()
    )

    expected_negatives = sum(
        sample.is_negative for sample in manifest.values()
    )

    if len(coco_images) != EXPECTED_IMAGE_COUNTS[split]:
        raise RuntimeError(
            f"{split.upper()}: generated image count mismatch"
        )

    if len(coco_annotations) != expected_annotations:
        raise RuntimeError(
            f"{split.upper()}: generated annotation count mismatch: "
            f"expected={expected_annotations}, actual={len(coco_annotations)}"
        )

    actual_negatives = len(coco_images) - len(
        {annotation["image_id"] for annotation in coco_annotations}
    )

    if actual_negatives != expected_negatives:
        raise RuntimeError(
            f"{split.upper()}: generated negative count mismatch: "
            f"expected={expected_negatives}, actual={actual_negatives}"
        )

    return coco, {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "negatives": actual_negatives,
    }


def write_json_temp(final_path: Path, data: dict) -> Path:
    temp_path = final_path.with_name(final_path.name + ".tmp")

    if temp_path.exists():
        temp_path.unlink()

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    return temp_path


def commit_outputs(coco_by_split: dict[str, dict]) -> None:
    ANNOTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_paths = {
        split: ANNOTATION_OUTPUT_DIR / f"instances_{split}.json"
        for split in SPLITS
    }

    backup_paths = {
        split: final_paths[split].with_name(
            final_paths[split].name + ".bak"
        )
        for split in SPLITS
    }

    for backup_path in backup_paths.values():
        if backup_path.exists():
            raise RuntimeError(
                f"Stale adapter transaction backup exists: {backup_path}"
            )

    temp_paths = {}
    moved_backups = []
    installed_outputs = []

    try:
        for split in SPLITS:
            temp_paths[split] = write_json_temp(
                final_paths[split],
                coco_by_split[split],
            )

        for split in SPLITS:
            final_path = final_paths[split]
            backup_path = backup_paths[split]

            if final_path.exists():
                os.replace(final_path, backup_path)
                moved_backups.append((final_path, backup_path))

        for split in SPLITS:
            final_path = final_paths[split]
            os.replace(temp_paths[split], final_path)
            installed_outputs.append(final_path)

    except Exception:
        for final_path in installed_outputs:
            if final_path.exists():
                final_path.unlink()

        for final_path, backup_path in reversed(moved_backups):
            if backup_path.exists():
                os.replace(backup_path, final_path)

        raise

    else:
        for _, backup_path in moved_backups:
            if backup_path.exists():
                backup_path.unlink()

    finally:
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink()


def main() -> None:
    test_json = ANNOTATION_OUTPUT_DIR / "instances_test.json"

    if test_json.exists():
        raise RuntimeError(
            f"TEST JSON must not exist during model development: {test_json}"
        )

    print("=== AVAX YOLOX COCO ADAPTER ===")
    print(f"Canonical dataset: {DATASET_ROOT}")
    print(f"Canonical manifest: {MANIFEST_PATH}")
    print(f"Adapter output: {OUTPUT_ROOT}")
    print()

    manifest = load_development_manifest()
    sources = preflight_canonical_sources(manifest)

    coco_by_split = {}
    summaries = {}

    for split in SPLITS:
        images, labels = sources[split]

        coco, summary = build_split(
            split,
            manifest[split],
            images,
            labels,
        )

        coco_by_split[split] = coco
        summaries[split] = summary

    commit_outputs(coco_by_split)

    for split in SPLITS:
        summary = summaries[split]

        print(f"{split.upper()}:")
        print(f"  Images: {summary['images']}")
        print(f"  Plate instances: {summary['annotations']}")
        print(f"  Negatives: {summary['negatives']}")
        print(
            f"  JSON: "
            f"{ANNOTATION_OUTPUT_DIR / f'instances_{split}.json'}"
        )
        print()

    print("TEST adapter: NOT CREATED")
    print("Canonical dataset: UNCHANGED")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()