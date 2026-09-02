import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "baseline_v1"
MANIFEST_PATH = DATASET_ROOT / "metadata" / "baseline_split_manifest.csv"

ADAPTER_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "datasets" / "baseline_v1_coco"
ANNOTATION_DIR = ADAPTER_ROOT / "annotations"

SPLITS = ("train", "val")

EXPECTED_COUNTS = {
    "train": {
        "images": 7622,
        "annotations": 9809,
        "negatives": 404,
    },
    "val": {
        "images": 625,
        "annotations": 857,
        "negatives": 50,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCE_CLASS_ID = 0
CATEGORY_ID = 1
CATEGORY_NAME = "license_plate"


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise RuntimeError(f"Invalid boolean value in canonical manifest: {value!r}")


def load_manifest() -> dict[str, dict[str, dict]]:
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
                    f"{split.upper()}: duplicate canonical_id in manifest: {canonical_id}"
                )

            is_negative = parse_bool(row["is_negative"])
            plate_instance_count = int(float(row["plate_instance_count"]))

            if is_negative and plate_instance_count != 0:
                raise RuntimeError(
                    f"{canonical_id}: negative sample has "
                    f"plate_instance_count={plate_instance_count}"
                )

            if not is_negative and plate_instance_count <= 0:
                raise RuntimeError(
                    f"{canonical_id}: positive sample has invalid "
                    f"plate_instance_count={plate_instance_count}"
                )

            samples[split][canonical_id] = {
                "is_negative": is_negative,
                "plate_instance_count": plate_instance_count,
            }

    for split in SPLITS:
        expected = EXPECTED_COUNTS[split]

        manifest_images = len(samples[split])
        manifest_annotations = sum(
            sample["plate_instance_count"]
            for sample in samples[split].values()
        )
        manifest_negatives = sum(
            sample["is_negative"]
            for sample in samples[split].values()
        )

        if manifest_images != expected["images"]:
            raise RuntimeError(
                f"{split.upper()}: manifest image count mismatch: "
                f"expected={expected['images']}, actual={manifest_images}"
            )

        if manifest_annotations != expected["annotations"]:
            raise RuntimeError(
                f"{split.upper()}: manifest annotation count mismatch: "
                f"expected={expected['annotations']}, actual={manifest_annotations}"
            )

        if manifest_negatives != expected["negatives"]:
            raise RuntimeError(
                f"{split.upper()}: manifest negative count mismatch: "
                f"expected={expected['negatives']}, actual={manifest_negatives}"
            )

    return samples


def scan_canonical_split(split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical image directory: {image_dir}")

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical label directory: {label_dir}")

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_paths = [
        path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    ]

    images = {}
    labels = {}

    for path in image_paths:
        if path.stem in images:
            raise RuntimeError(
                f"{split.upper()}: duplicate canonical image stem: {path.stem}"
            )

        images[path.stem] = path

    for path in label_paths:
        if path.stem in labels:
            raise RuntimeError(
                f"{split.upper()}: duplicate canonical label stem: {path.stem}"
            )

        labels[path.stem] = path

    return images, labels


def parse_yolo_label(
    label_path: Path,
    expected_instances: int,
    is_negative: bool,
) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing canonical label: {label_path}")

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if is_negative:
        if expected_instances != 0:
            raise RuntimeError(
                f"Negative sample has non-zero expected instances: {label_path}"
            )

        if lines:
            raise RuntimeError(
                f"Canonical negative has a non-empty label: {label_path}"
            )

        return []

    if len(lines) != expected_instances:
        raise RuntimeError(
            f"Canonical label count mismatch for {label_path}: "
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
                f"Unexpected YOLO class ID {class_id}: "
                f"{label_path}:{line_number}"
            )

        values = (x_center, y_center, width, height)

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Non-finite YOLO bbox: {label_path}:{line_number}"
            )

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise RuntimeError(
                f"Invalid normalized YOLO bbox: {label_path}:{line_number}"
            )

        boxes.append(values)

    return boxes


def yolo_to_coco_bbox(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = box

    box_width = width * image_width
    box_height = height * image_height
    x_min = (x_center - width / 2.0) * image_width
    y_min = (y_center - height / 2.0) * image_height

    return x_min, y_min, box_width, box_height


def assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            f"{context}: expected={expected}, actual={actual}"
        )


def validate_split(split: str, manifest: dict[str, dict]) -> dict:
    expected = EXPECTED_COUNTS[split]

    images, labels = scan_canonical_split(split)

    expected_ids = set(manifest)
    physical_image_ids = set(images)
    physical_label_ids = set(labels)

    if physical_image_ids != expected_ids:
        missing = sorted(expected_ids - physical_image_ids)
        extra = sorted(physical_image_ids - expected_ids)

        raise RuntimeError(
            f"{split.upper()}: canonical image identity mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    if physical_label_ids != expected_ids:
        missing = sorted(expected_ids - physical_label_ids)
        extra = sorted(physical_label_ids - expected_ids)

        raise RuntimeError(
            f"{split.upper()}: canonical label identity mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    json_path = ANNOTATION_DIR / f"instances_{split}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing COCO JSON: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = coco.get("categories")

    expected_categories = [
        {
            "id": CATEGORY_ID,
            "name": CATEGORY_NAME,
            "supercategory": CATEGORY_NAME,
        }
    ]

    if categories != expected_categories:
        raise RuntimeError(
            f"{split.upper()}: invalid COCO category definition: {categories}"
        )

    coco_images = coco.get("images")

    if not isinstance(coco_images, list):
        raise RuntimeError(
            f"{split.upper()}: COCO images field is missing or invalid"
        )

    coco_annotations = coco.get("annotations")

    if not isinstance(coco_annotations, list):
        raise RuntimeError(
            f"{split.upper()}: COCO annotations field is missing or invalid"
        )

    if len(coco_images) != expected["images"]:
        raise RuntimeError(
            f"{split.upper()}: COCO image count mismatch: "
            f"expected={expected['images']}, actual={len(coco_images)}"
        )

    if len(coco_annotations) != expected["annotations"]:
        raise RuntimeError(
            f"{split.upper()}: COCO annotation count mismatch: "
            f"expected={expected['annotations']}, actual={len(coco_annotations)}"
        )

    image_records = {}
    seen_file_names = set()
    coco_canonical_ids = set()

    for record in coco_images:
        image_id = record.get("id")
        file_name = record.get("file_name")

        if not isinstance(image_id, int):
            raise RuntimeError(
                f"{split.upper()}: invalid COCO image ID: {image_id}"
            )

        if image_id in image_records:
            raise RuntimeError(
                f"{split.upper()}: duplicate COCO image ID: {image_id}"
            )

        if not isinstance(file_name, str) or not file_name:
            raise RuntimeError(
                f"{split.upper()}: invalid COCO file_name for image ID {image_id}"
            )

        if file_name in seen_file_names:
            raise RuntimeError(
                f"{split.upper()}: duplicate COCO file_name: {file_name}"
            )

        if Path(file_name).name != file_name:
            raise RuntimeError(
                f"{split.upper()}: COCO file_name must reference the exact "
                f"canonical split file, not a nested path: {file_name}"
            )

        canonical_id = Path(file_name).stem

        if canonical_id not in manifest:
            raise RuntimeError(
                f"{split.upper()}: COCO references non-canonical sample: "
                f"{canonical_id}"
            )

        expected_image_path = images[canonical_id]

        if expected_image_path.name != file_name:
            raise RuntimeError(
                f"{split.upper()}: COCO filename mismatch for {canonical_id}: "
                f"expected={expected_image_path.name}, actual={file_name}"
            )

        with Image.open(expected_image_path) as image:
            actual_width, actual_height = image.size

        if record.get("width") != actual_width:
            raise RuntimeError(
                f"{split.upper()}: width mismatch for {canonical_id}"
            )

        if record.get("height") != actual_height:
            raise RuntimeError(
                f"{split.upper()}: height mismatch for {canonical_id}"
            )

        image_records[image_id] = {
            "canonical_id": canonical_id,
            "width": actual_width,
            "height": actual_height,
        }

        seen_file_names.add(file_name)
        coco_canonical_ids.add(canonical_id)

    if coco_canonical_ids != expected_ids:
        missing = sorted(expected_ids - coco_canonical_ids)
        extra = sorted(coco_canonical_ids - expected_ids)

        raise RuntimeError(
            f"{split.upper()}: COCO image identity mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    annotations_by_image = defaultdict(list)
    annotation_ids = set()

    for annotation in coco_annotations:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")

        if not isinstance(annotation_id, int):
            raise RuntimeError(
                f"{split.upper()}: invalid annotation ID: {annotation_id}"
            )

        if annotation_id in annotation_ids:
            raise RuntimeError(
                f"{split.upper()}: duplicate annotation ID: {annotation_id}"
            )

        if image_id not in image_records:
            raise RuntimeError(
                f"{split.upper()}: annotation {annotation_id} references "
                f"unknown image ID {image_id}"
            )

        if annotation.get("category_id") != CATEGORY_ID:
            raise RuntimeError(
                f"{split.upper()}: annotation {annotation_id} has "
                f"unexpected category_id={annotation.get('category_id')}"
            )

        bbox = annotation.get("bbox")

        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(
                f"{split.upper()}: invalid bbox in annotation {annotation_id}"
            )

        try:
            bbox = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{split.upper()}: non-numeric bbox in annotation {annotation_id}"
            ) from exc

        if not all(math.isfinite(value) for value in bbox):
            raise RuntimeError(
                f"{split.upper()}: non-finite bbox in annotation {annotation_id}"
            )

        if bbox[2] <= 0 or bbox[3] <= 0:
            raise RuntimeError(
                f"{split.upper()}: non-positive bbox size in "
                f"annotation {annotation_id}"
            )

        area = annotation.get("area")

        if not isinstance(area, (int, float)) or not math.isfinite(float(area)):
            raise RuntimeError(
                f"{split.upper()}: invalid area in annotation {annotation_id}"
            )

        assert_close(
            float(area),
            bbox[2] * bbox[3],
            f"{split.upper()} annotation {annotation_id} area",
        )

        if annotation.get("iscrowd") != 0:
            raise RuntimeError(
                f"{split.upper()}: annotation {annotation_id} "
                f"has unexpected iscrowd"
            )

        annotations_by_image[image_id].append(bbox)
        annotation_ids.add(annotation_id)

    actual_negative_count = 0

    for image_id, image_record in image_records.items():
        canonical_id = image_record["canonical_id"]
        sample = manifest[canonical_id]

        source_boxes = parse_yolo_label(
            labels[canonical_id],
            sample["plate_instance_count"],
            sample["is_negative"],
        )

        expected_bboxes = sorted(
            yolo_to_coco_bbox(
                box,
                image_record["width"],
                image_record["height"],
            )
            for box in source_boxes
        )

        actual_bboxes = sorted(
            tuple(bbox)
            for bbox in annotations_by_image.get(image_id, [])
        )

        if not actual_bboxes:
            actual_negative_count += 1

        if len(actual_bboxes) != len(expected_bboxes):
            raise RuntimeError(
                f"{split.upper()}: annotation count mismatch for "
                f"{canonical_id}: expected={len(expected_bboxes)}, "
                f"actual={len(actual_bboxes)}"
            )

        for box_number, (actual_bbox, expected_bbox) in enumerate(
            zip(actual_bboxes, expected_bboxes),
            start=1,
        ):
            for coordinate_index, (actual, expected_value) in enumerate(
                zip(actual_bbox, expected_bbox)
            ):
                assert_close(
                    actual,
                    expected_value,
                    f"{split.upper()} {canonical_id} bbox "
                    f"{box_number} coordinate {coordinate_index}",
                )

    if actual_negative_count != expected["negatives"]:
        raise RuntimeError(
            f"{split.upper()}: COCO negative count mismatch: "
            f"expected={expected['negatives']}, "
            f"actual={actual_negative_count}"
        )

    return {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "negatives": actual_negative_count,
    }


def main() -> None:
    print("=== YOLOX COCO ADAPTER VALIDATION ===")
    print(f"Canonical dataset: {DATASET_ROOT}")
    print(f"Canonical manifest: {MANIFEST_PATH}")
    print(f"Adapter: {ADAPTER_ROOT}")
    print()

    test_json = ANNOTATION_DIR / "instances_test.json"

    if test_json.exists():
        raise RuntimeError(
            f"TEST adapter must not exist during model development: {test_json}"
        )

    manifest = load_manifest()

    for split in SPLITS:
        result = validate_split(split, manifest[split])

        print(f"{split.upper()}: PASS")
        print(f"  Images: {result['images']}")
        print(f"  Plate instances: {result['annotations']}")
        print(f"  Negatives: {result['negatives']}")
        print()

    print("TEST adapter: ABSENT")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()