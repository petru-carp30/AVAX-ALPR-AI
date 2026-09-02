from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "kaggle_plate_license_recognition"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROBOFLOW_SUFFIX_PATTERN = re.compile(r"\.rf\.[0-9a-fA-F]{32}\.[^.]+$")


def calculate_file_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def get_source_key(filename: str) -> str:
    return ROBOFLOW_SUFFIX_PATTERN.sub("", filename)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    total_images = 0
    total_annotations = 0
    total_negative_images = 0

    split_files = {}
    split_source_keys = {}
    split_hashes = {}

    all_category_names = {}
    invalid_boxes = []
    missing_images = []
    extra_images = []
    corrupt_images = []
    dimension_mismatches = []

    print("=== KAGGLE COCO DATASET AUDIT ===")

    for split in SPLITS:
        split_directory = DATASET_ROOT / split
        annotation_path = split_directory / "_annotations.coco.json"

        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")

        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        coco = load_coco(annotation_path)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        image_by_id = {image["id"]: image for image in images}
        category_by_id = {category["id"]: category["name"] for category in categories}

        all_category_names.update(category_by_id)

        annotations_by_image = defaultdict(list)
        category_counts = Counter()

        for annotation in annotations:
            annotations_by_image[annotation["image_id"]].append(annotation)
            category_name = category_by_id.get(annotation["category_id"], f"UNKNOWN_{annotation['category_id']}")
            category_counts[category_name] += 1

        disk_images = {
            path.name: path
            for path in split_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        json_filenames = {image["file_name"] for image in images}

        for filename in sorted(json_filenames - set(disk_images)):
            missing_images.append((split, filename))

        for filename in sorted(set(disk_images) - json_filenames):
            extra_images.append((split, filename))

        split_hashes[split] = defaultdict(list)
        split_source_keys[split] = defaultdict(list)
        split_files[split] = disk_images

        negative_images = 0

        for image_info in images:
            filename = image_info["file_name"]
            image_path = disk_images.get(filename)

            if not annotations_by_image.get(image_info["id"]):
                negative_images += 1

            if image_path is None:
                continue

            try:
                with Image.open(image_path) as image:
                    actual_width, actual_height = image.size
                    image.verify()

                if actual_width != image_info["width"] or actual_height != image_info["height"]:
                    dimension_mismatches.append(
                        (
                            split,
                            filename,
                            image_info["width"],
                            image_info["height"],
                            actual_width,
                            actual_height,
                        )
                    )
            except Exception as error:
                corrupt_images.append((split, filename, str(error)))
                continue

            file_hash = calculate_file_hash(image_path)
            split_hashes[split][file_hash].append(filename)
            split_source_keys[split][get_source_key(filename)].append(filename)

        for annotation in annotations:
            image_info = image_by_id.get(annotation["image_id"])

            if image_info is None:
                invalid_boxes.append((split, annotation.get("id"), "Unknown image_id"))
                continue

            category_name = category_by_id.get(annotation["category_id"])

            if category_name is None:
                invalid_boxes.append((split, annotation.get("id"), "Unknown category_id"))
                continue

            bbox = annotation.get("bbox")

            if not isinstance(bbox, list) or len(bbox) != 4:
                invalid_boxes.append((split, annotation.get("id"), "Invalid bbox format"))
                continue

            x, y, width, height = bbox

            if width <= 0 or height <= 0:
                invalid_boxes.append((split, annotation.get("id"), f"Non-positive bbox: {bbox}"))
                continue

            if x < 0 or y < 0 or x + width > image_info["width"] or y + height > image_info["height"]:
                invalid_boxes.append((split, annotation.get("id"), f"Bounding box outside image: {bbox}"))

        total_images += len(images)
        total_annotations += len(annotations)
        total_negative_images += negative_images

        print()
        print(f"=== {split.upper()} ===")
        print(f"COCO images: {len(images)}")
        print(f"Disk images: {len(disk_images)}")
        print(f"Annotations: {len(annotations)}")
        print(f"Negative images: {negative_images}")
        print(f"Categories: {len(categories)}")

        for category_name, count in sorted(category_counts.items()):
            print(f"  {category_name}: {count}")

    print()
    print("=== TOTAL ===")
    print(f"Images: {total_images}")
    print(f"Annotations: {total_annotations}")
    print(f"Negative images: {total_negative_images}")
    print(f"Category names: {sorted(set(all_category_names.values()))}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Extra images: {len(extra_images)}")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Dimension mismatches: {len(dimension_mismatches)}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")

    print()
    print("=== EXACT CROSS-SPLIT DUPLICATES ===")

    exact_duplicate_count = 0

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1:]:
            overlapping_hashes = set(split_hashes[first_split]) & set(split_hashes[second_split])

            print(f"{first_split} <-> {second_split}: {len(overlapping_hashes)}")

            for file_hash in overlapping_hashes:
                exact_duplicate_count += 1

                if exact_duplicate_count <= 20:
                    print(f"  {first_split}: {split_hashes[first_split][file_hash]}")
                    print(f"  {second_split}: {split_hashes[second_split][file_hash]}")

    print()
    print("=== SOURCE-KEY CROSS-SPLIT OVERLAP ===")

    source_overlap_count = 0

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1:]:
            overlapping_keys = set(split_source_keys[first_split]) & set(split_source_keys[second_split])

            print(f"{first_split} <-> {second_split}: {len(overlapping_keys)}")

            for source_key in sorted(overlapping_keys):
                source_overlap_count += 1

                if source_overlap_count <= 30:
                    print(f"  Source: {source_key}")
                    print(f"  {first_split}: {split_source_keys[first_split][source_key]}")
                    print(f"  {second_split}: {split_source_keys[second_split][source_key]}")

    if missing_images:
        print()
        print("=== MISSING IMAGES ===")
        for item in missing_images[:20]:
            print(item)

    if corrupt_images:
        print()
        print("=== CORRUPT IMAGES ===")
        for item in corrupt_images[:20]:
            print(item)

    if invalid_boxes:
        print()
        print("=== INVALID BOUNDING BOXES ===")
        for item in invalid_boxes[:20]:
            print(item)


if __name__ == "__main__":
    main()