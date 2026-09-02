from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "kaggle_plate_license_recognition"
SPLITS = ("train", "valid", "test")
TARGET_CLASS = "LicensePlate"

ROBOFLOW_SUFFIX_PATTERN = re.compile(r"\.rf\.[0-9a-fA-F]{32}\.[^.]+$")


def get_source_key(filename: str) -> str:
    return ROBOFLOW_SUFFIX_PATTERN.sub("", filename)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        digest = sha256()
        digest.update(rgb_image.size[0].to_bytes(4, "little"))
        digest.update(rgb_image.size[1].to_bytes(4, "little"))
        digest.update(rgb_image.tobytes())
        return digest.hexdigest()


def normalize_boxes(annotations: list[dict]) -> tuple:
    boxes = []

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        boxes.append((round(x, 3), round(y, 3), round(width, 3), round(height, 3)))

    return tuple(sorted(boxes))


def main() -> None:
    source_groups = defaultdict(list)

    for split in SPLITS:
        split_directory = DATASET_ROOT / split
        coco = load_coco(split_directory / "_annotations.coco.json")

        category_by_id = {category["id"]: category["name"] for category in coco["categories"]}
        annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            if category_by_id.get(annotation["category_id"]) == TARGET_CLASS:
                annotations_by_image[annotation["image_id"]].append(annotation)

        for image in coco["images"]:
            plate_annotations = annotations_by_image.get(image["id"], [])

            if not plate_annotations:
                continue

            image_path = split_directory / image["file_name"]
            source_key = get_source_key(image["file_name"])

            source_groups[source_key].append(
                {
                    "split": split,
                    "filename": image["file_name"],
                    "path": image_path,
                    "boxes": normalize_boxes(plate_annotations),
                }
            )

    multiplicity_counts = Counter()
    groups_with_multiple_variants = 0
    groups_with_identical_pixels = 0
    groups_with_different_pixels = 0
    groups_with_annotation_disagreement = 0

    different_pixel_examples = []
    annotation_disagreement_examples = []

    for source_key, variants in source_groups.items():
        multiplicity_counts[len(variants)] += 1

        if len(variants) == 1:
            continue

        groups_with_multiple_variants += 1

        pixel_hashes = {calculate_pixel_hash(variant["path"]) for variant in variants}

        if len(pixel_hashes) == 1:
            groups_with_identical_pixels += 1
        else:
            groups_with_different_pixels += 1

            if len(different_pixel_examples) < 20:
                different_pixel_examples.append((source_key, variants))

        box_sets = {variant["boxes"] for variant in variants}

        if len(box_sets) > 1:
            groups_with_annotation_disagreement += 1

            if len(annotation_disagreement_examples) < 20:
                annotation_disagreement_examples.append((source_key, variants))

    total_variants = sum(len(variants) for variants in source_groups.values())

    print("=== KAGGLE SOURCE GROUP AUDIT ===")
    print(f"Detector image variants: {total_variants}")
    print(f"Unique source groups: {len(source_groups)}")
    print(f"Groups with multiple variants: {groups_with_multiple_variants}")
    print(f"Multi-variant groups with identical pixels: {groups_with_identical_pixels}")
    print(f"Multi-variant groups with different pixels: {groups_with_different_pixels}")
    print(f"Groups with LicensePlate annotation disagreement: {groups_with_annotation_disagreement}")

    print()
    print("=== SOURCE GROUP MULTIPLICITY ===")

    for multiplicity, count in sorted(multiplicity_counts.items()):
        print(f"{multiplicity} variant(s): {count} source groups")

    if different_pixel_examples:
        print()
        print("=== DIFFERENT PIXEL EXAMPLES ===")

        for source_key, variants in different_pixel_examples:
            print(f"Source: {source_key}")

            for variant in variants:
                print(f"  {variant['split']}: {variant['filename']}")

    if annotation_disagreement_examples:
        print()
        print("=== ANNOTATION DISAGREEMENT EXAMPLES ===")

        for source_key, variants in annotation_disagreement_examples:
            print(f"Source: {source_key}")

            for variant in variants:
                print(f"  {variant['split']}: {variant['filename']}")
                print(f"    Boxes: {variant['boxes']}")


if __name__ == "__main__":
    main()