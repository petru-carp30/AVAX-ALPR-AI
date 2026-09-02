from collections import Counter, defaultdict
from pathlib import Path
import json
import re


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


def main() -> None:
    total_images = 0
    total_detector_images = 0
    total_non_detector_images = 0
    total_plate_instances = 0
    total_multi_plate_images = 0

    detector_source_keys = {}
    detector_filenames = {}

    print("=== KAGGLE DETECTOR SUBSET AUDIT ===")

    for split in SPLITS:
        annotation_path = DATASET_ROOT / split / "_annotations.coco.json"
        coco = load_coco(annotation_path)

        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        category_by_id = {category["id"]: category["name"] for category in categories}
        annotations_by_image = defaultdict(list)

        for annotation in annotations:
            annotations_by_image[annotation["image_id"]].append(annotation)

        detector_images = []
        non_detector_images = []
        multi_plate_images = []
        plate_instances = 0
        non_detector_class_counts = Counter()

        detector_source_keys[split] = defaultdict(list)
        detector_filenames[split] = set()

        for image in images:
            image_annotations = annotations_by_image.get(image["id"], [])
            plate_annotations = [
                annotation
                for annotation in image_annotations
                if category_by_id.get(annotation["category_id"]) == TARGET_CLASS
            ]

            if plate_annotations:
                detector_images.append(image)
                detector_filenames[split].add(image["file_name"])
                detector_source_keys[split][get_source_key(image["file_name"])].append(image["file_name"])
                plate_instances += len(plate_annotations)

                if len(plate_annotations) > 1:
                    multi_plate_images.append((image["file_name"], len(plate_annotations)))
            else:
                non_detector_images.append(image)

                for annotation in image_annotations:
                    class_name = category_by_id.get(annotation["category_id"], "UNKNOWN")
                    non_detector_class_counts[class_name] += 1

        print()
        print(f"=== {split.upper()} ===")
        print(f"All images: {len(images)}")
        print(f"Images with LicensePlate bbox: {len(detector_images)}")
        print(f"Images without LicensePlate bbox: {len(non_detector_images)}")
        print(f"LicensePlate instances: {plate_instances}")
        print(f"Images with multiple LicensePlate boxes: {len(multi_plate_images)}")
        print(f"Unique detector source groups: {len(detector_source_keys[split])}")

        if non_detector_class_counts:
            print("Classes found in images without LicensePlate bbox:")
            for class_name, count in sorted(non_detector_class_counts.items()):
                print(f"  {class_name}: {count}")

        print("Example images without LicensePlate bbox:")
        for image in non_detector_images[:15]:
            print(f"  {image['file_name']}")

        if multi_plate_images:
            print("Example multi-plate images:")
            for filename, count in multi_plate_images[:10]:
                print(f"  {filename}: {count} plates")

        total_images += len(images)
        total_detector_images += len(detector_images)
        total_non_detector_images += len(non_detector_images)
        total_plate_instances += plate_instances
        total_multi_plate_images += len(multi_plate_images)

    print()
    print("=== TOTAL DETECTOR SUBSET ===")
    print(f"All images: {total_images}")
    print(f"Images with LicensePlate bbox: {total_detector_images}")
    print(f"Images without LicensePlate bbox: {total_non_detector_images}")
    print(f"LicensePlate instances: {total_plate_instances}")
    print(f"Images with multiple LicensePlate boxes: {total_multi_plate_images}")

    print()
    print("=== DETECTOR SOURCE-KEY CROSS-SPLIT OVERLAP ===")

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1:]:
            overlapping_keys = set(detector_source_keys[first_split]) & set(detector_source_keys[second_split])

            print(f"{first_split} <-> {second_split}: {len(overlapping_keys)}")

            for source_key in sorted(overlapping_keys)[:15]:
                print(f"  Source: {source_key}")
                print(f"    {first_split}: {detector_source_keys[first_split][source_key]}")
                print(f"    {second_split}: {detector_source_keys[second_split][source_key]}")


if __name__ == "__main__":
    main()