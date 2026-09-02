from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "elpd"
VOC_ROOT = DATASET_ROOT / "PASCAL_VOC"
IMAGE_DIRECTORY = VOC_ROOT / "JPEGImages"
ANNOTATION_DIRECTORY = VOC_ROOT / "Annotations"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def calculate_file_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def discover_images(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def discover_annotations(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".xml"
    }


def main() -> None:
    images = discover_images(IMAGE_DIRECTORY)
    annotations = discover_annotations(ANNOTATION_DIRECTORY)

    image_ids = set(images)
    annotation_ids = set(annotations)

    missing_annotations = sorted(image_ids - annotation_ids)
    orphan_annotations = sorted(annotation_ids - image_ids)

    corrupt_images = []
    invalid_xml_files = []
    invalid_boxes = []
    empty_annotations = []

    class_counts = Counter()
    dimension_counts = Counter()
    hash_groups = defaultdict(list)

    total_instances = 0
    multi_plate_images = 0

    print("=== ELPD DATASET AUDIT ===")
    print(f"Images: {len(images)}")
    print(f"XML annotations: {len(annotations)}")

    for image_id, image_path in sorted(images.items()):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                dimension_counts[(width, height)] += 1
                image.verify()
        except Exception as error:
            corrupt_images.append((image_path.name, str(error)))
            continue

        hash_groups[calculate_file_hash(image_path)].append(image_path.name)

    for image_id in sorted(image_ids & annotation_ids):
        annotation_path = annotations[image_id]

        try:
            root = ET.parse(annotation_path).getroot()
        except Exception as error:
            invalid_xml_files.append((annotation_path.name, str(error)))
            continue

        size = root.find("size")

        if size is None:
            invalid_xml_files.append((annotation_path.name, "Missing size element"))
            continue

        try:
            xml_width = int(size.findtext("width", "0"))
            xml_height = int(size.findtext("height", "0"))
        except ValueError:
            invalid_xml_files.append((annotation_path.name, "Invalid image dimensions"))
            continue

        objects = root.findall("object")

        if not objects:
            empty_annotations.append(annotation_path.name)
            continue

        if len(objects) > 1:
            multi_plate_images += 1

        total_instances += len(objects)

        for object_index, obj in enumerate(objects, start=1):
            class_name = obj.findtext("name", "").strip()
            class_counts[class_name or "<EMPTY>"] += 1

            bbox = obj.find("bndbox")

            if bbox is None:
                invalid_boxes.append((annotation_path.name, object_index, "Missing bndbox"))
                continue

            try:
                x_min = float(bbox.findtext("xmin", "nan"))
                y_min = float(bbox.findtext("ymin", "nan"))
                x_max = float(bbox.findtext("xmax", "nan"))
                y_max = float(bbox.findtext("ymax", "nan"))
            except ValueError:
                invalid_boxes.append((annotation_path.name, object_index, "Invalid numeric bbox"))
                continue

            if not (0 <= x_min < x_max <= xml_width and 0 <= y_min < y_max <= xml_height):
                invalid_boxes.append(
                    (
                        annotation_path.name,
                        object_index,
                        f"bbox=({x_min}, {y_min}, {x_max}, {y_max}), image=({xml_width}, {xml_height})",
                    )
                )

    duplicate_groups = {
        file_hash: filenames
        for file_hash, filenames in hash_groups.items()
        if len(filenames) > 1
    }

    print()
    print("=== ANNOTATION SUMMARY ===")
    print(f"License-plate instances: {total_instances}")
    print(f"Multi-object images: {multi_plate_images}")
    print(f"Empty annotations / negatives: {len(empty_annotations)}")
    print(f"Classes: {dict(class_counts)}")

    print()
    print("=== VALIDATION ===")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Invalid XML files: {len(invalid_xml_files)}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Orphan annotations: {len(orphan_annotations)}")

    print()
    print("=== IMAGE DIMENSIONS ===")

    for dimensions, count in dimension_counts.most_common(15):
        print(f"{dimensions[0]}x{dimensions[1]}: {count}")

    print()
    print("=== EXACT DUPLICATES ===")
    print(f"Exact duplicate groups: {len(duplicate_groups)}")

    for filenames in list(duplicate_groups.values())[:20]:
        print(f"  {filenames}")

    if invalid_boxes:
        print()
        print("=== INVALID BBOX EXAMPLES ===")

        for item in invalid_boxes[:30]:
            print(item)

    if invalid_xml_files:
        print()
        print("=== INVALID XML EXAMPLES ===")

        for item in invalid_xml_files[:20]:
            print(item)

    if corrupt_images:
        print()
        print("=== CORRUPT IMAGE EXAMPLES ===")

        for filename, error in corrupt_images:
            print(f"{filename}: {error}")


if __name__ == "__main__":
    main()