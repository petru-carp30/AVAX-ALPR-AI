from collections import Counter
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "romanian_lp" / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_annotations(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def calculate_file_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def validate_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as error:
        return str(error)


def parse_annotation(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    filename = root.findtext("filename")

    size_node = root.find("size")
    annotation_width = int(size_node.findtext("width")) if size_node is not None else None
    annotation_height = int(size_node.findtext("height")) if size_node is not None else None

    objects = []

    for object_node in root.findall("object"):
        class_name = object_node.findtext("name")
        box_node = object_node.find("bndbox")

        if box_node is None:
            objects.append({"class_name": class_name, "box": None})
            continue

        box = {
            "xmin": float(box_node.findtext("xmin")),
            "ymin": float(box_node.findtext("ymin")),
            "xmax": float(box_node.findtext("xmax")),
            "ymax": float(box_node.findtext("ymax")),
        }

        objects.append({"class_name": class_name, "box": box})

    return {
        "filename": filename,
        "width": annotation_width,
        "height": annotation_height,
        "objects": objects,
    }


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    images = find_images(DATASET_ROOT)
    annotations = find_annotations(DATASET_ROOT)

    images_by_name = {path.name: path for path in images}

    corrupt_images = []
    invalid_annotations = []
    invalid_boxes = []
    orphan_annotations = []
    annotated_images = set()
    class_counts = Counter()
    plate_instances = 0

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Images discovered: {len(images)}")
    print(f"XML annotations discovered: {len(annotations)}")

    for image_path in images:
        error = validate_image(image_path)

        if error:
            corrupt_images.append((image_path, error))

    for annotation_path in annotations:
        try:
            annotation = parse_annotation(annotation_path)
        except Exception as error:
            invalid_annotations.append((annotation_path, str(error)))
            continue

        filename = annotation["filename"]

        if not filename:
            invalid_annotations.append((annotation_path, "Missing filename"))
            continue

        image_path = images_by_name.get(filename)

        if image_path is None:
            orphan_annotations.append((annotation_path, filename))
            continue

        annotated_images.add(image_path)

        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
        except Exception:
            continue

        if annotation["width"] != actual_width or annotation["height"] != actual_height:
            invalid_annotations.append(
                (
                    annotation_path,
                    f"Annotation size {annotation['width']}x{annotation['height']} does not match image size {actual_width}x{actual_height}",
                )
            )

        for object_index, detected_object in enumerate(annotation["objects"]):
            class_name = detected_object["class_name"]
            box = detected_object["box"]

            class_counts[class_name] += 1
            plate_instances += 1

            if box is None:
                invalid_boxes.append((annotation_path, object_index, "Missing bounding box"))
                continue

            xmin = box["xmin"]
            ymin = box["ymin"]
            xmax = box["xmax"]
            ymax = box["ymax"]

            if xmax <= xmin or ymax <= ymin:
                invalid_boxes.append((annotation_path, object_index, f"Invalid dimensions: {box}"))
                continue

            if xmin < 0 or ymin < 0 or xmax > actual_width or ymax > actual_height:
                invalid_boxes.append((annotation_path, object_index, f"Bounding box outside image bounds: {box}"))

    unannotated_images = [path for path in images if path not in annotated_images]

    hashes = {}
    duplicate_groups = []

    for image_path in images:
        file_hash = calculate_file_hash(image_path)
        hashes.setdefault(file_hash, []).append(image_path)

    for paths in hashes.values():
        if len(paths) > 1:
            duplicate_groups.append(paths)

    print()
    print("=== DATASET INVENTORY ===")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Plate/object instances: {plate_instances}")
    print(f"Images without valid XML pairing: {len(unannotated_images)}")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Invalid XML annotations: {len(invalid_annotations)}")
    print(f"Orphan XML annotations: {len(orphan_annotations)}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")
    print(f"Exact duplicate image groups: {len(duplicate_groups)}")

    print()
    print("=== SOURCE CLASSES ===")

    if class_counts:
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
    else:
        print("No classes discovered.")

    if corrupt_images:
        print()
        print("=== CORRUPT IMAGES ===")
        for path, error in corrupt_images:
            print(f"{path}: {error}")

    if invalid_annotations:
        print()
        print("=== INVALID ANNOTATIONS ===")
        for path, error in invalid_annotations:
            print(f"{path}: {error}")

    if orphan_annotations:
        print()
        print("=== ORPHAN ANNOTATIONS ===")
        for path, filename in orphan_annotations:
            print(f"{path}: image '{filename}' not found")

    if invalid_boxes:
        print()
        print("=== INVALID BOUNDING BOXES ===")
        for path, object_index, error in invalid_boxes:
            print(f"{path}, object {object_index}: {error}")

    if duplicate_groups:
        print()
        print("=== EXACT DUPLICATE IMAGES ===")
        for group_index, paths in enumerate(duplicate_groups, start=1):
            print(f"Group {group_index}:")
            for path in paths:
                print(f"  {path}")


if __name__ == "__main__":
    main()