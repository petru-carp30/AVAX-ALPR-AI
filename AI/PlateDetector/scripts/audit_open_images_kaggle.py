from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import math
import re

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"
SPLITS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OPEN_IMAGES_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")


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


def discover_labels(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    annotations = []
    errors = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as error:
        return [], [f"Unable to read label: {error}"]

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) != 5:
            errors.append(f"Line {line_number}: expected 5 values, found {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"Line {line_number}: invalid numeric value")
            continue

        values = (x_center, y_center, width, height)

        if not all(math.isfinite(value) for value in values):
            errors.append(f"Line {line_number}: non-finite bbox value")
            continue

        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            errors.append(f"Line {line_number}: center outside normalized range")

        if not 0 < width <= 1 or not 0 < height <= 1:
            errors.append(f"Line {line_number}: invalid normalized width/height")

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        epsilon = 1e-6

        if x_min < -epsilon or y_min < -epsilon or x_max > 1 + epsilon or y_max > 1 + epsilon:
            errors.append(f"Line {line_number}: bbox extends outside image")

        annotations.append((class_id, x_center, y_center, width, height))

    return annotations, errors


def main() -> None:
    total_images = 0
    total_labels = 0
    total_instances = 0
    total_negative_images = 0
    total_multi_plate_images = 0

    split_hashes = {}
    split_image_ids = {}

    invalid_labels = []
    corrupt_images = []
    missing_labels = []
    orphan_labels = []
    invalid_image_ids = []

    class_counts = Counter()
    dimension_counts = Counter()

    print("=== OPEN IMAGES KAGGLE DATASET AUDIT ===")

    for split in SPLITS:
        image_directory = DATASET_ROOT / "images" / split
        label_directory = DATASET_ROOT / "labels" / split

        if not image_directory.exists():
            raise FileNotFoundError(f"Image directory not found: {image_directory}")

        if not label_directory.exists():
            raise FileNotFoundError(f"Label directory not found: {label_directory}")

        images = discover_images(image_directory)
        labels = discover_labels(label_directory)

        image_ids = set(images)
        label_ids = set(labels)

        split_hashes[split] = defaultdict(list)
        split_image_ids[split] = image_ids

        split_instances = 0
        split_negative_images = 0
        split_multi_plate_images = 0

        for image_id in sorted(image_ids):
            image_path = images[image_id]

            if not OPEN_IMAGES_ID_PATTERN.fullmatch(image_id):
                invalid_image_ids.append((split, image_id))

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    dimension_counts[(width, height)] += 1
                    image.verify()
            except Exception as error:
                corrupt_images.append((split, image_path.name, str(error)))
                continue

            split_hashes[split][calculate_file_hash(image_path)].append(image_path.name)

        for image_id in sorted(image_ids - label_ids):
            missing_labels.append((split, image_id))
            split_negative_images += 1

        for label_id in sorted(label_ids - image_ids):
            orphan_labels.append((split, label_id))

        for image_id in sorted(image_ids & label_ids):
            annotations, errors = parse_label(labels[image_id])

            if errors:
                invalid_labels.append((split, labels[image_id].name, errors))

            if not annotations:
                split_negative_images += 1
                continue

            split_instances += len(annotations)

            if len(annotations) > 1:
                split_multi_plate_images += 1

            for annotation in annotations:
                class_counts[annotation[0]] += 1

        print()
        print(f"=== {split.upper()} ===")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Instances: {split_instances}")
        print(f"Negative/unlabeled images: {split_negative_images}")
        print(f"Multi-plate images: {split_multi_plate_images}")
        print(f"Missing labels: {len(image_ids - label_ids)}")
        print(f"Orphan labels: {len(label_ids - image_ids)}")

        total_images += len(images)
        total_labels += len(labels)
        total_instances += split_instances
        total_negative_images += split_negative_images
        total_multi_plate_images += split_multi_plate_images

    print()
    print("=== TOTAL ===")
    print(f"Images: {total_images}")
    print(f"Label files: {total_labels}")
    print(f"Instances: {total_instances}")
    print(f"Negative/unlabeled images: {total_negative_images}")
    print(f"Multi-plate images: {total_multi_plate_images}")
    print(f"Classes: {dict(sorted(class_counts.items()))}")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Invalid label files: {len(invalid_labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Orphan labels: {len(orphan_labels)}")
    print(f"Invalid Open Images style IDs: {len(invalid_image_ids)}")

    print()
    print("=== IMAGE DIMENSIONS ===")

    for dimensions, count in dimension_counts.most_common(15):
        print(f"{dimensions[0]}x{dimensions[1]}: {count}")

    print()
    print("=== SOURCE-ID CROSS-SPLIT OVERLAP ===")

    overlapping_ids = split_image_ids["train"] & split_image_ids["val"]

    print(f"train <-> val: {len(overlapping_ids)}")

    for image_id in sorted(overlapping_ids)[:20]:
        print(f"  {image_id}")

    print()
    print("=== EXACT DUPLICATES ===")

    train_hashes = split_hashes["train"]
    val_hashes = split_hashes["val"]

    train_internal_duplicates = {
        file_hash: filenames
        for file_hash, filenames in train_hashes.items()
        if len(filenames) > 1
    }

    val_internal_duplicates = {
        file_hash: filenames
        for file_hash, filenames in val_hashes.items()
        if len(filenames) > 1
    }

    cross_split_hashes = set(train_hashes) & set(val_hashes)

    print(f"Train duplicate groups: {len(train_internal_duplicates)}")
    print(f"Val duplicate groups: {len(val_internal_duplicates)}")
    print(f"Exact duplicate groups across train/val: {len(cross_split_hashes)}")

    for file_hash in list(cross_split_hashes)[:20]:
        print(f"  train: {train_hashes[file_hash]}")
        print(f"  val: {val_hashes[file_hash]}")

    if invalid_labels:
        print()
        print("=== INVALID LABEL EXAMPLES ===")

        for split, filename, errors in invalid_labels[:20]:
            print(f"{split}/{filename}")
            for error in errors:
                print(f"  {error}")

    if corrupt_images:
        print()
        print("=== CORRUPT IMAGE EXAMPLES ===")

        for item in corrupt_images[:20]:
            print(item)


if __name__ == "__main__":
    main()