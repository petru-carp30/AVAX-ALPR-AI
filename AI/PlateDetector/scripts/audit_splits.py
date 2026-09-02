from collections import defaultdict
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "romanian_lp" / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def calculate_file_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def get_sequence_name(path: Path) -> str:
    filename = path.name

    if "#t=" in filename:
        return filename.split("#t=", 1)[0]

    return path.stem


def main() -> None:
    train_directory = DATASET_ROOT / "train" / "images"
    valid_directory = DATASET_ROOT / "valid" / "images"

    if not train_directory.exists():
        raise FileNotFoundError(f"Train directory not found: {train_directory}")

    if not valid_directory.exists():
        raise FileNotFoundError(f"Validation directory not found: {valid_directory}")

    train_images = find_images(train_directory)
    valid_images = find_images(valid_directory)

    print("=== SPLIT INVENTORY ===")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(valid_images)}")
    print(f"Total images: {len(train_images) + len(valid_images)}")

    train_sequences = defaultdict(list)
    valid_sequences = defaultdict(list)

    for image_path in train_images:
        train_sequences[get_sequence_name(image_path)].append(image_path)

    for image_path in valid_images:
        valid_sequences[get_sequence_name(image_path)].append(image_path)

    overlapping_sequences = sorted(set(train_sequences) & set(valid_sequences))

    print()
    print("=== SEQUENCE INVENTORY ===")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(valid_sequences)}")
    print(f"Sequences present in both splits: {len(overlapping_sequences)}")

    if overlapping_sequences:
        print()
        print("=== CROSS-SPLIT SEQUENCE LEAKAGE ===")

        for sequence in overlapping_sequences:
            print()
            print(f"Sequence: {sequence}")
            print(f"Train frames: {len(train_sequences[sequence])}")
            print(f"Validation frames: {len(valid_sequences[sequence])}")

            for image_path in train_sequences[sequence]:
                print(f"  TRAIN: {image_path.name}")

            for image_path in valid_sequences[sequence]:
                print(f"  VALID: {image_path.name}")

    train_hashes = defaultdict(list)
    valid_hashes = defaultdict(list)

    for image_path in train_images:
        train_hashes[calculate_file_hash(image_path)].append(image_path)

    for image_path in valid_images:
        valid_hashes[calculate_file_hash(image_path)].append(image_path)

    overlapping_hashes = sorted(set(train_hashes) & set(valid_hashes))

    print()
    print("=== EXACT CROSS-SPLIT DUPLICATES ===")
    print(f"Exact duplicate groups across train/validation: {len(overlapping_hashes)}")

    for file_hash in overlapping_hashes:
        print()

        for image_path in train_hashes[file_hash]:
            print(f"  TRAIN: {image_path.name}")

        for image_path in valid_hashes[file_hash]:
            print(f"  VALID: {image_path.name}")


if __name__ == "__main__":
    main()