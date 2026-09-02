from hashlib import sha256
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "elpd"

TARGET_FILES = (
    "filename_prefix_00795_.png",
    "filename_prefix_00857_.png",
    "filename_prefix_01634_.png",
    "filename_prefix_01699_.png",
)

IMAGE_DIRECTORIES = {
    "PASCAL_VOC": DATASET_ROOT / "PASCAL_VOC" / "JPEGImages",
    "COCO": DATASET_ROOT / "COCO" / "images" / "Train",
    "YOLO": DATASET_ROOT / "YOLO" / "images" / "Train",
}


def calculate_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "FILE NOT FOUND"

    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.load()

        return True, f"VALID {dimensions[0]}x{dimensions[1]} sha256={calculate_hash(path)}"
    except Exception as error:
        return False, f"INVALID: {error}"


def main() -> None:
    print("=== ELPD CORRUPT REPLICA CHECK ===")

    for filename in TARGET_FILES:
        print()
        print(filename)

        for format_name, directory in IMAGE_DIRECTORIES.items():
            path = directory / filename
            valid, result = validate_image(path)
            status = "PASS" if valid else "FAIL"

            print(f"  {format_name}: {status} - {result}")


if __name__ == "__main__":
    main()