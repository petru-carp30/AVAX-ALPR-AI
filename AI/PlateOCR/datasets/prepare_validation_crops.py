from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DETECTOR_DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateOCR" / "datasets" / "validation_mvp"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_CSV = OUTPUT_ROOT / "ground_truth.csv"

TARGET_CANDIDATES = 100
RANDOM_SEED = 42
PADDING_RATIO = 0.08

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

PLATE_CLASS_NAMES = {
    "license_plate",
    "license plate",
    "licence_plate",
    "licence plate",
    "plate",
    "number_plate",
    "number plate",
}

NORMALIZED_PLATE_CLASS_NAMES = {
    name.strip().lower().replace("-", "_") for name in PLATE_CLASS_NAMES
}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def find_image_for_xml(xml_path: Path) -> Path | None:
    try:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")

        if filename:
            direct_candidate = xml_path.parent / filename

            if direct_candidate.exists():
                return direct_candidate

        stem = Path(filename).stem if filename else xml_path.stem

        search_directories = [
            xml_path.parent,
            xml_path.parent.parent / "JPEGImages",
            xml_path.parent.parent / "images",
            xml_path.parent.parent / "Images",
        ]

        for directory in search_directories:
            for extension in VALID_IMAGE_EXTENSIONS:
                candidate = directory / f"{stem}{extension}"

                if candidate.exists():
                    return candidate

    except ET.ParseError:
        return None

    return None


def read_plate_boxes(xml_path: Path) -> list[tuple[int, int, int, int]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    boxes = []

    for obj in root.findall("object"):
        class_name = normalize_class_name(obj.findtext("name", ""))

        if class_name not in NORMALIZED_PLATE_CLASS_NAMES:
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        try:
            xmin = int(float(bbox.findtext("xmin", "0")))
            ymin = int(float(bbox.findtext("ymin", "0")))
            xmax = int(float(bbox.findtext("xmax", "0")))
            ymax = int(float(bbox.findtext("ymax", "0")))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box

    box_width = xmax - xmin
    box_height = ymax - ymin

    pad_x = int(round(box_width * PADDING_RATIO))
    pad_y = int(round(box_height * PADDING_RATIO))

    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(image_width, xmax + pad_x)
    ymax = min(image_height, ymax + pad_y)

    return xmin, ymin, xmax, ymax


def discover_samples() -> list[tuple[Path, Path, tuple[int, int, int, int]]]:
    samples = []

    xml_files = list(DETECTOR_DATASETS_ROOT.rglob("*.xml"))

    print(f"Scanning: {DETECTOR_DATASETS_ROOT}")
    print(f"XML files found: {len(xml_files)}")

    for xml_path in xml_files:
        image_path = find_image_for_xml(xml_path)

        if image_path is None:
            continue

        boxes = read_plate_boxes(xml_path)

        for box in boxes:
            samples.append((image_path, xml_path, box))

    return samples


def clear_previous_output() -> None:
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    for image_path in OUTPUT_IMAGES.iterdir():
        if image_path.is_file():
            image_path.unlink()

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()


def main() -> None:
    if not DETECTOR_DATASETS_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {DETECTOR_DATASETS_ROOT}"
        )

    clear_previous_output()

    samples = discover_samples()

    if not samples:
        raise RuntimeError(
            "No Pascal VOC license plate annotations were found. "
            "Check dataset paths and object class names."
        )

    print(f"Plate instances found: {len(samples)}")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    selected_samples = samples[:TARGET_CANDIDATES]

    rows = []

    for index, (image_path, xml_path, box) in enumerate(selected_samples, start=1):
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                expanded_box = expand_box(
                    box,
                    image.width,
                    image.height,
                )

                crop = image.crop(expanded_box)

                output_filename = f"plate_{index:04d}.jpg"
                output_path = OUTPUT_IMAGES / output_filename

                crop.save(output_path, quality=95)

                rows.append(
                    {
                        "filename": output_filename,
                        "plate_text": "",
                        "use_for_validation": "",
                        "notes": "",
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "source_image": str(image_path.relative_to(PROJECT_ROOT)),
                        "source_annotation": str(xml_path.relative_to(PROJECT_ROOT)),
                    }
                )

                print(
                    f"[{index:03d}/{len(selected_samples)}] "
                    f"{output_filename} "
                    f"{crop.width}x{crop.height}"
                )

        except Exception as exc:
            print(f"SKIPPED: {image_path}")
            print(f"Reason: {exc}")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "plate_text",
                "use_for_validation",
                "notes",
                "crop_width",
                "crop_height",
                "source_image",
                "source_annotation",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Validation candidate generation complete.")
    print(f"Candidates created: {len(rows)}")
    print(f"Images: {OUTPUT_IMAGES}")
    print(f"Ground truth: {OUTPUT_CSV}")
    print()
    print("MANUAL REVIEW:")
    print("Readable plate:")
    print("  plate_text = correct plate")
    print("  use_for_validation = yes")
    print()
    print("Unreadable or uncertain plate:")
    print("  plate_text = empty")
    print("  use_for_validation = no")
    print("  notes = unreadable")
    print()
    print("Target: select 50 reliable validation crops.")
    print("Do not guess uncertain characters.")


if __name__ == "__main__":
    main()