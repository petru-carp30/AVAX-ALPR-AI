from collections import defaultdict
from pathlib import Path
import json
import random

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "kaggle_plate_license_recognition"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "documentation" / "dataset_review"
SPLITS = ("train", "valid", "test")
TARGET_CLASS = "LicensePlate"
SAMPLE_COUNT = 40
SEED = 42


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_samples() -> list[dict]:
    samples = []

    for split in SPLITS:
        split_directory = DATASET_ROOT / split
        coco = load_coco(split_directory / "_annotations.coco.json")

        category_by_id = {category["id"]: category["name"] for category in coco["categories"]}
        annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            if category_by_id.get(annotation["category_id"]) == TARGET_CLASS:
                annotations_by_image[annotation["image_id"]].append(annotation)

        for image_info in coco["images"]:
            annotations = annotations_by_image.get(image_info["id"], [])

            if not annotations:
                continue

            samples.append(
                {
                    "split": split,
                    "path": split_directory / image_info["file_name"],
                    "filename": image_info["file_name"],
                    "annotations": annotations,
                }
            )

    return samples


def draw_sample(sample: dict, tile_size: int = 320) -> Image.Image:
    with Image.open(sample["path"]) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        for annotation in sample["annotations"]:
            x, y, width, height = annotation["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="red", width=4)

        image.thumbnail((tile_size, tile_size - 40))

        tile = Image.new("RGB", (tile_size, tile_size), "white")
        x_offset = (tile_size - image.width) // 2
        tile.paste(image, (x_offset, 0))

        draw = ImageDraw.Draw(tile)
        label = f"{sample['split']} | {len(sample['annotations'])} plate(s)"
        draw.text((8, tile_size - 32), label, fill="black")

        return tile


def create_sheet(samples: list[dict], output_path: Path, columns: int = 4, tile_size: int = 320) -> None:
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")

    for index, sample in enumerate(samples):
        tile = draw_sample(sample, tile_size)
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(tile, (x, y))

    sheet.save(output_path, quality=92)


def calculate_suspicion_score(sample: dict) -> float:
    score = 0.0

    with Image.open(sample["path"]) as image:
        image_area = image.width * image.height

    for annotation in sample["annotations"]:
        _, _, width, height = annotation["bbox"]
        aspect_ratio = width / height
        area_ratio = (width * height) / image_area

        if aspect_ratio < 1.2:
            score += 3.0

        if area_ratio > 0.25:
            score += 3.0

        if area_ratio < 0.001:
            score += 1.0

    if len(sample["annotations"]) >= 3:
        score += 2.0

    return score


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()

    random_generator = random.Random(SEED)
    random_samples = random_generator.sample(samples, min(SAMPLE_COUNT, len(samples)))

    suspicious_samples = sorted(samples, key=calculate_suspicion_score, reverse=True)[:SAMPLE_COUNT]

    create_sheet(random_samples, OUTPUT_ROOT / "random_samples.jpg")
    create_sheet(suspicious_samples, OUTPUT_ROOT / "suspicious_samples.jpg")

    print(f"Detector samples discovered: {len(samples)}")
    print(f"Random review sheet: {OUTPUT_ROOT / 'random_samples.jpg'}")
    print(f"Suspicious review sheet: {OUTPUT_ROOT / 'suspicious_samples.jpg'}")


if __name__ == "__main__":
    main()