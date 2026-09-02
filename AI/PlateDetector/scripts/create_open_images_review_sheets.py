from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "documentation" / "dataset_review"

SPLITS = ("train", "val")
SEED = 42
RANDOM_SAMPLE_COUNT = 48
EDGE_SAMPLE_COUNT = 48
TILE_WIDTH = 320
TILE_HEIGHT = 240
COLUMNS = 4


def load_annotations(label_path: Path):
    annotations = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def collect_samples():
    samples = []

    for split in SPLITS:
        image_directory = DATASET_ROOT / "images" / split
        label_directory = DATASET_ROOT / "labels" / split

        for image_path in sorted(image_directory.glob("*")):
            if not image_path.is_file():
                continue

            label_path = label_directory / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            annotations = load_annotations(label_path)

            if not annotations:
                continue

            min_area = min(annotation[3] * annotation[4] for annotation in annotations)
            max_area = max(annotation[3] * annotation[4] for annotation in annotations)
            max_edge_proximity = max(
                max(
                    0.0,
                    annotation[3] / 2 - annotation[1],
                    annotation[4] / 2 - annotation[2],
                    annotation[1] + annotation[3] / 2 - 1.0,
                    annotation[2] + annotation[4] / 2 - 1.0,
                )
                for annotation in annotations
            )

            samples.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "annotations": annotations,
                    "plate_count": len(annotations),
                    "min_area": min_area,
                    "max_area": max_area,
                    "edge_score": max_edge_proximity,
                }
            )

    return samples


def draw_sample(sample):
    with Image.open(sample["image_path"]) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for _, x_center, y_center, width, height in sample["annotations"]:
        x_min = max(0, int((x_center - width / 2) * image_width))
        y_min = max(0, int((y_center - height / 2) * image_height))
        x_max = min(image_width - 1, int((x_center + width / 2) * image_width))
        y_max = min(image_height - 1, int((y_center + height / 2) * image_height))

        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=max(2, image_width // 500))

    image.thumbnail((TILE_WIDTH, TILE_HEIGHT - 30))

    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
    x_offset = (TILE_WIDTH - image.width) // 2
    tile.paste(image, (x_offset, 0))

    label = f"{sample['split']} | {sample['image_path'].stem} | plates={sample['plate_count']}"
    ImageDraw.Draw(tile).text((5, TILE_HEIGHT - 24), label, fill="black")

    return tile


def create_sheet(samples, output_path):
    rows = (len(samples) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * TILE_WIDTH, rows * TILE_HEIGHT), "white")

    for index, sample in enumerate(samples):
        tile = draw_sample(sample)
        x = (index % COLUMNS) * TILE_WIDTH
        y = (index // COLUMNS) * TILE_HEIGHT
        sheet.paste(tile, (x, y))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main():
    samples = collect_samples()

    random_generator = random.Random(SEED)
    random_samples = random_generator.sample(samples, min(RANDOM_SAMPLE_COUNT, len(samples)))

    edge_samples = sorted(
        samples,
        key=lambda sample: (
            sample["edge_score"],
            sample["plate_count"],
            sample["max_area"],
        ),
        reverse=True,
    )[:EDGE_SAMPLE_COUNT]

    random_output = OUTPUT_ROOT / "open_images_random_samples.jpg"
    edge_output = OUTPUT_ROOT / "open_images_edge_samples.jpg"

    create_sheet(random_samples, random_output)
    create_sheet(edge_samples, edge_output)

    print(f"Samples discovered: {len(samples)}")
    print(f"Random review sheet: {random_output}")
    print(f"Edge review sheet: {edge_output}")


if __name__ == "__main__":
    main()