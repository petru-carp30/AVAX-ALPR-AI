from pathlib import Path
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "elpd"
VOC_ROOT = DATASET_ROOT / "PASCAL_VOC"
IMAGE_DIRECTORY = VOC_ROOT / "JPEGImages"
ANNOTATION_DIRECTORY = VOC_ROOT / "Annotations"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "documentation" / "dataset_review"

SEED = 42
RANDOM_SAMPLE_COUNT = 48
TILE_WIDTH = 320
TILE_HEIGHT = 320
COLUMNS = 4


def load_objects(annotation_path: Path):
    root = ET.parse(annotation_path).getroot()
    objects = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        objects.append(
            (
                float(bbox.findtext("xmin", "0")),
                float(bbox.findtext("ymin", "0")),
                float(bbox.findtext("xmax", "0")),
                float(bbox.findtext("ymax", "0")),
            )
        )

    return objects


def validate_image(image_path: Path):
    try:
        with Image.open(image_path) as image:
            image.load()
        return True
    except Exception:
        return False


def collect_samples():
    positive_samples = []
    negative_samples = []
    corrupt_samples = []

    for annotation_path in sorted(ANNOTATION_DIRECTORY.glob("*.xml")):
        image_path = IMAGE_DIRECTORY / f"{annotation_path.stem}.png"

        if not image_path.exists():
            continue

        objects = load_objects(annotation_path)

        if not validate_image(image_path):
            corrupt_samples.append(
                {
                    "image_path": image_path,
                    "objects": objects,
                }
            )
            continue

        sample = {
            "image_path": image_path,
            "objects": objects,
        }

        if objects:
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)

    return positive_samples, negative_samples, corrupt_samples


def draw_sample(sample):
    with Image.open(sample["image_path"]) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)

    for x_min, y_min, x_max, y_max in sample["objects"]:
        draw.rectangle(
            (
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max),
            ),
            outline="red",
            width=max(2, image.width // 500),
        )

    image.thumbnail((TILE_WIDTH, TILE_HEIGHT - 30))

    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
    x_offset = (TILE_WIDTH - image.width) // 2
    tile.paste(image, (x_offset, 0))

    label = f"{sample['image_path'].stem} | plates={len(sample['objects'])}"
    ImageDraw.Draw(tile).text((5, TILE_HEIGHT - 24), label, fill="black")

    return tile


def create_sheet(samples, output_path):
    if not samples:
        return

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
    positive_samples, negative_samples, corrupt_samples = collect_samples()

    random_generator = random.Random(SEED)
    random_positive_samples = random_generator.sample(
        positive_samples,
        min(RANDOM_SAMPLE_COUNT, len(positive_samples)),
    )

    corrupt_instances = sum(len(sample["objects"]) for sample in corrupt_samples)

    random_output = OUTPUT_ROOT / "elpd_random_positive_samples.jpg"
    negative_output = OUTPUT_ROOT / "elpd_negative_samples.jpg"

    create_sheet(random_positive_samples, random_output)
    create_sheet(negative_samples, negative_output)

    print("=== ELPD REVIEW SUMMARY ===")
    print(f"Usable positive images: {len(positive_samples)}")
    print(f"Usable negative images: {len(negative_samples)}")
    print(f"Corrupt images excluded: {len(corrupt_samples)}")
    print(f"Instances inside corrupt images: {corrupt_instances}")
    print(f"Usable total images: {len(positive_samples) + len(negative_samples)}")
    print(f"Random positive review sheet: {random_output}")
    print(f"Negative review sheet: {negative_output}")

    if corrupt_samples:
        print()
        print("=== CORRUPT SAMPLE DETAILS ===")

        for sample in corrupt_samples:
            print(
                f"{sample['image_path'].name}: "
                f"annotated_instances={len(sample['objects'])}"
            )


if __name__ == "__main__":
    main()