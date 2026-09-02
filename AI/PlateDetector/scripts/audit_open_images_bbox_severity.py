from collections import Counter
from pathlib import Path
import math


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"
SPLITS = ("train", "val")


def calculate_overshoot(x_center: float, y_center: float, width: float, height: float) -> float:
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return max(
        max(0.0, -x_min),
        max(0.0, -y_min),
        max(0.0, x_max - 1.0),
        max(0.0, y_max - 1.0),
    )


def get_bucket(overshoot: float) -> str:
    if overshoot <= 0:
        return "valid"

    if overshoot <= 0.0001:
        return "<= 0.0001"

    if overshoot <= 0.001:
        return "<= 0.001"

    if overshoot <= 0.01:
        return "<= 0.01"

    if overshoot <= 0.05:
        return "<= 0.05"

    return "> 0.05"


def main() -> None:
    bucket_counts = Counter()
    invalid_boxes = []
    invalid_files = set()
    total_boxes = 0

    for split in SPLITS:
        label_directory = DATASET_ROOT / "labels" / split

        for label_path in label_directory.glob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()

                if not stripped:
                    continue

                parts = stripped.split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    continue

                if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
                    continue

                total_boxes += 1

                overshoot = calculate_overshoot(x_center, y_center, width, height)
                bucket = get_bucket(overshoot)
                bucket_counts[bucket] += 1

                if overshoot > 0:
                    invalid_files.add((split, label_path.name))
                    invalid_boxes.append(
                        {
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                            "class_id": class_id,
                            "bbox": (x_center, y_center, width, height),
                            "overshoot": overshoot,
                        }
                    )

    invalid_boxes.sort(key=lambda item: item["overshoot"], reverse=True)

    print("=== OPEN IMAGES BBOX SEVERITY AUDIT ===")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")
    print(f"Label files affected: {len(invalid_files)}")

    print()
    print("=== OVERSHOOT DISTRIBUTION ===")

    for bucket in ("valid", "<= 0.0001", "<= 0.001", "<= 0.01", "<= 0.05", "> 0.05"):
        print(f"{bucket}: {bucket_counts[bucket]}")

    print()
    print("=== WORST INVALID BOXES ===")

    for item in invalid_boxes[:30]:
        print(
            f"{item['split']}/{item['file']} "
            f"line={item['line']} "
            f"overshoot={item['overshoot']:.8f} "
            f"bbox={item['bbox']}"
        )


if __name__ == "__main__":
    main()