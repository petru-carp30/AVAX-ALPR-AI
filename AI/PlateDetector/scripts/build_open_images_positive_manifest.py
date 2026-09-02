from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"
DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"

OUTPUT_ROOT = DATASETS_ROOT / "manifests" / "open_images_lp_kaggle"
IMAGE_MANIFEST_PATH = OUTPUT_ROOT / "open_images_positive_manifest.csv"
BOX_MANIFEST_PATH = OUTPUT_ROOT / "open_images_positive_boxes.csv"

EXPECTED_IMAGES = 5368
EXPECTED_INSTANCES = 7852
EXPECTED_CLASS_IDS = {0}
EXPECTED_CLIPPABLE_BOXES = 192

PROVENANCE_PATH = DATASETS_ROOT / "provenance" / "open_images_lp_kaggle" / "open_images_provenance.csv"

REQUIRED_PROVENANCE_COLUMNS = [
    "ImageID",
    "License",
    "OriginalURL",
    "OriginalLandingURL",
    "Author",
    "AuthorProfileURL",
]

COLUMN_ALIASES = {
    "imageid": "ImageID",
    "license": "License",
    "originalurl": "OriginalURL",
    "originallandingurl": "OriginalLandingURL",
    "author": "Author",
    "authorprofileurl": "AuthorProfileURL",
    "title": "Title",
    "thumbnail300kurl": "Thumbnail300KURL",
    "originalmd5": "OriginalMD5",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for column in frame.columns:
        normalized = column.strip().lower()

        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]

    return frame.rename(columns=rename_map)


def load_provenance_index() -> pd.DataFrame:
    if not PROVENANCE_PATH.exists():
        raise FileNotFoundError(
            f"Open Images provenance file not found: {PROVENANCE_PATH}"
        )

    frame = pd.read_csv(
        PROVENANCE_PATH,
        dtype=str,
        low_memory=False,
    )

    required_columns = [
        "ImageID",
        "LocalKaggleSplit",
        "LocalFileName",
        "OpenImagesSplit",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "Title",
        "SourceMetadataURL",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Provenance CSV is missing required columns: {missing_columns}"
        )

    frame["ImageID"] = frame["ImageID"].str.strip().str.lower()

    if len(frame) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} provenance records, found {len(frame)}."
        )

    duplicate_ids = frame["ImageID"].duplicated(keep=False)

    if duplicate_ids.any():
        duplicates = frame.loc[
            duplicate_ids,
            "ImageID",
        ].tolist()

        raise RuntimeError(
            f"Duplicate ImageIDs found in provenance CSV: {duplicates[:20]}"
        )

    for column in REQUIRED_PROVENANCE_COLUMNS:
        missing_values = int(frame[column].isna().sum())

        if missing_values:
            raise RuntimeError(
                f"Provenance CSV contains {missing_values} missing values in {column}."
            )

    frame["_source_csv"] = PROVENANCE_PATH.relative_to(PROJECT_ROOT).as_posix()

    print("\nOpen Images provenance:")
    print(f"  Records: {len(frame)}")
    print(f"  Unique ImageIDs: {frame['ImageID'].nunique()}")
    print(f"  Duplicate ImageIDs: {frame['ImageID'].duplicated().sum()}")
    print(f"  Source: {PROVENANCE_PATH}")

    return frame


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.strip().split()

    if len(parts) != 5:
        raise RuntimeError(f"Invalid YOLO label line: {line}")

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    return class_id, x_center, y_center, width, height


def convert_to_corners(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0

    return x_min, y_min, x_max, y_max


def clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def overshoot_amount(x_min: float, y_min: float, x_max: float, y_max: float) -> float:
    return max(
        0.0 - x_min,
        0.0 - y_min,
        x_max - 1.0,
        y_max - 1.0,
        0.0,
    )


def overshoot_bucket(value: float) -> str:
    if value == 0.0:
        return "valid"
    if value <= 1e-4:
        return "<=1e-4"
    if value <= 1e-3:
        return "<=1e-3"
    if value <= 1e-2:
        return "<=1e-2"
    if value <= 5e-2:
        return "<=5e-2"
    return ">5e-2"


def collect_split(split_name: str) -> tuple[list[dict], list[dict]]:
    image_root = RAW_ROOT / "images" / split_name
    label_root = RAW_ROOT / "labels" / split_name

    image_paths = sorted(image_root.glob("*.jpg"))
    label_paths = sorted(label_root.glob("*.txt"))

    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    if missing_labels:
        raise RuntimeError(f"{split_name}: {len(missing_labels)} images are missing labels.")

    if orphan_labels:
        raise RuntimeError(f"{split_name}: {len(orphan_labels)} orphan labels found.")

    image_records = []
    box_records = []

    for image_path in image_paths:
        label_path = label_root / f"{image_path.stem}.txt"

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except Exception as exception:
            raise RuntimeError(f"Corrupt image: {image_path}: {exception}") from exception

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        if not lines:
            raise RuntimeError(f"Unexpected empty label file in positive source: {label_path}")

        per_image_box_count = 0
        per_image_requires_clipping = 0
        per_image_non_clippable = 0
        per_image_max_overshoot = 0.0

        for box_index, line in enumerate(lines, start=1):
            class_id, x_center, y_center, box_width, box_height = parse_yolo_label_line(line)

            x_min, y_min, x_max, y_max = convert_to_corners(x_center, y_center, box_width, box_height)

            overshoot = overshoot_amount(x_min, y_min, x_max, y_max)
            bucket = overshoot_bucket(overshoot)

            clipped_x_min = clip01(x_min)
            clipped_y_min = clip01(y_min)
            clipped_x_max = clip01(x_max)
            clipped_y_max = clip01(y_max)

            clipped_width = clipped_x_max - clipped_x_min
            clipped_height = clipped_y_max - clipped_y_min

            requires_clipping = overshoot > 0.0
            non_clippable = clipped_width <= 0.0 or clipped_height <= 0.0 or bucket == ">5e-2"

            if requires_clipping:
                per_image_requires_clipping += 1

            if non_clippable:
                per_image_non_clippable += 1

            per_image_max_overshoot = max(per_image_max_overshoot, overshoot)

            box_records.append(
                {
                    "source_dataset": "open_images_v7",
                    "source_split": split_name,
                    "image_id": image_path.stem.lower(),
                    "source_image_name": image_path.name,
                    "source_label_name": label_path.name,
                    "box_index": box_index,
                    "class_id": class_id,
                    "x_center_raw": x_center,
                    "y_center_raw": y_center,
                    "width_raw": box_width,
                    "height_raw": box_height,
                    "x_min_raw": x_min,
                    "y_min_raw": y_min,
                    "x_max_raw": x_max,
                    "y_max_raw": y_max,
                    "requires_clipping": requires_clipping,
                    "non_clippable": non_clippable,
                    "overshoot": overshoot,
                    "overshoot_bucket": bucket,
                    "x_min_clipped": clipped_x_min,
                    "y_min_clipped": clipped_y_min,
                    "x_max_clipped": clipped_x_max,
                    "y_max_clipped": clipped_y_max,
                    "width_clipped": clipped_width,
                    "height_clipped": clipped_height,
                    "real_synthetic": "REAL",
                    "is_negative": False,
                    "canonical_class": "license_plate",
                }
            )

            per_image_box_count += 1

        image_records.append(
            {
                "source_dataset": "open_images_v7",
                "source_split": split_name,
                "image_id": image_path.stem.lower(),
                "source_image_name": image_path.name,
                "source_label_name": label_path.name,
                "source_image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_label_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                "image_width": width,
                "image_height": height,
                "plate_instance_count": per_image_box_count,
                "clip_required_box_count": per_image_requires_clipping,
                "non_clippable_box_count": per_image_non_clippable,
                "max_overshoot": per_image_max_overshoot,
                "real_synthetic": "REAL",
                "is_negative": False,
                "canonical_class": "license_plate",
            }
        )

    return image_records, box_records


def print_distribution(frame: pd.DataFrame, column: str) -> None:
    print(f"\n{column}:")
    for value, count in frame[column].value_counts().sort_index().items():
        print(f"  {value}: {count}")


def main() -> None:
    provenance = load_provenance_index()

    image_records = []
    box_records = []

    for split_name in ("train", "val"):
        split_images, split_boxes = collect_split(split_name)
        image_records.extend(split_images)
        box_records.extend(split_boxes)

    image_frame = pd.DataFrame(image_records)
    box_frame = pd.DataFrame(box_records)

    image_frame = image_frame.merge(
        provenance,
        left_on="image_id",
        right_on="ImageID",
        how="left",
        validate="one_to_one",
    )

    missing_provenance = image_frame["License"].isna().sum()
    if missing_provenance:
        missing_ids = image_frame.loc[image_frame["License"].isna(), "image_id"].tolist()[:20]
        raise RuntimeError(
            f"{missing_provenance} Open Images positives are missing provenance metadata. "
            f"Examples: {missing_ids}"
        )

    duplicate_image_ids = image_frame["image_id"].duplicated().sum()
    class_ids = set(box_frame["class_id"].unique().tolist())
    total_instances = int(box_frame.shape[0])
    total_images = int(image_frame.shape[0])

    clippable_boxes = int(box_frame["requires_clipping"].sum())
    non_clippable_boxes = int(box_frame["non_clippable"].sum())

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_frame = image_frame.sort_values(["source_split", "image_id"], kind="stable").reset_index(drop=True)
    box_frame = box_frame.sort_values(["source_split", "image_id", "box_index"], kind="stable").reset_index(drop=True)

    image_frame.to_csv(IMAGE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    box_frame.to_csv(BOX_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print("\n=== OPEN IMAGES POSITIVE MANIFEST ===")
    print(f"Images: {total_images}")
    print(f"Plate instances: {total_instances}")
    print(f"Duplicate image IDs: {duplicate_image_ids}")
    print(f"Unique class IDs: {sorted(class_ids)}")
    print(f"Boxes requiring clipping: {clippable_boxes}")
    print(f"Non-clippable boxes: {non_clippable_boxes}")

    print_distribution(image_frame, "source_split")
    print_distribution(box_frame, "overshoot_bucket")

    print("\nProvenance completeness:")
    for column in REQUIRED_PROVENANCE_COLUMNS:
        missing = int(image_frame[column].isna().sum())
        print(f"  {column}: missing={missing}")

    print("\nExpected baseline checks:")
    print(f"  Images == {EXPECTED_IMAGES}: {total_images == EXPECTED_IMAGES}")
    print(f"  Instances == {EXPECTED_INSTANCES}: {total_instances == EXPECTED_INSTANCES}")
    print(f"  Class IDs == {sorted(EXPECTED_CLASS_IDS)}: {class_ids == EXPECTED_CLASS_IDS}")
    print(f"  Clippable boxes == {EXPECTED_CLIPPABLE_BOXES}: {clippable_boxes == EXPECTED_CLIPPABLE_BOXES}")
    print(f"  Non-clippable boxes == 0: {non_clippable_boxes == 0}")

    if total_images != EXPECTED_IMAGES:
        raise RuntimeError("Open Images image count does not match audited baseline.")

    if total_instances != EXPECTED_INSTANCES:
        raise RuntimeError("Open Images instance count does not match audited baseline.")

    if class_ids != EXPECTED_CLASS_IDS:
        raise RuntimeError("Unexpected class IDs found in Open Images labels.")

    if non_clippable_boxes != 0:
        raise RuntimeError("Found non-clippable Open Images boxes; review required.")

    print("\nClipping policy:")
    print("  Raw labels remain untouched: YES")
    print("  Tiny normalized overshoot is retained in raw metadata: YES")
    print("  Derived canonical annotations will clip to [0, 1]: YES")

    print("\nRESULT: OPEN IMAGES POSITIVE SOURCE MANIFEST READY")
    print(f"Image manifest: {IMAGE_MANIFEST_PATH}")
    print(f"Box manifest: {BOX_MANIFEST_PATH}")


if __name__ == "__main__":
    main()