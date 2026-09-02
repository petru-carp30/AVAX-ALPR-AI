from __future__ import annotations

import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"

ROMANIAN_MANIFEST = DATASETS_ROOT / "manifests" / "romanian_lp" / "romanian_positive_manifest.csv"
OPEN_IMAGES_MANIFEST = DATASETS_ROOT / "manifests" / "open_images_lp_kaggle" / "open_images_positive_manifest.csv"
OPEN_IMAGES_BOXES = DATASETS_ROOT / "manifests" / "open_images_lp_kaggle" / "open_images_positive_boxes.csv"
KAGGLE_MANIFEST = DATASETS_ROOT / "manifests" / "kaggle_plate_license_recognition" / "kaggle_positive_manifest.csv"
KAGGLE_BOXES = DATASETS_ROOT / "manifests" / "kaggle_plate_license_recognition" / "kaggle_positive_boxes.csv"
ELPD_MANIFEST = DATASETS_ROOT / "manifests" / "elpd" / "elpd_positive_manifest.csv"
ELPD_BOXES = DATASETS_ROOT / "manifests" / "elpd" / "elpd_positive_boxes.csv"
NEGATIVE_MANIFEST = DATASETS_ROOT / "manifests" / "open_images_negative_pool" / "open_images_negative_pool_manifest.csv"

NEGATIVE_BATCH1_ROOT = DATASETS_ROOT / "raw" / "open_images_negative_candidates" / "images"
NEGATIVE_BATCH2_ROOT = DATASETS_ROOT / "raw" / "open_images_negative_candidates_batch2" / "images"

OUTPUT_ROOT = DATASETS_ROOT / "derived" / "canonical_pool"
BUILD_ROOT = DATASETS_ROOT / "derived" / "canonical_pool_build"

EXPECTED_TOTAL_IMAGES = 9299
EXPECTED_POSITIVE_IMAGES = 8795
EXPECTED_NEGATIVE_IMAGES = 504
EXPECTED_REAL_IMAGES = 7013
EXPECTED_SYNTHETIC_IMAGES = 2286
EXPECTED_PLATE_INSTANCES = 12083

EXCLUDED_ROMANIAN = {
    "dayride_type1_001.mp4#t=558.jpg",
    "dayride_type1_001.mp4#t=809.jpg",
}


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def canonical_id(prefix: str, identity: str) -> str:
    return f"{prefix}__{stable_hash(identity)}"


def write_label(path: Path, boxes: list[dict]) -> None:
    lines = [f"0 {box['x_center']:.8f} {box['y_center']:.8f} {box['width']:.8f} {box['height']:.8f}" for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_yolo_box(box: dict, source: str) -> None:
    values = [box["x_center"], box["y_center"], box["width"], box["height"]]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise RuntimeError(f"Out-of-range normalized bbox for {source}: {box}")
    if box["width"] <= 0.0 or box["height"] <= 0.0:
        raise RuntimeError(f"Zero/negative normalized bbox for {source}: {box}")
    x1 = box["x_center"] - box["width"] / 2.0
    y1 = box["y_center"] - box["height"] / 2.0
    x2 = box["x_center"] + box["width"] / 2.0
    y2 = box["y_center"] + box["height"] / 2.0
    tolerance = 1e-7
    if x1 < -tolerance or y1 < -tolerance or x2 > 1.0 + tolerance or y2 > 1.0 + tolerance:
        raise RuntimeError(f"Normalized bbox exceeds image for {source}: {box}")


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, image_width: float, image_height: float) -> dict:
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    return {"x_center": x_center, "y_center": y_center, "width": width, "height": height}


def xywh_to_yolo(x: float, y: float, width: float, height: float, image_width: float, image_height: float) -> dict:
    return {
        "x_center": (x + width / 2.0) / image_width,
        "y_center": (y + height / 2.0) / image_height,
        "width": width / image_width,
        "height": height / image_height,
    }


def normalized_xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "x_center": (xmin + xmax) / 2.0,
        "y_center": (ymin + ymax) / 2.0,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }


def parse_pascal_boxes(annotation_path: Path, image_width: float, image_height: float) -> list[dict]:
    root = ET.parse(annotation_path).getroot()
    boxes = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise RuntimeError(f"Missing bndbox: {annotation_path}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        boxes.append(xyxy_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height))

    return boxes


def copy_sample(source_path: Path, canonical_name: str, boxes: list[dict], images_root: Path, labels_root: Path) -> tuple[Path, Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source image missing: {source_path}")

    extension = source_path.suffix.lower()
    destination_image = images_root / f"{canonical_name}{extension}"
    destination_label = labels_root / f"{canonical_name}.txt"

    shutil.copy2(source_path, destination_image)
    write_label(destination_label, boxes)

    return destination_image, destination_label


def append_box_records(box_records: list[dict], canonical_name: str, source_dataset: str, boxes: list[dict]) -> None:
    for index, box in enumerate(boxes, start=1):
        validate_yolo_box(box, f"{source_dataset}:{canonical_name}:{index}")
        box_records.append({
            "canonical_id": canonical_name,
            "box_index": index,
            "class_id": 0,
            "canonical_class": "license_plate",
            "x_center": box["x_center"],
            "y_center": box["y_center"],
            "width": box["width"],
            "height": box["height"],
            "source_dataset": source_dataset,
        })


def process_romanian(sample_records: list[dict], box_records: list[dict], images_root: Path, labels_root: Path) -> None:
    frame = pd.read_csv(ROMANIAN_MANIFEST)

    frame = frame[~frame["source_image_name"].isin(EXCLUDED_ROMANIAN)].copy()

    if len(frame) != 532:
        raise RuntimeError(f"Expected 532 Romanian positives, found {len(frame)}.")

    for row in frame.itertuples(index=False):
        source_path = PROJECT_ROOT / row.source_image_path
        annotation_path = PROJECT_ROOT / row.source_annotation_path
        canonical_name = canonical_id("romanian", row.source_image_path)
        boxes = parse_pascal_boxes(annotation_path, row.image_width, row.image_height)

        for box in boxes:
            validate_yolo_box(box, row.source_image_name)

        destination_image, destination_label = copy_sample(source_path, canonical_name, boxes, images_root, labels_root)
        append_box_records(box_records, canonical_name, "romanian_public_lp", boxes)

        sample_records.append({
            "canonical_id": canonical_name,
            "canonical_image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_label_path": destination_label.relative_to(PROJECT_ROOT).as_posix(),
            "source_dataset": "romanian_public_lp",
            "source_image": row.source_image_name,
            "source_path": row.source_image_path,
            "source_group": row.source_sequence,
            "provenance_id": f"{row.source_sequence}#frame={row.source_frame}",
            "real_synthetic": "REAL",
            "allowed_split": "TRAIN_VAL_TEST",
            "is_negative": False,
            "plate_instance_count": len(boxes),
            "license_reference": row.license_reference,
        })


def process_open_images(sample_records: list[dict], box_records: list[dict], images_root: Path, labels_root: Path) -> None:
    images = pd.read_csv(OPEN_IMAGES_MANIFEST, dtype={"image_id": str})
    boxes_frame = pd.read_csv(OPEN_IMAGES_BOXES, dtype={"image_id": str})
    grouped_boxes = {image_id: group for image_id, group in boxes_frame.groupby("image_id")}

    if len(images) != 5368:
        raise RuntimeError(f"Expected 5368 Open Images positives, found {len(images)}.")

    for row in images.itertuples(index=False):
        source_path = PROJECT_ROOT / row.source_image_path
        canonical_name = f"openimages__{row.image_id}"
        source_boxes = grouped_boxes.get(row.image_id)

        if source_boxes is None:
            raise RuntimeError(f"No boxes found for Open Images positive: {row.image_id}")

        boxes = [normalized_xyxy_to_yolo(float(box.x_min_clipped), float(box.y_min_clipped), float(box.x_max_clipped), float(box.y_max_clipped)) for box in source_boxes.itertuples(index=False)]

        for box in boxes:
            validate_yolo_box(box, row.image_id)

        destination_image, destination_label = copy_sample(source_path, canonical_name, boxes, images_root, labels_root)
        append_box_records(box_records, canonical_name, "open_images_v7", boxes)

        sample_records.append({
            "canonical_id": canonical_name,
            "canonical_image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_label_path": destination_label.relative_to(PROJECT_ROOT).as_posix(),
            "source_dataset": "open_images_v7",
            "source_image": row.source_image_name,
            "source_path": row.source_image_path,
            "source_group": row.image_id,
            "provenance_id": row.image_id,
            "real_synthetic": "REAL",
            "allowed_split": "TRAIN_VAL_TEST",
            "is_negative": False,
            "plate_instance_count": len(boxes),
            "license_reference": row.License,
        })


def process_kaggle(sample_records: list[dict], box_records: list[dict], images_root: Path, labels_root: Path) -> None:
    images = pd.read_csv(KAGGLE_MANIFEST)
    boxes_frame = pd.read_csv(KAGGLE_BOXES)
    grouped_boxes = {(group, filename): rows for (group, filename), rows in boxes_frame.groupby(["source_group", "filename"])}

    if len(images) != 609:
        raise RuntimeError(f"Expected 609 Kaggle positives, found {len(images)}.")

    for row in images.itertuples(index=False):
        source_path = PROJECT_ROOT / row.source_image_path
        canonical_name = canonical_id("kaggle", f"{row.source_group}|{row.filename}")
        source_boxes = grouped_boxes.get((row.source_group, row.filename))

        if source_boxes is None:
            raise RuntimeError(f"No boxes found for Kaggle positive: {row.filename}")

        boxes = [xywh_to_yolo(float(box.x), float(box.y), float(box.width), float(box.height), float(box.image_width), float(box.image_height)) for box in source_boxes.itertuples(index=False)]

        for box in boxes:
            validate_yolo_box(box, row.filename)

        destination_image, destination_label = copy_sample(source_path, canonical_name, boxes, images_root, labels_root)
        append_box_records(box_records, canonical_name, "kaggle_plate_license_recognition", boxes)

        sample_records.append({
            "canonical_id": canonical_name,
            "canonical_image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_label_path": destination_label.relative_to(PROJECT_ROOT).as_posix(),
            "source_dataset": "kaggle_plate_license_recognition",
            "source_image": row.filename,
            "source_path": row.source_image_path,
            "source_group": row.source_group,
            "provenance_id": f"{row.source_group}|{row.filename}",
            "real_synthetic": "REAL",
            "allowed_split": "TRAIN_ONLY",
            "is_negative": False,
            "plate_instance_count": len(boxes),
            "license_reference": "Kaggle source metadata retained in source manifest",
        })


def process_elpd(sample_records: list[dict], box_records: list[dict], images_root: Path, labels_root: Path) -> None:
    images = pd.read_csv(ELPD_MANIFEST)
    boxes_frame = pd.read_csv(ELPD_BOXES)
    grouped_boxes = {name: rows for name, rows in boxes_frame.groupby("source_image_name")}

    if len(images) != 2286:
        raise RuntimeError(f"Expected 2286 ELPD positives, found {len(images)}.")

    for row in images.itertuples(index=False):
        source_path = PROJECT_ROOT / row.source_image_path
        canonical_name = canonical_id("elpd", row.source_image_name)
        source_boxes = grouped_boxes.get(row.source_image_name)

        if source_boxes is None:
            raise RuntimeError(f"No boxes found for ELPD positive: {row.source_image_name}")

        boxes = [xyxy_to_yolo(float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax), float(box.image_width), float(box.image_height)) for box in source_boxes.itertuples(index=False)]

        for box in boxes:
            validate_yolo_box(box, row.source_image_name)

        destination_image, destination_label = copy_sample(source_path, canonical_name, boxes, images_root, labels_root)
        append_box_records(box_records, canonical_name, "elpd", boxes)

        sample_records.append({
            "canonical_id": canonical_name,
            "canonical_image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_label_path": destination_label.relative_to(PROJECT_ROOT).as_posix(),
            "source_dataset": "elpd",
            "source_image": row.source_image_name,
            "source_path": row.source_image_path,
            "source_group": row.source_image_name,
            "provenance_id": row.source_image_name,
            "real_synthetic": "SYNTHETIC",
            "allowed_split": "TRAIN_ONLY",
            "is_negative": False,
            "plate_instance_count": len(boxes),
            "license_reference": row.license_reference,
        })


def process_negatives(sample_records: list[dict], images_root: Path, labels_root: Path) -> None:
    frame = pd.read_csv(NEGATIVE_MANIFEST, dtype={"ImageID": str})

    if len(frame) != EXPECTED_NEGATIVE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_NEGATIVE_IMAGES} Open Images negatives, found {len(frame)}.")

    for row in frame.itertuples(index=False):
        batch = int(row.NegativeBatch)
        source_root = NEGATIVE_BATCH1_ROOT if batch == 1 else NEGATIVE_BATCH2_ROOT if batch == 2 else None

        if source_root is None:
            raise RuntimeError(f"Unexpected negative batch: {batch}")

        source_path = source_root / f"{row.ImageID}.jpg"
        canonical_name = f"openimagesneg__{row.ImageID}"
        destination_image, destination_label = copy_sample(source_path, canonical_name, [], images_root, labels_root)

        sample_records.append({
            "canonical_id": canonical_name,
            "canonical_image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_label_path": destination_label.relative_to(PROJECT_ROOT).as_posix(),
            "source_dataset": "open_images_v7_negative_pool",
            "source_image": source_path.name,
            "source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_group": row.ImageID,
            "provenance_id": row.ImageID,
            "real_synthetic": "REAL",
            "allowed_split": "TRAIN_VAL_TEST",
            "is_negative": True,
            "plate_instance_count": 0,
            "license_reference": row.License,
        })


def main() -> None:
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)

    images_root = BUILD_ROOT / "images"
    labels_root = BUILD_ROOT / "labels"
    metadata_root = BUILD_ROOT / "metadata"

    images_root.mkdir(parents=True)
    labels_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)

    sample_records = []
    box_records = []

    print("Processing Romanian Public LP...")
    process_romanian(sample_records, box_records, images_root, labels_root)

    print("Processing Open Images positives...")
    process_open_images(sample_records, box_records, images_root, labels_root)

    print("Processing filtered Kaggle positives...")
    process_kaggle(sample_records, box_records, images_root, labels_root)

    print("Processing ELPD positives...")
    process_elpd(sample_records, box_records, images_root, labels_root)

    print("Processing Open Images negatives...")
    process_negatives(sample_records, images_root, labels_root)

    samples = pd.DataFrame(sample_records)
    boxes = pd.DataFrame(box_records)

    samples["canonical_image_path"] = samples["canonical_image_path"].str.replace("canonical_pool_build","canonical_pool", regex=False)
    samples["canonical_label_path"] = samples["canonical_label_path"].str.replace("canonical_pool_build","canonical_pool", regex=False)

    duplicate_ids = int(samples["canonical_id"].duplicated().sum())
    positive_count = int((~samples["is_negative"]).sum())
    negative_count = int(samples["is_negative"].sum())
    real_count = int((samples["real_synthetic"] == "REAL").sum())
    synthetic_count = int((samples["real_synthetic"] == "SYNTHETIC").sum())
    total_instances = int(samples["plate_instance_count"].sum())

    if len(samples) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_IMAGES} canonical images, found {len(samples)}.")
    if positive_count != EXPECTED_POSITIVE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_POSITIVE_IMAGES} positives, found {positive_count}.")
    if negative_count != EXPECTED_NEGATIVE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_NEGATIVE_IMAGES} negatives, found {negative_count}.")
    if real_count != EXPECTED_REAL_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_REAL_IMAGES} REAL images, found {real_count}.")
    if synthetic_count != EXPECTED_SYNTHETIC_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_SYNTHETIC_IMAGES} SYNTHETIC images, found {synthetic_count}.")
    if total_instances != EXPECTED_PLATE_INSTANCES or len(boxes) != EXPECTED_PLATE_INSTANCES:
        raise RuntimeError(f"Expected {EXPECTED_PLATE_INSTANCES} plate instances, found samples={total_instances}, boxes={len(boxes)}.")
    if duplicate_ids:
        raise RuntimeError(f"Duplicate canonical IDs found: {duplicate_ids}.")

    image_files = list(images_root.iterdir())
    label_files = list(labels_root.glob("*.txt"))

    if len(image_files) != EXPECTED_TOTAL_IMAGES or len(label_files) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(f"Canonical file count mismatch: images={len(image_files)}, labels={len(label_files)}.")

    samples.to_csv(metadata_root / "canonical_samples.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(metadata_root / "canonical_boxes.csv", index=False, encoding="utf-8-sig")
    (metadata_root / "classes.txt").write_text("0 license_plate\n", encoding="utf-8")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    BUILD_ROOT.rename(OUTPUT_ROOT)

    print("\n=== AVAX CANONICAL DETECTOR POOL ===")
    print(f"Total images: {len(samples)}")
    print(f"Positive images: {positive_count}")
    print(f"Negative images: {negative_count}")
    print(f"Plate instances: {total_instances}")
    print(f"REAL images: {real_count}")
    print(f"SYNTHETIC images: {synthetic_count}")
    print(f"Duplicate canonical IDs: {duplicate_ids}")

    print("\nSource composition:")
    for source, count in samples["source_dataset"].value_counts().items():
        source_instances = int(samples.loc[samples["source_dataset"] == source, "plate_instance_count"].sum())
        print(f"  {source}: {count} images / {source_instances} plate instances")

    print("\nCanonical format:")
    print("  Class: 0 = license_plate")
    print("  Labels: YOLO normalized XYWH")
    print("  Negative labels: empty TXT")
    print("  Images copied without re-encoding: YES")
    print("  Raw sources modified: NO")
    print("  AVAX split assigned: NO")

    print(f"\nCanonical pool: {OUTPUT_ROOT}")
    print("RESULT: CANONICAL DETECTOR POOL READY FOR GROUP / NEAR-DUPLICATE ANALYSIS")


if __name__ == "__main__":
    main()