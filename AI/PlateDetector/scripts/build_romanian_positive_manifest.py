from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "romanian_lp"
DATASET_ROOT = RAW_ROOT / "dataset"

OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "manifests" / "romanian_lp"
OUTPUT_PATH = OUTPUT_ROOT / "romanian_positive_manifest.csv"

EXPECTED_IMAGES = 534
EXPECTED_INSTANCES = 652
EXPECTED_SEQUENCES = 4

LICENSE_PATH = RAW_ROOT / "LICENSE"
README_PATH = RAW_ROOT / "README.md"

SEQUENCE_PATTERN = re.compile(r"^(?P<sequence>.+?\.mp4)#t=(?P<frame>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_source_identity(stem: str) -> tuple[str, int]:
    match = SEQUENCE_PATTERN.match(stem)

    if match is None:
        raise RuntimeError(f"Unexpected Romanian filename format: {stem}")

    return match.group("sequence"), int(match.group("frame"))


def parse_annotation(annotation_path: Path) -> dict:
    root = ET.parse(annotation_path).getroot()

    xml_width = int(root.findtext("size/width", default="0"))
    xml_height = int(root.findtext("size/height", default="0"))

    objects = []
    invalid_boxes = []

    for object_element in root.findall("object"):
        class_name = object_element.findtext("name", default="").strip()

        box = object_element.find("bndbox")

        if box is None:
            invalid_boxes.append("missing_bndbox")
            continue

        xmin = float(box.findtext("xmin", default="nan"))
        ymin = float(box.findtext("ymin", default="nan"))
        xmax = float(box.findtext("xmax", default="nan"))
        ymax = float(box.findtext("ymax", default="nan"))

        valid = (
            xmin >= 0
            and ymin >= 0
            and xmax > xmin
            and ymax > ymin
            and xmax <= xml_width
            and ymax <= xml_height
        )

        if not valid:
            invalid_boxes.append(
                f"{class_name}:{xmin},{ymin},{xmax},{ymax}"
            )

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "xml_width": xml_width,
        "xml_height": xml_height,
        "objects": objects,
        "invalid_boxes": invalid_boxes,
    }


def collect_split(split_name: str) -> list[dict]:
    image_root = DATASET_ROOT / split_name / "images"
    annotation_root = DATASET_ROOT / split_name / "annots"

    image_paths = sorted(image_root.glob("*.jpg"))
    annotation_paths = sorted(annotation_root.glob("*.xml"))

    image_stems = {path.stem for path in image_paths}
    annotation_stems = {path.stem for path in annotation_paths}

    missing_annotations = sorted(image_stems - annotation_stems)
    orphan_annotations = sorted(annotation_stems - image_stems)

    if missing_annotations:
        raise RuntimeError(
            f"{split_name}: {len(missing_annotations)} images are missing annotations."
        )

    if orphan_annotations:
        raise RuntimeError(
            f"{split_name}: {len(orphan_annotations)} orphan annotations found."
        )

    records = []

    for image_path in image_paths:
        annotation_path = annotation_root / f"{image_path.stem}.xml"

        sequence, frame_number = parse_source_identity(image_path.stem)
        annotation = parse_annotation(annotation_path)

        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image.verify()
        except Exception as exception:
            raise RuntimeError(
                f"Corrupt image: {image_path}: {exception}"
            ) from exception

        dimension_match = (
            actual_width == annotation["xml_width"]
            and actual_height == annotation["xml_height"]
        )

        object_classes = [
            obj["class_name"]
            for obj in annotation["objects"]
        ]

        records.append(
            {
                "source_dataset": "romanian_public_lp",
                "source_split": split_name,
                "source_image_name": image_path.name,
                "source_annotation_name": annotation_path.name,
                "source_sequence": sequence,
                "source_frame": frame_number,
                "source_image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_annotation_path": annotation_path.relative_to(PROJECT_ROOT).as_posix(),
                "image_sha256": sha256_file(image_path),
                "annotation_sha256": sha256_file(annotation_path),
                "image_width": actual_width,
                "image_height": actual_height,
                "xml_width": annotation["xml_width"],
                "xml_height": annotation["xml_height"],
                "dimension_match": dimension_match,
                "plate_instance_count": len(annotation["objects"]),
                "object_classes": ";".join(sorted(set(object_classes))),
                "invalid_box_count": len(annotation["invalid_boxes"]),
                "real_synthetic": "REAL",
                "is_negative": False,
                "canonical_class": "license_plate",
                "license_reference": "AI/PlateDetector/datasets/raw/romanian_lp/LICENSE",
                "readme_reference": "AI/PlateDetector/datasets/raw/romanian_lp/README.md",
            }
        )

    return records


def print_distribution(frame: pd.DataFrame, column: str) -> None:
    print(f"\n{column}:")

    for value, count in frame[column].value_counts().sort_index().items():
        print(f"  {value}: {count}")


def main() -> None:
    if not LICENSE_PATH.exists():
        raise FileNotFoundError(f"LICENSE not found: {LICENSE_PATH}")

    if not README_PATH.exists():
        raise FileNotFoundError(f"README not found: {README_PATH}")

    records = []

    for split_name in ("train", "valid"):
        records.extend(collect_split(split_name))

    frame = pd.DataFrame(records)

    duplicate_images = frame["image_sha256"].duplicated().sum()
    invalid_boxes = int(frame["invalid_box_count"].sum())
    dimension_mismatches = int((~frame["dimension_match"]).sum())

    sequence_splits = defaultdict(set)

    for row in frame.itertuples(index=False):
        sequence_splits[row.source_sequence].add(row.source_split)

    upstream_leakage_sequences = {
        sequence: splits
        for sequence, splits in sequence_splits.items()
        if len(splits) > 1
    }

    class_counter = Counter()

    for value in frame["object_classes"]:
        for class_name in str(value).split(";"):
            if class_name:
                class_counter[class_name] += 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    frame = frame.sort_values(
        ["source_sequence", "source_frame"],
        kind="stable",
    ).reset_index(drop=True)

    frame.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== ROMANIAN PUBLIC LP POSITIVE MANIFEST ===")
    print(f"Images: {len(frame)}")
    print(f"Plate instances: {frame['plate_instance_count'].sum()}")
    print(f"Unique source sequences: {frame['source_sequence'].nunique()}")
    print(f"Duplicate image hashes: {duplicate_images}")
    print(f"Invalid boxes: {invalid_boxes}")
    print(f"Dimension mismatches: {dimension_mismatches}")

    print_distribution(frame, "source_split")
    print_distribution(frame, "source_sequence")

    print("\nObject classes:")
    for class_name, count in class_counter.items():
        print(f"  {class_name}: {count}")

    print("\nUpstream sequence leakage:")
    if upstream_leakage_sequences:
        for sequence, splits in sorted(upstream_leakage_sequences.items()):
            print(f"  {sequence}: {', '.join(sorted(splits))}")
    else:
        print("  NONE")

    print("\nLicensing references:")
    print(f"  LICENSE SHA256: {sha256_file(LICENSE_PATH)}")
    print(f"  README SHA256: {sha256_file(README_PATH)}")
    print("  Raw files remain untouched: YES")

    expected_ok = (
        len(frame) == EXPECTED_IMAGES
        and int(frame["plate_instance_count"].sum()) == EXPECTED_INSTANCES
        and frame["source_sequence"].nunique() == EXPECTED_SEQUENCES
        and invalid_boxes == 0
        and dimension_mismatches == 0
    )

    print("\nExpected baseline checks:")
    print(f"  Images == {EXPECTED_IMAGES}: {len(frame) == EXPECTED_IMAGES}")
    print(
        f"  Instances == {EXPECTED_INSTANCES}: "
        f"{int(frame['plate_instance_count'].sum()) == EXPECTED_INSTANCES}"
    )
    print(
        f"  Sequences == {EXPECTED_SEQUENCES}: "
        f"{frame['source_sequence'].nunique() == EXPECTED_SEQUENCES}"
    )

    if not expected_ok:
        raise RuntimeError(
            "Romanian manifest validation does not match the audited baseline."
        )

    print("\nRESULT: ROMANIAN POSITIVE SOURCE MANIFEST READY")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()