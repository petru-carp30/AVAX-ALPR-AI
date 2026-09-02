from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "elpd"
OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "manifests" / "elpd"

FULL_MANIFEST_PATH = OUTPUT_ROOT / "elpd_filter_manifest.csv"
ACCEPTED_MANIFEST_PATH = OUTPUT_ROOT / "elpd_positive_manifest.csv"
BOX_MANIFEST_PATH = OUTPUT_ROOT / "elpd_positive_boxes.csv"

EXPECTED_TOTAL_IMAGES = 2329
EXPECTED_ACCEPTED_IMAGES = 2286
EXPECTED_ACCEPTED_INSTANCES = 2947
EXPECTED_EMPTY_IMAGES = 39

EXPECTED_CORRUPT_IMAGES = {
    "filename_prefix_00795_.png",
    "filename_prefix_00857_.png",
    "filename_prefix_01634_.png",
    "filename_prefix_01699_.png",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_pascal_voc_root() -> Path:
    candidates = []

    for annotations_dir in RAW_ROOT.rglob("Annotations"):
        pascal_root = annotations_dir.parent
        if (pascal_root / "JPEGImages").is_dir():
            candidates.append(pascal_root)

    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Pascal VOC root, found {len(candidates)}: {candidates}")

    return candidates[0]


def find_license_reference() -> str:
    for path in RAW_ROOT.rglob("*"):
        if path.is_file() and (path.name.lower().startswith("license") or path.name.lower().startswith("copying")):
            return path.relative_to(PROJECT_ROOT).as_posix()
    return ""


def parse_annotation(path: Path) -> dict:
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width", default="0"))
    height = int(root.findtext("size/height", default="0"))
    objects = []

    for element in root.findall("object"):
        class_name = element.findtext("name", default="").strip()
        box = element.find("bndbox")

        if box is None:
            raise RuntimeError(f"Missing bndbox in {path}")

        xmin = float(box.findtext("xmin", default="nan"))
        ymin = float(box.findtext("ymin", default="nan"))
        xmax = float(box.findtext("xmax", default="nan"))
        ymax = float(box.findtext("ymax", default="nan"))

        if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin or xmax > width or ymax > height:
            raise RuntimeError(f"Invalid bbox in {path}: {(xmin, ymin, xmax, ymax)}")

        objects.append({"source_class": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return {"width": width, "height": height, "objects": objects}


def main() -> None:
    pascal_root = find_pascal_voc_root()
    image_root = pascal_root / "JPEGImages"
    annotation_root = pascal_root / "Annotations"
    license_reference = find_license_reference()

    print(f"Pascal VOC root: {pascal_root}")

    image_paths = sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    annotation_paths = sorted(annotation_root.glob("*.xml"))

    images_by_stem = {path.stem: path for path in image_paths}
    annotations_by_stem = {path.stem: path for path in annotation_paths}

    missing_annotations = sorted(set(images_by_stem) - set(annotations_by_stem))
    orphan_annotations = sorted(set(annotations_by_stem) - set(images_by_stem))

    if missing_annotations:
        raise RuntimeError(f"Missing annotations: {len(missing_annotations)}. Examples: {missing_annotations[:10]}")

    if orphan_annotations:
        raise RuntimeError(f"Orphan annotations: {len(orphan_annotations)}. Examples: {orphan_annotations[:10]}")

    image_records = []
    box_records = []
    corrupt_images = set()
    empty_images = set()
    source_classes = set()
    corrupt_source_instances = 0

    for image_path in image_paths:
        annotation_path = annotations_by_stem[image_path.stem]
        annotation = parse_annotation(annotation_path)
        objects = annotation["objects"]

        for obj in objects:
            source_classes.add(obj["source_class"])

        corrupt = False
        actual_width = None
        actual_height = None

        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image.verify()
        except Exception:
            corrupt = True
            corrupt_images.add(image_path.name)
            corrupt_source_instances += len(objects)

        dimension_match = not corrupt and actual_width == annotation["width"] and actual_height == annotation["height"]

        if not corrupt and not dimension_match:
            raise RuntimeError(f"Dimension mismatch: {image_path.name}")

        if corrupt:
            status = "excluded"
            reason = "corrupt_image"
        elif not objects:
            status = "excluded"
            reason = "empty_annotation_incomplete"
            empty_images.add(image_path.name)
        else:
            status = "accepted"
            reason = ""

        image_records.append({
            "source_dataset": "elpd",
            "source_image_name": image_path.name,
            "source_annotation_name": annotation_path.name,
            "source_image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_annotation_path": annotation_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_image_sha256": sha256_file(image_path),
            "source_annotation_sha256": sha256_file(annotation_path),
            "image_width": actual_width if actual_width is not None else annotation["width"],
            "image_height": actual_height if actual_height is not None else annotation["height"],
            "plate_instance_count": len(objects),
            "filter_status": status,
            "filter_reason": reason,
            "real_synthetic": "SYNTHETIC",
            "allowed_split": "TRAIN_ONLY",
            "canonical_class": "license_plate",
            "license_name": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "license_reference": license_reference,
        })

        if status == "accepted":
            for box_index, obj in enumerate(objects, start=1):
                box_records.append({
                    "source_dataset": "elpd",
                    "source_image_name": image_path.name,
                    "box_index": box_index,
                    "source_class": obj["source_class"],
                    "canonical_class": "license_plate",
                    "xmin": obj["xmin"],
                    "ymin": obj["ymin"],
                    "xmax": obj["xmax"],
                    "ymax": obj["ymax"],
                    "image_width": annotation["width"],
                    "image_height": annotation["height"],
                    "real_synthetic": "SYNTHETIC",
                    "allowed_split": "TRAIN_ONLY",
                })

    full_manifest = pd.DataFrame(image_records)
    accepted_manifest = full_manifest[full_manifest["filter_status"] == "accepted"].copy()
    box_manifest = pd.DataFrame(box_records)

    if len(full_manifest) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_IMAGES} source images, found {len(full_manifest)}.")

    if corrupt_images != EXPECTED_CORRUPT_IMAGES:
        raise RuntimeError(f"Corrupt image set differs from audited baseline. Found: {sorted(corrupt_images)}")

    if len(empty_images) != EXPECTED_EMPTY_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_EMPTY_IMAGES} empty-label images, found {len(empty_images)}.")

    if len(accepted_manifest) != EXPECTED_ACCEPTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED_IMAGES} accepted positives, found {len(accepted_manifest)}.")

    if len(box_manifest) != EXPECTED_ACCEPTED_INSTANCES:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED_INSTANCES} accepted instances, found {len(box_manifest)}.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    full_manifest.to_csv(FULL_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    accepted_manifest.to_csv(ACCEPTED_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    box_manifest.to_csv(BOX_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print("\n=== ELPD FILTERED POSITIVE MANIFEST ===")
    print(f"Source images: {len(full_manifest)}")
    print(f"Accepted positives: {len(accepted_manifest)}")
    print(f"Accepted plate instances: {len(box_manifest)}")
    print(f"Corrupt excluded: {len(corrupt_images)}")
    print(f"Empty-label excluded: {len(empty_images)}")
    print(f"Instances inside corrupt images: {corrupt_source_instances}")
    print(f"Source classes: {sorted(source_classes)}")

    print("\nCorrupt images:")
    for filename in sorted(corrupt_images):
        print(f"  {filename}")

    print("\nPolicy:")
    print("  RealSynthetic: SYNTHETIC")
    print("  Allowed split: TRAIN_ONLY")
    print(f"  License: {LICENSE_NAME}")
    print(f"  License URL: {LICENSE_URL}")
    print(f"  Local license reference: {license_reference or '<not found>'}")
    print("  Empty labels treated as detector negatives: NO")
    print("  Raw source remains untouched: YES")

    print("\nRESULT: ELPD FILTERED POSITIVE SOURCE MANIFEST READY")
    print(f"Full filter manifest: {FULL_MANIFEST_PATH}")
    print(f"Accepted manifest: {ACCEPTED_MANIFEST_PATH}")
    print(f"Accepted boxes: {BOX_MANIFEST_PATH}")


if __name__ == "__main__":
    main()