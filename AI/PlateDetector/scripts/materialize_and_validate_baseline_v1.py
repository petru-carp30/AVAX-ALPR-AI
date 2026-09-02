from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"
CANONICAL_ROOT = DATASETS_ROOT / "derived" / "canonical_pool"
BASELINE_ROOT = DATASETS_ROOT / "derived" / "baseline_v1"
MANIFEST_PATH = BASELINE_ROOT / "metadata" / "baseline_split_manifest.csv"
FINGERPRINTS_PATH = CANONICAL_ROOT / "metadata" / "similarity" / "canonical_image_fingerprints.csv"
NEAR_CANDIDATES_PATH = CANONICAL_ROOT / "metadata" / "similarity" / "near_duplicate_candidates.csv"
STAGING_ROOT = BASELINE_ROOT.with_name("baseline_v1.__staging__")
IMAGES_BACKUP = BASELINE_ROOT.with_name("baseline_v1.__images_backup__")
LABELS_BACKUP = BASELINE_ROOT.with_name("baseline_v1.__labels_backup__")

EXPECTED_TOTAL = 8916
EXPECTED_SPLITS = {"train": 7622, "val": 625, "test": 669}
EXPECTED_POSITIVES = 8412
EXPECTED_NEGATIVES = 504
EXPECTED_INSTANCES = 11580
EXPECTED_TRAIN_REAL = 5717
EXPECTED_TRAIN_SYNTHETIC = 1905
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_COLUMNS = {
    "canonical_id", "canonical_image_path", "canonical_label_path", "split_status",
    "avax_split", "source_dataset", "source_group", "real_synthetic",
    "allowed_split", "is_negative", "plate_instance_count",
}


def as_bool(value) -> bool:
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"Invalid boolean value: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(path: Path, expected_instances: int, is_negative: bool) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if is_negative:
        if expected_instances != 0 or lines:
            raise RuntimeError(f"Invalid canonical negative label: {path}")
        return

    if expected_instances <= 0 or len(lines) != expected_instances:
        raise RuntimeError(
            f"Positive label count mismatch: {path}, expected={expected_instances}, actual={len(lines)}"
        )

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"Invalid YOLO format: {path}:{line_number}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise RuntimeError(f"Invalid YOLO numeric value: {path}:{line_number}") from exc
        if class_id != 0:
            raise RuntimeError(f"Unexpected class ID {class_id}: {path}:{line_number}")
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            raise RuntimeError(f"Non-finite bbox: {path}:{line_number}")
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise RuntimeError(f"Out-of-range bbox: {path}:{line_number}")


def load_manifest() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing authoritative manifest: {MANIFEST_PATH}")

    frame = pd.read_csv(MANIFEST_PATH)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

    status = frame["split_status"].astype(str).str.strip().str.lower()
    unknown = set(status) - {"included", "excluded"}
    if unknown:
        raise RuntimeError(f"Unexpected split_status values: {sorted(unknown)}")

    included = frame[status == "included"].copy()
    excluded = frame[status == "excluded"].copy()
    if len(included) != EXPECTED_TOTAL:
        raise RuntimeError(f"Included count mismatch: expected={EXPECTED_TOTAL}, actual={len(included)}")

    included["canonical_id"] = included["canonical_id"].astype(str).str.strip()
    included["avax_split"] = included["avax_split"].astype(str).str.strip().str.lower()
    included["real_synthetic"] = included["real_synthetic"].astype(str).str.strip().str.upper()
    included["allowed_split"] = included["allowed_split"].astype(str).str.strip().str.upper()
    included["plate_instance_count"] = pd.to_numeric(included["plate_instance_count"], errors="raise")

    if included["canonical_id"].duplicated().any():
        raise RuntimeError("Duplicate canonical_id found")
    if not (included["plate_instance_count"] % 1 == 0).all():
        raise RuntimeError("Non-integer plate_instance_count found")
    included["plate_instance_count"] = included["plate_instance_count"].astype(int)

    for split, expected in EXPECTED_SPLITS.items():
        actual = int((included["avax_split"] == split).sum())
        if actual != expected:
            raise RuntimeError(f"{split.upper()} count mismatch: expected={expected}, actual={actual}")

    negative = included["is_negative"].map(as_bool)
    positives = int((~negative).sum())
    negatives = int(negative.sum())
    instances = int(included["plate_instance_count"].sum())
    if (positives, negatives, instances) != (EXPECTED_POSITIVES, EXPECTED_NEGATIVES, EXPECTED_INSTANCES):
        raise RuntimeError(
            f"Manifest totals mismatch: positives={positives}, negatives={negatives}, instances={instances}"
        )
    if (negative & (included["plate_instance_count"] != 0)).any():
        raise RuntimeError("Negative sample with non-zero plate_instance_count")
    if ((~negative) & (included["plate_instance_count"] <= 0)).any():
        raise RuntimeError("Positive sample with non-positive plate_instance_count")

    train = included[included["avax_split"] == "train"]
    val = included[included["avax_split"] == "val"]
    test = included[included["avax_split"] == "test"]
    train_real = int((train["real_synthetic"] == "REAL").sum())
    train_synthetic = int((train["real_synthetic"] == "SYNTHETIC").sum())

    if (train_real, train_synthetic) != (EXPECTED_TRAIN_REAL, EXPECTED_TRAIN_SYNTHETIC):
        raise RuntimeError(f"TRAIN composition mismatch: real={train_real}, synthetic={train_synthetic}")
    if train_synthetic / len(train) > 0.25:
        raise RuntimeError("TRAIN synthetic ratio exceeds 25%")
    if (val["real_synthetic"] != "REAL").any() or (test["real_synthetic"] != "REAL").any():
        raise RuntimeError("VAL or TEST contains synthetic samples")
    if not included[
        (included["allowed_split"] == "TRAIN_ONLY") & (included["avax_split"] != "train")
    ].empty:
        raise RuntimeError("TRAIN_ONLY sample found outside TRAIN")

    groups = included["source_dataset"].astype(str) + "::" + included["source_group"].astype(str)
    leakage = included.assign(_group=groups).groupby("_group")["avax_split"].nunique()
    if (leakage > 1).any():
        raise RuntimeError(f"Source-group leakage detected: {int((leakage > 1).sum())}")

    if not FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(f"Missing fingerprint metadata: {FINGERPRINTS_PATH}")
    fingerprints = pd.read_csv(FINGERPRINTS_PATH)[["canonical_id", "pixel_sha256"]]
    if fingerprints["canonical_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate canonical_id in fingerprint metadata")

    merged = included[["canonical_id", "avax_split"]].merge(
        fingerprints, on="canonical_id", how="left", validate="one_to_one"
    )
    if merged["pixel_sha256"].isna().any():
        raise RuntimeError("Missing fingerprint for included canonical sample")
    pixel_leakage = merged.groupby("pixel_sha256")["avax_split"].nunique()
    if (pixel_leakage > 1).any():
        raise RuntimeError(f"Exact-pixel leakage detected: {int((pixel_leakage > 1).sum())}")

    if not NEAR_CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing near-duplicate metadata: {NEAR_CANDIDATES_PATH}")
    try:
        near_candidates = len(pd.read_csv(NEAR_CANDIDATES_PATH))
    except pd.errors.EmptyDataError:
        near_candidates = 0
    if near_candidates:
        raise RuntimeError(f"Unresolved near-duplicate candidates: {near_candidates}")

    return included, excluded, {
        "positives": positives,
        "negatives": negatives,
        "instances": instances,
        "train_real": train_real,
        "train_synthetic": train_synthetic,
        "group_leakage": 0,
        "pixel_leakage": 0,
        "near_candidates": 0,
    }


def canonical_source(value) -> Path:
    path = (PROJECT_ROOT / str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical source missing: {path}")
    if not path.is_relative_to(CANONICAL_ROOT.resolve()):
        raise RuntimeError(f"Manifest source is outside canonical_pool: {path}")
    return path


def preflight_sources(included: pd.DataFrame) -> dict[str, tuple[str, Path, Path, bool, int]]:
    samples = {}

    for row in included.itertuples(index=False):
        canonical_id = str(row.canonical_id)
        image = canonical_source(row.canonical_image_path)
        label = canonical_source(row.canonical_label_path)

        if image.stem != canonical_id or label.stem != canonical_id:
            raise RuntimeError(
                f"Canonical filename mismatch: id={canonical_id}, image={image.name}, label={label.name}"
            )
        if image.suffix.lower() not in IMAGE_EXTENSIONS or label.suffix.lower() != ".txt":
            raise RuntimeError(f"Unsupported canonical file type for {canonical_id}")

        is_negative = as_bool(row.is_negative)
        instances = int(row.plate_instance_count)
        validate_label(label, instances, is_negative)

        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as exc:
            raise RuntimeError(f"Unreadable canonical image: {image}") from exc

        samples[canonical_id] = (str(row.avax_split), image, label, is_negative, instances)

    if len(samples) != EXPECTED_TOTAL:
        raise RuntimeError(f"Source preflight count mismatch: {len(samples)}")
    return samples


def validate_root(
    root: Path,
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    samples: dict[str, tuple[str, Path, Path, bool, int]],
    verify_bytes: bool = True,
) -> dict:
    total_images = total_labels = total_instances = total_negatives = 0
    all_ids = set()

    for split, expected_count in EXPECTED_SPLITS.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"Missing {split} directories under {root}")

        image_entries = list(image_dir.iterdir())
        label_entries = list(label_dir.iterdir())
        if any(p.is_dir() for p in image_entries) or any(p.is_dir() for p in label_entries):
            raise RuntimeError(f"{split.upper()} contains unexpected nested directories")
        if any(p.suffix.lower() not in IMAGE_EXTENSIONS for p in image_entries if p.is_file()):
            raise RuntimeError(f"{split.upper()} images directory contains unexpected file types")
        if any(p.suffix.lower() != ".txt" for p in label_entries if p.is_file()):
            raise RuntimeError(f"{split.upper()} labels directory contains unexpected file types")

        images = [p for p in image_entries if p.is_file()]
        labels = [p for p in label_entries if p.is_file()]
        if len(images) != expected_count or len(labels) != expected_count:
            raise RuntimeError(
                f"{split.upper()} physical count mismatch: "
                f"images={len(images)}, labels={len(labels)}, expected={expected_count}"
            )

        expected_ids = set(included.loc[included["avax_split"] == split, "canonical_id"].astype(str))
        image_ids = {p.stem for p in images}
        label_ids = {p.stem for p in labels}
        if image_ids != expected_ids:
            raise RuntimeError(
                f"{split.upper()} image identities mismatch: "
                f"missing={sorted(expected_ids - image_ids)[:5]}, extra={sorted(image_ids - expected_ids)[:5]}"
            )
        if label_ids != expected_ids:
            raise RuntimeError(
                f"{split.upper()} label identities mismatch: "
                f"missing={sorted(expected_ids - label_ids)[:5]}, extra={sorted(label_ids - expected_ids)[:5]}"
            )

        for canonical_id in expected_ids:
            _, source_image, source_label, is_negative, instances = samples[canonical_id]
            image = image_dir / source_image.name
            label = label_dir / source_label.name
            validate_label(label, instances, is_negative)
            if verify_bytes and (
                sha256(image) != sha256(source_image) or sha256(label) != sha256(source_label)
            ):
                raise RuntimeError(f"Materialized bytes differ from canonical source: {canonical_id}")
            total_instances += instances
            total_negatives += int(is_negative)

        total_images += len(images)
        total_labels += len(labels)
        all_ids.update(image_ids)

    if (total_images, total_labels, total_instances, total_negatives) != (
        EXPECTED_TOTAL, EXPECTED_TOTAL, EXPECTED_INSTANCES, EXPECTED_NEGATIVES
    ):
        raise RuntimeError(
            f"Physical totals mismatch: images={total_images}, labels={total_labels}, "
            f"instances={total_instances}, negatives={total_negatives}"
        )

    excluded_ids = set(excluded["canonical_id"].astype(str))
    if excluded_ids & all_ids:
        raise RuntimeError(f"Excluded samples materialized: {sorted(excluded_ids & all_ids)[:5]}")

    return {
        "images": total_images,
        "labels": total_labels,
        "instances": total_instances,
        "negatives": total_negatives,
    }


def build_staging(samples: dict[str, tuple[str, Path, Path, bool, int]]) -> None:
    if STAGING_ROOT.exists() or IMAGES_BACKUP.exists() or LABELS_BACKUP.exists():
        raise RuntimeError("Stale staging/backup exists; inspect and remove it before retrying")

    for split in EXPECTED_SPLITS:
        (STAGING_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (STAGING_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, canonical_id in enumerate(sorted(samples), start=1):
        split, image, label, _, _ = samples[canonical_id]
        shutil.copy2(image, STAGING_ROOT / "images" / split / image.name)
        shutil.copy2(label, STAGING_ROOT / "labels" / split / label.name)
        if index % 500 == 0 or index == len(samples):
            print(f"Materialized staging {index}/{len(samples)}")


def replace_materialization(
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    samples: dict[str, tuple[str, Path, Path, bool, int]],
) -> dict:
    images = BASELINE_ROOT / "images"
    labels = BASELINE_ROOT / "labels"
    staged_images = STAGING_ROOT / "images"
    staged_labels = STAGING_ROOT / "labels"

    os.replace(images, IMAGES_BACKUP)
    try:
        os.replace(labels, LABELS_BACKUP)
        os.replace(staged_images, images)
        os.replace(staged_labels, labels)
    except Exception:
        shutil.rmtree(images, ignore_errors=True)
        shutil.rmtree(labels, ignore_errors=True)
        if IMAGES_BACKUP.exists():
            os.replace(IMAGES_BACKUP, images)
        if LABELS_BACKUP.exists():
            os.replace(LABELS_BACKUP, labels)
        raise

    try:
        result = validate_root(BASELINE_ROOT, included, excluded, samples)
    except Exception:
        shutil.rmtree(images, ignore_errors=True)
        shutil.rmtree(labels, ignore_errors=True)
        os.replace(IMAGES_BACKUP, images)
        os.replace(LABELS_BACKUP, labels)
        raise

    shutil.rmtree(IMAGES_BACKUP)
    shutil.rmtree(LABELS_BACKUP)
    shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore baseline_v1 physical files from the accepted canonical manifest."
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    print("=== AVAX baseline_v1 PHYSICAL RESTORATION ===")
    print(f"Authoritative manifest: {MANIFEST_PATH}")
    print(f"Canonical source root: {CANONICAL_ROOT}")
    print(f"Physical baseline root: {BASELINE_ROOT}")

    included, excluded, summary = load_manifest()
    print(
        f"\nManifest: PASS | images={EXPECTED_TOTAL} | positives={summary['positives']} | "
        f"negatives={summary['negatives']} | instances={summary['instances']} | "
        f"TRAIN/VAL/TEST={EXPECTED_SPLITS['train']}/{EXPECTED_SPLITS['val']}/{EXPECTED_SPLITS['test']}"
    )

    samples = preflight_sources(included)
    print(f"Canonical source preflight: PASS ({len(samples)}/{EXPECTED_TOTAL})")

    if args.preflight_only:
        print("No files written.")
        print("RESULT: PREFLIGHT PASS")
        return

    build_staging(samples)
    staging = validate_root(STAGING_ROOT, included, excluded, samples)
    print(
        f"Staging: PASS | images={staging['images']} | labels={staging['labels']} | "
        f"instances={staging['instances']} | negatives={staging['negatives']}"
    )

    final = replace_materialization(included, excluded, samples)
    print("\n=== RESTORATION COMPLETE ===")
    print(f"Images: {final['images']} | Labels: {final['labels']}")
    print(
        f"TRAIN: {EXPECTED_SPLITS['train']} | VAL: {EXPECTED_SPLITS['val']} | TEST: {EXPECTED_SPLITS['test']}"
    )
    print(f"Plate instances: {final['instances']} | Negatives: {final['negatives']}")
    print("Canonical IDs/splits: PRESERVED")
    print("Manifest/metadata/provenance: UNCHANGED")
    print("Raw datasets modified: NO")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
