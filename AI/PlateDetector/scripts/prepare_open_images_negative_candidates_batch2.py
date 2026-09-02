from __future__ import annotations

import random
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]

POSITIVE_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"

BATCH1_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_negative_candidates"
BATCH1_METADATA_PATH = BATCH1_ROOT / "negative_candidates_metadata.csv"
SOURCE_METADATA_ROOT = BATCH1_ROOT / "metadata"

BATCH2_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_negative_candidates_batch2"
IMAGE_ROOT = BATCH2_ROOT / "images"

RANDOM_SEED = 20260825

CATEGORY_CONFIG = {
    "road_gate_scene": {
        "classes": ["Traffic sign", "Traffic light", "Stop sign", "Parking meter"],
        "quota": 60,
    },
    "heavy_equipment_candidate": {
        "classes": ["Truck"],
        "quota": 50,
    },
    "vehicle_no_visible_plate_candidate": {
        "classes": ["Car", "Truck", "Van", "Bus", "Motorcycle"],
        "quota": 40,
    },
}


def collect_positive_ids() -> set[str]:
    image_ids: set[str] = set()

    for split in ("train", "val"):
        image_dir = POSITIVE_ROOT / "images" / split

        if not image_dir.exists():
            continue

        for path in image_dir.iterdir():
            if path.is_file():
                image_ids.add(path.stem.lower())

    return image_ids


def collect_batch1_ids() -> set[str]:
    if not BATCH1_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Batch 1 metadata not found: {BATCH1_METADATA_PATH}"
        )

    frame = pd.read_csv(
        BATCH1_METADATA_PATH,
        dtype={"ImageID": str},
    )

    return set(frame["ImageID"].str.lower())


def load_class_map() -> tuple[dict[str, str], dict[str, str]]:
    path = SOURCE_METADATA_ROOT / "oidv7-class-descriptions-boxable.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Open Images class metadata not found: {path}"
        )

    frame = pd.read_csv(
        path,
        header=None,
        names=["LabelName", "ClassName"],
        dtype=str,
    )

    name_to_mid = dict(zip(frame["ClassName"], frame["LabelName"]))
    mid_to_name = dict(zip(frame["LabelName"], frame["ClassName"]))

    return name_to_mid, mid_to_name


def load_boxes(split: str) -> pd.DataFrame:
    path = SOURCE_METADATA_ROOT / f"{split}-annotations-bbox.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Open Images bbox metadata not found: {path}"
        )

    frame = pd.read_csv(
        path,
        usecols=["ImageID", "LabelName"],
        dtype={"ImageID": str, "LabelName": str},
    )

    frame["ImageID"] = frame["ImageID"].str.lower()

    return frame


def load_image_metadata(split: str) -> pd.DataFrame:
    path = SOURCE_METADATA_ROOT / f"{split}-images-with-rotation.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Open Images image metadata not found: {path}"
        )

    frame = pd.read_csv(path, dtype=str)
    frame["ImageID"] = frame["ImageID"].str.lower()

    return frame


def build_candidate_pool(
    split: str,
    boxes: pd.DataFrame,
    name_to_mid: dict[str, str],
    mid_to_name: dict[str, str],
    excluded_ids: set[str],
) -> list[dict[str, str]]:
    plate_mid = name_to_mid.get("Vehicle registration plate")

    if plate_mid is None:
        raise RuntimeError(
            "Vehicle registration plate class was not found."
        )

    plate_image_ids = set(
        boxes.loc[
            boxes["LabelName"] == plate_mid,
            "ImageID",
        ]
    )

    records: list[dict[str, str]] = []

    for category, config in CATEGORY_CONFIG.items():
        class_mids = []

        for class_name in config["classes"]:
            class_mid = name_to_mid.get(class_name)

            if class_mid is None:
                print(f"Class not available, skipping: {class_name}")
                continue

            class_mids.append(class_mid)

        relevant = boxes[
            boxes["LabelName"].isin(class_mids)
            & ~boxes["ImageID"].isin(plate_image_ids)
            & ~boxes["ImageID"].isin(excluded_ids)
        ].copy()

        relevant["ClassName"] = relevant["LabelName"].map(mid_to_name)

        grouped = relevant.groupby("ImageID")["ClassName"].apply(
            lambda values: ";".join(sorted(set(values)))
        )

        for image_id, matched_classes in grouped.items():
            records.append(
                {
                    "ImageID": image_id,
                    "SourceSplit": split,
                    "CandidateCategory": category,
                    "MatchedClasses": matched_classes,
                }
            )

    return records


def sample_candidates(records: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)

    if frame.empty:
        raise RuntimeError("No Batch 2 candidates were discovered.")

    random_generator = random.Random(RANDOM_SEED)

    selected_rows = []
    selected_ids: set[str] = set()

    for category, config in CATEGORY_CONFIG.items():
        category_frame = frame[
            frame["CandidateCategory"] == category
        ].copy()

        category_frame = category_frame[
            ~category_frame["ImageID"].isin(selected_ids)
        ]

        indices = list(category_frame.index)
        random_generator.shuffle(indices)

        quota = config["quota"]
        selected_indices = indices[:quota]

        if len(selected_indices) < quota:
            print(
                f"WARNING: {category} requested {quota}, "
                f"but only {len(selected_indices)} were available."
            )

        for index in selected_indices:
            row = category_frame.loc[index]
            selected_rows.append(row)
            selected_ids.add(str(row["ImageID"]))

    result = pd.DataFrame(selected_rows)

    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return result


def attach_provenance(candidates: pd.DataFrame) -> pd.DataFrame:
    result_frames = []

    for split in ("validation", "test"):
        split_candidates = candidates[
            candidates["SourceSplit"] == split
        ].copy()

        if split_candidates.empty:
            continue

        metadata = load_image_metadata(split)

        merged = split_candidates.merge(
            metadata,
            on="ImageID",
            how="left",
            validate="one_to_one",
        )

        result_frames.append(merged)

    result = pd.concat(
        result_frames,
        ignore_index=True,
    )

    if result["OriginalURL"].isna().any():
        raise RuntimeError(
            "At least one Batch 2 candidate is missing provenance metadata."
        )

    result["SourceDataset"] = "open_images_v7"
    result["RealSynthetic"] = "REAL"
    result["IsNegativeCandidate"] = True
    result["NegativeBatch"] = 2

    return result


def download_images(candidates: pd.DataFrame) -> list[str]:
    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    bucket = boto3.resource(
        "s3",
        config=Config(signature_version=UNSIGNED),
    ).Bucket("open-images-dataset")

    failures: list[str] = []

    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc="Downloading Batch 2 candidates",
    ):
        image_id = str(row.ImageID)
        split = str(row.SourceSplit)

        target = IMAGE_ROOT / f"{image_id}.jpg"

        if target.exists():
            continue

        try:
            bucket.download_file(
                f"{split}/{image_id}.jpg",
                str(target),
            )
        except Exception as exception:
            print(
                f"\nDownload failed: "
                f"{split}/{image_id}: {exception}"
            )
            failures.append(image_id)

    return failures


def validate_images(candidates: pd.DataFrame) -> list[str]:
    invalid_ids = []

    for image_id in tqdm(
        candidates["ImageID"],
        desc="Validating Batch 2 images",
    ):
        path = IMAGE_ROOT / f"{image_id}.jpg"

        if not path.exists():
            invalid_ids.append(image_id)
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid_ids.append(image_id)

    return invalid_ids


def print_summary(frame: pd.DataFrame) -> None:
    print("\n=== OPEN IMAGES NEGATIVE BATCH 2 ===")
    print(f"Candidate images: {len(frame)}")

    print("\nCandidate categories:")
    for category, count in frame["CandidateCategory"].value_counts().items():
        print(f"  {category}: {count}")

    print("\nSource splits:")
    for split, count in frame["SourceSplit"].value_counts().items():
        print(f"  {split}: {count}")

    print("\nLicense distribution:")
    for license_value, count in frame["License"].fillna("<missing>").value_counts().items():
        print(f"  {license_value}: {count}")

    print("\nIMPORTANT:")
    print("These are Batch 2 CANDIDATES only.")
    print("Every image still requires manual visual audit.")


def main() -> None:
    BATCH2_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_ids = collect_positive_ids()
    batch1_ids = collect_batch1_ids()

    excluded_ids = positive_ids | batch1_ids

    print(f"Positive Open Images IDs excluded: {len(positive_ids)}")
    print(f"Batch 1 candidate IDs excluded: {len(batch1_ids)}")
    print(f"Total unique excluded IDs: {len(excluded_ids)}")

    name_to_mid, mid_to_name = load_class_map()

    records = []

    for split in ("validation", "test"):
        print(f"\nProcessing Open Images split: {split}")

        boxes = load_boxes(split)

        records.extend(
            build_candidate_pool(
                split=split,
                boxes=boxes,
                name_to_mid=name_to_mid,
                mid_to_name=mid_to_name,
                excluded_ids=excluded_ids,
            )
        )

    candidates = sample_candidates(records)
    candidates = attach_provenance(candidates)

    metadata_path = (
        BATCH2_ROOT
        / "negative_candidates_batch2_metadata.csv"
    )

    candidates.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    failures = download_images(candidates)
    invalid_ids = validate_images(candidates)

    print(f"\nDownload failures: {len(failures)}")
    print(f"Corrupt/missing images: {len(invalid_ids)}")

    if invalid_ids:
        invalid_path = (
            BATCH2_ROOT
            / "corrupt_or_missing_batch2.txt"
        )

        invalid_path.write_text(
            "\n".join(invalid_ids),
            encoding="utf-8",
        )

    print_summary(candidates)

    print(f"\nMetadata: {metadata_path}")
    print(f"Images: {IMAGE_ROOT}")


if __name__ == "__main__":
    main()