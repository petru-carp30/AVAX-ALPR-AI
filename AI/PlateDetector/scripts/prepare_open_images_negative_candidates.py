from __future__ import annotations

import random
from pathlib import Path

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]

POSITIVE_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_lp_kaggle"
RAW_NEGATIVE_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_negative_candidates"
METADATA_ROOT = RAW_NEGATIVE_ROOT / "metadata"
IMAGE_ROOT = RAW_NEGATIVE_ROOT / "images"

RANDOM_SEED = 20260824

CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"

SPLIT_URLS = {
    "validation": {
        "boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "metadata": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    },
    "test": {
        "boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "metadata": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    },
}

CATEGORY_CONFIG = {
    "vehicle_no_visible_plate_candidate": {
        "classes": ["Car", "Truck", "Van", "Bus", "Motorcycle"],
        "quota": 350,
    },
    "heavy_equipment_candidate": {
        "classes": ["Truck", "Tractor", "Forklift", "Crane"],
        "quota": 180,
    },
    "people": {
        "classes": ["Person", "Man", "Woman", "Boy", "Girl"],
        "quota": 150,
    },
    "road_gate_scene": {
        "classes": ["Traffic sign", "Traffic light", "Stop sign", "Parking meter"],
        "quota": 200,
    },
    "text_like_object": {
        "classes": ["Billboard", "Poster", "Traffic sign"],
        "quota": 160,
    },
    "barrier_background": {
        "classes": ["Fence", "Door", "Window", "Building"],
        "quota": 160,
    },
}


def download_file(url: str, target: Path) -> None:
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading metadata: {target.name}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        with target.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def collect_positive_open_images_ids() -> set[str]:
    image_ids: set[str] = set()

    for split in ("train", "val"):
        image_dir = POSITIVE_ROOT / "images" / split

        if not image_dir.exists():
            continue

        for path in image_dir.iterdir():
            if path.is_file():
                image_ids.add(path.stem.lower())

    return image_ids


def load_class_map() -> tuple[dict[str, str], dict[str, str]]:
    path = METADATA_ROOT / "oidv7-class-descriptions-boxable.csv"
    download_file(CLASS_DESCRIPTIONS_URL, path)

    frame = pd.read_csv(
        path,
        header=None,
        names=["LabelName", "ClassName"],
        dtype=str,
    )

    name_to_mid = dict(zip(frame["ClassName"], frame["LabelName"]))
    mid_to_name = dict(zip(frame["LabelName"], frame["ClassName"]))

    return name_to_mid, mid_to_name


def load_split_boxes(split: str) -> pd.DataFrame:
    path = METADATA_ROOT / f"{split}-annotations-bbox.csv"
    download_file(SPLIT_URLS[split]["boxes"], path)

    return pd.read_csv(
        path,
        usecols=["ImageID", "LabelName"],
        dtype={"ImageID": str, "LabelName": str},
    )


def load_split_metadata(split: str) -> pd.DataFrame:
    path = METADATA_ROOT / f"{split}-images-with-rotation.csv"
    download_file(SPLIT_URLS[split]["metadata"], path)

    frame = pd.read_csv(path, dtype=str)
    frame["ImageID"] = frame["ImageID"].str.lower()
    return frame


def available_mids(
    class_names: list[str],
    name_to_mid: dict[str, str],
) -> list[str]:
    result = []

    for class_name in class_names:
        mid = name_to_mid.get(class_name)

        if mid is None:
            print(f"Class not available, skipping: {class_name}")
            continue

        result.append(mid)

    return result


def build_candidate_index(
    split: str,
    boxes: pd.DataFrame,
    name_to_mid: dict[str, str],
    mid_to_name: dict[str, str],
    excluded_ids: set[str],
) -> list[dict[str, str]]:
    plate_mid = name_to_mid["Vehicle registration plate"]

    boxes["ImageID"] = boxes["ImageID"].str.lower()

    plate_image_ids = set(
        boxes.loc[boxes["LabelName"] == plate_mid, "ImageID"]
    )

    records: list[dict[str, str]] = []

    for category, config in CATEGORY_CONFIG.items():
        mids = available_mids(config["classes"], name_to_mid)

        if not mids:
            continue

        relevant = boxes[
            boxes["LabelName"].isin(mids)
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
    random_generator = random.Random(RANDOM_SEED)

    records_frame = pd.DataFrame(records)

    if records_frame.empty:
        raise RuntimeError("No negative candidates were discovered.")

    selected_rows: list[pd.Series] = []
    selected_ids: set[str] = set()

    for category, config in CATEGORY_CONFIG.items():
        category_rows = records_frame[
            records_frame["CandidateCategory"] == category
        ].copy()

        category_rows = category_rows[
            ~category_rows["ImageID"].isin(selected_ids)
        ]

        indices = list(category_rows.index)
        random_generator.shuffle(indices)

        quota = config["quota"]
        selected_indices = indices[:quota]

        if len(selected_indices) < quota:
            print(
                f"WARNING: category {category} requested {quota}, "
                f"available {len(selected_indices)}"
            )

        for index in selected_indices:
            row = category_rows.loc[index]
            selected_rows.append(row)
            selected_ids.add(str(row["ImageID"]))

    result = pd.DataFrame(selected_rows)

    if result.empty:
        raise RuntimeError("Candidate sampling produced no images.")

    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return result


def attach_provenance(candidates: pd.DataFrame) -> pd.DataFrame:
    result_frames = []

    for split in ("validation", "test"):
        split_candidates = candidates[
            candidates["SourceSplit"] == split
        ].copy()

        if split_candidates.empty:
            continue

        metadata = load_split_metadata(split)

        merged = split_candidates.merge(
            metadata,
            on="ImageID",
            how="left",
            validate="one_to_one",
        )

        result_frames.append(merged)

    result = pd.concat(result_frames, ignore_index=True)

    missing_metadata = result["OriginalURL"].isna().sum()

    if missing_metadata:
        raise RuntimeError(
            f"{missing_metadata} candidate images are missing Open Images provenance."
        )

    result["SourceDataset"] = "open_images_v7"
    result["RealSynthetic"] = "REAL"
    result["IsNegativeCandidate"] = True

    return result


def download_images(candidates: pd.DataFrame) -> list[str]:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    bucket = boto3.resource(
        "s3",
        config=Config(signature_version=UNSIGNED),
    ).Bucket("open-images-dataset")

    failures: list[str] = []

    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc="Downloading candidates",
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
            print(f"\nDownload failed: {split}/{image_id}: {exception}")
            failures.append(image_id)

    return failures


def validate_images(candidates: pd.DataFrame) -> list[str]:
    corrupt_ids: list[str] = []

    for image_id in tqdm(
        candidates["ImageID"],
        desc="Validating downloaded images",
    ):
        path = IMAGE_ROOT / f"{image_id}.jpg"

        if not path.exists():
            corrupt_ids.append(image_id)
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt_ids.append(image_id)

    return corrupt_ids


def print_summary(frame: pd.DataFrame) -> None:
    print("\n=== OPEN IMAGES NEGATIVE CANDIDATE PREPARATION ===")
    print(f"Candidate images: {len(frame)}")

    print("\nCandidate categories:")
    for category, count in frame["CandidateCategory"].value_counts().items():
        print(f"  {category}: {count}")

    print("\nSource splits:")
    for split, count in frame["SourceSplit"].value_counts().items():
        print(f"  {split}: {count}")

    print("\nLicense metadata:")
    for license_value, count in frame["License"].fillna("<missing>").value_counts().items():
        print(f"  {license_value}: {count}")

    required_columns = [
        "ImageID",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "Title",
    ]

    for column in required_columns:
        missing = frame[column].isna().sum()
        print(f"Missing {column}: {missing}")

    print("\nIMPORTANT:")
    print("These are CANDIDATES only.")
    print("No image is a detector negative until visual review confirms")
    print("that NO license plate is visibly present.")


def main() -> None:
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    positive_ids = collect_positive_open_images_ids()
    print(f"Existing positive Open Images IDs excluded: {len(positive_ids)}")

    name_to_mid, mid_to_name = load_class_map()

    if "Vehicle registration plate" not in name_to_mid:
        raise RuntimeError("Vehicle registration plate class was not found.")

    records: list[dict[str, str]] = []

    for split in ("validation", "test"):
        print(f"\nProcessing Open Images split: {split}")
        boxes = load_split_boxes(split)

        records.extend(
            build_candidate_index(
                split=split,
                boxes=boxes,
                name_to_mid=name_to_mid,
                mid_to_name=mid_to_name,
                excluded_ids=positive_ids,
            )
        )

    candidates = sample_candidates(records)
    candidates = attach_provenance(candidates)

    metadata_path = RAW_NEGATIVE_ROOT / "negative_candidates_metadata.csv"
    candidates.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    failures = download_images(candidates)

    if failures:
        print(f"\nDownload failures: {len(failures)}")

    corrupt_ids = validate_images(candidates)

    if corrupt_ids:
        print(f"Corrupt/missing downloads: {len(corrupt_ids)}")
        corrupt_path = RAW_NEGATIVE_ROOT / "corrupt_or_missing.txt"
        corrupt_path.write_text("\n".join(corrupt_ids), encoding="utf-8")
    else:
        print("Corrupt/missing downloads: 0")

    print_summary(candidates)

    print(f"\nMetadata written to: {metadata_path}")
    print(f"Images written to: {IMAGE_ROOT}")


if __name__ == "__main__":
    main()