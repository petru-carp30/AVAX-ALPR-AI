from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import re
import numpy as np

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "kaggle_plate_license_recognition"
AUDIT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "audits" / "kaggle_plate_license_recognition"

REVIEW_PATH = AUDIT_ROOT / "kaggle_source_group_review.csv"

SPLITS = ("train", "valid", "test")
TARGET_CLASS = "LicensePlate"

ROBOFLOW_SUFFIX_PATTERN = re.compile(r"\.rf\.[0-9a-fA-F]{32}\.[^.]+$")

WINDOW_NAME = "AVAX Kaggle Detector Group Audit"
THUMB_WIDTH = 1200
THUMB_HEIGHT = 760
HEADER_HEIGHT = 150
COLUMNS = 1
MAX_VARIANTS_PER_PAGE = 1


def get_source_key(filename: str) -> str:
    return ROBOFLOW_SUFFIX_PATTERN.sub("", filename)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_groups() -> dict[str, list[dict]]:
    groups = defaultdict(list)

    for split in SPLITS:
        split_root = DATASET_ROOT / split
        coco = load_coco(split_root / "_annotations.coco.json")

        category_by_id = {
            category["id"]: category["name"]
            for category in coco["categories"]
        }

        annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            if category_by_id.get(annotation["category_id"]) == TARGET_CLASS:
                annotations_by_image[annotation["image_id"]].append(annotation)

        for image_info in coco["images"]:
            annotations = annotations_by_image.get(image_info["id"], [])

            if not annotations:
                continue

            filename = image_info["file_name"]
            source_group = get_source_key(filename)

            groups[source_group].append({
                "split": split,
                "filename": filename,
                "path": split_root / filename,
                "annotations": annotations,
                "width": image_info["width"],
                "height": image_info["height"],
            })

    return groups


def calculate_suspicion_score(variants: list[dict]) -> float:
    score = 0.0

    for variant in variants:
        image_area = variant["width"] * variant["height"]

        for annotation in variant["annotations"]:
            _, _, width, height = annotation["bbox"]

            aspect_ratio = width / height
            area_ratio = (width * height) / image_area

            if aspect_ratio < 1.2:
                score += 3.0

            if area_ratio > 0.25:
                score += 3.0

            if area_ratio < 0.001:
                score += 1.0

        if len(variant["annotations"]) >= 3:
            score += 2.0

    return score


def render_variant(variant: dict) -> cv2.Mat:
    image = cv2.imread(str(variant["path"]))

    if image is None:
        tile = cv2.imread(str(variant["path"]))
        raise RuntimeError(f"Could not load image: {variant['path']}")

    original_height, original_width = image.shape[:2]

    scale = min(
        THUMB_WIDTH / original_width,
        (THUMB_HEIGHT - 55) / original_height,
    )

    resized_width = max(1, int(original_width * scale))
    resized_height = max(1, int(original_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    tile = cv2.copyMakeBorder(
        resized,
        0,
        THUMB_HEIGHT - resized_height,
        0,
        THUMB_WIDTH - resized_width,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    for annotation in variant["annotations"]:
        x, y, width, height = annotation["bbox"]

        x1 = int(round(x * scale))
        y1 = int(round(y * scale))
        x2 = int(round((x + width) * scale))
        y2 = int(round((y + height) * scale))

        cv2.rectangle(
            tile,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    label = (
        f"{variant['split']} | "
        f"{variant['filename'][:48]} | "
        f"{len(variant['annotations'])} plate(s)"
    )

    cv2.rectangle(
        tile,
        (0, THUMB_HEIGHT - 55),
        (THUMB_WIDTH, THUMB_HEIGHT),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        tile,
        label,
        (8, THUMB_HEIGHT - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return tile


def render_group(
    source_group: str,
    variants: list[dict],
    group_index: int,
    total_groups: int,
    reviewed_count: int,
    accepted_count: int,
    page_index: int,
) -> cv2.Mat:
    page_count = math.ceil(len(variants) / MAX_VARIANTS_PER_PAGE)

    start = page_index * MAX_VARIANTS_PER_PAGE
    end = min(start + MAX_VARIANTS_PER_PAGE, len(variants))

    page_variants = variants[start:end]

    rows = math.ceil(len(page_variants) / COLUMNS)

    canvas_width = COLUMNS * THUMB_WIDTH
    canvas_height = HEADER_HEIGHT + rows * THUMB_HEIGHT

    canvas = np.full(
    (canvas_height, canvas_width, 3),
    15,
    dtype=np.uint8,
    )

    lines = [
        f"Group {group_index}/{total_groups} | Source: {source_group}",
        f"Variants: {len(variants)} | Page {page_index + 1}/{page_count} | Reviewed: {reviewed_count} | Accepted: {accepted_count}",
        "A=accept | M=mosaic/collage | C=plate/OCR crop | X=reject other | U=unsure | [ ]=page | Q=save+quit",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (20, 35 + index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for local_index, variant in enumerate(page_variants):
        tile = render_variant(variant)

        column = local_index % COLUMNS
        row = local_index // COLUMNS

        x = column * THUMB_WIDTH
        y = HEADER_HEIGHT + row * THUMB_HEIGHT

        canvas[
            y:y + THUMB_HEIGHT,
            x:x + THUMB_WIDTH,
        ] = tile

    return canvas


def save_review(groups: dict[str, list[dict]], status_map: dict[str, str]) -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    records = []

    for source_group, variants in groups.items():
        records.append({
            "source_group": source_group,
            "variant_count": len(variants),
            "plate_instance_count": sum(
                len(variant["annotations"])
                for variant in variants
            ),
            "upstream_splits": ";".join(
                sorted({variant["split"] for variant in variants})
            ),
            "review_status": status_map.get(source_group, ""),
        })

    frame = pd.DataFrame(records)
    frame = frame.sort_values("source_group").reset_index(drop=True)

    frame.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(groups: dict[str, list[dict]], status_map: dict[str, str]) -> None:
    counts = pd.Series(
        list(status_map.values()),
        dtype="object",
    ).value_counts()

    total_detector_images = sum(len(variants) for variants in groups.values())

    accepted_groups = {
        source_group
        for source_group, status in status_map.items()
        if status == "accepted_training_positive"
    }

    accepted_images = sum(
        len(groups[source_group])
        for source_group in accepted_groups
    )

    accepted_instances = sum(
        len(variant["annotations"])
        for source_group in accepted_groups
        for variant in groups[source_group]
    )

    print("\n=== KAGGLE DETECTOR GROUP VISUAL AUDIT ===")
    print(f"Total source groups: {len(groups)}")
    print(f"Total detector image variants: {total_detector_images}")

    print("\nReview results:")
    for status, count in counts.items():
        if status:
            print(f"  {status}: {count}")

    print("\nAccepted training subset:")
    print(f"  Source groups: {len(accepted_groups)}")
    print(f"  Image variants: {accepted_images}")
    print(f"  LicensePlate instances: {accepted_instances}")

    print(f"\nReview CSV: {REVIEW_PATH}")


def main() -> None:
    groups = collect_groups()

    if len(groups) != 909:
        raise RuntimeError(
            f"Expected 909 detector source groups, found {len(groups)}."
        )

    total_variants = sum(len(variants) for variants in groups.values())

    if total_variants != 1539:
        raise RuntimeError(
            f"Expected 1539 detector image variants, found {total_variants}."
        )

    if REVIEW_PATH.exists():
        previous = pd.read_csv(
            REVIEW_PATH,
            dtype={"source_group": str},
        )

        status_map = dict(
            zip(
                previous["source_group"],
                previous["review_status"].fillna(""),
            )
        )
    else:
        status_map = {}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -calculate_suspicion_score(item[1]),
            item[0],
        ),
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(WINDOW_NAME, 1400, 950)

    reviewed_count = sum(
        bool(status) and status != "unsure"
        for status in status_map.values()
    )

    accepted_count = sum(
        status == "accepted_training_positive"
        for status in status_map.values()
    )

    for group_index, (source_group, variants) in enumerate(
        ordered_groups,
        start=1,
    ):
        existing_status = status_map.get(source_group, "")

        if existing_status and existing_status != "unsure":
            continue

        page_index = 0
        page_count = math.ceil(
            len(variants) / MAX_VARIANTS_PER_PAGE
        )

        while True:
            display = render_group(
                source_group=source_group,
                variants=variants,
                group_index=group_index,
                total_groups=len(groups),
                reviewed_count=reviewed_count,
                accepted_count=accepted_count,
                page_index=page_index,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("a"), ord("A")):
                status_map[source_group] = "accepted_training_positive"
                reviewed_count += 1
                accepted_count += 1
                break

            if key in (ord("m"), ord("M")):
                status_map[source_group] = "reject_mosaic_collage"
                reviewed_count += 1
                break

            if key in (ord("c"), ord("C")):
                status_map[source_group] = "reject_plate_or_ocr_crop"
                reviewed_count += 1
                break

            if key in (ord("x"), ord("X")):
                status_map[source_group] = "reject_other"
                reviewed_count += 1
                break

            if key in (ord("u"), ord("U")):
                status_map[source_group] = "unsure"
                break

            if key == ord("]"):
                page_index = min(
                    page_index + 1,
                    page_count - 1,
                )
                continue

            if key == ord("["):
                page_index = max(
                    page_index - 1,
                    0,
                )
                continue

            if key in (ord("q"), ord("Q")):
                save_review(groups, status_map)
                print_summary(groups, status_map)
                cv2.destroyAllWindows()
                return

        save_review(groups, status_map)

    save_review(groups, status_map)
    print_summary(groups, status_map)

    cv2.destroyAllWindows()

    unsure_count = sum(
        status == "unsure"
        for status in status_map.values()
    )

    if unsure_count:
        print(f"\nRESULT: {unsure_count} SOURCE GROUP(S) STILL REQUIRE REVIEW")
    else:
        print("\nRESULT: KAGGLE SOURCE GROUP VISUAL AUDIT COMPLETE")


if __name__ == "__main__":
    main()