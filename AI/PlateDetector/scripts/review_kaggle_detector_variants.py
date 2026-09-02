from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "datasets"
    / "raw"
    / "kaggle_plate_license_recognition"
)

AUDIT_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "datasets"
    / "audits"
    / "kaggle_plate_license_recognition"
)

GROUP_REVIEW_PATH = AUDIT_ROOT / "kaggle_source_group_review.csv"
VARIANT_REVIEW_PATH = AUDIT_ROOT / "kaggle_variant_review.csv"

SPLITS = ("train", "valid", "test")
TARGET_CLASS = "LicensePlate"

ROBOFLOW_SUFFIX_PATTERN = re.compile(
    r"\.rf\.[0-9a-fA-F]{32}\.[^.]+$"
)

WINDOW_NAME = "AVAX Kaggle Detector Variant Audit"

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 760
HEADER_HEIGHT = 150


def get_source_key(filename: str) -> str:
    return ROBOFLOW_SUFFIX_PATTERN.sub("", filename)


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_variants() -> dict[str, list[dict]]:
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
            if (
                category_by_id.get(annotation["category_id"])
                == TARGET_CLASS
            ):
                annotations_by_image[
                    annotation["image_id"]
                ].append(annotation)

        for image_info in coco["images"]:
            annotations = annotations_by_image.get(
                image_info["id"],
                [],
            )

            if not annotations:
                continue

            filename = image_info["file_name"]
            source_group = get_source_key(filename)

            groups[source_group].append(
                {
                    "source_group": source_group,
                    "split": split,
                    "filename": filename,
                    "path": split_root / filename,
                    "annotations": annotations,
                    "width": image_info["width"],
                    "height": image_info["height"],
                }
            )

    for source_group in groups:
        groups[source_group] = sorted(
            groups[source_group],
            key=lambda item: (
                item["split"],
                item["filename"],
            ),
        )

    return groups


def load_existing_variant_reviews() -> dict[tuple[str, str], str]:
    if not VARIANT_REVIEW_PATH.exists():
        return {}

    frame = pd.read_csv(
        VARIANT_REVIEW_PATH,
        dtype=str,
    )

    return {
        (
            row.source_group,
            row.filename,
        ): (
            ""
            if pd.isna(row.review_status)
            else row.review_status
        )
        for row in frame.itertuples(index=False)
    }


def migrate_single_variant_group_reviews(
    groups: dict[str, list[dict]],
    status_map: dict[tuple[str, str], str],
) -> int:
    if not GROUP_REVIEW_PATH.exists():
        return 0

    group_review = pd.read_csv(
        GROUP_REVIEW_PATH,
        dtype=str,
    )

    group_status_map = dict(
        zip(
            group_review["source_group"],
            group_review["review_status"].fillna(""),
        )
    )

    migrated = 0

    for source_group, variants in groups.items():
        if len(variants) != 1:
            continue

        group_status = group_status_map.get(
            source_group,
            "",
        )

        if not group_status or group_status == "unsure":
            continue

        variant = variants[0]
        key = (
            source_group,
            variant["filename"],
        )

        if status_map.get(key):
            continue

        status_map[key] = group_status
        migrated += 1

    return migrated


def render_variant(
    variant: dict,
    group_index: int,
    total_groups: int,
    variant_index: int,
    variant_count: int,
    reviewed_variants: int,
    accepted_variants: int,
    current_status: str
) -> np.ndarray:
    image = cv2.imread(str(variant["path"]))

    if image is None:
        raise RuntimeError(
            f"Could not load image: {variant['path']}"
        )

    original_height, original_width = image.shape[:2]

    available_height = DISPLAY_HEIGHT - HEADER_HEIGHT

    scale = min(
        DISPLAY_WIDTH / original_width,
        available_height / original_height,
    )

    resized_width = max(
        1,
        int(original_width * scale),
    )

    resized_height = max(
        1,
        int(original_height * scale),
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    for annotation in variant["annotations"]:
        x, y, width, height = annotation["bbox"]

        x1 = int(round(x * scale))
        y1 = int(round(y * scale))
        x2 = int(round((x + width) * scale))
        y2 = int(round((y + height) * scale))

        cv2.rectangle(
            resized,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

    canvas = np.full(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3,
        ),
        15,
        dtype=np.uint8,
    )

    x_offset = (
        DISPLAY_WIDTH - resized_width
    ) // 2

    y_offset = HEADER_HEIGHT

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    lines = [
        f"Group {group_index}/{total_groups} | Source: {variant['source_group']}",
        f"Variant {variant_index}/{variant_count} | Split: {variant['split']} | Plates: {len(variant['annotations'])}",
        f"Reviewed: {reviewed_variants} | Accepted: {accepted_variants} | Current: {current_status or 'UNREVIEWED'}",
        "A=accept | M=mosaic | C=OCR crop | X=reject | U=unsure | B=back | N=next | Q=save+quit",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (
                20,
                30 + index * 32,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        variant["filename"][:110],
        (
            20,
            DISPLAY_HEIGHT - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def save_reviews(
    groups: dict[str, list[dict]],
    status_map: dict[tuple[str, str], str],
) -> None:
    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for source_group, variants in groups.items():
        variant_count = len(variants)

        for variant_index, variant in enumerate(
            variants,
            start=1,
        ):
            key = (
                source_group,
                variant["filename"],
            )

            records.append(
                {
                    "source_group": source_group,
                    "variant_index": variant_index,
                    "variant_count": variant_count,
                    "split": variant["split"],
                    "filename": variant["filename"],
                    "plate_instance_count": len(
                        variant["annotations"]
                    ),
                    "review_status": status_map.get(
                        key,
                        "",
                    ),
                }
            )

    frame = pd.DataFrame(records)

    frame = frame.sort_values(
        [
            "source_group",
            "variant_index",
        ],
        kind="stable",
    )

    frame.to_csv(
        VARIANT_REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    groups: dict[str, list[dict]],
    status_map: dict[tuple[str, str], str],
) -> None:
    statuses = [
        status
        for status in status_map.values()
        if status
    ]

    counts = pd.Series(
        statuses,
        dtype="object",
    ).value_counts()

    accepted_keys = {
        key
        for key, status in status_map.items()
        if status == "accepted_training_positive"
    }

    accepted_instances = 0

    groups_with_accepted_variants = set()

    for source_group, variants in groups.items():
        for variant in variants:
            key = (
                source_group,
                variant["filename"],
            )

            if key not in accepted_keys:
                continue

            accepted_instances += len(
                variant["annotations"]
            )

            groups_with_accepted_variants.add(
                source_group
            )

    total_variants = sum(
        len(variants)
        for variants in groups.values()
    )

    reviewed = sum(1 for status in status_map.values() if status and status != "unsure")

    unsure = sum(
        status == "unsure"
        for status in status_map.values()
    )

    print(
        "\n=== KAGGLE DETECTOR VARIANT AUDIT ==="
    )

    print(
        f"Total source groups: {len(groups)}"
    )

    print(
        f"Total detector variants: {total_variants}"
    )

    print(
        f"Reviewed variants: {reviewed}"
    )

    print(
        f"Unsure variants: {unsure}"
    )

    print("\nReview results:")

    for status, count in counts.items():
        print(
            f"  {status}: {count}"
        )

    print("\nAccepted training subset:")

    print(
        f"  Accepted variants: {len(accepted_keys)}"
    )

    print(
        "  Source groups represented: "
        f"{len(groups_with_accepted_variants)}"
    )

    print(
        "  LicensePlate instances: "
        f"{accepted_instances}"
    )

    print(
        f"\nReview CSV: {VARIANT_REVIEW_PATH}"
    )

def get_review_counts(status_map: dict[tuple[str, str], str]) -> tuple[int, int]:
    reviewed = sum(1 for status in status_map.values() if status and status != "unsure")
    accepted = sum(1 for status in status_map.values() if status == "accepted_training_positive")
    return reviewed, accepted


def main() -> None:
    groups = collect_variants()
    total_variants = sum(len(variants) for variants in groups.values())

    if len(groups) != 909:
        raise RuntimeError(f"Expected 909 source groups, found {len(groups)}.")

    if total_variants != 1539:
        raise RuntimeError(f"Expected 1539 detector variants, found {total_variants}.")

    status_map = load_existing_variant_reviews()
    migrated = migrate_single_variant_group_reviews(groups, status_map)

    print(f"Single-variant decisions migrated: {migrated}")
    print("Use B/N to navigate backward/forward. Existing verdicts can be overwritten.")

    ordered_groups = sorted(groups.items(), key=lambda item: item[0])
    items = []

    for group_index, (source_group, variants) in enumerate(ordered_groups, start=1):
        for variant_index, variant in enumerate(variants, start=1):
            items.append({
                "group_index": group_index,
                "variant_index": variant_index,
                "variant_count": len(variants),
                "variant": variant,
            })

    save_reviews(groups, status_map)

    first_unreviewed = 0

    for index, item in enumerate(items):
        variant = item["variant"]
        status = status_map.get((variant["source_group"], variant["filename"]), "")

        if not status or status == "unsure":
            first_unreviewed = index
            break
    else:
        first_unreviewed = len(items) - 1

    current_index = first_unreviewed

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 950)

    while 0 <= current_index < len(items):
        item = items[current_index]
        variant = item["variant"]
        review_key = (variant["source_group"], variant["filename"])
        current_status = status_map.get(review_key, "")

        reviewed_variants, accepted_variants = get_review_counts(status_map)

        display = render_variant(
            variant=variant,
            group_index=item["group_index"],
            total_groups=len(groups),
            variant_index=item["variant_index"],
            variant_count=item["variant_count"],
            reviewed_variants=reviewed_variants,
            accepted_variants=accepted_variants,
            current_status=current_status,
        )

        cv2.imshow(WINDOW_NAME, display)
        key_press = cv2.waitKey(0) & 0xFF

        if key_press in (ord("a"), ord("A")):
            status_map[review_key] = "accepted_training_positive"
            save_reviews(groups, status_map)
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("m"), ord("M")):
            status_map[review_key] = "reject_mosaic_collage"
            save_reviews(groups, status_map)
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("c"), ord("C")):
            status_map[review_key] = "reject_plate_or_ocr_crop"
            save_reviews(groups, status_map)
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("x"), ord("X")):
            status_map[review_key] = "reject_other"
            save_reviews(groups, status_map)
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("u"), ord("U")):
            status_map[review_key] = "unsure"
            save_reviews(groups, status_map)
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("b"), ord("B")):
            current_index = max(current_index - 1, 0)
            continue

        if key_press in (ord("n"), ord("N")):
            current_index = min(current_index + 1, len(items) - 1)
            continue

        if key_press in (ord("q"), ord("Q")):
            save_reviews(groups, status_map)
            print_summary(groups, status_map)
            cv2.destroyAllWindows()
            return

    save_reviews(groups, status_map)
    print_summary(groups, status_map)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()