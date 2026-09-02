from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_negative_candidates_batch2"
AUDIT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "audits" / "open_images_negative_pool" / "batch2"

METADATA_PATH = RAW_ROOT / "negative_candidates_batch2_metadata.csv"
IMAGE_ROOT = RAW_ROOT / "images"

REVIEW_PATH = AUDIT_ROOT / "negative_review_batch2.csv"
ACCEPTED_PATH = AUDIT_ROOT / "accepted_negatives_batch2_metadata.csv"

MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 850
HEADER_HEIGHT = 135


def resize_for_display(image):
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def add_header(image, row, audited_count, accepted_count, total):
    display = cv2.copyMakeBorder(
        image,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    lines = [
        f"ImageID: {row['ImageID']}",
        f"Category: {row['CandidateCategory']} | Matched: {row['MatchedClasses']}",
        f"Audited: {audited_count}/{total} | Accepted: {accepted_count}",
        "A = accept negative | P = plate visible | X = reject other | U = unsure | Q = save and quit",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (20, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return display


def save_results(metadata, review):
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    accepted_ids = set(
        review.loc[
            review["ReviewStatus"] == "accepted_negative",
            "ImageID",
        ]
    )

    accepted = metadata[
        metadata["ImageID"].isin(accepted_ids)
    ].copy()

    accepted = accepted.merge(
        review[["ImageID", "ReviewStatus"]],
        on="ImageID",
        how="left",
        validate="one_to_one",
    )

    accepted.to_csv(
        ACCEPTED_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(metadata, review):
    audited = review[
        review["ReviewStatus"].notna()
        & (review["ReviewStatus"] != "")
    ].copy()

    accepted_review = audited[
        audited["ReviewStatus"] == "accepted_negative"
    ]

    accepted_ids = set(accepted_review["ImageID"])

    accepted = metadata[
        metadata["ImageID"].isin(accepted_ids)
    ].copy()

    print("\n=== OPEN IMAGES NEGATIVE BATCH 2 VISUAL AUDIT ===")
    print(f"Total candidates: {len(metadata)}")
    print(f"Audited: {len(audited)}")
    print(f"Accepted negatives: {len(accepted)}")

    print("\nReview results:")
    for status, count in audited["ReviewStatus"].value_counts().items():
        print(f"  {status}: {count}")

    if not accepted.empty:
        print("\nAccepted category distribution:")
        for category, count in accepted["CandidateCategory"].value_counts().items():
            print(f"  {category}: {count}")

        print("\nAccepted source distribution:")
        for split, count in accepted["SourceSplit"].value_counts().items():
            print(f"  {split}: {count}")

        print("\nAccepted license distribution:")
        for license_value, count in accepted["License"].value_counts().items():
            print(f"  {license_value}: {count}")

    combined_total = 456 + len(accepted)

    print("\nCombined negative pool estimate:")
    print(f"  Batch 1 confirmed: 456")
    print(f"  Batch 2 accepted: {len(accepted)}")
    print(f"  Combined: {combined_total}")

    if combined_total >= 500:
        print("\nRESULT: MASTER NEGATIVE POOL BASELINE RANGE REACHED")
    else:
        print("\nRESULT: NEGATIVE POOL STILL BELOW APPROXIMATE 500 TARGET")


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH, dtype={"ImageID": str})

    if REVIEW_PATH.exists():
        review = pd.read_csv(REVIEW_PATH, dtype={"ImageID": str})
    else:
        review = pd.DataFrame({
            "ImageID": metadata["ImageID"],
            "ReviewStatus": "",
        })

    status_map = dict(
        zip(
            review["ImageID"],
            review["ReviewStatus"].fillna(""),
        )
    )

    audited_count = sum(bool(status) for status in status_map.values())
    accepted_count = sum(
        status == "accepted_negative"
        for status in status_map.values()
    )

    cv2.namedWindow(
        "AVAX Open Images Negative Batch 2 Audit",
        cv2.WINDOW_NORMAL,
    )

    total = len(metadata)

    for _, row in metadata.iterrows():
        image_id = row["ImageID"]

        if status_map.get(image_id):
            continue

        image_path = IMAGE_ROOT / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            status_map[image_id] = "reject_missing_or_corrupt"
            audited_count += 1
            continue

        image = resize_for_display(image)
        display = add_header(
            image,
            row,
            audited_count,
            accepted_count,
            total,
        )

        cv2.imshow(
            "AVAX Open Images Negative Batch 2 Audit",
            display,
        )

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("a"), ord("A")):
                status_map[image_id] = "accepted_negative"
                accepted_count += 1
                audited_count += 1
                break

            if key in (ord("p"), ord("P")):
                status_map[image_id] = "reject_plate_visible"
                audited_count += 1
                break

            if key in (ord("x"), ord("X")):
                status_map[image_id] = "reject_other"
                audited_count += 1
                break

            if key in (ord("u"), ord("U")):
                status_map[image_id] = "unsure"
                audited_count += 1
                break

            if key in (ord("q"), ord("Q")):
                review["ReviewStatus"] = review["ImageID"].map(status_map)
                save_results(metadata, review)
                print_summary(metadata, review)
                cv2.destroyAllWindows()
                return

        review["ReviewStatus"] = review["ImageID"].map(status_map)
        save_results(metadata, review)

    review["ReviewStatus"] = review["ImageID"].map(status_map)
    save_results(metadata, review)
    print_summary(metadata, review)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()