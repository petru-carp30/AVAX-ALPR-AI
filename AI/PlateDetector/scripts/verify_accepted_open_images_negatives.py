from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "open_images_negative_candidates"
AUDIT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "audits" / "open_images_negative_pool"

METADATA_PATH = RAW_ROOT / "negative_candidates_metadata.csv"
IMAGE_ROOT = RAW_ROOT / "images"
REVIEW_PATH = AUDIT_ROOT / "negative_review.csv"
ACCEPTED_PATH = AUDIT_ROOT / "accepted_negatives_metadata.csv"

MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 900


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


def add_header(image, row, index, total):
    header_height = 125
    header = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    lines = [
        f"QC {index}/{total} | ImageID: {row['ImageID']}",
        f"Category: {row['CandidateCategory']} | Matched: {row['MatchedClasses']}",
        "A = confirm negative | P = plate visible | X = reject other | U = unsure | Q = save and quit",
    ]

    for line_index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (20, 32 + line_index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return header


def save_results(metadata, review):
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    accepted_ids = set(
        review.loc[review["ReviewStatus"] == "accepted_negative", "ImageID"]
    )

    accepted = metadata[
        metadata["ImageID"].isin(accepted_ids)
    ].copy()

    accepted = accepted.merge(
        review[["ImageID", "ReviewStatus"]],
        on="ImageID",
        how="left",
    )

    accepted.to_csv(ACCEPTED_PATH, index=False, encoding="utf-8-sig")


def print_summary(review):
    counts = review["ReviewStatus"].fillna("").value_counts()

    print("\n=== OPEN IMAGES ACCEPTED NEGATIVE QC ===")
    print(f"Accepted negatives after QC: {counts.get('accepted_negative', 0)}")
    print(f"Reject plate visible: {counts.get('reject_plate_visible', 0)}")
    print(f"Reject other: {counts.get('reject_other', 0)}")
    print(f"Unsure: {counts.get('unsure', 0)}")


def main():
    metadata = pd.read_csv(METADATA_PATH, dtype={"ImageID": str})
    review = pd.read_csv(REVIEW_PATH, dtype={"ImageID": str})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = AUDIT_ROOT / f"negative_review_before_qc_{timestamp}.csv"
    shutil.copy2(REVIEW_PATH, backup_path)

    print(f"Backup created: {backup_path}")

    accepted_ids = review.loc[
        review["ReviewStatus"] == "accepted_negative",
        "ImageID",
    ].tolist()

    accepted_metadata = metadata[
        metadata["ImageID"].isin(accepted_ids)
    ].copy()

    accepted_metadata = accepted_metadata.set_index("ImageID").loc[accepted_ids].reset_index()

    status_map = dict(zip(review["ImageID"], review["ReviewStatus"]))

    cv2.namedWindow("AVAX Accepted Negative QC", cv2.WINDOW_NORMAL)

    total = len(accepted_metadata)

    for index, row in accepted_metadata.iterrows():
        image_id = row["ImageID"]
        image_path = IMAGE_ROOT / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            status_map[image_id] = "reject_missing_or_corrupt"
            continue

        image = resize_for_display(image)
        display = add_header(image, row, index + 1, total)

        cv2.imshow("AVAX Accepted Negative QC", display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("a"), ord("A")):
                status_map[image_id] = "accepted_negative"
                break

            if key in (ord("p"), ord("P")):
                status_map[image_id] = "reject_plate_visible"
                break

            if key in (ord("x"), ord("X")):
                status_map[image_id] = "reject_other"
                break

            if key in (ord("u"), ord("U")):
                status_map[image_id] = "unsure"
                break

            if key in (ord("q"), ord("Q")):
                review["ReviewStatus"] = review["ImageID"].map(status_map)
                save_results(metadata, review)
                print_summary(review)
                cv2.destroyAllWindows()
                return

        review["ReviewStatus"] = review["ImageID"].map(status_map)
        save_results(metadata, review)

    review["ReviewStatus"] = review["ImageID"].map(status_map)
    save_results(metadata, review)
    print_summary(review)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()