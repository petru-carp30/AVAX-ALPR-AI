from __future__ import annotations

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

TARGET_ACCEPTED = 800
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 900


def resize_for_display(image):
    height, width = image.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0,
    )

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_information(image, row, accepted_count, audited_count):
    display = image.copy()

    lines = [
        f"ImageID: {row['ImageID']}",
        f"Category: {row['CandidateCategory']}",
        f"Matched: {row['MatchedClasses']}",
        f"Audited: {audited_count} | Accepted: {accepted_count}/{TARGET_ACCEPTED}",
        "A = accept negative | P = reject: plate visible | X = reject other",
        "U = unsure/review later | Q = save and quit",
    ]

    y = 30

    for line in lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return display


def save_review(metadata, review):
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

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

    accepted.to_csv(
        ACCEPTED_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(metadata, review):
    audited = review[
        review["ReviewStatus"].notna()
        & (review["ReviewStatus"] != "")
    ]

    accepted = audited[
        audited["ReviewStatus"] == "accepted_negative"
    ]

    print("\n=== OPEN IMAGES NEGATIVE VISUAL AUDIT ===")
    print(f"Total candidates: {len(metadata)}")
    print(f"Audited: {len(audited)}")
    print(f"Accepted negatives: {len(accepted)}")

    print("\nReview results:")
    for status, count in audited["ReviewStatus"].value_counts().items():
        print(f"  {status}: {count}")

    if not accepted.empty:
        accepted_metadata = metadata.merge(
            accepted[["ImageID"]],
            on="ImageID",
            how="inner",
        )

        print("\nAccepted category distribution:")
        for category, count in accepted_metadata["CandidateCategory"].value_counts().items():
            print(f"  {category}: {count}")

        print("\nAccepted source distribution:")
        for split, count in accepted_metadata["SourceSplit"].value_counts().items():
            print(f"  {split}: {count}")

        print("\nAccepted license distribution:")
        for license_value, count in accepted_metadata["License"].fillna("<missing>").value_counts().items():
            print(f"  {license_value}: {count}")


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Candidate metadata not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={"ImageID": str},
    )

    if REVIEW_PATH.exists():
        review = pd.read_csv(
            REVIEW_PATH,
            dtype={"ImageID": str},
        )
    else:
        review = pd.DataFrame(
            {
                "ImageID": metadata["ImageID"],
                "ReviewStatus": "",
            }
        )

    review_status = dict(
        zip(review["ImageID"], review["ReviewStatus"].fillna(""))
    )

    accepted_count = sum(
        status == "accepted_negative"
        for status in review_status.values()
    )

    audited_count = sum(
        bool(status)
        for status in review_status.values()
    )

    cv2.namedWindow(
        "AVAX Open Images Negative Audit",
        cv2.WINDOW_NORMAL,
    )

    for _, row in metadata.iterrows():
        image_id = row["ImageID"]

        if review_status.get(image_id):
            continue

        if accepted_count >= TARGET_ACCEPTED:
            print(f"\nTarget of {TARGET_ACCEPTED} accepted negatives reached.")
            break

        image_path = IMAGE_ROOT / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            review_status[image_id] = "reject_missing_or_corrupt"
            audited_count += 1
            continue

        image = resize_for_display(image)
        display = draw_information(
            image,
            row,
            accepted_count,
            audited_count,
        )

        cv2.imshow(
            "AVAX Open Images Negative Audit",
            display,
        )

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("a"), ord("A")):
                review_status[image_id] = "accepted_negative"
                accepted_count += 1
                audited_count += 1
                break

            if key in (ord("p"), ord("P")):
                review_status[image_id] = "reject_plate_visible"
                audited_count += 1
                break

            if key in (ord("x"), ord("X")):
                review_status[image_id] = "reject_other"
                audited_count += 1
                break

            if key in (ord("u"), ord("U")):
                review_status[image_id] = "unsure"
                audited_count += 1
                break

            if key in (ord("q"), ord("Q")):
                review["ReviewStatus"] = review["ImageID"].map(review_status)
                save_review(metadata, review)
                print_summary(metadata, review)
                cv2.destroyAllWindows()
                return

        review["ReviewStatus"] = review["ImageID"].map(review_status)
        save_review(metadata, review)

    review["ReviewStatus"] = review["ImageID"].map(review_status)
    save_review(metadata, review)
    print_summary(metadata, review)

    cv2.destroyAllWindows()

    if accepted_count < 500:
        print("\nRESULT: NEGATIVE POOL NOT SUFFICIENT")
        print("Fewer than 500 audited real negatives were accepted.")
    elif accepted_count <= 1000:
        print("\nRESULT: NEGATIVE POOL TARGET SATISFIED")
    else:
        print("\nRESULT: REVIEW REQUIRED")
        print("Accepted count exceeds Master baseline target.")


if __name__ == "__main__":
    main()