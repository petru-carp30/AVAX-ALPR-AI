from collections import Counter
from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROVENANCE_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "provenance" / "open_images_lp_kaggle" / "open_images_provenance.csv"

FIELDS_TO_AUDIT = (
    "ImageID",
    "OriginalURL",
    "OriginalLandingURL",
    "License",
    "AuthorProfileURL",
    "Author",
    "Title",
)


def is_missing(value: str | None) -> bool:
    return value is None or not value.strip()


def main() -> None:
    with PROVENANCE_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print("=== OPEN IMAGES PROVENANCE METADATA AUDIT ===")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {fieldnames}")

    print()
    print("=== FIELD COMPLETENESS ===")

    for field in FIELDS_TO_AUDIT:
        if field not in fieldnames:
            print(f"{field}: COLUMN NOT PRESENT")
            continue

        missing = sum(1 for row in rows if is_missing(row.get(field)))
        present = len(rows) - missing

        print(f"{field}: present={present}, missing={missing}")

    if "License" in fieldnames:
        license_counts = Counter(
            row["License"].strip() if not is_missing(row.get("License")) else "<MISSING>"
            for row in rows
        )

        print()
        print("=== LICENSE DISTRIBUTION ===")

        for license_value, count in license_counts.most_common():
            print(f"{count}: {license_value}")

    image_ids = [row.get("ImageID", "").strip() for row in rows]
    unique_image_ids = set(image_ids)

    print()
    print("=== ID CONSISTENCY ===")
    print(f"ImageID rows: {len(image_ids)}")
    print(f"Unique ImageIDs: {len(unique_image_ids)}")
    print(f"Duplicate ImageID rows: {len(image_ids) - len(unique_image_ids)}")

    if "Author" in fieldnames:
        unique_authors = {
            row["Author"].strip()
            for row in rows
            if not is_missing(row.get("Author"))
        }

        print()
        print("=== AUTHOR SUMMARY ===")
        print(f"Unique non-empty authors: {len(unique_authors)}")


if __name__ == "__main__":
    main()