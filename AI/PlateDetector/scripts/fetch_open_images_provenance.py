from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

OPEN_IMAGES_SOURCES = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}

LOCAL_SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")

PROVENANCE_FIELDS = (
    "OriginalURL",
    "OriginalLandingURL",
    "License",
    "AuthorProfileURL",
    "Author",
    "Title",
)


def configure_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile local Kaggle Open Images files with official Open Images metadata."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def collect_local_images(
    dataset_root: Path,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    entries: list[dict[str, str]] = []
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)

    for local_split in LOCAL_SPLITS:
        image_dir = dataset_root / "images" / local_split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            image_id = path.stem.lower()

            if not IMAGE_ID_PATTERN.fullmatch(image_id):
                raise ValueError(f"Invalid Open Images-style filename: {path.name}")

            entry = {
                "ImageID": image_id,
                "LocalKaggleSplit": local_split,
                "LocalFileName": path.name,
            }

            entries.append(entry)
            by_id[image_id].append(entry)

    if not entries:
        raise RuntimeError(f"No local images found under: {dataset_root / 'images'}")

    return entries, by_id


def scan_official_source(
    upstream_split: str,
    url: str,
    wanted_ids: set[str],
    matches: dict[str, dict[str, dict[str, str]]],
    timeout: int,
    retries: int,
) -> None:
    print(f"Fetching official metadata for upstream split '{upstream_split}'...")

    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url,
                headers={"User-Agent": "AVAX-ALPR-Provenance-Audit/1.0"},
            )

            with urlopen(request, timeout=timeout) as response:
                text_stream = io.TextIOWrapper(
                    response,
                    encoding="utf-8-sig",
                    newline="",
                )
                reader = csv.DictReader(text_stream)

                required_columns = {"ImageID", *PROVENANCE_FIELDS}
                missing_columns = required_columns.difference(reader.fieldnames or [])

                if missing_columns:
                    raise RuntimeError(
                        f"Official metadata for '{upstream_split}' is missing columns: "
                        f"{', '.join(sorted(missing_columns))}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    image_id = row["ImageID"].strip().lower()

                    if image_id in wanted_ids:
                        matches[image_id][upstream_split] = {
                            field: row.get(field, "")
                            for field in PROVENANCE_FIELDS
                        }

                    if row_number % 1_000_000 == 0:
                        matched_here = sum(
                            1
                            for source_matches in matches.values()
                            if upstream_split in source_matches
                        )

                        print(
                            f"  scanned={row_number - 1:,} rows, "
                            f"matched_local_ids={matched_here:,}"
                        )

            matched_here = sum(
                1
                for source_matches in matches.values()
                if upstream_split in source_matches
            )

            print(
                f"  matched local IDs in '{upstream_split}': "
                f"{matched_here:,}"
            )
            return

        except (URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to read official metadata for "
                    f"'{upstream_split}' after {retries} attempts."
                ) from exc

            delay_seconds = 2 ** attempt

            print(
                f"  network error on attempt {attempt}/{retries}: {exc}. "
                f"Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    local_entries: list[dict[str, str]],
    local_by_id: dict[str, list[dict[str, str]]],
    matches: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = output_dir / "open_images_provenance.csv"
    missing_path = output_dir / "missing_open_images_ids.csv"
    duplicates_path = output_dir / "duplicate_open_images_ids.csv"
    summary_path = output_dir / "open_images_provenance_summary.json"

    provenance_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, str]] = []

    for local_entry in local_entries:
        image_id = local_entry["ImageID"]
        source_matches = matches.get(image_id, {})

        if not source_matches:
            missing_rows.append(local_entry)
            continue

        for upstream_split in OPEN_IMAGES_SOURCES:
            if upstream_split not in source_matches:
                continue

            provenance_rows.append(
                {
                    **local_entry,
                    "OpenImagesSplit": upstream_split,
                    **source_matches[upstream_split],
                    "SourceMetadataURL": OPEN_IMAGES_SOURCES[upstream_split],
                }
            )

    for image_id, source_matches in sorted(matches.items()):
        if len(source_matches) <= 1:
            continue

        duplicate_rows.append(
            {
                "ImageID": image_id,
                "LocalKaggleSplits": "|".join(
                    sorted(
                        {
                            entry["LocalKaggleSplit"]
                            for entry in local_by_id[image_id]
                        }
                    )
                ),
                "OpenImagesSplits": "|".join(
                    split
                    for split in OPEN_IMAGES_SOURCES
                    if split in source_matches
                ),
            }
        )

    provenance_fields = [
        "ImageID",
        "LocalKaggleSplit",
        "LocalFileName",
        "OpenImagesSplit",
        *PROVENANCE_FIELDS,
        "SourceMetadataURL",
    ]

    write_csv(
        provenance_path,
        provenance_fields,
        provenance_rows,
    )

    write_csv(
        missing_path,
        ["ImageID", "LocalKaggleSplit", "LocalFileName"],
        missing_rows,
    )

    write_csv(
        duplicates_path,
        ["ImageID", "LocalKaggleSplits", "OpenImagesSplits"],
        duplicate_rows,
    )

    matched_local_images = sum(
        1
        for entry in local_entries
        if matches.get(entry["ImageID"])
    )

    matches_by_upstream_split = {
        split: sum(
            1
            for entry in local_entries
            if split in matches.get(entry["ImageID"], {})
        )
        for split in OPEN_IMAGES_SOURCES
    }

    summary = {
        "expected_local_images": len(local_entries),
        "unique_local_image_ids": len(local_by_id),
        "matched_official_metadata": matched_local_images,
        "matches_by_upstream_split": matches_by_upstream_split,
        "missing_provenance": len(missing_rows),
        "duplicate_ids_across_upstream_sources": len(duplicate_rows),
        "provenance_csv": str(provenance_path),
        "missing_ids_csv": str(missing_path),
        "duplicate_ids_csv": str(duplicates_path),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary


def print_summary(summary: dict[str, object]) -> None:
    print("\n=== PROVENANCE SUMMARY ===")
    print(f"Expected local images: {summary['expected_local_images']}")
    print(f"Unique local ImageIDs: {summary['unique_local_image_ids']}")
    print(f"Matched official metadata: {summary['matched_official_metadata']}")
    print("Matches by upstream split:")

    for split, count in summary["matches_by_upstream_split"].items():
        print(f"  {split}: {count}")

    print(f"Missing provenance: {summary['missing_provenance']}")

    print(
        "Duplicate IDs across upstream metadata sources: "
        f"{summary['duplicate_ids_across_upstream_sources']}"
    )

    print(f"Provenance CSV: {summary['provenance_csv']}")
    print(f"Missing IDs CSV: {summary['missing_ids_csv']}")
    print(f"Duplicate IDs CSV: {summary['duplicate_ids_csv']}")


def main() -> None:
    configure_csv_field_limit()
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    local_entries, local_by_id = collect_local_images(dataset_root)
    wanted_ids = set(local_by_id)

    matches: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    print(f"Expected local images: {len(local_entries):,}")
    print(f"Unique local ImageIDs: {len(wanted_ids):,}")

    for upstream_split, url in OPEN_IMAGES_SOURCES.items():
        scan_official_source(
            upstream_split=upstream_split,
            url=url,
            wanted_ids=wanted_ids,
            matches=matches,
            timeout=args.timeout,
            retries=args.retries,
        )

    summary = write_outputs(
        output_dir=output_dir,
        local_entries=local_entries,
        local_by_id=local_by_id,
        matches=matches,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()