from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "datasets"
    / "manifests"
    / "romanian_lp"
    / "romanian_positive_manifest.csv"
)


def parse_objects(annotation_path: Path) -> list[tuple]:
    root = ET.parse(annotation_path).getroot()
    objects = []

    for object_element in root.findall("object"):
        class_name = object_element.findtext("name", default="").strip()
        box = object_element.find("bndbox")

        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        objects.append(
            (
                class_name,
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

    return sorted(objects)


def main() -> None:
    frame = pd.read_csv(
        MANIFEST_PATH,
        dtype={"image_sha256": str},
    )

    duplicate_mask = frame["image_sha256"].duplicated(
        keep=False
    )

    duplicates = frame[
        duplicate_mask
    ].copy()

    print("\n=== ROMANIAN EXACT DUPLICATE AUDIT ===")

    if duplicates.empty:
        print("Exact duplicate image groups: 0")
        print("RESULT: NO EXACT DUPLICATES")
        return

    grouped = duplicates.groupby("image_sha256")

    print(f"Exact duplicate groups: {grouped.ngroups}")
    print(f"Images involved: {len(duplicates)}")

    for group_index, (image_hash, group) in enumerate(
        grouped,
        start=1,
    ):
        print(f"\n--- GROUP {group_index} ---")
        print(f"Image SHA256: {image_hash}")
        print(f"Images: {len(group)}")

        annotation_sets = []

        for row in group.itertuples(index=False):
            image_path = PROJECT_ROOT / row.source_image_path
            annotation_path = PROJECT_ROOT / row.source_annotation_path

            objects = parse_objects(annotation_path)
            annotation_sets.append(objects)

            print()
            print(f"Image: {row.source_image_name}")
            print(f"Upstream split: {row.source_split}")
            print(f"Sequence: {row.source_sequence}")
            print(f"Frame: {row.source_frame}")
            print(f"Plate instances: {row.plate_instance_count}")
            print(f"Annotation SHA256: {row.annotation_sha256}")
            print(f"Image exists: {image_path.exists()}")
            print(f"Annotation exists: {annotation_path.exists()}")
            print(f"Objects: {objects}")

        same_sequence = group["source_sequence"].nunique() == 1
        same_annotations = all(
            objects == annotation_sets[0]
            for objects in annotation_sets
        )

        print()
        print(f"Same source sequence: {same_sequence}")
        print(f"Same annotations: {same_annotations}")

        if same_sequence and same_annotations:
            print(
                "Assessment: exact repeated frame with matching annotations."
            )
        elif same_sequence:
            print(
                "Assessment: exact repeated frame within one sequence, "
                "but annotations differ."
            )
        else:
            print(
                "Assessment: exact image occurs across different source sequences."
            )

    print("\nPolicy:")
    print("Exact duplicate images must remain in one AVAX split.")
    print("Raw source files remain untouched.")

    print("\nRESULT: ROMANIAN EXACT DUPLICATE AUDIT COMPLETE")


if __name__ == "__main__":
    main()