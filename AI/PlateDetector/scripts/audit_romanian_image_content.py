from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "datasets"
    / "audits"
    / "romanian_lp"
)

OUTPUT_PATH = OUTPUT_ROOT / "romanian_image_content_audit.csv"

BLACK_MAX_THRESHOLD = 2
NEAR_UNIFORM_STD_THRESHOLD = 1.0


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8)

    minimum = int(pixels.min())
    maximum = int(pixels.max())
    mean = float(pixels.mean())
    std = float(pixels.std())

    is_all_black = maximum == 0
    is_near_black = maximum <= BLACK_MAX_THRESHOLD
    is_near_uniform = std <= NEAR_UNIFORM_STD_THRESHOLD

    return {
        "pixel_min": minimum,
        "pixel_max": maximum,
        "pixel_mean": mean,
        "pixel_std": std,
        "is_all_black": is_all_black,
        "is_near_black": is_near_black,
        "is_near_uniform": is_near_uniform,
    }


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    records = []

    for index, row in manifest.iterrows():
        image_path = PROJECT_ROOT / row["source_image_path"]

        result = inspect_image(image_path)

        records.append({
            "source_image_name": row["source_image_name"],
            "source_sequence": row["source_sequence"],
            "source_frame": row["source_frame"],
            "source_split": row["source_split"],
            "plate_instance_count": row["plate_instance_count"],
            **result,
        })

        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(manifest)} images")

    audit = pd.DataFrame(records)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    all_black = audit[audit["is_all_black"]]
    near_black = audit[audit["is_near_black"]]
    near_uniform = audit[audit["is_near_uniform"]]

    print("\n=== ROMANIAN IMAGE CONTENT AUDIT ===")
    print(f"Images audited: {len(audit)}")
    print(f"All-black images: {len(all_black)}")
    print(f"Near-black images: {len(near_black)}")
    print(f"Near-uniform images: {len(near_uniform)}")

    if not all_black.empty:
        print("\nAll-black images:")
        for row in all_black.itertuples(index=False):
            print(
                f"  {row.source_image_name} | "
                f"instances={row.plate_instance_count} | "
                f"min={row.pixel_min} max={row.pixel_max} "
                f"mean={row.pixel_mean:.4f} std={row.pixel_std:.4f}"
            )

    suspicious = audit[
        audit["is_near_uniform"]
        & ~audit["is_all_black"]
    ]

    if not suspicious.empty:
        print("\nOther near-uniform images requiring review:")
        for row in suspicious.itertuples(index=False):
            print(
                f"  {row.source_image_name} | "
                f"instances={row.plate_instance_count} | "
                f"min={row.pixel_min} max={row.pixel_max} "
                f"mean={row.pixel_mean:.4f} std={row.pixel_std:.4f}"
            )

    print(f"\nAudit CSV: {OUTPUT_PATH}")

    if len(all_black) == 2:
        expected = {
            "dayride_type1_001.mp4#t=558.jpg",
            "dayride_type1_001.mp4#t=809.jpg",
        }

        actual = set(all_black["source_image_name"])

        if actual == expected:
            print("\nRESULT: TWO KNOWN BLANK SOURCE IMAGES CONFIRMED")
        else:
            print("\nRESULT: TWO BLANK IMAGES FOUND, BUT IDENTITIES REQUIRE REVIEW")
    elif len(all_black) == 0:
        print("\nRESULT: NO ALL-BLACK IMAGES FOUND")
    else:
        print("\nRESULT: ADDITIONAL BLANK SOURCE IMAGES REQUIRE REVIEW")


if __name__ == "__main__":
    main()