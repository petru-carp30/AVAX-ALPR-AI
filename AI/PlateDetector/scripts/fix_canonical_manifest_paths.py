from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool" / "metadata" / "canonical_samples.csv"

OLD_PREFIX = "AI/PlateDetector/datasets/derived/canonical_pool_build/"
NEW_PREFIX = "AI/PlateDetector/datasets/derived/canonical_pool/"


def main() -> None:
    frame = pd.read_csv(SAMPLES_PATH)

    frame["canonical_image_path"] = frame["canonical_image_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)
    frame["canonical_label_path"] = frame["canonical_label_path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

    missing_images = [path for path in frame["canonical_image_path"] if not (PROJECT_ROOT / path).exists()]
    missing_labels = [path for path in frame["canonical_label_path"] if not (PROJECT_ROOT / path).exists()]

    if missing_images:
        raise RuntimeError(f"Missing canonical images after path repair: {len(missing_images)}. Examples: {missing_images[:10]}")

    if missing_labels:
        raise RuntimeError(f"Missing canonical labels after path repair: {len(missing_labels)}. Examples: {missing_labels[:10]}")

    if len(frame) != 9299:
        raise RuntimeError(f"Expected 9299 canonical samples, found {len(frame)}.")

    frame.to_csv(SAMPLES_PATH, index=False, encoding="utf-8-sig")

    print("\n=== CANONICAL MANIFEST PATH REPAIR ===")
    print(f"Samples: {len(frame)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")
    print("canonical_pool_build references remaining:", frame["canonical_image_path"].str.contains("canonical_pool_build", regex=False).sum())
    print("\nRESULT: CANONICAL MANIFEST PATHS REPAIRED")


if __name__ == "__main__":
    main()