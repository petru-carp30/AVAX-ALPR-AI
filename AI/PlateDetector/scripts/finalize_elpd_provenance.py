from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets"

PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "elpd"
PROVENANCE_PATH = PROVENANCE_ROOT / "elpd_provenance.csv"

MANIFEST_ROOT = DATASETS_ROOT / "manifests" / "elpd"
MANIFEST_PATHS = [
    MANIFEST_ROOT / "elpd_filter_manifest.csv",
    MANIFEST_ROOT / "elpd_positive_manifest.csv",
    MANIFEST_ROOT / "elpd_positive_boxes.csv",
]

SOURCE_TITLE = "European License Plate Dataset (ELPD) 2329 Images"
SOURCE_URL = "https://www.kaggle.com/datasets/tielemarvin/european-license-plate-dataset-elpd"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_REFERENCE = "AI/PlateDetector/datasets/provenance/elpd/elpd_provenance.csv"


def main() -> None:
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)

    provenance = pd.DataFrame([{
        "source_dataset": "elpd",
        "source_title": SOURCE_TITLE,
        "source_url": SOURCE_URL,
        "license_name": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "real_synthetic": "SYNTHETIC",
        "allowed_split": "TRAIN_ONLY",
        "raw_source_modified": False,
    }])

    provenance.to_csv(PROVENANCE_PATH, index=False, encoding="utf-8-sig")

    for manifest_path in MANIFEST_PATHS:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        frame = pd.read_csv(manifest_path)
        frame["source_dataset_url"] = SOURCE_URL
        frame["license_name"] = LICENSE_NAME
        frame["license_url"] = LICENSE_URL
        frame["license_reference"] = LICENSE_REFERENCE
        frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\n=== ELPD PROVENANCE FINALIZATION ===")
    print(f"Source: {SOURCE_TITLE}")
    print(f"Source URL: {SOURCE_URL}")
    print(f"License: {LICENSE_NAME}")
    print(f"License URL: {LICENSE_URL}")
    print(f"Provenance file: {PROVENANCE_PATH}")
    print(f"Updated manifests: {len(MANIFEST_PATHS)}")
    print("Raw source remains untouched: YES")
    print("\nRESULT: ELPD PROVENANCE READY")


if __name__ == "__main__":
    main()