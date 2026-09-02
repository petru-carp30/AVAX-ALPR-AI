from materialize_and_validate_baseline_v1 import (
    BASELINE_ROOT,
    EXPECTED_SPLITS,
    load_manifest,
    preflight_sources,
    validate_root,
)


def main() -> None:
    print("=== AVAX baseline_v1 POST-MATERIALIZATION VALIDATION ===")

    included, excluded, summary = load_manifest()
    samples = preflight_sources(included)
    result = validate_root(BASELINE_ROOT, included, excluded, samples)

    print("Manifest: PASS")
    print(f"Images: {result['images']}")
    print(f"Labels: {result['labels']}")
    print(f"TRAIN: {EXPECTED_SPLITS['train']}")
    print(f"VAL: {EXPECTED_SPLITS['val']}")
    print(f"TEST: {EXPECTED_SPLITS['test']}")
    print(f"Positive images: {summary['positives']}")
    print(f"Negative images: {result['negatives']}")
    print(f"Plate instances: {result['instances']}")
    print(f"Source-group leakage: {summary['group_leakage']}")
    print(f"Exact-pixel leakage: {summary['pixel_leakage']}")
    print(f"Unresolved near-duplicate candidates: {summary['near_candidates']}")
    print("Canonical IDs/splits: PRESERVED")
    print("Materialized files vs canonical sources: BYTE-IDENTICAL")
    print("Negative labels: EMPTY BY MANIFEST CONTRACT")
    print("YOLO class IDs/bboxes: PASS")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
