from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool"
SAMPLES_PATH = CANONICAL_ROOT / "metadata" / "canonical_samples.csv"
OUTPUT_ROOT = CANONICAL_ROOT / "metadata" / "similarity"

FINGERPRINTS_PATH = OUTPUT_ROOT / "canonical_image_fingerprints.csv"
EXACT_GROUPS_PATH = OUTPUT_ROOT / "exact_duplicate_groups.csv"
NEAR_CANDIDATES_PATH = OUTPUT_ROOT / "near_duplicate_candidates.csv"

EXPECTED_IMAGES = 9299

PHASH_THRESHOLD = 8
DHASH_THRESHOLD = 10
ASPECT_RATIO_REL_THRESHOLD = 0.15


def decoded_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(rgb.width.to_bytes(4, "little"))
        digest.update(rgb.height.to_bytes(4, "little"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def compute_phash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not load image: {path}")

    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_frequency = dct[:8, :8]
    median = np.median(low_frequency.flatten()[1:])
    return bits_to_int(low_frequency > median)


def compute_dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not load image: {path}")

    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    return bits_to_int(resized[:, 1:] > resized[:, :-1])


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def relative_aspect_difference(first: float, second: float) -> float:
    return abs(first - second) / max(first, second)


class BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return

        node = self.root

        while True:
            distance = hamming_distance(value, node.value)

            if distance == 0:
                node.indices.append(index)
                return

            if distance not in node.children:
                node.children[distance] = BKNode(value, index)
                return

            node = node.children[distance]

    def query(self, value: int, max_distance: int) -> list[tuple[int, list[int]]]:
        if self.root is None:
            return []

        results = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)

            if distance <= max_distance:
                results.append((distance, node.indices))

            minimum = distance - max_distance
            maximum = distance + max_distance

            for edge_distance, child in node.children.items():
                if minimum <= edge_distance <= maximum:
                    stack.append(child)

        return results


def build_fingerprints(samples: pd.DataFrame) -> pd.DataFrame:
    records = []

    for index, row in enumerate(samples.itertuples(index=False), start=1):
        path = PROJECT_ROOT / row.canonical_image_path

        with Image.open(path) as image:
            width, height = image.size

        records.append({
            "canonical_id": row.canonical_id,
            "canonical_image_path": row.canonical_image_path,
            "source_dataset": row.source_dataset,
            "source_group": row.source_group,
            "leakage_source_group": f"{row.source_dataset}::{row.source_group}",
            "real_synthetic": row.real_synthetic,
            "allowed_split": row.allowed_split,
            "is_negative": row.is_negative,
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "pixel_sha256": decoded_pixel_hash(path),
            "phash": f"{compute_phash(path):016x}",
            "dhash": f"{compute_dhash(path):016x}",
        })

        if index % 500 == 0 or index == len(samples):
            print(f"Fingerprinted {index}/{len(samples)} images")

    return pd.DataFrame(records)


def build_exact_groups(frame: pd.DataFrame) -> pd.DataFrame:
    records = []

    for pixel_hash, group in frame.groupby("pixel_sha256"):
        if len(group) < 2:
            continue

        group_id = f"exact__{pixel_hash[:16]}"
        leakage_groups = group["leakage_source_group"].nunique()

        for row in group.itertuples(index=False):
            records.append({
                "exact_group_id": group_id,
                "pixel_sha256": pixel_hash,
                "canonical_id": row.canonical_id,
                "source_dataset": row.source_dataset,
                "source_group": row.source_group,
                "leakage_source_group": row.leakage_source_group,
                "allowed_split": row.allowed_split,
                "cross_source_group": leakage_groups > 1,
            })

    return pd.DataFrame(records)


def build_near_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    tree = BKTree()
    records = []
    phashes = [int(value, 16) for value in frame["phash"]]
    dhashes = [int(value, 16) for value in frame["dhash"]]

    for current_index, current in enumerate(frame.itertuples(index=False)):
        current_phash = phashes[current_index]
        current_dhash = dhashes[current_index]

        for phash_distance, previous_indices in tree.query(current_phash, PHASH_THRESHOLD):
            for previous_index in previous_indices:
                previous = frame.iloc[previous_index]

                if previous["leakage_source_group"] == current.leakage_source_group:
                    continue

                if previous["allowed_split"] == "TRAIN_ONLY" and current.allowed_split == "TRAIN_ONLY":
                    continue

                if previous["pixel_sha256"] == current.pixel_sha256:
                    continue

                aspect_difference = relative_aspect_difference(float(previous["aspect_ratio"]), float(current.aspect_ratio))

                if aspect_difference > ASPECT_RATIO_REL_THRESHOLD:
                    continue

                dhash_distance = hamming_distance(dhashes[previous_index], current_dhash)

                if dhash_distance > DHASH_THRESHOLD:
                    continue

                records.append({
                    "canonical_id_a": previous["canonical_id"],
                    "canonical_id_b": current.canonical_id,
                    "source_dataset_a": previous["source_dataset"],
                    "source_dataset_b": current.source_dataset,
                    "source_group_a": previous["source_group"],
                    "source_group_b": current.source_group,
                    "allowed_split_a": previous["allowed_split"],
                    "allowed_split_b": current.allowed_split,
                    "phash_distance": phash_distance,
                    "dhash_distance": dhash_distance,
                    "aspect_ratio_relative_difference": aspect_difference,
                })

        tree.add(current_phash, current_index)

        if (current_index + 1) % 500 == 0 or current_index + 1 == len(frame):
            print(f"Near-duplicate search {current_index + 1}/{len(frame)} images")

    return pd.DataFrame(records)


def main() -> None:
    samples = pd.read_csv(SAMPLES_PATH)

    if len(samples) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} canonical samples, found {len(samples)}.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Building image fingerprints...")
    fingerprints = build_fingerprints(samples)
    fingerprints.to_csv(FINGERPRINTS_PATH, index=False, encoding="utf-8-sig")

    print("\nFinding exact pixel duplicates...")
    exact_groups = build_exact_groups(fingerprints)
    exact_groups.to_csv(EXACT_GROUPS_PATH, index=False, encoding="utf-8-sig")

    print("\nFinding perceptual near-duplicate candidates...")
    near_candidates = build_near_candidates(fingerprints)
    near_candidates.to_csv(NEAR_CANDIDATES_PATH, index=False, encoding="utf-8-sig")

    exact_group_count = exact_groups["exact_group_id"].nunique() if not exact_groups.empty else 0
    exact_image_count = exact_groups["canonical_id"].nunique() if not exact_groups.empty else 0
    cross_group_exact_count = exact_groups.loc[exact_groups["cross_source_group"], "exact_group_id"].nunique() if not exact_groups.empty else 0

    print("\n=== CANONICAL SIMILARITY ANALYSIS ===")
    print(f"Canonical images: {len(fingerprints)}")
    print(f"Unique decoded pixel hashes: {fingerprints['pixel_sha256'].nunique()}")
    print(f"Leakage source groups: {fingerprints['leakage_source_group'].nunique()}")

    print("\nExact duplicates:")
    print(f"  Exact duplicate groups: {exact_group_count}")
    print(f"  Images involved: {exact_image_count}")
    print(f"  Groups crossing source-group boundaries: {cross_group_exact_count}")

    print("\nNear-duplicate candidates:")
    print(f"  Candidate pairs: {len(near_candidates)}")
    print(f"  pHash threshold: <= {PHASH_THRESHOLD}")
    print(f"  dHash threshold: <= {DHASH_THRESHOLD}")
    print(f"  Aspect ratio relative difference: <= {ASPECT_RATIO_REL_THRESHOLD:.2f}")

    if not near_candidates.empty:
        print("\nCandidate source pairs:")
        pair_counts = Counter(tuple(sorted((row.source_dataset_a, row.source_dataset_b))) for row in near_candidates.itertuples(index=False))
        for pair, count in pair_counts.most_common():
            print(f"  {pair[0]} <-> {pair[1]}: {count}")

        print("\nDistance distribution:")
        for distance, count in sorted(near_candidates["phash_distance"].value_counts().items()):
            print(f"  pHash {distance}: {count}")

    print("\nOutputs:")
    print(f"  Fingerprints: {FINGERPRINTS_PATH}")
    print(f"  Exact groups: {EXACT_GROUPS_PATH}")
    print(f"  Near candidates: {NEAR_CANDIDATES_PATH}")
    print("\nRESULT: SIMILARITY CANDIDATE ANALYSIS COMPLETE")
    print("NOTE: Near-duplicate candidates are NOT yet automatically merged into leakage groups.")


if __name__ == "__main__":
    main()