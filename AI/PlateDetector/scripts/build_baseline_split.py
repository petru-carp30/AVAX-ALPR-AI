from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool"
INPUT_PATH = CANONICAL_ROOT / "metadata" / "baseline_candidate_manifest.csv"

OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "baseline_v1" / "metadata"
OUTPUT_MANIFEST = OUTPUT_ROOT / "baseline_split_manifest.csv"
POLICY_PATH = OUTPUT_ROOT / "split_policy.json"

EXPECTED_CANONICAL_IMAGES = 9299
EXPECTED_ELIGIBLE_IMAGES = 9297
MAX_SYNTHETIC_RATIO = 0.25

ROMANIAN_SPLITS = {
    "dayride_type1_001.mp4": "train",
    "dayride_type1_003.mp4": "train",
    "dayride_type1_002.mp4": "val",
    "nightride_type3_001.mp4": "test",
}


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_open_images(frame: pd.DataFrame) -> dict[str, str]:
    if frame["source_group"].nunique() != len(frame):
        raise RuntimeError("Open Images split candidates do not have one unique source group per image.")

    ordered = frame.copy()
    ordered["split_hash"] = ordered["source_group"].astype(str).map(stable_key)
    ordered = ordered.sort_values(["split_hash", "canonical_id"], kind="stable").reset_index(drop=True)

    val_count = round(len(ordered) * 0.10)
    test_count = round(len(ordered) * 0.10)
    assignments = {}

    for index, row in ordered.iterrows():
        if index < val_count:
            split = "val"
        elif index < val_count + test_count:
            split = "test"
        else:
            split = "train"
        assignments[row["canonical_id"]] = split

    return assignments


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)

    if len(frame) != EXPECTED_CANONICAL_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_CANONICAL_IMAGES} canonical records, found {len(frame)}.")

    eligible_mask = frame["baseline_status"] == "eligible_candidate"

    if int(eligible_mask.sum()) != EXPECTED_ELIGIBLE_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE_IMAGES} eligible candidates, found {int(eligible_mask.sum())}.")

    frame["avax_split"] = ""
    frame["split_status"] = "excluded"
    frame["split_exclusion_reason"] = frame["baseline_exclusion_reason"].fillna("")

    eligible = frame[eligible_mask].copy()

    romanian = eligible[eligible["source_dataset"] == "romanian_public_lp"]

    unknown_sequences = set(romanian["source_group"]) - set(ROMANIAN_SPLITS)
    if unknown_sequences:
        raise RuntimeError(f"Unknown Romanian source sequences: {sorted(unknown_sequences)}")

    for source_group, split in ROMANIAN_SPLITS.items():
        ids = romanian.loc[romanian["source_group"] == source_group, "canonical_id"]
        frame.loc[frame["canonical_id"].isin(ids), "avax_split"] = split
        frame.loc[frame["canonical_id"].isin(ids), "split_status"] = "included"

    for source_dataset in ("open_images_v7", "open_images_v7_negative_pool"):
        source_frame = eligible[eligible["source_dataset"] == source_dataset]
        assignments = assign_open_images(source_frame)

        for canonical_id, split in assignments.items():
            frame.loc[frame["canonical_id"] == canonical_id, "avax_split"] = split
            frame.loc[frame["canonical_id"] == canonical_id, "split_status"] = "included"

    kaggle_ids = eligible.loc[eligible["source_dataset"] == "kaggle_plate_license_recognition", "canonical_id"]
    frame.loc[frame["canonical_id"].isin(kaggle_ids), "avax_split"] = "train"
    frame.loc[frame["canonical_id"].isin(kaggle_ids), "split_status"] = "included"

    real_train_count = len(frame[(frame["split_status"] == "included") & (frame["avax_split"] == "train") & (frame["real_synthetic"] == "REAL")])
    max_synthetic_images = math.floor(real_train_count / 3)

    elpd = eligible[eligible["source_dataset"] == "elpd"].copy()
    elpd["split_hash"] = elpd["canonical_id"].map(stable_key)
    elpd = elpd.sort_values(["split_hash", "canonical_id"], kind="stable").reset_index(drop=True)

    synthetic_keep_count = min(len(elpd), max_synthetic_images)
    keep_elpd_ids = set(elpd.iloc[:synthetic_keep_count]["canonical_id"])
    exclude_elpd_ids = set(elpd.iloc[synthetic_keep_count:]["canonical_id"])

    frame.loc[frame["canonical_id"].isin(keep_elpd_ids), "avax_split"] = "train"
    frame.loc[frame["canonical_id"].isin(keep_elpd_ids), "split_status"] = "included"
    frame.loc[frame["canonical_id"].isin(exclude_elpd_ids), "split_status"] = "excluded"
    frame.loc[frame["canonical_id"].isin(exclude_elpd_ids), "split_exclusion_reason"] = "baseline_synthetic_cap"

    included = frame[frame["split_status"] == "included"].copy()

    if (included["avax_split"] == "").any():
        raise RuntimeError("Included samples without AVAX split found.")

    train = included[included["avax_split"] == "train"]
    val = included[included["avax_split"] == "val"]
    test = included[included["avax_split"] == "test"]

    synthetic_train = int((train["real_synthetic"] == "SYNTHETIC").sum())
    real_train = int((train["real_synthetic"] == "REAL").sum())
    synthetic_ratio = synthetic_train / len(train)

    if synthetic_ratio > MAX_SYNTHETIC_RATIO:
        raise RuntimeError(f"Synthetic training ratio exceeds limit: {synthetic_ratio:.6f}")

    if (val["real_synthetic"] != "REAL").any() or (test["real_synthetic"] != "REAL").any():
        raise RuntimeError("VAL/TEST contains synthetic samples.")

    train_only_outside_train = included[(included["allowed_split"] == "TRAIN_ONLY") & (included["avax_split"] != "train")]
    if not train_only_outside_train.empty:
        raise RuntimeError(f"TRAIN_ONLY samples outside train: {len(train_only_outside_train)}")

    included["leakage_group"] = included["source_dataset"].astype(str) + "::" + included["source_group"].astype(str)
    leakage = included.groupby("leakage_group")["avax_split"].nunique()
    leaking_groups = leakage[leakage > 1]

    if not leaking_groups.empty:
        raise RuntimeError(f"Source-group leakage detected: {len(leaking_groups)} groups.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_MANIFEST, index=False, encoding="utf-8-sig")

    for split_name in ("train", "val", "test"):
        ids = frame.loc[(frame["split_status"] == "included") & (frame["avax_split"] == split_name), "canonical_id"]
        (OUTPUT_ROOT / f"{split_name}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    policy = {
        "open_images_split": {"train": 0.80, "val": 0.10, "test": 0.10},
        "romanian_sequence_split": ROMANIAN_SPLITS,
        "kaggle": "TRAIN_ONLY",
        "elpd": "TRAIN_ONLY_WITH_SYNTHETIC_CAP",
        "max_training_synthetic_ratio": MAX_SYNTHETIC_RATIO,
        "validation_and_test_real_only": True,
        "near_duplicate_candidates_found": 0,
        "source_group_leakage_allowed": False,
    }

    POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print("\n=== AVAX BASELINE LEAKAGE-SAFE SPLIT ===")
    print(f"Included images: {len(included)}")
    print(f"Excluded images: {len(frame) - len(included)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    print("\nTraining composition:")
    print(f"  REAL: {real_train}")
    print(f"  SYNTHETIC: {synthetic_train}")
    print(f"  Synthetic ratio: {synthetic_ratio:.4%}")

    print("\nELPD:")
    print(f"  Available: {len(elpd)}")
    print(f"  Included: {synthetic_keep_count}")
    print(f"  Excluded by synthetic cap: {len(exclude_elpd_ids)}")

    print("\nRomanian sequence assignment:")
    for source_group, split in ROMANIAN_SPLITS.items():
        count = len(included[(included["source_dataset"] == "romanian_public_lp") & (included["source_group"] == source_group)])
        print(f"  {source_group}: {split.upper()} ({count} images)")

    print("\nLeakage validation:")
    print(f"  Source groups crossing splits: {len(leaking_groups)}")
    print(f"  TRAIN_ONLY outside train: {len(train_only_outside_train)}")
    print(f"  Synthetic images in VAL: {(val['real_synthetic'] == 'SYNTHETIC').sum()}")
    print(f"  Synthetic images in TEST: {(test['real_synthetic'] == 'SYNTHETIC').sum()}")

    print("\nSplit source composition:")
    for split_name, split_frame in (("TRAIN", train), ("VAL", val), ("TEST", test)):
        print(f"\n  {split_name}:")
        for source, count in split_frame["source_dataset"].value_counts().items():
            instances = int(split_frame.loc[split_frame["source_dataset"] == source, "plate_instance_count"].sum())
            print(f"    {source}: {count} images / {instances} plate instances")

    print(f"\nManifest: {OUTPUT_MANIFEST}")
    print(f"Policy: {POLICY_PATH}")
    print("RESULT: BASELINE SPLIT ASSIGNED AND LEAKAGE VALIDATED")


if __name__ == "__main__":
    main()