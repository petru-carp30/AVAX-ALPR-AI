from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import postprocess

from evaluate_val_thresholds import (
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_VAL_IMAGES,
    EXPECTED_VAL_NEGATIVES,
    EXP_PATH,
    MATCH_IOU_THRESHOLD,
    PROJECT_ROOT,
    box_iou,
    calculate_sha256,
    get_ground_truth_boxes,
)


OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "evaluation" / "val_visual_review"
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.csv"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"

BATCH_SIZE = 32
NMS_THRESHOLD = 0.45

CANDIDATES = {
    "recall": 0.175,
    "balanced": 0.225,
    "f1": 0.375,
}

CATEGORY_NAMES = [
    "recovered_0175_vs_0375",
    "recovered_0225_vs_0375",
    "persistent_fn_0175",
    "extra_fp_0225_vs_0375",
    "negative_fp",
    "multi_plate_error",
]

PANEL_MAX_WIDTH = 700


def tensor_checksum(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cache_checksums(samples: list[dict]) -> list[str]:
    return [tensor_checksum(sample["raw_output"]) for sample in samples]


def detailed_match(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict:
    matched_ground_truth = set()
    matched_predictions = set()
    matches = []
    false_positive_indices = []
    duplicate_indices = []

    if predicted_boxes.numel() == 0:
        return {
            "matches": [],
            "matched_gt": set(),
            "matched_predictions": set(),
            "fp_indices": [],
            "duplicate_indices": [],
            "fn_indices": list(range(ground_truth.shape[0])),
        }

    if ground_truth.numel() == 0:
        return {
            "matches": [],
            "matched_gt": set(),
            "matched_predictions": set(),
            "fp_indices": list(range(predicted_boxes.shape[0])),
            "duplicate_indices": [],
            "fn_indices": [],
        }

    order = torch.argsort(predicted_scores, descending=True)
    overlaps = box_iou(predicted_boxes, ground_truth)

    for prediction_index_tensor in order:
        prediction_index = int(prediction_index_tensor.item())
        best_iou, best_ground_truth_index_tensor = overlaps[prediction_index].max(dim=0)

        if best_iou.item() < MATCH_IOU_THRESHOLD:
            false_positive_indices.append(prediction_index)
            continue

        ground_truth_index = int(best_ground_truth_index_tensor.item())

        if ground_truth_index in matched_ground_truth:
            false_positive_indices.append(prediction_index)
            duplicate_indices.append(prediction_index)
            continue

        matched_ground_truth.add(ground_truth_index)
        matched_predictions.add(prediction_index)

        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": ground_truth_index,
                "iou": float(best_iou.item()),
            }
        )

    false_negative_indices = [
        index
        for index in range(ground_truth.shape[0])
        if index not in matched_ground_truth
    ]

    return {
        "matches": matches,
        "matched_gt": matched_ground_truth,
        "matched_predictions": matched_predictions,
        "fp_indices": false_positive_indices,
        "duplicate_indices": duplicate_indices,
        "fn_indices": false_negative_indices,
    }


def evaluate_candidate(sample: dict, dataset, exp, confidence_threshold: float) -> dict:
    raw_output = sample["raw_output"].unsqueeze(0).detach().clone()

    processed = postprocess(
        raw_output,
        num_classes=1,
        conf_thre=confidence_threshold,
        nms_thre=NMS_THRESHOLD,
        class_agnostic=False,
    )[0]

    if processed is None:
        predicted_boxes = torch.empty((0, 4), dtype=torch.float32)
        predicted_scores = torch.empty((0,), dtype=torch.float32)
    else:
        predicted_boxes = processed[:, :4].clone().cpu()
        predicted_scores = (processed[:, 4] * processed[:, 5]).clone().cpu()

        scale = min(
            exp.test_size[0] / float(sample["image_height"]),
            exp.test_size[1] / float(sample["image_width"]),
        )

        predicted_boxes /= scale

    ground_truth = get_ground_truth_boxes(dataset, sample["image_id"])

    match = detailed_match(
        predicted_boxes,
        predicted_scores,
        ground_truth,
    )

    return {
        "confidence_threshold": confidence_threshold,
        "boxes": predicted_boxes,
        "scores": predicted_scores,
        "ground_truth": ground_truth,
        "match": match,
    }


def draw_box(
    image,
    box,
    color,
    label: str,
    thickness: int = 2,
):
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    text_y = max(y1 - 7, 18)

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def create_panel(
    original_image,
    candidate_name: str,
    candidate_result: dict,
):
    image = original_image.copy()

    ground_truth = candidate_result["ground_truth"]
    predicted_boxes = candidate_result["boxes"]
    predicted_scores = candidate_result["scores"]
    match = candidate_result["match"]

    for ground_truth_index, box in enumerate(ground_truth):
        if ground_truth_index in match["matched_gt"]:
            color = (0, 200, 0)
            label = f"GT#{ground_truth_index} MATCH"
        else:
            color = (255, 0, 255)
            label = f"GT#{ground_truth_index} FN"

        draw_box(
            image,
            box,
            color,
            label,
            thickness=3,
        )

    for prediction_index, box in enumerate(predicted_boxes):
        score = float(predicted_scores[prediction_index].item())

        if prediction_index in match["matched_predictions"]:
            color = (255, 200, 0)
            label = f"TP {score:.3f}"
        else:
            color = (0, 0, 255)
            label = f"FP {score:.3f}"

        draw_box(
            image,
            box,
            color,
            label,
            thickness=2,
        )

    header = (
        f"{candidate_name} "
        f"conf={candidate_result['confidence_threshold']:.3f} "
        f"NMS={NMS_THRESHOLD:.2f} "
        f"TP={len(match['matches'])} "
        f"FP={len(match['fp_indices'])} "
        f"FN={len(match['fn_indices'])}"
    )

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        header,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if image.shape[1] > PANEL_MAX_WIDTH:
        scale = PANEL_MAX_WIDTH / image.shape[1]

        image = cv2.resize(
            image,
            (
                int(round(image.shape[1] * scale)),
                int(round(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


def stack_panels(panels: list) -> object:
    max_height = max(panel.shape[0] for panel in panels)

    resized_panels = []

    for panel in panels:
        if panel.shape[0] == max_height:
            resized_panels.append(panel)
            continue

        scale = max_height / panel.shape[0]

        resized = cv2.resize(
            panel,
            (
                int(round(panel.shape[1] * scale)),
                max_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_panels.append(resized)

    return cv2.hconcat(resized_panels)


def save_review_image(
    category: str,
    image_id: int,
    file_name: str,
    original_image,
    candidate_results: dict,
) -> Path:
    category_root = OUTPUT_ROOT / category
    category_root.mkdir(parents=True, exist_ok=True)

    panels = [
        create_panel(
            original_image,
            candidate_name,
            candidate_results[candidate_name],
        )
        for candidate_name in ["recall", "balanced", "f1"]
    ]

    comparison = stack_panels(panels)

    safe_name = Path(file_name).stem
    output_path = category_root / f"{image_id}_{safe_name}.jpg"

    success = cv2.imwrite(str(output_path), comparison)

    if not success:
        raise RuntimeError(f"Failed to write review image: {output_path}")

    return output_path


def main() -> None:
    checkpoint_hash = calculate_sha256(CHECKPOINT_PATH)

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: {checkpoint_hash}"
        )

    exp = get_exp(str(EXP_PATH), None)

    if exp.val_ann != "instances_val.json":
        raise RuntimeError(
            f"Unexpected VAL annotation: {exp.val_ann}"
        )

    if exp.test_ann != "__TEST_BLOCKED_DO_NOT_USE__.json":
        raise RuntimeError(
            "TEST protection contract is not active"
        )

    dataset = exp.get_eval_dataset(testdev=False)

    if len(dataset) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_IMAGES} VAL images, found {len(dataset)}"
        )

    negative_count = sum(
        1
        for image_id in dataset.ids
        if len(dataset.coco.imgToAnns.get(int(image_id), [])) == 0
    )

    if negative_count != EXPECTED_VAL_NEGATIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_NEGATIVES} negatives, found {negative_count}"
        )

    loader = exp.get_eval_loader(
        batch_size=BATCH_SIZE,
        is_distributed=False,
        testdev=False,
    )

    model = exp.get_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.cuda()
    model.eval()

    cached_predictions = []

    print("Running one forward pass over VAL...")

    for images, _, info_images, image_ids in tqdm(loader):
        images = images.cuda(
            non_blocking=True,
        ).float()

        with torch.no_grad():
            outputs = model(images)

        for index in range(outputs.shape[0]):
            cached_predictions.append(
                {
                    "image_id": int(image_ids[index]),
                    "image_height": int(info_images[0][index]),
                    "image_width": int(info_images[1][index]),
                    "raw_output": outputs[index].detach().cpu(),
                }
            )

    if len(cached_predictions) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_IMAGES} cached predictions, "
            f"found {len(cached_predictions)}"
        )

    initial_checksums = cache_checksums(cached_predictions)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_counts = {
        category: 0
        for category in CATEGORY_NAMES
    }

    manifest_rows = []

    id_to_index = {
        int(image_id): index
        for index, image_id in enumerate(dataset.ids)
    }

    print("Building visual error review...")

    for sample in tqdm(cached_predictions):
        image_id = sample["image_id"]
        dataset_index = id_to_index[image_id]

        file_name = dataset.annotations[dataset_index][3]
        original_image = dataset.load_image(dataset_index)

        candidate_results = {
            candidate_name: evaluate_candidate(
                sample,
                dataset,
                exp,
                confidence_threshold,
            )
            for candidate_name, confidence_threshold in CANDIDATES.items()
        }

        recall_match = candidate_results["recall"]["match"]
        balanced_match = candidate_results["balanced"]["match"]
        f1_match = candidate_results["f1"]["match"]

        ground_truth_count = candidate_results["recall"]["ground_truth"].shape[0]

        categories = []

        recovered_recall = (
            recall_match["matched_gt"]
            - f1_match["matched_gt"]
        )

        recovered_balanced = (
            balanced_match["matched_gt"]
            - f1_match["matched_gt"]
        )

        if recovered_recall:
            categories.append("recovered_0175_vs_0375")

        if recovered_balanced:
            categories.append("recovered_0225_vs_0375")

        if recall_match["fn_indices"]:
            categories.append("persistent_fn_0175")

        if (
            ground_truth_count > 0
            and len(balanced_match["fp_indices"])
            > len(f1_match["fp_indices"])
        ):
            categories.append("extra_fp_0225_vs_0375")

        if (
            ground_truth_count == 0
            and len(recall_match["fp_indices"]) > 0
        ):
            categories.append("negative_fp")

        if (
            ground_truth_count >= 2
            and (
                recall_match["fn_indices"]
                or recall_match["fp_indices"]
                or balanced_match["fn_indices"]
                or balanced_match["fp_indices"]
                or f1_match["fn_indices"]
                or f1_match["fp_indices"]
            )
        ):
            categories.append("multi_plate_error")

        if not categories:
            continue

        saved_paths = {}

        for category in categories:
            output_path = save_review_image(
                category,
                image_id,
                file_name,
                original_image,
                candidate_results,
            )

            saved_paths[category] = str(
                output_path.relative_to(PROJECT_ROOT)
            )

            category_counts[category] += 1

        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "ground_truth_count": ground_truth_count,
                "categories": "|".join(categories),
                "recall_tp": len(recall_match["matches"]),
                "recall_fp": len(recall_match["fp_indices"]),
                "recall_fn": len(recall_match["fn_indices"]),
                "balanced_tp": len(balanced_match["matches"]),
                "balanced_fp": len(balanced_match["fp_indices"]),
                "balanced_fn": len(balanced_match["fn_indices"]),
                "f1_tp": len(f1_match["matches"]),
                "f1_fp": len(f1_match["fp_indices"]),
                "f1_fn": len(f1_match["fn_indices"]),
                "recovered_gt_0175_vs_0375": "|".join(
                    str(index)
                    for index in sorted(recovered_recall)
                ),
                "recovered_gt_0225_vs_0375": "|".join(
                    str(index)
                    for index in sorted(recovered_balanced)
                ),
                "review_paths": "|".join(
                    f"{category}:{path}"
                    for category, path in saved_paths.items()
                ),
            }
        )

    final_checksums = cache_checksums(cached_predictions)

    if final_checksums != initial_checksums:
        raise RuntimeError(
            "Cached raw predictions changed during visual review."
        )

    fieldnames = [
        "image_id",
        "file_name",
        "ground_truth_count",
        "categories",
        "recall_tp",
        "recall_fp",
        "recall_fn",
        "balanced_tp",
        "balanced_fp",
        "balanced_fn",
        "f1_tp",
        "f1_fp",
        "f1_fn",
        "recovered_gt_0175_vs_0375",
        "recovered_gt_0225_vs_0375",
        "review_paths",
    ]

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": 85,
        "val_images": EXPECTED_VAL_IMAGES,
        "val_negative_images": EXPECTED_VAL_NEGATIVES,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "candidates": CANDIDATES,
        "reviewed_unique_images": len(manifest_rows),
        "category_image_counts": category_counts,
        "cached_predictions_unchanged": True,
        "test_accessed": False,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n=== VAL VISUAL REVIEW GENERATED ===")
    print(f"Checkpoint SHA256: {checkpoint_hash}")
    print(f"VAL images: {EXPECTED_VAL_IMAGES}")
    print(f"VAL negatives: {EXPECTED_VAL_NEGATIVES}")
    print(f"NMS: {NMS_THRESHOLD}")
    print(f"Candidates: {CANDIDATES}")
    print(f"Unique review images: {len(manifest_rows)}")

    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("Cached raw predictions: UNCHANGED")
    print("TEST accessed: NO")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()