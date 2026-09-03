from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXP_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "experiments" / "yolox_nano_baseline_v1.py"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "runs"
    / "ai_wp_001_yolox_nano_baseline_v1_416_fp32_bs32"
    / "best_ckpt.pth"
)

OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "evaluation" / "val_threshold_analysis"
SUMMARY_PATH = OUTPUT_ROOT / "threshold_sweep.csv"

EXPECTED_CHECKPOINT_SHA256 = "50B49FADBB9F64F415752E878E60264426274FD6B869F89961EF0978E9CE6CBF"
EXPECTED_VAL_IMAGES = 625
EXPECTED_VAL_NEGATIVES = 50

CONFIDENCE_THRESHOLDS = [
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.325,
    0.35,
    0.375,
    0.40,
    0.425,
    0.45,
]

NMS_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
]

MATCH_IOU_THRESHOLD = 0.50
BATCH_SIZE = 32


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest().upper()


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 or second.numel() == 0:
        return torch.zeros((first.shape[0], second.shape[0]), dtype=torch.float32)

    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    first_area = (first[:, 2] - first[:, 0]).clamp(min=0) * (first[:, 3] - first[:, 1]).clamp(min=0)
    second_area = (second[:, 2] - second[:, 0]).clamp(min=0) * (second[:, 3] - second[:, 1]).clamp(min=0)

    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def get_ground_truth_boxes(dataset, image_id: int) -> torch.Tensor:
    annotations = dataset.coco.imgToAnns.get(image_id, [])
    boxes = []

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        boxes.append([x, y, x + width, y + height])

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)

    return torch.tensor(boxes, dtype=torch.float32)


def match_detections(predictions: torch.Tensor, scores: torch.Tensor, ground_truth: torch.Tensor) -> tuple[int, int, int, int]:
    if predictions.numel() == 0:
        return 0, 0, ground_truth.shape[0], 0

    if ground_truth.numel() == 0:
        return 0, predictions.shape[0], 0, 0

    order = torch.argsort(scores, descending=True)
    predictions = predictions[order]

    overlaps = box_iou(predictions, ground_truth)
    matched_ground_truth = set()

    true_positives = 0
    false_positives = 0
    duplicates = 0

    for prediction_index in range(predictions.shape[0]):
        best_iou, best_ground_truth_index = overlaps[prediction_index].max(dim=0)

        if best_iou.item() < MATCH_IOU_THRESHOLD:
            false_positives += 1
            continue

        ground_truth_index = int(best_ground_truth_index.item())

        if ground_truth_index in matched_ground_truth:
            false_positives += 1
            duplicates += 1
            continue

        matched_ground_truth.add(ground_truth_index)
        true_positives += 1

    false_negatives = ground_truth.shape[0] - len(matched_ground_truth)

    return true_positives, false_positives, false_negatives, duplicates

def tensor_checksum(tensor):
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()

    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(cpu_tensor.numpy().tobytes())

    return digest.hexdigest()


def cache_checksums(samples):
    return [
        tensor_checksum(sample["raw_output"])
        for sample in samples
    ]


def assert_cache_unchanged(samples, expected_checksums, stage):
    actual_checksums = cache_checksums(samples)

    if actual_checksums != expected_checksums:
        changed_indices = [
            index
            for index, (before, after) in enumerate(
                zip(expected_checksums, actual_checksums)
            )
            if before != after
        ]

        raise RuntimeError(
            f"Cached raw predictions changed during {stage}. "
            f"Changed sample indices: {changed_indices[:10]}"
        )

def evaluate_single_threshold(
    cached_predictions,
    dataset,
    exp,
    confidence_threshold: float,
    nms_threshold: float,
):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    duplicates = 0
    negative_fp_detections = 0
    negative_images_with_fp = 0

    for sample in cached_predictions:
        raw_output = sample["raw_output"].unsqueeze(0).detach().clone()

        processed = postprocess(
            raw_output,
            num_classes=1,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
            class_agnostic=False,
        )[0]

        if processed is None:
            predicted_boxes = torch.empty((0, 4), dtype=torch.float32)
            predicted_scores = torch.empty((0,), dtype=torch.float32)
        else:
            predicted_boxes = processed[:, :4].clone()
            predicted_scores = processed[:, 4] * processed[:, 5]

            scale = min(
                exp.test_size[0] / float(sample["image_height"]),
                exp.test_size[1] / float(sample["image_width"]),
            )

            predicted_boxes /= scale

        ground_truth = get_ground_truth_boxes(
            dataset,
            sample["image_id"],
        )

        tp, fp, fn, duplicate_count = match_detections(
            predicted_boxes,
            predicted_scores,
            ground_truth,
        )

        true_positives += tp
        false_positives += fp
        false_negatives += fn
        duplicates += duplicate_count

        if ground_truth.shape[0] == 0 and predicted_boxes.shape[0] > 0:
            negative_images_with_fp += 1
            negative_fp_detections += predicted_boxes.shape[0]

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "confidence_threshold": confidence_threshold,
        "nms_threshold": nms_threshold,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "duplicate_detections": duplicates,
        "negative_images_with_fp": negative_images_with_fp,
        "negative_fp_detections": negative_fp_detections,
    }


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint_hash = calculate_sha256(CHECKPOINT_PATH)

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: {checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    exp = get_exp(str(EXP_PATH), None)

    if exp.num_classes != 1:
        raise RuntimeError(f"Expected one class, found {exp.num_classes}")

    if exp.val_ann != "instances_val.json":
        raise RuntimeError(f"Unexpected VAL annotation file: {exp.val_ann}")

    if exp.test_ann != "__TEST_BLOCKED_DO_NOT_USE__.json":
        raise RuntimeError("TEST protection contract is not active")

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
            f"Expected {EXPECTED_VAL_NEGATIVES} VAL negatives, found {negative_count}"
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

    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda()
    model.eval()

    cached_predictions = []

    print("Running one forward pass over VAL...")

    for images, _, info_images, image_ids in tqdm(loader):
        images = images.cuda(non_blocking=True).float()

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
            f"Expected {EXPECTED_VAL_IMAGES} cached predictions, found {len(cached_predictions)}"
        )

    initial_cache_checksums = cache_checksums(cached_predictions)

    validation_sample = cached_predictions[0]
    before = validation_sample["raw_output"].clone()

    candidate = validation_sample["raw_output"].unsqueeze(0).clone()

    _ = postprocess(
        candidate,
        num_classes=1,
        conf_thre=0.01,
        nms_thre=0.45,
        class_agnostic=False,
    )

    if not torch.equal(validation_sample["raw_output"], before):
        raise RuntimeError(
            "Cached prediction mutation guard failed after postprocess()."
        )

    print("Cached postprocess mutation guard: PASS")
    print("Cached raw prediction integrity baseline: CREATED")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for nms_threshold in NMS_THRESHOLDS:
        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            duplicates = 0
            negative_fp_detections = 0
            negative_images_with_fp = 0

            for sample in cached_predictions:
                raw_output = sample["raw_output"].unsqueeze(0).detach().clone()

                processed = postprocess(
                    raw_output,
                    num_classes=1,
                    conf_thre=confidence_threshold,
                    nms_thre=nms_threshold,
                    class_agnostic=False,
                )[0]

                if processed is None:
                    predicted_boxes = torch.empty((0, 4), dtype=torch.float32)
                    predicted_scores = torch.empty((0,), dtype=torch.float32)
                else:
                    predicted_boxes = processed[:, :4].clone()
                    predicted_scores = processed[:, 4] * processed[:, 5]

                    scale = min(
                        exp.test_size[0] / float(sample["image_height"]),
                        exp.test_size[1] / float(sample["image_width"]),
                    )

                    predicted_boxes /= scale

                ground_truth = get_ground_truth_boxes(
                    dataset,
                    sample["image_id"],
                )

                tp, fp, fn, duplicate_count = match_detections(
                    predicted_boxes,
                    predicted_scores,
                    ground_truth,
                )

                true_positives += tp
                false_positives += fp
                false_negatives += fn
                duplicates += duplicate_count

                if ground_truth.shape[0] == 0 and predicted_boxes.shape[0] > 0:
                    negative_images_with_fp += 1
                    negative_fp_detections += predicted_boxes.shape[0]

            precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives > 0
                else 0.0
            )

            recall = (
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives > 0
                else 0.0
            )

            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            result = {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "match_iou_threshold": MATCH_IOU_THRESHOLD,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "duplicate_detections": duplicates,
                "negative_images_with_fp": negative_images_with_fp,
                "negative_fp_detections": negative_fp_detections,
            }

            results.append(result)

            print(
                f"conf={confidence_threshold:.2f} "
                f"nms={nms_threshold:.2f} "
                f"P={precision:.4f} "
                f"R={recall:.4f} "
                f"F1={f1:.4f} "
                f"FPneg={negative_fp_detections}"
            )

    assert_cache_unchanged(
        cached_predictions,
        initial_cache_checksums,
        "threshold sweep",
    )

    print("Cached raw predictions after sweep: UNCHANGED")

    reference_result = next(
        result
        for result in results
        if result["confidence_threshold"] == 0.40
        and result["nms_threshold"] == 0.45
    )

    independent_result = evaluate_single_threshold(
        cached_predictions,
        dataset,
        exp,
        confidence_threshold=0.40,
        nms_threshold=0.45,
    )

    if independent_result != reference_result:
        raise RuntimeError(
            "Independent threshold evaluation does not match sweep result."
        )

    assert_cache_unchanged(
        cached_predictions,
        initial_cache_checksums,
        "independent threshold validation",
    )

    print("Independent threshold row match: PASS")
    print("Cached raw predictions after independent validation: UNCHANGED")

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    best_f1 = max(results, key=lambda item: item["f1"])

    print("\n=== VAL THRESHOLD SWEEP COMPLETE ===")
    print(f"VAL images: {EXPECTED_VAL_IMAGES}")
    print(f"VAL negative images: {EXPECTED_VAL_NEGATIVES}")
    print(f"Checkpoint SHA256: {checkpoint_hash}")
    print("TEST accessed: NO")
    print(f"Results: {SUMMARY_PATH}")

    print("\nBest F1 candidate:")
    for key, value in best_f1.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()