from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from yolox.data import COCODataset, ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

from evaluate_val_thresholds_512 import (
    assert_cache_unchanged,
    cache_checksums,
    calculate_sha256,
    get_ground_truth_boxes,
    match_detections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXP_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "experiments" / "yolox_nano_baseline_v1_512.py"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "runs"
    / "ai_wp_001_yolox_nano_baseline_v1_512_fp32_bs32"
    / "best_ckpt.pth"
)

ADAPTER_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "datasets" / "baseline_v1_coco"
TEST_JSON_PATH = ADAPTER_ROOT / "annotations" / "instances_test.json"

PRE_TEST_SELECTION_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "evaluation"
    / "ai_wp_001_final_detector_selection.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "evaluation"
    / "ai_wp_001_final_test_results.json"
)

EXPECTED_CHECKPOINT_SHA256 = "9A0C6A9D8ED0B9CAD31212F7ADDECF2C35C4B2CFF932ACFF1D02DF34519746B0"

EXPECTED_TEST_IMAGES = 669
EXPECTED_TEST_INSTANCES = 914
EXPECTED_TEST_NEGATIVES = 50

BATCH_SIZE = 32

COCO_CONFIDENCE_THRESHOLD = 0.01
COCO_NMS_THRESHOLD = 0.65

RUNTIME_CONFIDENCE_THRESHOLD = 0.225
RUNTIME_NMS_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50


def validate_pre_test_freeze() -> None:
    if not PRE_TEST_SELECTION_PATH.is_file():
        raise FileNotFoundError(f"Missing pre-TEST selection: {PRE_TEST_SELECTION_PATH}")

    selection = json.loads(PRE_TEST_SELECTION_PATH.read_text(encoding="utf-8-sig"))

    if selection.get("status") != "FINAL_DETECTOR_SELECTED_BEFORE_TEST":
        raise RuntimeError("Detector was not frozen before TEST")

    if selection.get("test_accessed") is not False:
        raise RuntimeError("Pre-TEST selection does not state test_accessed=false")

    detector = selection.get("selected_detector", {})

    if detector.get("input_size") != [512, 512]:
        raise RuntimeError("Frozen input resolution mismatch")

    if detector.get("confidence_threshold") != RUNTIME_CONFIDENCE_THRESHOLD:
        raise RuntimeError("Frozen confidence threshold mismatch")

    if detector.get("nms_threshold") != RUNTIME_NMS_THRESHOLD:
        raise RuntimeError("Frozen NMS threshold mismatch")


def validate_test_json() -> None:
    if not TEST_JSON_PATH.is_file():
        raise FileNotFoundError(f"Missing TEST adapter: {TEST_JSON_PATH}")

    data = json.loads(TEST_JSON_PATH.read_text(encoding="utf-8"))

    image_count = len(data.get("images", []))
    instance_count = len(data.get("annotations", []))
    annotated_ids = {annotation["image_id"] for annotation in data.get("annotations", [])}
    negative_count = image_count - len(annotated_ids)

    if image_count != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"TEST image count mismatch: {image_count}")

    if instance_count != EXPECTED_TEST_INSTANCES:
        raise RuntimeError(f"TEST instance count mismatch: {instance_count}")

    if negative_count != EXPECTED_TEST_NEGATIVES:
        raise RuntimeError(f"TEST negative count mismatch: {negative_count}")


def decode_sample(
    raw_output: torch.Tensor,
    confidence_threshold: float,
    nms_threshold: float,
    image_height: int,
    image_width: int,
    test_size: tuple[int, int],
):
    processed = postprocess(
        raw_output.unsqueeze(0).detach().clone(),
        num_classes=1,
        conf_thre=confidence_threshold,
        nms_thre=nms_threshold,
        class_agnostic=False,
    )[0]

    if processed is None:
        return None

    processed = processed.detach().cpu().clone()

    scale = min(
        test_size[0] / float(image_height),
        test_size[1] / float(image_width),
    )

    processed[:, :4] /= scale

    return processed


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(
            f"FINAL TEST results already exist. Refusing to rerun one-shot evaluation: {OUTPUT_PATH}"
        )

    validate_pre_test_freeze()
    validate_test_json()

    checkpoint_hash = calculate_sha256(CHECKPOINT_PATH)

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: {checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    exp = get_exp(str(EXP_PATH), None)

    if exp.num_classes != 1:
        raise RuntimeError("Expected one detector class")

    if exp.input_size != (512, 512) or exp.test_size != (512, 512):
        raise RuntimeError("Expected frozen 512x512 detector")

    if exp.test_ann != "__TEST_BLOCKED_DO_NOT_USE__.json":
        raise RuntimeError("Development TEST protection contract was modified")

    dataset = COCODataset(
        data_dir=str(ADAPTER_ROOT),
        json_file="instances_test.json",
        name="test",
        img_size=exp.test_size,
        preproc=ValTransform(legacy=False),
    )

    if len(dataset) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected TEST dataset size: {len(dataset)}")

    negative_count = sum(
        1
        for image_id in dataset.ids
        if len(dataset.coco.imgToAnns.get(int(image_id), [])) == 0
    )

    instance_count = sum(
        len(dataset.coco.imgToAnns.get(int(image_id), []))
        for image_id in dataset.ids
    )

    if negative_count != EXPECTED_TEST_NEGATIVES:
        raise RuntimeError(f"Unexpected TEST negative count: {negative_count}")

    if instance_count != EXPECTED_TEST_INSTANCES:
        raise RuntimeError(f"Unexpected TEST instance count: {instance_count}")

    sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
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
    total_forward_seconds = 0.0

    print("=== FINAL TEST ONE-SHOT ===")
    print("TEST is now being accessed for final evaluation.")
    print("Running exactly one model forward pass over TEST...")

    for images, _, info_images, image_ids in tqdm(loader):
        images = images.cuda(non_blocking=True).float()

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model(images)

        torch.cuda.synchronize()
        total_forward_seconds += time.perf_counter() - start_time

        for index in range(outputs.shape[0]):
            cached_predictions.append(
                {
                    "image_id": int(image_ids[index]),
                    "image_height": int(info_images[0][index]),
                    "image_width": int(info_images[1][index]),
                    "raw_output": outputs[index].detach().cpu(),
                }
            )

    if len(cached_predictions) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Cached TEST prediction count mismatch: {len(cached_predictions)}"
        )

    initial_checksums = cache_checksums(cached_predictions)

    coco_predictions = []

    for sample in cached_predictions:
        processed = decode_sample(
            sample["raw_output"],
            COCO_CONFIDENCE_THRESHOLD,
            COCO_NMS_THRESHOLD,
            sample["image_height"],
            sample["image_width"],
            exp.test_size,
        )

        if processed is None:
            continue

        for detection in processed:
            x1, y1, x2, y2 = detection[:4].tolist()
            score = float((detection[4] * detection[5]).item())
            class_index = int(detection[6].item())
            category_id = int(dataset.class_ids[class_index])

            coco_predictions.append(
                {
                    "image_id": sample["image_id"],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    ground_truth_coco = COCO(str(TEST_JSON_PATH))

    if not coco_predictions:
        raise RuntimeError("Detector produced no TEST predictions")

    detection_coco = ground_truth_coco.loadRes(coco_predictions)

    coco_evaluator = COCOeval(
        ground_truth_coco,
        detection_coco,
        "bbox",
    )

    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    map_50_95 = float(coco_evaluator.stats[0])
    map_50 = float(coco_evaluator.stats[1])

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    duplicate_detections = 0
    negative_images_with_fp = 0
    negative_fp_detections = 0

    for sample in cached_predictions:
        processed = decode_sample(
            sample["raw_output"],
            RUNTIME_CONFIDENCE_THRESHOLD,
            RUNTIME_NMS_THRESHOLD,
            sample["image_height"],
            sample["image_width"],
            exp.test_size,
        )

        if processed is None:
            predicted_boxes = torch.empty((0, 4), dtype=torch.float32)
            predicted_scores = torch.empty((0,), dtype=torch.float32)
        else:
            predicted_boxes = processed[:, :4].clone()
            predicted_scores = (processed[:, 4] * processed[:, 5]).clone()

        ground_truth = get_ground_truth_boxes(
            dataset,
            sample["image_id"],
        )

        tp, fp, fn, duplicates = match_detections(
            predicted_boxes,
            predicted_scores,
            ground_truth,
        )

        true_positives += tp
        false_positives += fp
        false_negatives += fn
        duplicate_detections += duplicates

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

    assert_cache_unchanged(
        cached_predictions,
        initial_checksums,
        "FINAL TEST postprocessing",
    )

    average_forward_ms = (
        total_forward_seconds / EXPECTED_TEST_IMAGES * 1000.0
    )

    results = {
        "work_package": "AI-WP-001",
        "evaluation": "FINAL_TEST_ONE_SHOT",
        "test_accessed": True,
        "post_test_tuning_performed": False,
        "checkpoint": {
            "file": "best_ckpt.pth",
            "selected_epoch": 60,
            "sha256": checkpoint_hash,
        },
        "model": {
            "architecture": "YOLOX-Nano",
            "input_size": [512, 512],
        },
        "test_dataset": {
            "images": EXPECTED_TEST_IMAGES,
            "plate_instances": EXPECTED_TEST_INSTANCES,
            "negative_images": EXPECTED_TEST_NEGATIVES,
        },
        "coco_protocol": {
            "confidence_threshold": COCO_CONFIDENCE_THRESHOLD,
            "nms_threshold": COCO_NMS_THRESHOLD,
            "map_50_95": map_50_95,
            "map_50": map_50,
        },
        "runtime_operating_point": {
            "confidence_threshold": RUNTIME_CONFIDENCE_THRESHOLD,
            "nms_threshold": RUNTIME_NMS_THRESHOLD,
            "match_iou_threshold": MATCH_IOU_THRESHOLD,
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duplicate_detections": duplicate_detections,
            "negative_images_with_fp": negative_images_with_fp,
            "negative_fp_detections": negative_fp_detections,
        },
        "desktop_timing": {
            "average_model_forward_ms_per_image": average_forward_ms,
            "device": torch.cuda.get_device_name(0),
            "note": "Desktop CUDA timing only. Not an Android benchmark.",
        },
        "cached_raw_predictions_unchanged": True,
        "android_latency": "NOT YET MEASURED",
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== FINAL TEST COMPLETE ===")
    print(f"TEST images: {EXPECTED_TEST_IMAGES}")
    print(f"TEST plate instances: {EXPECTED_TEST_INSTANCES}")
    print(f"TEST negative images: {EXPECTED_TEST_NEGATIVES}")
    print(f"mAP@0.50:0.95: {map_50_95:.6f}")
    print(f"mAP@0.50: {map_50:.6f}")
    print(f"TP: {true_positives}")
    print(f"FP: {false_positives}")
    print(f"FN: {false_negatives}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Negative images with FP: {negative_images_with_fp}")
    print(f"Negative FP detections: {negative_fp_detections}")
    print(f"Desktop average forward: {average_forward_ms:.3f} ms/image")
    print("Cached raw predictions: UNCHANGED")
    print("Post-TEST tuning: NO")
    print("Android latency: NOT YET MEASURED")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()