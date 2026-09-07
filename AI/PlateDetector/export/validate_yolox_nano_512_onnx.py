from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import postprocess, replace_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXP_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "experiments"
    / "yolox_nano_baseline_v1_512.py"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "runs"
    / "ai_wp_001_yolox_nano_baseline_v1_512_fp32_bs32"
    / "best_ckpt.pth"
)

ONNX_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "export"
    / "avax_plate_detector_yolox_nano_512_v1.onnx"
)

VAL_IMAGE_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "datasets"
    / "derived"
    / "baseline_v1"
    / "images"
    / "val"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "export"
    / "avax_plate_detector_yolox_nano_512_v1_validation.json"
)

EXPECTED_CHECKPOINT_SHA256 = "9A0C6A9D8ED0B9CAD31212F7ADDECF2C35C4B2CFF932ACFF1D02DF34519746B0"
EXPECTED_ONNX_SHA256 = "B42313FE76EFCD98332430A25FE6BEEC553FC5FE7633957917453836AEA81793"

INPUT_NAME = "images"
OUTPUT_NAME = "output"
EXPECTED_INPUT_SHAPE = [1, 3, 512, 512]
EXPECTED_OUTPUT_SHAPE = [1, 5376, 6]

CONFIDENCE_THRESHOLD = 0.225
NMS_THRESHOLD = 0.45

RAW_RTOL = 1e-3
RAW_ATOL = 1e-3
MAX_BOX_DIFFERENCE_PX = 0.5
MAX_SCORE_DIFFERENCE = 1e-3

REPRESENTATIVE_SAMPLE_COUNT = 6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest().upper()


def select_representative_images() -> list[Path]:
    images = sorted(
        path
        for path in VAL_IMAGE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < REPRESENTATIVE_SAMPLE_COUNT:
        raise RuntimeError(
            f"Not enough VAL images: found={len(images)}, required={REPRESENTATIVE_SAMPLE_COUNT}"
        )

    indices = np.linspace(
        0,
        len(images) - 1,
        REPRESENTATIVE_SAMPLE_COUNT,
        dtype=int,
    )

    return [images[int(index)] for index in indices]


def prepare_input(image_path: Path) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    original_height, original_width = image.shape[:2]

    tensor, scale = preproc(
        image,
        (512, 512),
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if list(tensor.shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"Unexpected preprocessing shape for {image_path.name}: {tensor.shape}"
        )

    if tensor.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected preprocessing dtype for {image_path.name}: {tensor.dtype}"
        )

    return tensor, {
        "file_name": image_path.name,
        "original_width": original_width,
        "original_height": original_height,
        "resize_scale": float(scale),
    }


def sorted_detections(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(output.copy())

    processed = postprocess(
        tensor,
        num_classes=1,
        conf_thre=CONFIDENCE_THRESHOLD,
        nms_thre=NMS_THRESHOLD,
        class_agnostic=False,
    )[0]

    if processed is None:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    processed = processed.detach().cpu()

    boxes = processed[:, :4].numpy()
    scores = (processed[:, 4] * processed[:, 5]).numpy()

    order = np.argsort(-scores)

    return boxes[order], scores[order]


def main() -> None:
    if calculate_sha256(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA256 mismatch")

    onnx_hash = calculate_sha256(ONNX_PATH)

    if onnx_hash != EXPECTED_ONNX_SHA256:
        raise RuntimeError(
            f"ONNX SHA256 mismatch: {onnx_hash} != {EXPECTED_ONNX_SHA256}"
        )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    opset = next(
        entry.version
        for entry in onnx_model.opset_import
        if entry.domain in {"", "ai.onnx"}
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    if len(session_inputs) != 1:
        raise RuntimeError(f"Expected one ONNX input, found {len(session_inputs)}")

    if len(session_outputs) != 1:
        raise RuntimeError(f"Expected one ONNX output, found {len(session_outputs)}")

    input_metadata = session_inputs[0]
    output_metadata = session_outputs[0]

    if input_metadata.name != INPUT_NAME:
        raise RuntimeError(f"Unexpected ONNX input name: {input_metadata.name}")

    if list(input_metadata.shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Unexpected ONNX input shape: {input_metadata.shape}")

    if input_metadata.type != "tensor(float)":
        raise RuntimeError(f"Unexpected ONNX input dtype: {input_metadata.type}")

    if output_metadata.name != OUTPUT_NAME:
        raise RuntimeError(f"Unexpected ONNX output name: {output_metadata.name}")

    if list(output_metadata.shape) != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected ONNX output shape: {output_metadata.shape}")

    exp = get_exp(str(EXP_PATH), None)
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

    model.eval()

    model = replace_module(
        model,
        nn.SiLU,
        SiLU,
    )

    model.head.decode_in_inference = True

    sample_results = []
    raw_allclose_pass = True
    detection_consistency_pass = True

    overall_max_abs_difference = 0.0
    overall_mean_abs_difference = 0.0
    overall_max_box_difference = 0.0
    overall_max_score_difference = 0.0

    representative_images = select_representative_images()

    for image_path in representative_images:
        input_tensor, metadata = prepare_input(image_path)

        with torch.no_grad():
            pytorch_output = model(
                torch.from_numpy(input_tensor)
            ).cpu().numpy()

        onnx_output = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: input_tensor},
        )[0]

        if list(pytorch_output.shape) != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected PyTorch output shape: {pytorch_output.shape}"
            )

        if list(onnx_output.shape) != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected ONNX Runtime output shape: {onnx_output.shape}"
            )

        if not np.isfinite(pytorch_output).all():
            raise RuntimeError(f"Non-finite PyTorch output: {image_path.name}")

        if not np.isfinite(onnx_output).all():
            raise RuntimeError(f"Non-finite ONNX output: {image_path.name}")

        absolute_difference = np.abs(
            pytorch_output - onnx_output
        )

        max_abs_difference = float(
            absolute_difference.max()
        )

        mean_abs_difference = float(
            absolute_difference.mean()
        )

        raw_match = bool(
            np.allclose(
                pytorch_output,
                onnx_output,
                rtol=RAW_RTOL,
                atol=RAW_ATOL,
            )
        )

        pytorch_boxes, pytorch_scores = sorted_detections(
            pytorch_output
        )

        onnx_boxes, onnx_scores = sorted_detections(
            onnx_output
        )

        detection_count_match = (
            len(pytorch_boxes) == len(onnx_boxes)
        )

        max_box_difference = 0.0
        max_score_difference = 0.0

        if detection_count_match and len(pytorch_boxes) > 0:
            max_box_difference = float(
                np.abs(
                    pytorch_boxes - onnx_boxes
                ).max()
            )

            max_score_difference = float(
                np.abs(
                    pytorch_scores - onnx_scores
                ).max()
            )

        detection_match = (
            detection_count_match
            and max_box_difference <= MAX_BOX_DIFFERENCE_PX
            and max_score_difference <= MAX_SCORE_DIFFERENCE
        )

        raw_allclose_pass &= raw_match
        detection_consistency_pass &= detection_match

        overall_max_abs_difference = max(
            overall_max_abs_difference,
            max_abs_difference,
        )

        overall_mean_abs_difference = max(
            overall_mean_abs_difference,
            mean_abs_difference,
        )

        overall_max_box_difference = max(
            overall_max_box_difference,
            max_box_difference,
        )

        overall_max_score_difference = max(
            overall_max_score_difference,
            max_score_difference,
        )

        sample_results.append(
            {
                **metadata,
                "raw_allclose": raw_match,
                "max_abs_difference": max_abs_difference,
                "mean_abs_difference": mean_abs_difference,
                "pytorch_detection_count": int(len(pytorch_boxes)),
                "onnx_detection_count": int(len(onnx_boxes)),
                "detection_count_match": detection_count_match,
                "max_box_difference_px": max_box_difference,
                "max_score_difference": max_score_difference,
                "detection_consistency": detection_match,
            }
        )

    validation_pass = (
        raw_allclose_pass
        and detection_consistency_pass
    )

    results = {
        "artifact": {
            "filename": ONNX_PATH.name,
            "size_bytes": ONNX_PATH.stat().st_size,
            "sha256": onnx_hash,
            "opset": opset,
        },
        "runtime": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnxruntime_provider": "CPUExecutionProvider",
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "input": {
            "name": input_metadata.name,
            "shape": list(input_metadata.shape),
            "dtype": input_metadata.type,
        },
        "output": {
            "name": output_metadata.name,
            "shape": list(output_metadata.shape),
            "decoded_in_model": True,
        },
        "comparison": {
            "sample_count": len(sample_results),
            "source_split": "VAL",
            "raw_rtol": RAW_RTOL,
            "raw_atol": RAW_ATOL,
            "max_allowed_box_difference_px": MAX_BOX_DIFFERENCE_PX,
            "max_allowed_score_difference": MAX_SCORE_DIFFERENCE,
            "raw_allclose_pass": raw_allclose_pass,
            "detection_consistency_pass": detection_consistency_pass,
            "overall_max_abs_difference": overall_max_abs_difference,
            "overall_max_mean_abs_difference": overall_mean_abs_difference,
            "overall_max_box_difference_px": overall_max_box_difference,
            "overall_max_score_difference": overall_max_score_difference,
        },
        "samples": sample_results,
        "onnx_checker": "PASS",
        "validation_result": "PASS" if validation_pass else "FAIL",
        "android_latency": "NOT YET MEASURED",
    }

    RESULT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== AVAX ONNX VALIDATION ===")
    print(f"Artifact: {ONNX_PATH.name}")
    print(f"Size bytes: {ONNX_PATH.stat().st_size}")
    print(f"SHA256: {onnx_hash}")
    print(f"Opset: {opset}")
    print(f"ONNX Runtime: {ort.__version__}")
    print("Provider: CPUExecutionProvider")
    print(f"Input: {input_metadata.name} {list(input_metadata.shape)} {input_metadata.type}")
    print(f"Output: {output_metadata.name} {list(output_metadata.shape)}")
    print(f"Representative samples: {len(sample_results)}")
    print(f"Raw tensor consistency: {'PASS' if raw_allclose_pass else 'FAIL'}")
    print(f"Detection consistency: {'PASS' if detection_consistency_pass else 'FAIL'}")
    print(f"Max raw abs difference: {overall_max_abs_difference:.8f}")
    print(f"Max box difference: {overall_max_box_difference:.8f} px")
    print(f"Max score difference: {overall_max_score_difference:.8f}")
    print(f"ONNX checker: PASS")
    print(f"VALIDATION RESULT: {'PASS' if validation_pass else 'FAIL'}")
    print("Android latency: NOT YET MEASURED")
    print(f"Results: {RESULT_PATH}")

    if not validation_pass:
        raise RuntimeError(
            "ONNX validation failed. Do not proceed to mobile contract."
        )


if __name__ == "__main__":
    main()