from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "export"
    / "avax_plate_detector_yolox_nano_512_v1.onnx"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "9A0C6A9D8ED0B9CAD31212F7ADDECF2C35C4B2CFF932ACFF1D02DF34519746B0"
)

INPUT_NAME = "images"
OUTPUT_NAME = "output"
OPSET_VERSION = 17


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest().upper()


def main() -> None:
    checkpoint_hash = calculate_sha256(CHECKPOINT_PATH)

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: "
            f"{checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    exp = get_exp(str(EXP_PATH), None)

    if exp.num_classes != 1:
        raise RuntimeError(f"Expected num_classes=1, found {exp.num_classes}")

    if exp.test_size != (512, 512):
        raise RuntimeError(f"Expected test_size=(512, 512), found {exp.test_size}")

    model = exp.get_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    model = replace_module(
        model,
        nn.SiLU,
        SiLU,
    )

    model.head.decode_in_inference = True

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    dummy_input = torch.zeros(
        (1, 3, 512, 512),
        dtype=torch.float32,
    )

    with torch.no_grad():
        reference_output = model(dummy_input)

    if tuple(reference_output.shape) != (1, 5376, 6):
        raise RuntimeError(
            f"Unexpected PyTorch output shape: "
            f"{tuple(reference_output.shape)}"
        )

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VERSION,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)

    if len(onnx_model.graph.input) != 1:
        raise RuntimeError("Expected exactly one ONNX input")

    if len(onnx_model.graph.output) != 1:
        raise RuntimeError("Expected exactly one ONNX output")

    input_tensor = onnx_model.graph.input[0]
    output_tensor = onnx_model.graph.output[0]

    artifact_hash = calculate_sha256(OUTPUT_PATH)

    print("=== AVAX ONNX EXPORT COMPLETE ===")
    print(f"Artifact: {OUTPUT_PATH}")
    print(f"Size bytes: {OUTPUT_PATH.stat().st_size}")
    print(f"SHA256: {artifact_hash}")
    print(f"Opset: {OPSET_VERSION}")
    print(f"Input name: {input_tensor.name}")
    print("Input shape: [1, 3, 512, 512]")
    print("Input dtype: float32")
    print(f"Output name: {output_tensor.name}")
    print("Output shape: [1, 5376, 6]")
    print("Decode in model: YES")
    print("Dynamic shapes: NO")
    print("ONNX checker: PASS")
    print("Android latency: NOT YET MEASURED")


if __name__ == "__main__":
    main()