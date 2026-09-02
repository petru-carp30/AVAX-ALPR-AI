from pathlib import Path

import torch
from yolox.exp import get_exp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "third_party" / "YOLOX" / "exps" / "default" / "yolox_nano.py"

INPUT_HEIGHT = 416
INPUT_WIDTH = 416


def describe_output(output) -> str:
    if isinstance(output, torch.Tensor):
        return str(tuple(output.shape))

    if isinstance(output, (list, tuple)):
        shapes = []

        for item in output:
            if isinstance(item, torch.Tensor):
                shapes.append(tuple(item.shape))
            else:
                shapes.append(type(item).__name__)

        return str(shapes)

    return type(output).__name__


def main() -> None:
    print("=== YOLOX-NANO CUDA SMOKE TEST ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Experiment: {EXP_PATH}")

    exp = get_exp(str(EXP_PATH), None)
    model = exp.get_model().to(device).eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    input_tensor = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)

    with torch.inference_mode():
        output = model(input_tensor)

    torch.cuda.synchronize()

    print(f"Model parameters: {parameter_count}")
    print(f"Input shape: {tuple(input_tensor.shape)}")
    print(f"Output shape: {describe_output(output)}")
    print("YOLOX-Nano CUDA forward: PASS")


if __name__ == "__main__":
    main()