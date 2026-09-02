from pathlib import Path

import torch

from yolox.exp import get_exp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_PATH = PROJECT_ROOT / "AI" / "PlateDetector" / "training" / "experiments" / "yolox_nano_baseline_v1.py"

EXPECTED_TRAIN_IMAGES = 7622
EXPECTED_VAL_IMAGES = 625
EXPECTED_CLASSES = 1


def main():
    print("=== AVAX YOLOX-NANO BASELINE EXP SMOKE TEST ===")

    exp = get_exp(str(EXP_PATH), None)

    print(f"Experiment: {exp.exp_name}")
    print(f"Classes: {exp.num_classes}")
    print(f"Input size: {exp.input_size}")
    print(f"Test size: {exp.test_size}")
    print(f"Data root: {exp.data_dir}")
    print(f"Train annotation: {exp.train_ann}")
    print(f"Val annotation: {exp.val_ann}")
    print(f"Seed: {exp.seed}")

    if exp.num_classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Expected {EXPECTED_CLASSES} class, got {exp.num_classes}")

    train_dataset = exp.get_dataset()
    val_dataset = exp.get_eval_dataset()

    print(f"TRAIN images: {len(train_dataset)}")
    print(f"VAL images: {len(val_dataset)}")

    if len(train_dataset) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"TRAIN count mismatch: expected {EXPECTED_TRAIN_IMAGES}, got {len(train_dataset)}")

    if len(val_dataset) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(f"VAL count mismatch: expected {EXPECTED_VAL_IMAGES}, got {len(val_dataset)}")

    train_image, train_target, _, _ = train_dataset[0]
    val_image, val_target, _, _ = val_dataset[0]

    print(f"TRAIN sample image shape: {train_image.shape}")
    print(f"TRAIN sample target shape: {train_target.shape}")
    print(f"VAL sample image shape: {val_image.shape}")
    print(f"VAL sample target shape: {val_target.shape}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    model = exp.get_model().to(device).eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    input_tensor = torch.randn(1, 3, 416, 416, device=device)

    with torch.inference_mode():
        output = model(input_tensor)

    torch.cuda.synchronize()

    print(f"Model parameters: {parameter_count}")
    print(f"Model output shape: {tuple(output.shape)}")

    if output.shape[-1] != 6:
        raise RuntimeError(f"Expected YOLOX output width 6 for one class, got {output.shape[-1]}")

    try:
        exp.get_eval_dataset(testdev=True)
    except RuntimeError as error:
        print(f"TEST guard: PASS - {error}")
    else:
        raise RuntimeError("TEST guard did not block testdev access")

    print("RESULT: PASS")


if __name__ == "__main__":
    main()