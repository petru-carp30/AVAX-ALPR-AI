from __future__ import annotations

import gc
from pathlib import Path

import torch
from loguru import logger
from yolox.exp import get_exp


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (parent / "AI" / "PlateDetector").is_dir():
            return parent

    raise RuntimeError(f"Could not locate AVAX ALPR project root from: {script_path}")


PROJECT_ROOT = find_project_root()

EXP_PATH = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "experiments"
    / "yolox_nano_baseline_v1.py"
)

BATCH_SIZES = (16, 24, 32)
DEVICE = torch.device("cuda:0")
USE_AMP = False

GB = 1024 ** 3

YOLOX_ASSIGNMENT_OOM_TEXT = (
    "OOM RuntimeError is raised due to the huge memory cost during label assignment"
)


class YoloXOomFallbackDetector:
    def __init__(self) -> None:
        self.detected = False

    def __call__(self, message) -> None:
        if YOLOX_ASSIGNMENT_OOM_TEXT in str(message):
            self.detected = True


def memory_gb(value: int) -> float:
    return value / GB


def is_cuda_oom(exception: BaseException) -> bool:
    if isinstance(exception, torch.OutOfMemoryError):
        return True

    message = str(exception).lower()

    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
        or "cublas_status_alloc_failed" in message
    )


def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shutdown_loader_iterator(data_iterator) -> None:
    if data_iterator is None:
        return

    shutdown = getattr(data_iterator, "_shutdown_workers", None)

    if callable(shutdown):
        shutdown()


def run_batch_test(batch_size: int) -> str:
    exp = None
    model = None
    optimizer = None
    train_loader = None
    data_iterator = None
    inputs = None
    targets = None
    outputs = None
    loss = None
    scaler = None

    oom_detector = YoloXOomFallbackDetector()
    log_sink_id = logger.add(oom_detector, level="ERROR")

    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    print()
    print(f"=== BATCH SIZE {batch_size} ===")

    try:
        exp = get_exp(str(EXP_PATH), None)

        if getattr(exp, "dataset", None) is not None:
            raise RuntimeError(
                "Fresh Exp unexpectedly contains cached dataset state."
            )

        if "model" in exp.__dict__:
            raise RuntimeError(
                "Fresh Exp unexpectedly contains cached model state."
            )

        if "optimizer" in exp.__dict__:
            raise RuntimeError(
                "Fresh Exp unexpectedly contains cached optimizer state."
            )

        model = exp.get_model()
        model = model.to(DEVICE)
        model.train()

        optimizer = exp.get_optimizer(batch_size)

        train_loader = exp.get_data_loader(
            batch_size=batch_size,
            is_distributed=False,
            no_aug=False,
            cache_img=None,
        )

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError(
                f"Unexpected YOLOX batch structure: {type(batch)}"
            )

        inputs = batch[0]
        targets = batch[1]

        if inputs.shape[0] != batch_size:
            raise RuntimeError(
                f"Unexpected batch size: requested={batch_size}, "
                f"actual={inputs.shape[0]}"
            )

        inputs = inputs.to(
            device=DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        targets = targets.to(
            device=DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        targets.requires_grad_(False)

        inputs, targets = exp.preprocess(
            inputs,
            targets,
            exp.input_size,
        )

        optimizer.zero_grad(set_to_none=True)

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP,
        ):
            outputs = model(inputs, targets)

            if not isinstance(outputs, dict):
                raise RuntimeError(
                    f"Unexpected YOLOX training output type: {type(outputs)}"
                )

            if "total_loss" not in outputs:
                raise RuntimeError(
                    "YOLOX training output does not contain 'total_loss'."
                )

            loss = outputs["total_loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for batch size {batch_size}: "
                f"{float(loss.detach().cpu())}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize(DEVICE)

        if oom_detector.detected:
            print(
                "RESULT: OOM/FALLBACK "
                "(YOLOX label assignment switched to CPU)"
            )
            return "oom"

        allocated = torch.cuda.memory_allocated(DEVICE)
        reserved = torch.cuda.memory_reserved(DEVICE)
        peak_allocated = torch.cuda.max_memory_allocated(DEVICE)
        peak_reserved = torch.cuda.max_memory_reserved(DEVICE)

        print(f"Loss: {loss.item():.4f}")
        print(f"Allocated: {memory_gb(allocated):.2f} GB")
        print(f"Reserved: {memory_gb(reserved):.2f} GB")
        print(
            f"Peak allocated: "
            f"{memory_gb(peak_allocated):.2f} GB"
        )
        print(
            f"Peak reserved: "
            f"{memory_gb(peak_reserved):.2f} GB"
        )
        print("RESULT: PASS")

        return "pass"

    except Exception as exception:
        if is_cuda_oom(exception):
            print(f"CUDA OOM: {exception}")
            print("RESULT: OOM")
            return "oom"

        print(
            f"RESULT: ERROR "
            f"({type(exception).__name__}: {exception})"
        )
        raise

    finally:
        logger.remove(log_sink_id)

        shutdown_loader_iterator(data_iterator)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        outputs = None
        loss = None
        inputs = None
        targets = None
        scaler = None
        optimizer = None
        model = None
        data_iterator = None
        train_loader = None
        exp = None

        cleanup_cuda()


def main() -> None:
    if not EXP_PATH.exists():
        raise FileNotFoundError(
            f"Missing AVAX YOLOX experiment: {EXP_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=== AVAX YOLOX TRAIN BATCH SMOKE TEST ===")
    print(f"Experiment: {EXP_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(
        f"VRAM: "
        f"{memory_gb(torch.cuda.get_device_properties(DEVICE).total_memory):.2f} GB"
    )
    print(f"AMP: {USE_AMP}")
    print(f"Batch sizes: {BATCH_SIZES}")

    results = {}

    for batch_size in BATCH_SIZES:
        result = run_batch_test(batch_size)
        results[batch_size] = result

        if result == "oom":
            print()
            print(
                f"Stopping after OOM/FALLBACK at batch size "
                f"{batch_size}."
            )
            break

    print()
    print("=== SUMMARY ===")

    for batch_size in BATCH_SIZES:
        if batch_size not in results:
            print(f"Batch {batch_size}: NOT EXECUTED")
        else:
            print(
                f"Batch {batch_size}: "
                f"{results[batch_size].upper()}"
            )

    print()
    print("Actual training: NOT STARTED")
    print("TEST: NOT ACCESSED")


if __name__ == "__main__":
    main()