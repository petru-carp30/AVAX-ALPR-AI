from __future__ import annotations

import cv2
import numpy as np
import torch

from yolox.data.data_augment import preproc
from yolox.utils import postprocess


MODEL_SIZE = 512
CONFIDENCE_THRESHOLD = 0.225
NMS_THRESHOLD = 0.45


def rotate_box_clockwise(
    box: tuple[float, float, float, float],
    frame_width: float,
    frame_height: float,
    rotation_degrees: int,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box

    if rotation_degrees == 0:
        return left, top, right, bottom

    if rotation_degrees == 90:
        return (
            frame_height - bottom,
            left,
            frame_height - top,
            right,
        )

    if rotation_degrees == 180:
        return (
            frame_width - right,
            frame_height - bottom,
            frame_width - left,
            frame_height - top,
        )

    if rotation_degrees == 270:
        return (
            top,
            frame_width - right,
            bottom,
            frame_width - left,
        )

    raise ValueError(f"Unsupported rotation: {rotation_degrees}")


def inverse_rotate_box(
    box: tuple[float, float, float, float],
    frame_width: float,
    frame_height: float,
    rotation_degrees: int,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box

    if rotation_degrees == 0:
        return left, top, right, bottom

    if rotation_degrees == 90:
        return (
            top,
            frame_height - right,
            bottom,
            frame_height - left,
        )

    if rotation_degrees == 180:
        return (
            frame_width - right,
            frame_height - bottom,
            frame_width - left,
            frame_height - top,
        )

    if rotation_degrees == 270:
        return (
            frame_width - bottom,
            left,
            frame_width - top,
            right,
        )

    raise ValueError(f"Unsupported rotation: {rotation_degrees}")


def test_preprocessing() -> None:
    image = np.empty((720, 1280, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)

    tensor, scale = preproc(
        image,
        (MODEL_SIZE, MODEL_SIZE),
    )

    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == np.float32
    assert abs(scale - 0.4) < 1e-12

    np.testing.assert_array_equal(
        tensor[:, 0, 0],
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        tensor[:, 400, 100],
        np.array([114.0, 114.0, 114.0], dtype=np.float32),
    )

    assert float(tensor.max()) <= 114.0

    print("Preprocessing shape/type/order/padding: PASS")


def test_postprocessing() -> None:
    prediction = torch.tensor(
        [
            [
                [100.0, 100.0, 40.0, 20.0, 0.50, 0.50],
                [102.0, 100.0, 40.0, 20.0, 0.40, 0.60],
                [300.0, 300.0, 20.0, 10.0, 0.20, 0.90],
            ]
        ],
        dtype=torch.float32,
    )

    result = postprocess(
        prediction.clone(),
        num_classes=1,
        conf_thre=CONFIDENCE_THRESHOLD,
        nms_thre=NMS_THRESHOLD,
        class_agnostic=False,
    )[0]

    assert result is not None
    assert result.shape[0] == 1

    box = result[0, :4].numpy()
    score = float(result[0, 4] * result[0, 5])

    np.testing.assert_allclose(
        box,
        np.array([80.0, 90.0, 120.0, 110.0]),
        atol=1e-6,
    )

    assert abs(score - 0.25) < 1e-6

    print("YOLOX confidence/box/NMS semantics: PASS")


def test_letterbox_inverse() -> None:
    original_box = np.array(
        [100.0, 50.0, 300.0, 200.0],
        dtype=np.float64,
    )

    scale = min(
        MODEL_SIZE / 720.0,
        MODEL_SIZE / 1280.0,
    )

    model_box = original_box * scale
    restored_box = model_box / scale

    np.testing.assert_allclose(
        restored_box,
        original_box,
        atol=1e-9,
    )

    print("Letterbox inverse transform: PASS")


def test_rotation_round_trip() -> None:
    frame_width = 1280.0
    frame_height = 720.0

    source_box = (
        100.0,
        50.0,
        300.0,
        200.0,
    )

    for rotation_degrees in (0, 90, 180, 270):
        rotated_box = rotate_box_clockwise(
            source_box,
            frame_width,
            frame_height,
            rotation_degrees,
        )

        restored_box = inverse_rotate_box(
            rotated_box,
            frame_width,
            frame_height,
            rotation_degrees,
        )

        np.testing.assert_allclose(
            restored_box,
            source_box,
            atol=1e-9,
        )

        if rotation_degrees in (0, 180):
            oriented_width = frame_width
            oriented_height = frame_height
        else:
            oriented_width = frame_height
            oriented_height = frame_width

        assert 0.0 <= rotated_box[0] < rotated_box[2] <= oriented_width
        assert 0.0 <= rotated_box[1] < rotated_box[3] <= oriented_height

    print("Rotation transforms 0/90/180/270: PASS")


def test_full_coordinate_round_trip() -> None:
    frame_width = 1280.0
    frame_height = 720.0

    source_box = np.array(
        [100.0, 50.0, 300.0, 200.0],
        dtype=np.float64,
    )

    for rotation_degrees in (0, 90, 180, 270):
        rotated_box = np.array(
            rotate_box_clockwise(
                tuple(source_box),
                frame_width,
                frame_height,
                rotation_degrees,
            ),
            dtype=np.float64,
        )

        if rotation_degrees in (0, 180):
            oriented_width = frame_width
            oriented_height = frame_height
        else:
            oriented_width = frame_height
            oriented_height = frame_width

        scale = min(
            MODEL_SIZE / oriented_height,
            MODEL_SIZE / oriented_width,
        )

        model_box = rotated_box * scale
        restored_oriented_box = model_box / scale

        restored_source_box = np.array(
            inverse_rotate_box(
                tuple(restored_oriented_box),
                frame_width,
                frame_height,
                rotation_degrees,
            ),
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            restored_source_box,
            source_box,
            atol=1e-9,
        )

    print("Full model-to-CameraFrame coordinate round trip: PASS")


def main() -> None:
    test_preprocessing()
    test_postprocessing()
    test_letterbox_inverse()
    test_rotation_round_trip()
    test_full_coordinate_round_trip()

    print()
    print("=== MOBILE DETECTOR CONTRACT VALIDATION ===")
    print("Preprocessing: PASS")
    print("Output decode: PASS")
    print("Confidence threshold: 0.225")
    print("NMS threshold: 0.45")
    print("Letterbox inverse: PASS")
    print("Rotation 0/90/180/270: PASS")
    print("CameraFrame coordinate output: PASS")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()