# AVAX ALPR Mobile Detector Contract v1

## Scope

This contract defines the frozen AI handoff for the AVAX ALPR license plate detector.

Detector responsibility:

CameraFrame
→ FrameProcessor
→ License Plate Detector
→ PlateDetection[]

The detector does not perform OCR, plate normalization, vehicle lookup, access decisions, access logging, tracking, or temporal voting.

---

## Model

Filename:

`avax_plate_detector_yolox_nano_512_v1.onnx`

Format:

ONNX

Architecture:

YOLOX-Nano

Input resolution:

512x512

Batch size:

1

Dynamic shapes:

No

ONNX opset:

17

File size:

3,703,049 bytes

SHA256:

`B42313FE76EFCD98332430A25FE6BEEC553FC5FE7633957917453836AEA81793`

Checkpoint source:

`best_ckpt.pth`

Selected training epoch:

60

Checkpoint SHA256:

`9A0C6A9D8ED0B9CAD31212F7ADDECF2C35C4B2CFF932ACFF1D02DF34519746B0`

---

## ONNX Input

Tensor name:

`images`

Shape:

`[1, 3, 512, 512]`

Layout:

NCHW

Tensor dtype:

float32

Logical image color order:

BGR

The CameraFrame must be converted to an 8-bit BGR image before detector preprocessing.

No RGB channel swap is applied by the detector preprocessing contract.

---

## Camera Rotation

Supported `rotationDegrees` values:

- 0
- 90
- 180
- 270

`rotationDegrees` is applied clockwise to the CameraFrame buffer before resize and letterbox preprocessing.

The detector therefore always receives the correctly oriented image.

Let the original CameraFrame buffer dimensions be:

`W = frame width`

`H = frame height`

After clockwise rotation:

For 0 or 180 degrees:

`orientedWidth = W`

`orientedHeight = H`

For 90 or 270 degrees:

`orientedWidth = H`

`orientedHeight = W`

---

## Preprocessing

Input image before preprocessing:

- color: BGR
- layout: HWC
- dtype: uint8
- value range: 0..255

Processing order:

1. Apply `rotationDegrees` clockwise.
2. Preserve aspect ratio.
3. Compute:

`scale = min(512 / orientedHeight, 512 / orientedWidth)`

4. Resize using bilinear interpolation equivalent to OpenCV `INTER_LINEAR`.
5. Resize dimensions use integer truncation:

`resizedWidth = int(orientedWidth * scale)`

`resizedHeight = int(orientedHeight * scale)`

6. Place the resized image at the top-left of a 512x512 canvas.
7. Pad only the unused right and/or bottom region.
8. Padding value for every BGR channel:

`114`

9. Convert layout:

`HWC → CHW`

10. Convert dtype:

`uint8 → float32`

11. Add batch dimension.

Final tensor:

`[1, 3, 512, 512] float32`

Normalization:

None.

Do not divide by 255.

Do not subtract mean values.

Do not divide by standard deviation values.

---

## ONNX Output

Tensor name:

`output`

Shape:

`[1, 5376, 6]`

The YOLOX grid/stride decode is already included inside the ONNX model.

Each candidate contains:

`[centerX, centerY, width, height, objectness, classConfidence]`

Coordinates are expressed in the 512x512 detector-input coordinate space.

The detector contains one class only:

`license_plate`

---

## Confidence

For each candidate:

`confidence = objectness * classConfidence`

Frozen confidence threshold:

`0.225`

Reject candidates where:

`confidence < 0.225`

---

## Box Conversion

Convert:

`[centerX, centerY, width, height]`

to:

`[left, top, right, bottom]`

using:

`left = centerX - width / 2`

`top = centerY - height / 2`

`right = centerX + width / 2`

`bottom = centerY + height / 2`

These coordinates are still in the 512x512 detector-input coordinate space.

---

## NMS

Frozen NMS IoU threshold:

`0.45`

NMS score:

`objectness * classConfidence`

Only one detector class exists, so all surviving candidates belong to `license_plate`.

---

## Undo Letterbox

Because the resized image is placed at the top-left and there is no left or top padding offset:

`orientedLeft = modelLeft / scale`

`orientedTop = modelTop / scale`

`orientedRight = modelRight / scale`

`orientedBottom = modelBottom / scale`

Clamp these coordinates to:

`[0, orientedWidth]`

and:

`[0, orientedHeight]`

respectively.

---

## Inverse Camera Rotation

The final result must be expressed in the original CameraFrame buffer coordinate system.

Let the oriented box after undo-letterbox be:

`[L, T, R, B]`

and the original CameraFrame dimensions be:

`W, H`.

### rotationDegrees = 0

`left = L`

`top = T`

`right = R`

`bottom = B`

### rotationDegrees = 90

`left = T`

`top = H - R`

`right = B`

`bottom = H - L`

### rotationDegrees = 180

`left = W - R`

`top = H - B`

`right = W - L`

`bottom = H - T`

### rotationDegrees = 270

`left = W - B`

`top = L`

`right = W - T`

`bottom = R`

After inverse rotation, clamp again to the original CameraFrame bounds:

`x ∈ [0, W]`

`y ∈ [0, H]`

---

## Final Semantic Output

The detector returns zero or more:

`PlateDetection`

Semantic fields:

`left: float`

`top: float`

`right: float`

`bottom: float`

`confidence: float`

Coordinate space:

Original CameraFrame buffer pixel coordinates.

Required invariants:

`0 <= left < right <= frameWidth`

`0 <= top < bottom <= frameHeight`

`0 <= confidence <= 1`

---

## Threading and Memory Boundary

Inference is invoked through the existing `FrameProcessor`.

The detector may synchronously read the current CameraFrame.

The detector must not retain CameraFrame image buffers after synchronous processing returns.

The detector must not persist raw camera frames.

The detector must not modify CameraX lifecycle behavior.

The existing CameraX `STRATEGY_KEEP_ONLY_LATEST` behavior remains unchanged.

---

## OCR Boundary

OCR remains a separate downstream component.

Detector output:

`PlateDetection[]`

Future downstream flow:

`PlateDetection`
→ plate crop
→ perspective correction / preprocessing
→ OCR
→ plate normalization

The detector must not return plate text.

---

## Access-Control Boundary

The AI detector must not return:

- Granted
- Denied
- Expired
- NotYetValid
- Unknown
- vehicle database information

Access decisions remain application responsibilities.

---

## Frozen Runtime Parameters

Input:

`512x512`

Confidence threshold:

`0.225`

NMS threshold:

`0.45`

These parameters are frozen for the accelerated MVP.

Changing them requires reopening detector development.

---

## Validation

ONNX checker:

PASS

ONNX Runtime:

1.29.0

Validation provider:

CPUExecutionProvider

Representative comparison images:

6 VAL images

PyTorch vs ONNX raw tensor consistency:

PASS

PyTorch vs ONNX detection consistency:

PASS

Maximum observed raw tensor absolute difference:

`0.0094757080078125`

Maximum observed box difference:

`0.00006103515625 px`

Maximum observed confidence difference:

`0.00000017881393432617188`

---

## Performance

Desktop final TEST model-forward timing:

approximately `1.567 ms/image`

Hardware:

NVIDIA GeForce RTX 5060 Laptop GPU

This is not an Android benchmark.

Android latency:

`NOT YET MEASURED`

Android latency, thermal behavior, runtime provider selection, and frame-selection strategy belong to:

`MOB-AI-WP-001 — On-device License Plate Detector Integration`