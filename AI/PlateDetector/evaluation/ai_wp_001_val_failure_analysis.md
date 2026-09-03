# AI-WP-001 VAL Failure Analysis

## Frozen baseline

- Architecture: YOLOX-Nano
- Input: 416x416
- Selected checkpoint: best_ckpt.pth
- Selected epoch: 85
- Confidence threshold: 0.225
- NMS threshold: 0.45
- Matching IoU used for threshold analysis: 0.50

## Primary observed limitation

The dominant failure mode is small and distant license plates.

The persistent false-negative review showed that lowering the confidence threshold alone does not recover many of these cases. The limiting factor is primarily spatial information available at 416x416 rather than insufficient training duration.

## Secondary observed limitations

- Multi-plate and crowded vehicle scenes
- Motion blur
- Difficult viewing angles
- Partial occlusion
- Very small background vehicles
- Unusual plate appearances

## Dataset quality observations

Visual review identified cases where visible license plates are missing from the source ground-truth annotations or where IoU-based matching classifies semantically valid plate detections as errors because localization does not reach IoU 0.50.

The canonical baseline_v1 dataset is not modified retrospectively. These observations are retained as dataset quality technical debt for a future corrected dataset version.

## Threshold decision

Confidence 0.225 / NMS 0.45 was selected instead of the F1-maximizing confidence around 0.375.

The lower threshold preserves additional real license-plate detections that are useful to the downstream OCR stage. Visual review showed that the additional detector errors at 0.225 include real but incompletely annotated plates and localization mismatches, while raising the threshold removes useful low-confidence plate detections.

## Training decision

The 416x416 baseline will not receive additional epochs.

Training completed for 120 epochs and the best checkpoint occurred at epoch 85. Further improvement will be evaluated through a separate higher-resolution challenger rather than extending the frozen baseline run.

## TEST policy

TEST has not been accessed during model development.

Checkpoint, confidence threshold, and NMS threshold are frozen before final TEST evaluation. TEST results must not be used to modify these values.
