# MASTER HANDOFF â€” AI-WP-002

## STATUS REQUEST

READY FOR MASTER REVIEW

AI-WP-002 must not be marked DONE by the AI workstream.

---

## OCR SELECTION

Selected engine:

Google ML Kit Text Recognition v2 â€” Latin bundled model

Android artifact:

`com.google.mlkit:text-recognition:16.0.1`

Reason selected:

- runs on-device;
- bundled OCR model;
- no cloud OCR dependency;
- simple Android integration;
- supports Latin A-Z and digits 0-9;
- suitable for cropped license plate images;
- avoids custom OCR model training for the MVP;
- significantly lower integration complexity than custom ONNX/TFLite OCR.

The selected configuration is frozen for the MVP.

---

## CANDIDATES REJECTED

### Tesseract4Android

Not selected for MVP.

Reason:

- additional native runtime/model management;
- traineddata distribution required;
- more integration complexity than ML Kit.

Retained only as a possible future fallback.

### PaddleOCR mobile

Not selected for MVP.

Reason:

- larger integration and model-management scope;
- unnecessary complexity for the accelerated MVP.

---

## OFFLINE BEHAVIOR

OCR recognition is performed locally on the Android device.

The bundled Latin model is packaged with the application dependency and does not require downloading the OCR model before first recognition.

Plate images are not uploaded for OCR.

No cloud OCR service is required.

---

## VALIDATION DATA

Validation was performed using 50 manually reviewed public license plate crops.

No private AVAX site imagery was used.

Final validation composition:

- total samples: 50
- Romanian public dataset samples: 8
- ELPD samples: 42
- unique plate strings: 50

Ground truth was manually reviewed.

Ambiguous, unreadable and partial plates were excluded from the official validation set.

---

## VALIDATION CONFIGURATION

OCR engine:

Google ML Kit Text Recognition v2 Latin

Artifact:

`com.google.mlkit:text-recognition:16.0.1`

Candidate selection:

The OCR output is evaluated line-by-line.

The most plate-like OCR line is selected using only OCR-output characteristics:

- alphanumeric content;
- plausible plate-string length;
- presence of both letters and digits;
- OCR confidence;
- penalty for excessively long text.

Ground truth is NOT used to select the OCR candidate.

---

## PREPROCESSING

Final MVP preprocessing:

1. detector produces plate crop;
2. preserve original crop when height is at least 32 px;
3. if crop height is below 32 px:
   - upscale while preserving aspect ratio;
   - target height = 96 px;
4. run ML Kit OCR.

No additional preprocessing is required for the MVP.

Not used:

- adaptive thresholding;
- aggressive contrast processing;
- perspective correction;
- deskew;
- country grammar;
- character substitutions.

---

## FINAL VALIDATION RESULTS

Configuration:

Selective upscale for crops below 32 px.

Results:

- samples: 50
- normalized exact matches: 35 / 50
- normalized exact accuracy: 70.00%
- total ground-truth characters: 369
- total edit distance: 23
- character error rate: 6.23%
- character accuracy: 93.77%
- empty OCR results: 0

Romanian subset:

- samples: 8
- exact normalized matches: 4 / 8
- exact normalized accuracy: 50.00%

ELPD subset:

- samples: 42
- exact normalized matches: 31 / 42
- exact normalized accuracy: 73.81%

Android latency:

NOT YET MEASURED

Actual mobile latency must be measured during MOB-AI-WP-002.

---

## PREPROCESSING EXPERIMENT HISTORY

### V2 â€” baseline

No resize.

Results:

- exact normalized: 32 / 50
- accuracy: 64.00%
- character accuracy: 84.01%
- empty results: 6
- Romanian exact: 2 / 8

### V3 â€” upscale crops below 48 px to 96 px

Results:

- exact normalized: 33 / 50
- accuracy: 66.00%
- character accuracy: 92.68%
- empty results: 0
- Romanian exact: 4 / 8

This configuration introduced regressions on several crops that were already OCR-readable.

### V4 â€” selective upscale below 32 px to 96 px

Final selected configuration.

Results:

- exact normalized: 35 / 50
- accuracy: 70.00%
- character accuracy: 93.77%
- empty results: 0
- Romanian exact: 4 / 8

V4 is frozen for MVP.

---

## COMMON FAILURE MODES

Representative OCR errors observed:

`NMEO7592 -> NME07592`

O / 0 confusion.

`CMFH4336 -> GMFH4336`

C / G confusion.

`B778NDA -> 8778NDA`

B / 8 confusion.

`B150PEC -> B50PEC`

Missing character.

`B01XEN -> BOIXEN`

0 / O / I-related ambiguity.

`WFVE1248 -> NFVE1248`

W / N confusion.

`BGDZ8332 -> BGDZ833`

Missing final digit.

`NWG7513 -> NAG7513`

W / A confusion.

Failure analysis by condition showed:

- overexposed: 1 / 6 exact
- angled: 10 / 12 exact
- small: 15 / 21 exact
- slight blur: 4 / 7 exact
- pixelated: 2 / 3 exact

Overexposure remains the clearest observed weak condition.

---

## INPUT CONTRACT

Input:

cropped license plate image derived from detector output.

Expected image:

- upright according to camera frame orientation;
- RGB/ARGB-compatible Android bitmap/input image;
- crop should contain the full plate whenever possible.

Recommended preprocessing:

If crop height < 32 px:

- resize to 96 px height;
- preserve aspect ratio.

Otherwise:

- use original crop.

No mandatory perspective correction for MVP.

---

## OUTPUT CONTRACT

Conceptual result:

```kotlin
data class PlateOcrResult(
    val text: String,
    val confidence: Float?
)
```

OCR responsibility:

- return recognized plate candidate text;
- preserve OCR confidence when available.

The raw OCR result should remain available for debugging.

PlateNormalizer remains downstream.

---

## POSTPROCESSING BOUNDARY

Allowed inside OCR stage:

- select the most plate-like OCR text line;
- uppercase/evaluation cleanup where required;
- preserve raw OCR text separately.

Not allowed inside OCR stage:

- vehicle lookup;
- access decision;
- database access;
- Granted / Denied logic;
- Romanian plate grammar enforcement;
- aggressive O/0, B/8, I/1 substitutions.

PlateNormalizer remains responsible for application-level plate normalization.

---

## FAILURE SEMANTICS

OCR failure is acceptable.

Possible failure outcomes:

- empty text;
- low confidence;
- incorrect candidate;
- runtime OCR failure.

The AI must not invent plate text.

If OCR is unusable, Guard may fall back to manual plate entry.

---

## ANDROID HANDOFF

Recommended dependency:

```kotlin
implementation("com.google.mlkit:text-recognition:16.0.1")
```

Recognizer:

```kotlin
TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
```

Minimum integration sequence:

1. receive detector bounding box;
2. crop plate from camera frame;
3. apply small crop padding if required;
4. if crop height < 32 px, upscale to 96 px height;
5. create ML Kit `InputImage`;
6. run Latin text recognition;
7. select the most plate-like OCR line;
8. return text and confidence;
9. pass OCR result to existing PlateNormalizer;
10. if result is unusable, allow manual entry.

The Android implementation belongs to MOB-AI-WP-002.

---

## THREADING / MEMORY

OCR must run off the UI thread.

Only the plate crop should be processed.

Do not run OCR on the complete camera frame when a detector crop is available.

Do not persist plate crops by default.

Do not upload plate crops.

Multi-frame OCR voting is deferred.

---

## LICENSING

Selected component:

Google ML Kit Text Recognition v2.

Usage is governed by the applicable Google ML Kit / Google APIs terms.

No unresolved commercial-use blocker was identified during AI-WP-002 review.

The mobile project should retain the appropriate third-party notices and dependency information.

The component must not be documented as an Apache-2.0 ML Kit model.

Validation datasets remain subject to their respective source licenses and provenance records.

---

## PERFORMANCE

Desktop performance benchmarking was not required.

Android functional execution was confirmed on a Pixel 6 Pro.

Android latency:

NOT YET MEASURED

Latency measurement belongs to MOB-AI-WP-002.

---

## DEFERRED

Deferred beyond MVP:

- custom OCR training;
- dedicated OCR dataset acquisition;
- advanced contrast enhancement;
- thresholding experiments;
- perspective correction;
- country-specific grammar;
- character substitution rules;
- multi-frame OCR voting;
- OCR accuracy optimization;
- quantized custom OCR networks.

These should only be reopened if field performance proves insufficient.

---

## FILES

Created / retained under AI workstream:

`AI/PlateOCR/datasets/prepare_validation_crops.py`

`AI/PlateOCR/datasets/validation_mvp/`

`AI/PlateOCR/evaluation/results/ocr_validation_results_v4_selective_upscale.csv`

`AI/PlateOCR/evaluation/results/ocr_validation_summary_v4_selective_upscale.txt`

`AI/PlateOCR/documentation/AI-WP-002-MASTER-HANDOFF.md`

Android instrumentation validation code was used only to evaluate the mobile OCR engine and belongs to the mobile test environment.

---

## REFERENCE COMMIT

64872bf2b0e709f56c30eba1e778ae352af076a2

---

## RECOMMENDED NEXT TASK

MOB-AI-WP-002 â€” OCR + Automatic Local Verification Pipeline

---

## STATUS REQUEST

READY FOR MASTER REVIEW

Do not mark AI-WP-002 DONE automatically.

