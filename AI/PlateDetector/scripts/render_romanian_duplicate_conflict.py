from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "raw" / "romanian_lp" / "dataset" / "train"

IMAGE_ROOT = RAW_ROOT / "images"
ANNOTATION_ROOT = RAW_ROOT / "annots"

OUTPUT_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "audits" / "romanian_lp"

FRAME_A = "dayride_type1_001.mp4#t=558"
FRAME_B = "dayride_type1_001.mp4#t=809"


def load_boxes(annotation_path: Path) -> list[tuple[float, float, float, float]]:
    root = ET.parse(annotation_path).getroot()
    boxes = []

    for object_element in root.findall("object"):
        box = object_element.find("bndbox")

        if box is None:
            continue

        boxes.append((
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax")),
        ))

    return boxes


def draw_boxes(image, boxes, title):
    result = image.copy()

    cv2.rectangle(result, (0, 0), (result.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(result, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for index, (xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
        x1 = int(round(xmin))
        y1 = int(round(ymin))
        x2 = int(round(xmax))
        y2 = int(round(ymax))

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(result, f"plate {index}", (x1, max(y1 - 10, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return result


def main() -> None:
    image_a_path = IMAGE_ROOT / f"{FRAME_A}.jpg"
    image_b_path = IMAGE_ROOT / f"{FRAME_B}.jpg"

    annotation_a_path = ANNOTATION_ROOT / f"{FRAME_A}.xml"
    annotation_b_path = ANNOTATION_ROOT / f"{FRAME_B}.xml"

    image_a = cv2.imread(str(image_a_path))
    image_b = cv2.imread(str(image_b_path))

    if image_a is None or image_b is None:
        raise RuntimeError("Could not load duplicate images.")

    if image_a.shape != image_b.shape:
        raise RuntimeError("Duplicate images unexpectedly have different dimensions.")

    boxes_a = load_boxes(annotation_a_path)
    boxes_b = load_boxes(annotation_b_path)

    rendered_a = draw_boxes(
        image_a,
        boxes_a,
        f"{FRAME_A} | annotation count: {len(boxes_a)}",
    )

    rendered_b = draw_boxes(
        image_b,
        boxes_b,
        f"{FRAME_B} | annotation count: {len(boxes_b)}",
    )

    comparison = cv2.hconcat([rendered_a, rendered_b])

    max_width = 1900

    if comparison.shape[1] > max_width:
        scale = max_width / comparison.shape[1]
        comparison = cv2.resize(
            comparison,
            (
                int(comparison.shape[1] * scale),
                int(comparison.shape[0] * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_ROOT / "romanian_exact_duplicate_annotation_conflict.jpg"

    cv2.imwrite(str(output_path), comparison)

    print("\n=== ROMANIAN DUPLICATE CONFLICT RENDER ===")
    print(f"Frame A boxes: {len(boxes_a)}")
    print(f"Frame B boxes: {len(boxes_b)}")
    print(f"Output: {output_path}")
    print("\nRESULT: MANUAL ANNOTATION ADJUDICATION REQUIRED")


if __name__ == "__main__":
    main()