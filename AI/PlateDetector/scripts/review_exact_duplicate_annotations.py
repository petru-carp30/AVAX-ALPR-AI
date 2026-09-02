from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = PROJECT_ROOT / "AI" / "PlateDetector" / "datasets" / "derived" / "canonical_pool"

SAMPLES_PATH = CANONICAL_ROOT / "metadata" / "canonical_samples.csv"
BOXES_PATH = CANONICAL_ROOT / "metadata" / "canonical_boxes.csv"
EXACT_PATH = CANONICAL_ROOT / "metadata" / "similarity" / "exact_duplicate_groups.csv"
OUTPUT_PATH = CANONICAL_ROOT / "metadata" / "similarity" / "exact_duplicate_adjudication.csv"

WINDOW_NAME = "AVAX Exact Duplicate Annotation Review"
PANEL_WIDTH = 800
PANEL_HEIGHT = 700
HEADER_HEIGHT = 120


def load_existing_reviews() -> dict[str, str]:
    if not OUTPUT_PATH.exists():
        return {}
    frame = pd.read_csv(OUTPUT_PATH, dtype=str)
    return dict(zip(frame["exact_group_id"], frame["keep_canonical_id"]))


def get_boxes(boxes: pd.DataFrame, canonical_id: str) -> list[tuple[float, float, float, float]]:
    frame = boxes[boxes["canonical_id"] == canonical_id].sort_values("box_index")
    return [(float(row.x_center), float(row.y_center), float(row.width), float(row.height)) for row in frame.itertuples(index=False)]


def draw_variant(image_path: Path, boxes: list[tuple[float, float, float, float]], title: str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    for x_center, y_center, box_width, box_height in boxes:
        xmin = int((x_center - box_width / 2) * width)
        ymin = int((y_center - box_height / 2) * height)
        xmax = int((x_center + box_width / 2) * width)
        ymax = int((y_center + box_height / 2) * height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    scale = min(PANEL_WIDTH / width, (PANEL_HEIGHT - HEADER_HEIGHT) / height)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 25, dtype=np.uint8)
    x_offset = (PANEL_WIDTH - resized.shape[1]) // 2
    y_offset = HEADER_HEIGHT + (PANEL_HEIGHT - HEADER_HEIGHT - resized.shape[0]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_reviews(groups: list[dict], decisions: dict[str, str]) -> None:
    records = []

    for group in groups:
        keep_id = decisions.get(group["exact_group_id"], "")
        ids = group["canonical_ids"]

        records.append({
            "exact_group_id": group["exact_group_id"],
            "canonical_id_1": ids[0],
            "canonical_id_2": ids[1],
            "keep_canonical_id": keep_id,
            "exclude_canonical_id": ids[1] if keep_id == ids[0] else ids[0] if keep_id == ids[1] else "",
            "review_status": "resolved" if keep_id else "unresolved",
        })

    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    samples = pd.read_csv(SAMPLES_PATH)
    boxes = pd.read_csv(BOXES_PATH)
    exact = pd.read_csv(EXACT_PATH)

    groups = []

    for group_id, frame in exact.groupby("exact_group_id"):
        ids = frame["canonical_id"].tolist()

        if len(ids) != 2:
            raise RuntimeError(f"Expected exactly 2 images in {group_id}, found {len(ids)}.")

        groups.append({"exact_group_id": group_id, "canonical_ids": ids})

    decisions = load_existing_reviews()
    current_index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1650, 850)

    while 0 <= current_index < len(groups):
        group = groups[current_index]
        first_id, second_id = group["canonical_ids"]

        first_sample = samples[samples["canonical_id"] == first_id].iloc[0]
        second_sample = samples[samples["canonical_id"] == second_id].iloc[0]

        first_panel = draw_variant(PROJECT_ROOT / first_sample["canonical_image_path"], get_boxes(boxes, first_id), f"1 = {first_id}")
        second_panel = draw_variant(PROJECT_ROOT / second_sample["canonical_image_path"], get_boxes(boxes, second_id), f"2 = {second_id}")

        display = np.hstack([first_panel, second_panel])
        current_choice = decisions.get(group["exact_group_id"], "UNRESOLVED")

        cv2.putText(display, f"Group {current_index + 1}/{len(groups)} | Current: {current_choice}", (20, PANEL_HEIGHT - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "1=keep left | 2=keep right | U=unresolved | B=back | N=next | Q=save+quit", (20, PANEL_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("1"):
            decisions[group["exact_group_id"]] = first_id
            save_reviews(groups, decisions)
            current_index = min(current_index + 1, len(groups) - 1)
        elif key == ord("2"):
            decisions[group["exact_group_id"]] = second_id
            save_reviews(groups, decisions)
            current_index = min(current_index + 1, len(groups) - 1)
        elif key in (ord("u"), ord("U")):
            decisions.pop(group["exact_group_id"], None)
            save_reviews(groups, decisions)
            current_index = min(current_index + 1, len(groups) - 1)
        elif key in (ord("b"), ord("B")):
            current_index = max(current_index - 1, 0)
        elif key in (ord("n"), ord("N")):
            current_index = min(current_index + 1, len(groups) - 1)
        elif key in (ord("q"), ord("Q")):
            save_reviews(groups, decisions)
            break

    save_reviews(groups, decisions)
    cv2.destroyAllWindows()

    resolved = len(decisions)
    print("\n=== EXACT DUPLICATE ANNOTATION ADJUDICATION ===")
    print(f"Groups: {len(groups)}")
    print(f"Resolved: {resolved}")
    print(f"Unresolved: {len(groups) - resolved}")

    for group in groups:
        keep_id = decisions.get(group["exact_group_id"], "")
        ids = group["canonical_ids"]
        exclude_id = ids[1] if keep_id == ids[0] else ids[0] if keep_id == ids[1] else ""
        print(f"\n{group['exact_group_id']}")
        print(f"  Keep: {keep_id or 'UNRESOLVED'}")
        print(f"  Exclude: {exclude_id or 'UNRESOLVED'}")

    print(f"\nReview CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()