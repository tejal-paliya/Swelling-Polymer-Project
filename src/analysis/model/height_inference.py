# src/analysis/model/height_inference.py
import os, glob, cv2
from ultralytics import YOLO

MODEL_PATH   = "src/analysis/model/best.pt"
SAVE_PARENT  = "data/frame_capture"        # where annotated frames will live
# ────────────────────────────────────────────────────────────────────────────────
def _select_box(boxes, cls_id):
    """
    Return the box (xyxy tensor) with the highest confidence for a given class.
    boxes: Results.boxes from Ultralytics, len(boxes) ≥ 1
    cls_id: integer class id to select
    """
    cand = [b for b in boxes if int(b.cls[0]) == cls_id]
    if not cand:
        return None
    # highest confidence first
    cand.sort(key=lambda b: float(b.conf[0]), reverse=True)
    return cand[0].xyxy[0]   # tensor of shape (4,)  (x1, y1, x2, y2)
# ────────────────────────────────────────────────────────────────────────────────
def run_height_analysis(experiment_dir: str):
    """
    For every *.png in `experiment_dir`:
        • detect bed_height and vial_base
        • draw a red vertical line between mid-points
        • save annotated frame in data/frame_capture/<exp_id>/
        • print 'frame idx ▸ height_px'
    """
    model = YOLO(MODEL_PATH)
    pngs  = sorted(glob.glob(os.path.join(experiment_dir, "*.png")))
    if not pngs:
        print(f"[height-analysis] No PNGs in {experiment_dir}")
        return

    # derive experiment id from first filename: frame_exp<id>_....
    stem     = os.path.basename(pngs[0]).split("_")
    exp_id   = stem[1][3:] if len(stem) > 1 and stem[1].startswith("exp") else "unknown"
    save_dir = os.path.join(SAVE_PARENT, exp_id)
    os.makedirs(save_dir, exist_ok=True)

    for idx, fp in enumerate(pngs):
        res   = model(fp, verbose=False)[0]          # one-image inference
        box_bed  = _select_box(res.boxes, cls_id=0)  # bed_height
        box_base = _select_box(res.boxes, cls_id=1)  # vial_base
        if box_bed is None or box_base is None:
            print(f"[warning] missing box in {os.path.basename(fp)}; skipped")
            continue

        # mid-points
        x_bed  = float(box_bed[0] + box_bed[2]) / 2
        y_bed  = float(box_bed[1] + box_bed[3]) / 2
        x_base = float(box_base[0] + box_base[2]) / 2
        y_base = float(box_base[1] + box_base[3]) / 2
        height_px = abs(y_base - y_bed)

        # annotate image --------------------------------------------------------
        img = cv2.imread(fp)
        # draw bounding boxes (green for bed, blue for base)
        cv2.rectangle(img, (int(box_bed[0]),  int(box_bed[1])),
                           (int(box_bed[2]),  int(box_bed[3])), (0,255,0), 2)
        cv2.rectangle(img, (int(box_base[0]), int(box_base[1])),
                           (int(box_base[2]), int(box_base[3])), (255,0,0), 2)
        # draw red vertical line (x-pos = average of the two mid-points)
        x_line = int((x_bed + x_base) / 2)
        cv2.line(img, (x_line, int(y_bed)), (x_line, int(y_base)), (0,0,255), 3)
        # put height text
        cv2.putText(img, f"{int(height_px)} px",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # save ------------------------------------------------------------------
        tgt = os.path.join(save_dir, os.path.basename(fp))
        cv2.imwrite(tgt, img)
        print(f"{os.path.basename(fp)} ▸ {int(height_px)} px")

    print(f"[height-analysis] annotated frames ➜ {save_dir}")
