# src/analysis/model/height_inference.py
import os, glob, cv2, csv
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH  = "src/analysis/model/best.pt"
SAVE_PARENT = "data/frame_capture"               # annotated frames + csv + plot
DATE_FMT    = "%Y%m%d_%H%M%S"                    # matches timestamp in filename
# ────────────────────────────────────────────────────────────────────────────
def _best_box(boxes, cid):
    """return highest-confidence box (xyxy tensor) for class id cid"""
    pick = [b for b in boxes if int(b.cls[0]) == cid]
    return max(pick, key=lambda b: float(b.conf[0])).xyxy[0] if pick else None
# ────────────────────────────────────────────────────────────────────────────
def _midpoint(box):                              # (x1 y1 x2 y2) tensor → (x,y)
    return float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2
# ────────────────────────────────────────────────────────────────────────────
def run_height_analysis(exp_dir: str):
    model   = YOLO(MODEL_PATH)
    frames  = sorted(glob.glob(os.path.join(exp_dir, "*.png")))
    if not frames:
        print(f"[height-analysis] no PNGs in {exp_dir}")
        return

    # experiment id extracted from first file name "frame_exp<ID>_..."
    exp_id   = os.path.basename(frames[0]).split("_")[1][3:]
    out_dir  = os.path.join(SAVE_PARENT, exp_id)
    os.makedirs(out_dir, exist_ok=True)

    times, heights = [], []
    t0 = None                                         # start-time reference

    for idx, fp in enumerate(frames):
        result    = model(fp, verbose=False)[0]
        box_bed   = _best_box(result.boxes, 0)        # class 0 = bed_height
        box_base  = _best_box(result.boxes, 1)        # class 1 = vial_base
        if box_bed is None or box_base is None:
            print(f"[warn] missing class in {os.path.basename(fp)} – skipped")
            continue

        (xb, yb)  = _midpoint(box_bed)
        (xx, yx)  = _midpoint(box_base)
        h_px      = abs(yx - yb)

        # ---------------------------------------------------------------- draw
        img = cv2.imread(fp)
        cv2.rectangle(img, (int(box_bed[0]), int(box_bed[1])),
                           (int(box_bed[2]), int(box_bed[3])), (0,255,0), 2)
        cv2.rectangle(img, (int(box_base[0]), int(box_base[1])),
                           (int(box_base[2]), int(box_base[3])), (255,0,0), 2)
        x_line = int((xb + xx) / 2)
        cv2.line(img, (x_line, int(yb)), (x_line, int(yx)), (0,0,255), 3)
        cv2.putText(img, f"{int(h_px)} px", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imwrite(os.path.join(out_dir, os.path.basename(fp)), img)

        # ---------------------------------------------------------------- time
        ts = os.path.splitext(fp)[0].split("_")[-1]          # YYYYMMDD_HHMMSS
        t_curr = datetime.strptime(ts, DATE_FMT)
        if t0 is None:
            t0 = t_curr
        times.append((t_curr - t0).total_seconds())
        heights.append(h_px)
        print(f"{os.path.basename(fp)} → {int(h_px)} px  @ {times[-1]:.1f}s")

    # save CSV ---------------------------------------------------------------
    csv_path = os.path.join(out_dir, "heights.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "height_px"])
        w.writerows(zip(times, heights))
    # plot -------------------------------------------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(times, heights, marker="o")
    plt.xlabel("Time since first frame (s)")
    plt.ylabel("Height (pixels)")
    plt.title(f"Experiment {exp_id}: Bed Height vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heights.png"), dpi=200)
    plt.close()

    print(f"[height-analysis] data ➜ {csv_path}")
    print(f"[height-analysis] plot ➜ {os.path.join(out_dir,'heights.png')}")
