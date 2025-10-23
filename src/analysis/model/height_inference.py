import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "src/analysis/model/best.pt"
OUT_DIRNAME = "frame_capture"                    # top-level subfolder for annotated frames

def _sort_by_index(names):
    """Utility: numeric sort on ..._####_ timestamp."""
    def key(fp):
        stem = os.path.splitext(os.path.basename(fp))[0]
        parts = stem.split("_")
        try:
            return int(parts[2])            #  frame_exp<id>_0007_...
        except (ValueError, IndexError):
            return 0
    return sorted(names, key=key)

def run_height_analysis(experiment_dir):
    """
    1. loads best.pt,
    2. iterates over PNGs in experiment_dir,
    3. writes annotated images to experiment_dir/../frame_capture/<exp_id>/,
    4. saves heights.csv and heights.png in that folder,
    5. returns list of heights.
    """
    model = YOLO(MODEL_PATH)
    frame_paths = _sort_by_index(glob.glob(os.path.join(experiment_dir, "*.png")))
    if not frame_paths:
        print(f"No PNG frames found in {experiment_dir}")
        return []

    # Derive numeric experiment id from the first filename
    first = os.path.basename(frame_paths[0])
    exp_id = first.split("_")[1][3:] if "exp" in first else "unknown"
    save_root = os.path.join("data", OUT_DIRNAME, exp_id)
    os.makedirs(save_root, exist_ok=True)

    heights, times = [], []
    for idx, fp in enumerate(frame_paths):
        # ------------------------------------------------------------------ inference
        results = model(fp)[0]                           # Results object for one frame
        boxes = results.boxes
        if len(boxes) < 2:
            print(f"Warning: <2 boxes in {fp}; skipping")
            continue
        # Extract class-wise y coordinates
        y_bed   = [int(b.xyxy[0][1]) for b in boxes if int(b.cls[0]) == 0]   # class 0 = bed_height
        y_base  = [int(b.xyxy[0][3]) for b in boxes if int(b.cls[0]) == 1]   # class 1 = vial_base
        if not y_bed or not y_base:
            print(f"Warning: missing one class in {fp}; skipping")
            continue
        bed_y  = min(y_bed)      # uppermost bed surface
        base_y = max(y_base)     # lowermost vial base
        height = base_y - bed_y
        # ------------------------------------------------------------------ annotate frame
        img = cv2.imread(fp)
        # Draw boxes already rendered by results.plot(boxes=True, labels=False), but we add our own:
        for b in boxes:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            color = (0,255,0) if int(b.cls[0])==0 else (0,0,255)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(img, f"h={height}px", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        # ------------------------------------------------------------------ save
        tgt = os.path.join(save_root, os.path.basename(fp))
        cv2.imwrite(tgt, img)
        heights.append(height)
        times.append(idx)                      # one index per frame

    # ---------------------------------------------------------------------- persist CSV
    csv_path = os.path.join(save_root, "heights.csv")
    np.savetxt(csv_path, np.column_stack([times, heights]),
               delimiter=",", header="frame_idx,height_px", fmt="%d")
    # ---------------------------------------------------------------------- plot
    plt.figure(figsize=(8,5))
    plt.plot(times, heights, marker="o")
    plt.xlabel("Frame index")
    plt.ylabel("Bed height (pixels)")
    plt.title(f"Experiment {exp_id}: height vs. time")
    plt.grid(True)
    plot_path = os.path.join(save_root, "heights.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Annotated frames  ➜ {save_root}")
    print(f"CSV data          ➜ {csv_path}")
    print(f"Height plot       ➜ {plot_path}")
    return heights
