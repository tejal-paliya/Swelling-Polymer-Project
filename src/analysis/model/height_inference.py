import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

def load_yolo_model():
    model = YOLO('src/analysis/model/best.pt')  # Update path if needed
    return model

CLASSES = ['bed_height', 'vial_base', 'vial_lid_top']  # Adjust if different

def list_images_in_folder(folder):
    files = [
        (os.path.join(folder, f), os.path.getmtime(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if f.lower().endswith('.png') and f.startswith('frame_')
    ]
    files_sorted = sorted(files, key=lambda x: x[1])
    return [f[0] for f in files_sorted]

class RunningMean:
    def __init__(self, window_size=5):
        self.N = window_size
        self.values = []
    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.N:
            self.values.pop(0)
        return np.mean(self.values)

def centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def run_yolo_on_image(model, img):
    results = model(img)
    output = []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    for box, class_idx, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        class_idx = int(class_idx)
        output.append({
            'class': CLASSES[class_idx],
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf)
        })
    return output

def extract_centroids(detections):
    centroids = {}
    for item in detections:
        class_name = item['class']
        cx, cy = centroid(item['bbox'])
        centroids[class_name] = (cx, cy)
    return centroids

def run_height_analysis(parent_experiment_folder):
    print(f"Searching for latest image folder in: {parent_experiment_folder}")
    last_subfolders = [os.path.join(parent_experiment_folder, sf) for sf in os.listdir(parent_experiment_folder)
                       if os.path.isdir(os.path.join(parent_experiment_folder, sf))]
    if last_subfolders:
        latest_images_dir = max(last_subfolders, key=os.path.getmtime)
    else:
        latest_images_dir = parent_experiment_folder
    print(f"Analyzing images in: {latest_images_dir}")

    image_files = list_images_in_folder(latest_images_dir)
    if not image_files:
        print("No PNG images found.")
        return
    print(f"Found {len(image_files)} image files.")

    mod_times = [os.path.getmtime(f) for f in image_files]
    t0 = mod_times[0]  # always first image

    vial_height_mm = float(input("Enter the ACTUAL vial height (base to lid top) in mm: "))
    smoothing_window = 5
    smoother = RunningMean(window_size=smoothing_window)
    model = load_yolo_model()

    results = []

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to open {img_path}")
            continue
        detections = run_yolo_on_image(model, img)
        centroids = extract_centroids(detections)
        if not all(cls in centroids for cls in CLASSES):
            print(f"Frame {idx}: Not all key regions detected.")
            continue
        base_pt = centroids['vial_base']
        bed_pt = centroids['bed_height']
        lid_pt = centroids['vial_lid_top']

        base_y, bed_y, lid_y = base_pt[1], bed_pt[1], lid_pt[1]
        vial_height_px = abs(lid_y - base_y)
        bed_height_px = abs(bed_y - base_y)
        if vial_height_px == 0:
            print(f"Frame {idx}: Zero vial height in pixel, skipping.")
            continue
        px_per_mm = vial_height_px / vial_height_mm
        bed_height_mm = bed_height_px / px_per_mm

        current_time = os.path.getmtime(img_path)
        timestamp = current_time - t0  # this is >= 0

        smoothed = smoother.update(bed_height_mm)
        results.append((timestamp, bed_height_mm, smoothed))

        # Annotate image: red dots and red line
        disp = img.copy()
        cv2.circle(disp, base_pt, 8, (0,0,255), -1)
        cv2.circle(disp, bed_pt, 8, (0,0,255), -1)
        cv2.circle(disp, lid_pt, 8, (0,0,255), -1)
        cv2.line(disp, base_pt, (base_pt[0], bed_pt[1]), (0,0,255), 2)
        # Annotate with 1 decimal
        text = f"Height: {bed_height_mm:.1f} mm (smoothed: {smoothed:.1f})"
        cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        ann_fname = f"annot_{os.path.basename(img_path)}"
        save_path = os.path.join(latest_images_dir, ann_fname)
        cv2.imwrite(save_path, disp)

        print(f"Frame {idx}: t={timestamp:.1f}s, height={bed_height_mm:.1f}mm, smoothed={smoothed:.1f}mm")

    # Save results as CSV (full precision)
    out_csv = os.path.join(latest_images_dir, "bed_height_vs_time.csv")
    df = pd.DataFrame(results, columns=["seconds", "bed_height_mm", "smoothed_mm"])
    df.to_csv(out_csv, index=False)
    print(f"Saved swelling kinetics CSV: {out_csv}")

    # For plotting, use rounded values to 1 dp:
    df_plot = df.copy()
    df_plot["bed_height_mm"] = df_plot["bed_height_mm"].round(1)
    df_plot["smoothed_mm"] = df_plot["smoothed_mm"].round(1)

    # Plot trend
    plt.figure(figsize=(8,5))
    plt.plot(df_plot["seconds"], df_plot["bed_height_mm"], '.', label="Bed height (raw)")
    plt.plot(df_plot["seconds"], df_plot["smoothed_mm"], '-', label="Bed height (smoothed, window {} frames)".format(smoothing_window))
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Bed Height (mm)")
    plt.legend()
    plt.grid(True)
    plt.title("Polymer Swelling Kinetics")
    plot_path = os.path.join(latest_images_dir, "bed_height_vs_time.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved swelling kinetics plot: {plot_path}")

# Direct script execution support
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = os.path.join("data", "raw")
    run_height_analysis(folder)
