import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

def load_yolo_model():
    model = YOLO('src/analysis/model/best.pt')
    return model

CLASSES = ['bed_height', 'vial_base', 'vial_lid_top']

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
    t0 = mod_times[0]

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
        has_base = 'vial_base' in centroids
        has_bed = 'bed_height' in centroids
        has_lid = 'vial_lid_top' in centroids

        current_time = os.path.getmtime(img_path)
        timestamp = current_time - t0

        if not (has_base and has_bed):
            print(f"Frame {idx}: vial_base or bed_height not detected – skipping.")
            continue

        base_pt = centroids['vial_base']
        bed_pt = centroids['bed_height']
        base_y, bed_y = base_pt[1], bed_pt[1]
        bed_height_px = abs(bed_y - base_y)

        if has_lid:
            lid_pt = centroids['vial_lid_top']
            lid_y = lid_pt[1]
            vial_height_px = abs(lid_y - base_y)
            if vial_height_px != 0:
                px_per_mm = vial_height_px / vial_height_mm
                bed_height_mm = bed_height_px / px_per_mm
                smoothed_value = smoother.update(bed_height_mm)
                results.append((timestamp, round(bed_height_mm,1), round(smoothed_value,1), bed_height_px))
                print(f"Frame {idx}: t={timestamp:.1f}s, height={bed_height_mm:.1f}mm, smoothed={smoothed_value:.1f}mm")
            else:
                results.append((timestamp, None, None, bed_height_px))
                print(f"Frame {idx}: t={timestamp:.1f}s, vial_height_px=0 (invalid), raw px={bed_height_px}")
        else:
            results.append((timestamp, None, None, bed_height_px))
            print(f"Frame {idx}: t={timestamp:.1f}s, vial lid not detected, raw px={bed_height_px}")

        # Annotate image in all cases
        disp = img.copy()
        cv2.circle(disp, base_pt, 8, (0,0,255), -1)
        cv2.circle(disp, bed_pt, 8, (0,0,255), -1)
        cv2.line(disp, base_pt, (base_pt[0], bed_pt[1]), (0,0,255), 2)
        # Write bed height either in mm or in pixels
        if has_lid and vial_height_px != 0:
            text = f"Height: {bed_height_mm:.1f} mm (smoothed: {smoothed_value:.1f})"
        else:
            text = f"Height: {bed_height_px:.1f} px"
        cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
        ann_fname = f"annot_{os.path.basename(img_path)}"
        save_path = os.path.join(latest_images_dir, ann_fname)
        cv2.imwrite(save_path, disp)

    # Save CSV
    out_csv = os.path.join(latest_images_dir, "bed_height_vs_time.csv")
    df = pd.DataFrame(results, columns=["seconds", "bed_height_mm", "smoothed_mm", "bed_height_px"])
    df.to_csv(out_csv, index=False)
    print(f"Saved swelling kinetics CSV: {out_csv}")

    # Plot both mm if available and px (always)
    plt.figure(figsize=(8,5))
    if df['bed_height_mm'].notnull().any():
        plt.plot(df["seconds"], df["bed_height_mm"], 'o', label="Bed height (mm, where lid detected)")
        plt.plot(df["seconds"], df["smoothed_mm"], '-', label="Bed height smoothed (mm)")
    plt.plot(df["seconds"], df["bed_height_px"], 's', label="Bed height (pixels, raw)")
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Bed Height (mm or pixels)")
    plt.legend()
    plt.grid(True)
    plt.title("Polymer Swelling Kinetics")
    plot_path = os.path.join(latest_images_dir, "bed_height_vs_time.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved swelling kinetics plot: {plot_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = os.path.join("data", "raw")
    run_height_analysis(folder)
