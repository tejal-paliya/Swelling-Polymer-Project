import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def analyze_bead_swelling_all_frames(data_root="data/raw"):
    """
    Analyzes *all* frames in the most recent experiment folder,
    detects the bead diameter in each frame (no blur applied),
    and plots the swelling (diameter in pixels) vs. frame index.
    """
    # Find most recent experiment folder
    experiment_folders = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_root) for d in dn]
    if not experiment_folders:
        print(f"No experiment folders found in {data_root}")
        return None
    latest_folder = max(experiment_folders, key=os.path.getmtime)

    # Find all images in this folder
    image_files = sorted(glob.glob(os.path.join(latest_folder, "frame_*.png")))
    if not image_files:
        print(f"No images found in {latest_folder}")
        return None

    diameters = []
    frame_indices = []

    for idx, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if img is None:
            diameters.append(np.nan)
            continue

        # Canny edge detection (no blur)
        edges = cv2.Canny(img, 50, 150)

        # Set expected range for bead radii (customize as needed)
        min_radius = 20
        max_radius = 100

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
            param1=50, param2=30,
            minRadius=min_radius, maxRadius=max_radius
        )

        bead_diameter_pixels = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            img_h, img_w = img.shape[:2]
            best_circle = min(
                circles[0],
                key=lambda c: abs(c[1] - img_h * 0.8) + abs(c[0] - img_w / 2)
            )
            x, y, r = best_circle
            bead_diameter_pixels = r * 2
        else:
            bead_diameter_pixels = np.nan  # No bead found in this frame

        diameters.append(bead_diameter_pixels)
        frame_indices.append(idx)

    # Plotting swelling graph
    plt.figure(figsize=(8,5))
    plt.plot(frame_indices, diameters, marker='o')
    plt.xlabel("Frame Number")
    plt.ylabel("Bead Diameter (pixels)")
    plt.title("Bead Swelling (Pixel Diameter vs. Frame)")
    plt.grid()
    plt.show()
    return diameters

# Example usage:
if __name__ == "__main__":
    analyze_bead_swelling_all_frames()
