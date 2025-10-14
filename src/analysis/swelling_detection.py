import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def analyze_bead_swelling_all_frames_robust(data_root="data/raw"):
    """
    Robustly analyzes all frames in the most recent experiment folder.
    Uses temporal consistency to filter out jumps in detected diameter.
    """
    experiment_folders = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_root) for d in dn]
    if not experiment_folders:
        print(f"No experiment folders found in {data_root}")
        return None
    latest_folder = max(experiment_folders, key=os.path.getmtime)

    image_files = sorted(glob.glob(os.path.join(latest_folder, "frame_*.png")))
    if not image_files:
        print(f"No images found in {latest_folder}")
        return None

    bead_data = []
    prev_x, prev_y, prev_r = None, None, None
    tolerance_center = 25  # max shift in pixels from previous center
    tolerance_radius = 8   # max change in pixels from previous radius

    for idx, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if img is None:
            bead_data.append(np.nan)
            continue

        edges = cv2.Canny(img, 50, 150)
        min_radius = 20
        max_radius = 100

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
            param1=50, param2=30,
            minRadius=min_radius, maxRadius=max_radius
        )

        # Filter by spatial and size consistency
        if circles is not None:
            circles = np.uint16(np.around(circles))
            img_h, img_w = img.shape[:2]
            if idx == 0:
                best_circle = min(
                    circles[0], key=lambda c: abs(c[1] - img_h * 0.8) + abs(c[0] - img_w / 2)
                )
            else:
                candidates = [
                    c for c in circles[0] if (
                        abs(c[0] - prev_x) < tolerance_center and
                        abs(c[1] - prev_y) < tolerance_center and
                        abs(c[2] - prev_r) < tolerance_radius
                    )
                ]
                if candidates:
                    best_circle = min(candidates, key=lambda c: (c[0]-prev_x)**2 + (c[1]-prev_y)**2)
                else:
                    # No consistent detection: use previous radius (and mark as interpolated)
                    bead_data.append(prev_r*2 if prev_r is not None else np.nan)
                    continue

            x, y, r = best_circle
            prev_x, prev_y, prev_r = x, y, r
            bead_data.append(r*2)
             # Draw the detected circle and save image
            marked_img = img.copy()
            cv2.circle(marked_img, (x, y), r, (0, 255, 0), 2)  # Draw green circle
            cv2.circle(marked_img, (x, y), 2, (0, 0, 255), 3)  # Draw red center point
            folder, fname = os.path.split(image_path)
            marked_path = os.path.join(folder, 'marked_' + fname)
            cv2.imwrite(marked_path, marked_img)
        else:
            # If no detection, use previous radius (stationary assumption)
            bead_data.append(prev_r*2 if prev_r is not None else np.nan)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(bead_data, marker='o')
    plt.xlabel("Frame Number")
    plt.ylabel("Bead Diameter (pixels)")
    plt.title("Bead Swelling (Consistent Detection)")
    plt.grid()
    plt.show()

    return bead_data
