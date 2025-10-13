import cv2
import numpy as np
import os
import glob

def detect_bead_size_no_blur(data_root="data/raw"):
    """
    Finds most recent experiment folder, analyzes the first image,
    detects the bead diameter in pixels (no blur applied).
    Displays detected diameter over the image.
    """
    # Find most recent experiment folder
    experiment_folders = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_root) for d in dn]
    if not experiment_folders:
        print(f"No experiment folders found in {data_root}")
        return None
    latest_folder = max(experiment_folders, key=os.path.getmtime)

    # Find the first image in this folder
    image_files = sorted(glob.glob(os.path.join(latest_folder, "frame_*.png")))
    if not image_files:
        print(f"No images found in {latest_folder}")
        return None
    first_image = image_files[0]
    print(f"Analyzing image: {first_image}")

    # Load the color image
    img = cv2.imread(first_image)
    if img is None:
        print(f"Error: Could not load image {first_image}")
        return None

    # Direct edge detection without blur
    edges = cv2.Canny(img, 50, 150)  # You may need to tune these thresholds

    # Tighten radius bounds (adjust to match your bead size in pixels)
    min_radius = 13   # Try just smaller than expected bead radius
    max_radius = 25   # Try just larger than expected bead radius

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
        param1=50, param2=40,
        minRadius=min_radius, maxRadius=max_radius
    )

    bead_diameter_pixels = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # If multiple, pick the one closest to bottom center (optional)
        img_h, img_w = img.shape[:2]
        best_circle = min(
            circles[0],
            key=lambda c: abs(c[1] - img_h * 0.8) + abs(c[0] - img_w / 2)
        )
        x, y, r = best_circle
        bead_diameter_pixels = r * 2
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.putText(img, f"Diameter: {bead_diameter_pixels} px", (x-40, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Detected bead diameter: {bead_diameter_pixels} pixels")
    else:
        print("No bead detected in the image.")

    cv2.imshow("Bead Detection (No Blur)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bead_diameter_pixels
