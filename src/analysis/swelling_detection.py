import cv2
import numpy as np
import os
import glob

def detect_bead_and_display_size(data_root="data/raw"):
    """
    Automatically finds the most recent experiment folder and analyzes the first image
    to detect the bead and display its pixel diameter (no grayscale conversion).
    Args:
        data_root (str): Root directory where experiment folders are stored.
    Returns:
        bead_diameter_pixels (float or None): Diameter of detected bead in pixels, or None if not found.
    """
    # Find the most recent experiment folder
    experiment_folders = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_root) for d in dn]
    if not experiment_folders:
        print(f"No experiment folders found in {data_root}")
        return None
    latest_folder = max(experiment_folders, key=os.path.getmtime)

    # Find the first image in the latest folder
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

    # Apply Gaussian blur to reduce noise (on color image)
    blur = cv2.GaussianBlur(img, (9, 9), 2)

    # Use Canny edge detection to highlight bead edges
    edges = cv2.Canny(blur, 50, 150)

    # Use Hough Circle Transform to detect circles (beads)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )

    bead_diameter_pixels = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find the largest circle (assume it's the bead)
        largest_circle = max(circles[0], key=lambda c: c[2])
        x, y, r = largest_circle
        bead_diameter_pixels = r * 2
        # Draw the detected bead on the image
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.putText(img, f"Diameter: {bead_diameter_pixels} px", (x-40, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Detected bead diameter: {bead_diameter_pixels} pixels")
    else:
        print("No bead detected in the image.")

    # Display the result
    cv2.imshow("Bead Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bead_diameter_pixels
