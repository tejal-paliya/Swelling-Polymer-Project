import cv2
import os

def select_roi(frame):
    """Opens a window for manual ROI selection on the frame."""
    roi = cv2.selectROI("Select ROI (Enter to confirm)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI (Enter to confirm)")
    return roi

def save_roi(roi, filepath):
    """Saves ROI coordinates as comma-separated values."""
    with open(filepath, "w") as f:
        f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}")

def load_roi(filepath):
    """Loads ROI from file, returning (x, y, w, h) or None."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        x, y, w, h = map(int, f.read().strip().split(","))
        return (x, y, w, h)

