import cv2
import os
import numpy as np
from typing import Tuple, Optional

def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Open an interactive window for manual ROI selection.
    
    Args:
        frame: Input image frame (numpy array)
    
    Returns:
        Tuple of (x, y, width, height) representing the selected ROI
    
    Note:
        User should press ENTER to confirm selection or ESC to cancel
    """
    roi = cv2.selectROI(
        "Select ROI (Enter to confirm)",
        frame,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow("Select ROI (Enter to confirm)")
    return roi

def save_roi(roi: Tuple[int, int, int, int], filepath: str) -> None:
    """
    Save ROI coordinates to a text file.
    
    Args:
        roi: Tuple of (x, y, width, height)
        filepath: Path where ROI file will be saved
    
    Format:
        Coordinates are saved as comma-separated values: x,y,w,h
    """
    with open(filepath, "w") as f:
        f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}")

def load_roi(filepath: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Load ROI coordinates from a text file.
    
    Args:
        filepath: Path to the ROI file
    
    Returns:
        Tuple of (x, y, width, height) if file exists, None otherwise
    
    Raises:
        ValueError: If the file format is invalid
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r") as f:
        content = f.read().strip()
        try:
            x, y, w, h = map(int, content.split(","))
            return (x, y, w, h)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid ROI file format in {filepath}: {e}")

