"""Helper functions for live detection visualization."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from ultralytics import YOLO


def run_detection_on_frame(
    model: YOLO,
    frame: np.ndarray,
    classes: list = None
) -> Dict[str, Tuple[int, int]]:
    """
    Run YOLO detection and extract centroids.
    
    Args:
        model: YOLO model
        frame: Input frame
        classes: List of class names (from model.names)
    
    Returns:
        Dictionary mapping class names to (x, y) centroids
    """
    if classes is None:
        classes = ['bed_height', 'solvent', 'vial_base', 'vial_lid_top']
    
    results = model(frame, verbose=False)
    
    if len(results[0].boxes) == 0:
        return {}
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()
    
    detections = {}
    
    for box, class_idx in zip(boxes, cls):
        class_idx = int(class_idx)
        if class_idx < len(model.names):
            class_name = model.names[class_idx]
            
            # Calculate centroid
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            detections[class_name] = (cx, cy)
    
    return detections
