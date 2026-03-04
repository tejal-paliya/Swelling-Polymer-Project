"""Camera capture module with live detection visualization."""

import cv2
import time
import os
from typing import Tuple, Optional
from pathlib import Path
from ultralytics import YOLO

from src.roi_selector import select_roi, save_roi, load_roi
from src.data_handler import make_save_dir, save_frame, save_metadata
from src.logger import setup_logger
from src.config import Config
from src.realtime_visualizer import RealtimeVisualizer
from src.live_detection_helper import run_detection_on_frame
from src.exceptions import CameraError

# Initialize logger
logger = setup_logger("capture_frames")


def run_camera_capture(
    polymer_solvent_pair: str,
    temperature: str,
    experiment_number: str,
    n_frames_per_minute: int,
    duration_minutes: int,
    camera_index: int = 0,
    base_folder: str = "data/raw",
    config: Config = None,
    show_live_detection: bool = True  # NEW: Enable/disable live detection
) -> None:
    """
    Capture time-lapse images with live detection visualization.
    
    Args:
        polymer_solvent_pair: Identifier for polymer-solvent combination
        temperature: Temperature condition
        experiment_number: Unique experiment identifier
        n_frames_per_minute: Number of frames to capture per minute
        duration_minutes: Total experiment duration in minutes
        camera_index: Camera device index
        base_folder: Root directory for saving data
        config: Configuration object
        show_live_detection: Whether to show live detection markers (NEW)
    """
    # Load config if not provided
    if config is None:
        config = Config()
    
    # Construct experiment path
    experiment_path = make_save_dir(
        base_folder,
        polymer_solvent_pair,
        temperature,
        experiment_number
    )
    logger.info(f"Experiment directory created: {experiment_path}")

    # Initialize camera — FLIR or OpenCV depending on config
    if config.use_flir:
        from src.flir_camera import FLIRCamera
        cap = FLIRCamera(
            camera_index=camera_index,
            frame_rate=config.flir_frame_rate,
            exposure_us=config.flir_exposure_us,
            gain_db=config.flir_gain_db,
            gamma=config.flir_gamma,
            use_polarization_color=config.flir_polarization_color,
            width=config.flir_width,
            height=config.flir_height,
        )
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        error_msg = f"Error: Unable to open camera with index {camera_index}"
        logger.error(error_msg)
        raise CameraError(error_msg)
    
    ret, frame = cap.read()

    if not ret:
        cap.release()
        error_msg = "Error: Unable to read initial frame from camera"
        logger.error(error_msg)
        raise CameraError(error_msg)
    
    logger.info("Camera initialized successfully")

    # Select and save ROI
    roi = select_roi(frame)
    save_roi(roi, os.path.join(experiment_path, "roi.txt"))
    x, y, w, h = roi
    logger.info(f"ROI selected: x={x}, y={y}, w={w}, h={h}")

    # Load YOLO model for live detection (if enabled)
    detection_model = None
    if show_live_detection:
        try:
            detection_model = YOLO(config.model_path)
            logger.info(f"Loaded detection model for live visualization")
        except Exception as e:
            logger.warning(f"Could not load detection model: {e}")
            logger.info("Continuing without live detection")
            show_live_detection = False

    # Save experiment metadata
    metadata = {
        "polymer_solvent_pair": polymer_solvent_pair,
        "temperature": temperature,
        "experiment_number": experiment_number,
        "roi": f"{x},{y},{w},{h}",
        "n_frames_per_minute": n_frames_per_minute,
        "duration_minutes": duration_minutes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "live_detection": show_live_detection
    }
    save_metadata(metadata, os.path.join(experiment_path, "metadata.txt"))
    logger.info("Metadata saved")

    # Calculate capture parameters
    total_frames = n_frames_per_minute * duration_minutes
    interval = 60.0 / n_frames_per_minute

    logger.info(f"Starting capture: {total_frames} frames over {duration_minutes} minutes")
    logger.info(f"Capture interval: {interval:.2f} seconds")
    if show_live_detection:
        logger.info("Live detection markers: ENABLED")

    # Initialize real-time visualizer
    visualizer = None
    if config.realtime_display_enabled:
        visualizer = RealtimeVisualizer(
            window_name=config.display_window_name,
            show_roi=config.display_show_roi,
            show_stats=config.display_show_stats,
            roi_color=config.display_roi_color,
            roi_thickness=config.display_roi_thickness,
            text_color=config.display_text_color,
            text_font=config.display_text_font,
            text_scale=config.display_text_scale,
            text_thickness=config.display_text_thickness,
            fps_target=config.display_fps_target,
            show_detection_markers=show_live_detection  # Pass flag
        )
        logger.info("Real-time visualization enabled")

    # Capture frames
    frames_saved = 0
    start_time = time.time()
    user_quit = False
    
    for frame_idx in range(total_frames):
        # Calculate timing for this frame
        target_time = start_time + (frame_idx * interval)
        current_time = time.time()
        
        # Update display while waiting
        if visualizer:
            time_to_next = max(0, target_time - current_time)
            
            # Show live feed during wait period
            while current_time < target_time:
                ret, frame = cap.read()
                if not ret:
                    current_time = time.time()
                    continue
                
                # Run detection if enabled
                detections = None
                if show_live_detection and detection_model is not None:
                    try:
                        roi_crop = frame[y:y + h, x:x + w]
                        roi_detections = run_detection_on_frame(detection_model, roi_crop)
                        detections = {
                            cls: (cx + x, cy + y)
                            for cls, (cx, cy) in roi_detections.items()
                        }
                    except Exception as e:
                        logger.warning(f"Detection failed: {e}")
                
                # Show frame with detections
                should_continue = visualizer.show_frame_with_detections(
                    frame=frame,
                    roi=roi,
                    detections=detections,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    elapsed_time=time.time() - start_time,
                    time_to_next=max(0, target_time - current_time),
                    status="Capturing"
                )
                
                if not should_continue:
                    user_quit = True
                    logger.info("User requested quit")
                    break
                
                # Small sleep to avoid busy waiting
                time.sleep(min(0.033, max(0.001, target_time - current_time)))
                current_time = time.time()
            
            if user_quit:
                break
        else:
            # No visualization, just wait
            wait_time = target_time - current_time
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to capture frame {frame_idx}. Skipping.")
            continue
        
        # Extract ROI
        roi_frame = frame[y:y + h, x:x + w]

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"frame_exp{experiment_number}_{frame_idx:04d}_{timestamp}.png"
        
        try:
            save_path = save_frame(roi_frame, experiment_path, fname)
            frames_saved += 1
            logger.info(f"Saved frame {frame_idx + 1}/{total_frames}: {fname}")
        except IOError as e:
            logger.error(f"Failed to save frame {frame_idx}: {e}")
            continue

    # Cleanup
    if visualizer:
        visualizer.destroy_window()
    
    cap.release()
    cv2.destroyAllWindows()
    
    if user_quit:
        logger.info(f"Capture stopped by user. {frames_saved}/{total_frames} frames saved to: {experiment_path}")
    else:
        logger.info(f"Capture complete! {frames_saved}/{total_frames} frames saved to: {experiment_path}")
