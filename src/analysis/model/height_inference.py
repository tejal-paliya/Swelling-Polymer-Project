"""
Height inference module for polymer bed swelling analysis.

This module uses a trained YOLO model to detect bed height, vial base, and vial lid
in captured images, then calculates the bed height over time.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from src.logger import setup_logger
from src.exceptions import ModelError, DetectionError, DataError
from src.smoothing_filters import create_smoother, apply_all_smoothers
from src.analysis.kinetic_analysis import run_kinetic_analysis
from src.config import Config
from src.analysis.model.temporal_predictor import (
    load_predictor,
    run_temporal_prediction,
    save_temporal_results
)

# Initialize logger
logger = setup_logger("height_inference")

# Class names for YOLO detection
CLASSES = ['bed_height', 'solvent', 'vial_base', 'vial_lid_top']


def load_yolo_model(model_path: str = 'src/analysis/model/best.pt') -> YOLO:
    """
    Load a trained YOLO model for bed height detection.
    
    Args:
        model_path: Path to the YOLO model weights file
    
    Returns:
        Loaded YOLO model instance
    
    Raises:
        ModelError: If model file is not found or cannot be loaded
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        raise ModelError(error_msg)
    
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        model = YOLO(str(model_file))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        error_msg = f"Failed to load YOLO model: {e}"
        logger.error(error_msg)
        raise ModelError(error_msg)


def list_images_in_folder(folder: str) -> List[str]:
    """
    List all frame images in a folder, sorted by modification time.
    
    Only includes PNG files that start with 'frame_'.
    
    Args:
        folder: Path to the folder containing images
    
    Returns:
        List of image file paths, sorted chronologically
    
    Raises:
        DataError: If folder doesn't exist or contains no valid images
    """
    folder_path = Path(folder)
    
    if not folder_path.exists():
        error_msg = f"Folder not found: {folder}"
        logger.error(error_msg)
        raise DataError(error_msg)
    
    if not folder_path.is_dir():
        error_msg = f"Path is not a directory: {folder}"
        logger.error(error_msg)
        raise DataError(error_msg)
    
    try:
        files = [
            (os.path.join(folder, f), os.path.getmtime(os.path.join(folder, f)))
            for f in os.listdir(folder)
            if f.lower().endswith('.png') and f.startswith('frame_')
        ]
        
        if not files:
            error_msg = f"No valid frame images found in {folder}"
            logger.warning(error_msg)
            return []
        
        files_sorted = sorted(files, key=lambda x: x[1])
        image_paths = [f[0] for f in files_sorted]
        
        logger.info(f"Found {len(image_paths)} frame images in {folder}")
        return image_paths
        
    except Exception as e:
        error_msg = f"Error listing images in {folder}: {e}"
        logger.error(error_msg)
        raise DataError(error_msg)


def centroid(box: List[float]) -> Tuple[int, int]:
    """
    Calculate the centroid (center point) of a bounding box.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        Tuple of (center_x, center_y) coordinates
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def run_yolo_on_image(model: YOLO, img: np.ndarray) -> List[Dict]:
    """
    Run YOLO inference on a single image.
    
    Args:
        model: Loaded YOLO model
        img: Input image as numpy array
    
    Returns:
        List of detections, each containing class name, bbox, and confidence
    """
    try:
        results = model(img)
        output = []
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, class_idx, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            class_idx = int(class_idx)
            
            if class_idx >= len(CLASSES):
                logger.warning(f"Invalid class index {class_idx}, skipping detection")
                continue
            
            output.append({
                'class': CLASSES[class_idx],
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf)
            })
        
        return output
        
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        raise DetectionError(f"YOLO inference failed: {e}")


def extract_centroids(detections: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Extract centroid coordinates for each detected class.
    
    Args:
        detections: List of detection dictionaries from run_yolo_on_image
    
    Returns:
        Dictionary mapping class names to their centroid coordinates
    """
    centroids = {}
    for item in detections:
        class_name = item['class']
        cx, cy = centroid(item['bbox'])
        centroids[class_name] = (cx, cy)
    
    return centroids


def annotate_image(
    img: np.ndarray,
    base_pt: Tuple[int, int],
    bed_pt: Tuple[int, int],
    bed_height_value: Optional[float],
    smoothed_value: Optional[float],
    has_lid: bool,
    vial_height_px: float,
    smoothing_method: str
) -> np.ndarray:
    """
    Draw annotations on image showing detected points and measurements.
    
    Args:
        img: Input image
        base_pt: Vial base centroid (x, y)
        bed_pt: Bed height centroid (x, y)
        bed_height_value: Measured bed height in mm (or None)
        smoothed_value: Smoothed bed height in mm (or None)
        has_lid: Whether vial lid was detected
        vial_height_px: Vial height in pixels
        smoothing_method: Name of smoothing method used
    
    Returns:
        Annotated image copy
    """
    disp = img.copy()
    
    # Draw detection points
    cv2.circle(disp, base_pt, 8, (0, 0, 255), -1)
    cv2.circle(disp, bed_pt, 8, (0, 0, 255), -1)
    
    # Draw measurement line
    cv2.line(disp, base_pt, (base_pt[0], bed_pt[1]), (0, 0, 255), 2)
    
    # Add text annotation
    if has_lid and vial_height_px != 0 and bed_height_value is not None:
        text = f"Height: {bed_height_value:.2f} mm ({smoothing_method}: {smoothed_value:.2f})"
    else:
        bed_height_px = abs(bed_pt[1] - base_pt[1])
        text = f"Height: {bed_height_px:.2f} px"
    
    cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    
    return disp


def save_results(
    results: List[Tuple],
    output_dir: str,
    filename_prefix: str = "bed_height_vs_time",
    smoothing_method: str = "running_mean"
) -> Tuple[str, str]:
    """
    Save analysis results as CSV and plot.
    
    Args:
        results: List of tuples (timestamp, bed_height_mm, smoothed_mm, bed_height_px)
        output_dir: Directory to save outputs
        filename_prefix: Prefix for output filenames
        smoothing_method: Name of smoothing method used
    
    Returns:
        Tuple of (csv_path, plot_path)
    """
    # Create DataFrame
    df = pd.DataFrame(
        results,
        columns=["seconds", "bed_height_mm", "smoothed_mm", "bed_height_px"]
    )
    
    # Round all numeric columns to 2 decimal places
    df = df.round(2)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV: {csv_path}")
    
    # Create two-panel figure
    has_mm_data = df['bed_height_mm'].notnull().any()
    
    fig, (ax_mm, ax_px) = plt.subplots(
        2, 1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={'hspace': 0.12}
    )
    
    # ── Panel 1: mm height ────────────────────────────────────────────────────
    if has_mm_data:
        ax_mm.scatter(
            df["seconds"],
            df["bed_height_mm"],
            s=16, alpha=0.45, color='#4C72B0',
            label="Raw (mm)", zorder=2
        )
        ax_mm.plot(
            df["seconds"],
            df["smoothed_mm"],
            '-', linewidth=2, color='#C44E52',
            label=f"Smoothed — {smoothing_method.replace('_', ' ').title()}"
        )
        ax_mm.set_ylabel("Bed Height (mm)", fontsize=12)
        ax_mm.legend(loc='lower right', fontsize=10)
        ax_mm.grid(True, alpha=0.25)
        ax_mm.set_title(
            f"Polymer Bed Swelling — {smoothing_method.replace('_', ' ').title()}",
            fontsize=14, pad=10
        )
    else:
        ax_mm.text(
            0.5, 0.5, "No calibrated mm data\n(lid not detected in any frame)",
            ha='center', va='center', transform=ax_mm.transAxes,
            fontsize=11, color='grey'
        )
        ax_mm.set_ylabel("Bed Height (mm)", fontsize=12)
        ax_mm.set_title("Polymer Bed Swelling (mm)", fontsize=14)
    
    # ── Panel 2: pixel height ─────────────────────────────────────────────────
    ax_px.scatter(
        df["seconds"],
        df["bed_height_px"],
        s=16, alpha=0.45, color='#55A868',
        label="Raw (pixels)", zorder=2
    )
    ax_px.set_xlabel("Time since start (seconds)", fontsize=12)
    ax_px.set_ylabel("Bed Height (pixels)", fontsize=12)
    ax_px.legend(loc='lower right', fontsize=10)
    ax_px.grid(True, alpha=0.25)
    ax_px.set_title("Bed Height — Pixel Measurements (uncalibrated)", fontsize=13)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{filename_prefix}.png")
    fig.savefig(plot_path, dpi=900, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved results plot: {plot_path}")
    
    return csv_path, plot_path


def save_comparison_results(
    raw_values: List[float],
    timestamps: List[float],
    smoothed_results: dict,
    output_dir: str
) -> Tuple[str, str]:
    """
    Save comparison of all smoothing methods.
    
    Args:
        raw_values: List of raw measurements (mm)
        timestamps: List of timestamps (seconds)
        smoothed_results: Dictionary mapping method names to smoothed values
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (csv_path, plot_path)
    """
    # Create DataFrame with all methods
    data = {
        'seconds': timestamps,
        'raw': raw_values
    }
    
    for method, values in smoothed_results.items():
        if method != 'raw':
            data[method] = values
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "bed_height_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV: {csv_path}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: All methods together
    ax1.plot(df['seconds'], df['raw'], 'o', label='Raw', markersize=3, alpha=0.3)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, method in enumerate(['running_mean', 'kalman', 'savgol', 'ema', 'median']):
        if method in df.columns:
            ax1.plot(
                df['seconds'],
                df[method],
                '-',
                label=method.replace('_', ' ').title(),
                linewidth=2,
                color=colors[i]
            )
    
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Bed Height (mm)", fontsize=12)
    ax1.set_title("Comparison of All Smoothing Methods", fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in on a section (middle 20%)
    mid_idx = len(df) // 2
    window = len(df) // 10
    start_idx = max(0, mid_idx - window)
    end_idx = min(len(df), mid_idx + window)
    
    df_zoom = df.iloc[start_idx:end_idx]
    
    ax2.plot(df_zoom['seconds'], df_zoom['raw'], 'o', label='Raw', markersize=5, alpha=0.5)
    
    for i, method in enumerate(['running_mean', 'kalman', 'savgol', 'ema', 'median']):
        if method in df_zoom.columns:
            ax2.plot(
                df_zoom['seconds'],
                df_zoom[method],
                '-',
                label=method.replace('_', ' ').title(),
                linewidth=2,
                color=colors[i],
                marker='.'
            )
    
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Bed Height (mm)", fontsize=12)
    ax2.set_title("Zoomed View (Middle Section)", fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "bed_height_comparison.png")
    plt.savefig(plot_path, dpi=900)
    plt.close()
    logger.info(f"Saved comparison plot: {plot_path}")
    
    return csv_path, plot_path


def run_height_analysis(
    parent_experiment_folder: str,
    config: Optional[Config] = None,
    concentration_wv_pct: Optional[float] = None,
    polymer_density_g_per_cm3: Optional[float] = None,
    chi: Optional[float] = None,
) -> None:
    """
    Run complete height analysis pipeline on captured images.

    This function:
    1. Loads images from the experiment folder
    2. Runs YOLO detection on each image
    3. Calculates bed height in pixels and mm
    4. Applies selected smoothing method
    5. Optionally compares all smoothing methods
    6. Saves annotated images, CSV data, and plots
    7. Runs kinetic analysis (PSO, Peppas, first-order, Flory-Rehner)

    Args:
        parent_experiment_folder:  Path to folder containing experiment images
        config:                    Configuration object (optional, will create if None)
        concentration_wv_pct:      Polymer concentration in w/v% (g per 100 mL).
                                   Written to kinetic_parameters.csv.
        polymer_density_g_per_cm3: Dry polymer density (g/cm3) for phi_0 conversion.
                                   PVA=1.26, chitosan=1.35, starch=1.50, gelcarin=1.60
        chi:                       Flory-Huggins interaction parameter.
                                   Enables Flory-Rehner crosslink density calculation.
                                   PVA-water=0.49, chitosan-water=0.45, gelcarin-water=0.43

    Raises:
        DataError:      If no images found or folder structure invalid
        ModelError:     If model loading fails
        DetectionError: If YOLO inference fails
    """
    logger.info(f"Starting height analysis for: {parent_experiment_folder}")
    
    # Load config if not provided
    if config is None:
        config = Config()
    
    # Get smoothing configuration
    smoothing_method = config.smoothing_method
    logger.info(f"Using smoothing method: {smoothing_method}")
    
    # Find the latest subfolder if multiple exist
    parent_path = Path(parent_experiment_folder)
    if not parent_path.exists():
        error_msg = f"Experiment folder not found: {parent_experiment_folder}"
        logger.error(error_msg)
        raise DataError(error_msg)
    
    subfolders = [
        os.path.join(parent_experiment_folder, sf)
        for sf in os.listdir(parent_experiment_folder)
        if os.path.isdir(os.path.join(parent_experiment_folder, sf))
    ]
    
    if subfolders:
        latest_images_dir = max(subfolders, key=os.path.getmtime)
        logger.info(f"Found subfolders, using latest: {latest_images_dir}")
    else:
        latest_images_dir = parent_experiment_folder
        logger.info(f"No subfolders found, using parent: {latest_images_dir}")
    
    # List all images
    image_files = list_images_in_folder(latest_images_dir)
    
    if not image_files:
        error_msg = "No PNG images found in the experiment folder"
        logger.error(error_msg)
        raise DataError(error_msg)
    
    logger.info(f"Found {len(image_files)} image files to analyze")
    
    # Get reference times
    mod_times = [os.path.getmtime(f) for f in image_files]
    t0 = mod_times[0]
    logger.info(f"Reference time (t0): {t0}")
    
    # Get vial height for calibration
    try:
        vial_height_mm = float(input("Enter the ACTUAL vial height (base to lid top) in mm: "))
        if vial_height_mm <= 0:
            raise ValueError("Vial height must be positive")
        logger.info(f"Vial height for calibration: {vial_height_mm} mm")
    except ValueError as e:
        logger.error(f"Invalid vial height input: {e}")
        raise DataError(f"Invalid vial height: {e}")
    
    # Create smoother based on selected method
    smoother_config = {
        'running_mean': {'window_size': config.running_mean_window},
        'kalman': {
            'process_variance': config.kalman_process_variance,
            'measurement_variance': config.kalman_measurement_variance,
            'initial_estimate': config.kalman_initial_estimate,
            'initial_error': config.kalman_initial_error
        },
        'savgol': {
            'window_length': config.savgol_window_length,
            'polyorder': config.savgol_polyorder
        },
        'ema': {'alpha': config.ema_alpha},
        'median': {'window_size': config.median_window_size}
    }
    
    smoother = create_smoother(smoothing_method, smoother_config[smoothing_method])
    logger.info(f"Initialized {smoothing_method} smoother")
    
    # Load model
    model = load_yolo_model(config.model_path)
    
    # Process each image
    results = []
    raw_values_mm = []  # For comparison
    timestamps = []
    frames_processed = 0
    frames_with_calibration = 0
    
    logger.info("Starting image processing...")
    
    for idx, img_path in enumerate(image_files):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue
        
        # Run YOLO detection
        try:
            detections = run_yolo_on_image(model, img)
        except DetectionError as e:
            logger.warning(f"Detection failed for frame {idx}: {e}")
            continue
        
        # Extract centroids
        centroids = extract_centroids(detections)
        
        has_base = 'vial_base' in centroids
        has_bed = 'bed_height' in centroids
        has_lid = 'vial_lid_top' in centroids
        
        # Calculate timestamp
        current_time = os.path.getmtime(img_path)
        timestamp = current_time - t0
        
        # Check if we have minimum required detections
        if not (has_base and has_bed):
            logger.warning(
                f"Frame {idx}: vial_base or bed_height not detected — skipping"
            )
            continue
        
        frames_processed += 1
        
        # Get detection points
        base_pt = centroids['vial_base']
        bed_pt = centroids['bed_height']
        base_y, bed_y = base_pt[1], bed_pt[1]
        bed_height_px = abs(bed_y - base_y)
        
        # Calculate height in mm if lid is detected
        bed_height_mm = None
        smoothed_value = None
        vial_height_px = 0
        
        if has_lid:
            lid_pt = centroids['vial_lid_top']
            lid_y = lid_pt[1]
            vial_height_px = abs(lid_y - base_y)
            
            if vial_height_px != 0:
                px_per_mm = vial_height_px / vial_height_mm
                bed_height_mm = bed_height_px / px_per_mm
                smoothed_value = smoother.update(bed_height_mm)
                frames_with_calibration += 1
                
                # Store for comparison
                raw_values_mm.append(bed_height_mm)
                timestamps.append(timestamp)
                
                results.append((
                    round(timestamp, 2),
                    round(bed_height_mm, 2),
                    round(smoothed_value, 2),
                    round(bed_height_px, 2)
                ))
                
                logger.info(
                    f"Frame {idx}: t={timestamp:.2f}s, "
                    f"height={bed_height_mm:.2f}mm, "
                    f"smoothed={smoothed_value:.2f}mm"
                )
            else:
                results.append((round(timestamp, 2), None, None, round(bed_height_px, 2)))
                logger.warning(
                    f"Frame {idx}: t={timestamp:.2f}s, "
                    f"vial_height_px=0 (invalid), raw px={bed_height_px:.2f}"
                )
        else:
            results.append((round(timestamp, 2), None, None, round(bed_height_px, 2)))
            logger.info(
                f"Frame {idx}: t={timestamp:.2f}s, "
                f"vial lid not detected, raw px={bed_height_px:.2f}"
            )
        
        # Annotate and save image
        annotated_img = annotate_image(
            img=img,
            base_pt=base_pt,
            bed_pt=bed_pt,
            bed_height_value=bed_height_mm,
            smoothed_value=smoothed_value,
            has_lid=has_lid,
            vial_height_px=vial_height_px,
            smoothing_method=smoothing_method
        )
        
        ann_fname = f"annot_{os.path.basename(img_path)}"
        save_path = os.path.join(latest_images_dir, ann_fname)
        cv2.imwrite(save_path, annotated_img)
    
    # Log processing summary
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Total frames: {len(image_files)}")
    logger.info(f"  Frames processed: {frames_processed}")
    logger.info(f"  Frames with calibration: {frames_with_calibration}")
    
    if not results:
        logger.warning("No valid results to save")
        return
    
    # Save results with selected smoothing method
    csv_path, plot_path = save_results(
        results,
        latest_images_dir,
        smoothing_method=smoothing_method
    )
    
    # Generate comparison if enabled
    if config.smoothing_comparison_enabled and len(raw_values_mm) > 0:
        logger.info("\nGenerating smoothing comparison...")
        
        try:
            smoothed_results = apply_all_smoothers(raw_values_mm, smoother_config)
            
            comp_csv, comp_plot = save_comparison_results(
                raw_values_mm,
                timestamps,
                smoothed_results,
                latest_images_dir
            )
            
            logger.info(f"Comparison CSV saved: {comp_csv}")
            logger.info(f"Comparison plot saved: {comp_plot}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison: {e}")
    
    # Run kinetic analysis if enabled
    if config.kinetic_analysis_enabled and len(raw_values_mm) >= config.kinetic_analysis_min_points:
        logger.info("\nRunning kinetic analysis...")
        if concentration_wv_pct is not None:
            logger.info(f"  Concentration      : {concentration_wv_pct:.2f} w/v%")
        if polymer_density_g_per_cm3 is not None:
            logger.info(f"  Polymer density    : {polymer_density_g_per_cm3} g/cm3")
        if chi is not None:
            logger.info(f"  Flory-Huggins chi  : {chi}")
        try:
            kinetic_result = run_kinetic_analysis(
                height_mm=np.array(raw_values_mm),
                time_s=np.array(timestamps),
                output_dir=latest_images_dir,
                concentration_wv_pct=concentration_wv_pct,
                polymer_density_g_per_cm3=polymer_density_g_per_cm3,
                chi=chi,
                V1_m3_per_mol=config.solvent_molar_volume,
            )
            if kinetic_result:
                kinetic_csv, kinetic_plot = kinetic_result
                logger.info(f"Kinetic parameters saved: {kinetic_csv}")
                logger.info(f"Kinetic plot saved:       {kinetic_plot}")
        except Exception as e:
            logger.error(f"Kinetic analysis failed: {e}")
    elif config.kinetic_analysis_enabled:
        logger.warning(
            f"Kinetic analysis skipped: only {len(raw_values_mm)} calibrated frames "
            f"(minimum required: {config.kinetic_analysis_min_points})"
        )
    if config.temporal_prediction_enabled:
        temporal_model_path = config.temporal_model_path
        if not os.path.exists(temporal_model_path):
            logger.warning(
                f"Temporal model not found at {temporal_model_path}. "
                f"Train it first using train_temporal_predictor.py"
            )
        else:
            logger.info("\nRunning 3D CNN temporal prediction...")
            try:
                temporal_model, device = load_predictor(temporal_model_path)

                temporal_results = run_temporal_prediction(
                    model=temporal_model,
                    frame_paths=image_files,
                    device=device,
                    known_heights_mm=raw_values_mm
                )

                if temporal_results:
                    t_csv, t_plot = save_temporal_results(
                        temporal_results=temporal_results,
                        output_dir=latest_images_dir,
                        frame_interval_s=config.temporal_frame_interval_s
                    )
                    logger.info(f"Temporal prediction CSV saved: {t_csv}")
                    logger.info(f"Temporal prediction plot saved: {t_plot}")

            except Exception as e:
                logger.error(f"Temporal prediction failed: {e}")
     
    logger.info("\n=== Analysis Complete ===")
    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"Plot saved: {plot_path}")
    logger.info(f"Annotated images saved in: {latest_images_dir}")


if __name__ == "__main__":
    """
    Allow running this module standalone for testing.

    Usage:
        python height_inference.py <experiment_folder> [chi] [density] [concentration]

    Examples:
        python height_inference.py data/raw/gelcarin-water/20/6
        python height_inference.py data/raw/gelcarin-water/20/6 0.43 1.60 3.0
    """
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = os.path.join("data", "raw")

    _chi     = float(sys.argv[2]) if len(sys.argv) > 2 else None
    _density = float(sys.argv[3]) if len(sys.argv) > 3 else None
    _conc    = float(sys.argv[4]) if len(sys.argv) > 4 else None

    try:
        run_height_analysis(
            folder,
            chi=_chi,
            polymer_density_g_per_cm3=_density,
            concentration_wv_pct=_conc,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)
