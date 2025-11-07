import cv2
import time
import os
from src.roi_selector import select_roi, save_roi, load_roi
from src.data_handler import make_save_dir, save_frame, save_metadata

def run_camera_capture(
    polymer_solvent_pair, temperature, experiment_number,
    n_frames_per_minute, duration_minutes, camera_index=0, base_folder="data/raw"
):
    """
    Captures images from USB camera with user metadata and ROI selection.
    Creates unique folder for each round of experiment, selects ROI, saves frames and experiment metadata.

    Arguments are:
        polymer_solvent_pair (str): Polymer-solvent input from user.
        temperature (str): Temperature input from user.
        experiment_number (str or int): Experiment ID from user.
        n_frames_per_minute (int): Number of frames to capture per minute.
        duration_minutes (int): Total experiment time in minutes.
        camera_index (int): Index of camera to use (default 0).
        base_folder (str): Root folder for saving images.
    """
    # Construct path: base/polymer_solvent_pair/temperature/experiment_number/
    experiment_path = make_save_dir(base_folder, polymer_solvent_pair, temperature, experiment_number)

    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read initial frame from camera.")
        cap.release()
        return

    # Select ROI on the initial frame
    roi = select_roi(frame)
    save_roi(roi, os.path.join(experiment_path, "roi.txt"))
    x, y, w, h = roi

    # Save metadata for experiment
    metadata = {
        "polymer_solvent_pair": polymer_solvent_pair,
        "temperature": temperature,
        "experiment_number": experiment_number,
        "roi": f"{x},{y},{w},{h}",
        "n_frames_per_minute": n_frames_per_minute,
        "duration_minutes": duration_minutes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_metadata(metadata, os.path.join(experiment_path, "metadata.txt"))

    total_frames = n_frames_per_minute * duration_minutes
    interval = 60.0 / n_frames_per_minute

    print(f"Capturing {total_frames} frames over {duration_minutes} minutes.")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to capture frame {frame_idx}. Skipping.")
            continue
        roi_frame = frame[y:y + h, x:x + w]

        # Logical save name: frame_{exp}_{idx:04d}_{timestamp}.png
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"frame_exp{experiment_number}_{frame_idx:04d}_{timestamp}.png"
        save_path = os.path.join(experiment_path, fname)
        save_frame(roi_frame, experiment_path, fname)

        print(f"Saved: {save_path}")
        time.sleep(interval)

    cap.release()
    print(f"All frames saved to: {experiment_path}")
