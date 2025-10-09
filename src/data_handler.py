import os
import cv2

def make_save_dir(base, polymer_solvent_pair, temperature, experiment_number):
    """
    Creates a directory structure for images, returns the path.
    base/polymer_solvent_pair/temperature/experiment_number/
    Automatically creates any missing folders.
    """
    path = os.path.join(base, polymer_solvent_pair, str(temperature), str(experiment_number))
    os.makedirs(path, exist_ok=True)
    return path

def save_frame(frame, save_dir, filename):
    """
    Saves a frame as a PNG to save_dir/filename.
    """
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, frame)
    return path

def save_metadata(metadata_dict, filepath):
    """
    Save experiment metadata as key: value pairs.
    """
    with open(filepath, "w") as f:
        for k, v in metadata_dict.items():
            f.write(f"{k}: {v}\n")
