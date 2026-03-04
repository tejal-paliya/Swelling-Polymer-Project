import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

def make_save_dir(
    base: str,
    polymer_solvent_pair: str,
    temperature: str,
    experiment_number: str
) -> str:
    """
    Create a directory structure for saving experiment images.
    
    Directory structure: base/polymer_solvent_pair/temperature/experiment_number/
    
    Args:
        base: Root directory for all experiments
        polymer_solvent_pair: Identifier for polymer-solvent combination
        temperature: Temperature condition (e.g., '25C')
        experiment_number: Unique experiment identifier
    
    Returns:
        Full path to the created directory
    
    Example:
        >>> make_save_dir('data/raw', 'PVA-Water', '25C', '1')
        'data/raw/PVA-Water/25C/1'
    """
    path = os.path.join(base, polymer_solvent_pair, str(temperature), str(experiment_number))
    os.makedirs(path, exist_ok=True)
    return path

def save_frame(frame: np.ndarray, save_dir: str, filename: str) -> str:
    """
    Save an image frame as PNG to specified directory.
    
    Args:
        frame: Image array (numpy array from cv2)
        save_dir: Directory to save the image
        filename: Name of the file (should include .png extension)
    
    Returns:
        Full path to the saved file
    
    Raises:
        IOError: If the frame cannot be saved
    """
    path = os.path.join(save_dir, filename)
    success = cv2.imwrite(path, frame)
    
    if not success:
        raise IOError(f"Failed to save frame to {path}")
    
    return path

def save_metadata(metadata_dict: Dict[str, Any], filepath: str) -> None:
    """
    Save experiment metadata to a text file.
    
    Metadata is saved as key-value pairs, one per line.
    
    Args:
        metadata_dict: Dictionary containing metadata
        filepath: Path where metadata file will be saved
    
    Example:
        >>> metadata = {'temperature': '25C', 'duration': 10}
        >>> save_metadata(metadata, 'experiment/metadata.txt')
    """
    with open(filepath, "w") as f:
        for k, v in metadata_dict.items():
            f.write(f"{k}: {v}\n")

