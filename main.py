"""
Main entry point for polymer bed swelling analysis.

This script orchestrates the complete workflow:
1. Capture time-lapse images from camera
2. Analyze images to measure bed height over time
"""

import os
import sys
import json
from pathlib import Path

from src.capture_frames import run_camera_capture
from src.analysis.model.height_inference import run_height_analysis
from src.config import Config
from src.logger import setup_logger
from src.exceptions import PolymerBedError

# Initialize logger
logger = setup_logger("main")

def get_user_input() -> dict:
    """
    Collect experiment parameters from user via command line.

    Returns:
        Dictionary containing all experiment parameters
    """
    print("\n=== Polymer Swelling Experiment Setup ===\n")

    params = {}

    try:
        params['polymer_solvent_pair'] = input(
            "Enter polymer-solvent pair (e.g., 'PVA-Water'): "
        ).strip()

        params['temperature'] = input(
            "Enter temperature (e.g., '25C'): "
        ).strip()

        params['experiment_number'] = input(
            "Enter experiment number (e.g., '1'): "
        ).strip()

        params['concentration_wv_pct'] = float(input(
            "Enter polymer concentration in w/v% (g per 100 mL, e.g. '5'): "
        ))

        params['polymer_density_g_per_cm3'] = float(input(
            "Enter dry polymer density in g/cm3 "
            "(PVA=1.26, chitosan (med)=0.3, chitosan (low)=0.2, meyprodor 30=1, meyprodor 50=1.2, gelcarin=1.50, genu beta pectin=1.60, CCS=0.5): "
        ))

        params['chi'] = float(input(
            "Enter Flory-Huggins chi parameter "
            "(PVA-water=0.49, chitosan (med)=-0.02, chitosan (low)=-0.01, meyprodor 30=0.05, meyprodor 50=-0.02, gelcarin=0.3, genu beta pectin=0.3, CCS=0.05): "
        ))

        params['n_frames_per_minute'] = int(input(
            "Enter number of frames per minute (e.g., '30'): "
        ))

        params['duration_minutes'] = int(input(
            "Enter duration in minutes (e.g., '10'): "
        ))

        # Validate inputs
        if not params['polymer_solvent_pair']:
            raise ValueError("Polymer-solvent pair cannot be empty")

        if not params['temperature']:
            raise ValueError("Temperature cannot be empty")

        if params['concentration_wv_pct'] <= 0:
            raise ValueError("Concentration must be positive")

        if params['polymer_density_g_per_cm3'] <= 0:
            raise ValueError("Polymer density must be positive")

        if not (-2.0 <= params['chi'] <= 2.0):
            raise ValueError("chi should be between -2 and 2")

        if params['n_frames_per_minute'] <= 0:
            raise ValueError("Frames per minute must be positive")

        if params['duration_minutes'] <= 0:
            raise ValueError("Duration must be positive")

        return params

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nExperiment setup cancelled by user")
        sys.exit(0)


def save_experiment_metadata(params: dict, experiment_path: str) -> None:
    """
    Write experiment metadata to a JSON file alongside the CSV outputs.

    This file enables multi-experiment comparison scripts to read
    concentration, temperature, chi and polymer information without parsing
    folder names or CSV files.

    Args:
        params:          Dictionary returned by get_user_input()
        experiment_path: Path to the experiment output folder
    """
    metadata = {
        "polymer_solvent_pair":      params["polymer_solvent_pair"],
        "temperature":               params["temperature"],
        "concentration_wv_pct":      params["concentration_wv_pct"],
        "polymer_density_g_per_cm3": params["polymer_density_g_per_cm3"],
        "chi":                       params["chi"],
        "experiment_number":         params["experiment_number"],
        "n_frames_per_minute":       params["n_frames_per_minute"],
        "duration_minutes":          params["duration_minutes"],
    }

    metadata_path = os.path.join(experiment_path, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Experiment metadata saved: {metadata_path}")


def main():
    """Main execution function."""
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")

        # Get user input
        params = get_user_input()

        # Log experiment parameters
        logger.info("Experiment parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

        # Run camera capture
        logger.info("\n=== Starting Camera Capture ===\n")
        run_camera_capture(
            polymer_solvent_pair=params['polymer_solvent_pair'],
            temperature=params['temperature'],
            experiment_number=params['experiment_number'],
            n_frames_per_minute=params['n_frames_per_minute'],
            duration_minutes=params['duration_minutes'],
            camera_index=config.camera_index,
            base_folder=config.base_folder,
            config=config
        )

        # Construct experiment path
        experiment_path = os.path.join(
            config.base_folder,
            params['polymer_solvent_pair'],
            str(params['temperature']),
            str(params['experiment_number'])
        )

        # Run height analysis — pass polymer/solvent parameters for kinetic analysis
        logger.info("\n=== Starting Height Analysis ===\n")
        run_height_analysis(
            experiment_path,
            concentration_wv_pct=params['concentration_wv_pct'],
            polymer_density_g_per_cm3=params['polymer_density_g_per_cm3'],
            chi=params['chi'],
        )

        # Save experiment metadata (concentration, temperature, polymer)
        save_experiment_metadata(params, experiment_path)

        logger.info("\n=== Experiment Complete ===\n")
        logger.info(f"Results saved to: {experiment_path}")

    except PolymerBedError as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
