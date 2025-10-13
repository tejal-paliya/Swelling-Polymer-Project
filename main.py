from src.capture_frames import run_camera_capture
if __name__ == "__main__":
    polymer_solvent_pair = input("Enter polymer-solvent pair (e.g., 'PolymerA-SolventB'): ")
    temperature = input("Enter temperature (e.g., '25C'): ")
    experiment_number = input("Enter experiment number (e.g., '1'): ")
    n_frames_per_minute = int(input("Enter number of frames per minute (e.g., '30'): "))
    duration_minutes = int(input("Enter duration in minutes (e.g., '10'): "))

run_camera_capture(
        polymer_solvent_pair,
        temperature,
        experiment_number,
        n_frames_per_minute,
        duration_minutes
)
