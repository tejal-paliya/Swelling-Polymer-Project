import cv2
import time
import os

def test_camera_capture(frames_per_minute=1, duration_minutes=1, save_folder="test_frames"):
    """
    Opens camera feed, shows live video, captures frames at intervals,
    and saves them in specified folder.

    Args:
        frames_per_minute (int): Number of frames to save per minute.
        duration_minutes (int): Duration to capture frames (minutes).
        save_folder (str): Folder to save captured frames.
    """

    interval = 60.0 / frames_per_minute
    total_frames = frames_per_minute * duration_minutes

    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)  # Change index if multiple cameras

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Starting camera capture for {duration_minutes} minutes, saving {total_frames} frames...")

    frames_captured = 0
    start_time = time.time()

    while frames_captured < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera Live Feed", frame)

        elapsed = time.time() - start_time
        expected_frames = int(elapsed / interval)

        if frames_captured < expected_frames:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{frames_captured:04d}_{timestamp}.png"
            filepath = os.path.join(save_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved {filepath}")
            frames_captured += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting due to user input")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Camera capture finished. {frames_captured} frames saved to {save_folder}")

if __name__ == "__main__":
    # Run test with default config: 1 frame per minute for 1 minute
    test_camera_capture(frames_per_minute=1, duration_minutes=1)


