"""Real-time visualization for camera capture with detection markers."""

import cv2
import time
import numpy as np
from typing import Tuple, Optional, Dict, List

class RealtimeVisualizer:
    """
    Handles real-time display of camera feed during capture.
    
    Features:
    - Shows ROI boundary
    - Displays capture statistics
    - Shows countdown to next capture
    - Shows detection markers (NEW)
    - Allows user to quit early
    """
    
    def __init__(
        self,
        window_name: str = "Live Capture",
        show_roi: bool = True,
        show_stats: bool = True,
        roi_color: Tuple[int, int, int] = (0, 255, 0),
        roi_thickness: int = 2,
        text_color: Tuple[int, int, int] = (0, 255, 0),
        text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
        text_scale: float = 0.6,
        text_thickness: int = 2,
        fps_target: int = 30,
        show_detection_markers: bool = True  # NEW
    ):
        """
        Initialize the visualizer.
        
        Args:
            window_name: Name of the display window
            show_roi: Whether to show ROI boundary
            show_stats: Whether to show statistics overlay
            roi_color: Color for ROI box in BGR
            roi_thickness: Thickness of ROI box
            text_color: Color for text overlay in BGR
            text_font: OpenCV font type
            text_scale: Font scale
            text_thickness: Text thickness
            fps_target: Target display refresh rate
            show_detection_markers: Whether to show detection markers (NEW)
        """
        self.window_name = window_name
        self.show_roi = show_roi
        self.show_stats = show_stats
        self.roi_color = tuple(roi_color)
        self.roi_thickness = roi_thickness
        self.text_color = tuple(text_color)
        self.text_font = text_font
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.fps_target = fps_target
        self.frame_delay = 1.0 / fps_target
        self.show_detection_markers = show_detection_markers  # NEW
        
        self.window_created = False
        
        # Colors for different detection markers
        self.marker_colors = {
            'vial_base': (255, 0, 0),      # Blue
            'bed_height': (0, 255, 255),   # Yellow
            'vial_lid_top': (0, 255, 0)    # Green
        }
    
    def create_window(self) -> None:
        """Create the display window."""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True
    
    def destroy_window(self) -> None:
        """Destroy the display window."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
    
    def draw_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Draw ROI boundary on frame.
        
        Args:
            frame: Input frame
            roi: ROI as (x, y, width, height)
        
        Returns:
            Frame with ROI drawn
        """
        if not self.show_roi:
            return frame
        
        x, y, w, h = roi
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            self.roi_color,
            self.roi_thickness
        )
        
        # Add "ROI" label
        label_pos = (x, y - 10 if y > 30 else y + h + 20)
        cv2.putText(
            frame,
            "ROI",
            label_pos,
            self.text_font,
            self.text_scale,
            self.roi_color,
            self.text_thickness
        )
        
        return frame
    
    def draw_detection_markers(
        self,
        frame: np.ndarray,
        detections: Dict[str, Tuple[int, int]],
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw horizontal markers for detected positions.
        
        Args:
            frame: Input frame
            detections: Dictionary mapping class names to (x, y) centroids
            roi: Optional ROI to extend markers across
        
        Returns:
            Frame with markers drawn
        """
        if not self.show_detection_markers or not detections:
            return frame
        
        h, w = frame.shape[:2]
        
        # Determine line extent
        if roi is not None:
            x_roi, y_roi, w_roi, h_roi = roi
            x_start = x_roi
            x_end = x_roi + w_roi
        else:
            x_start = 0
            x_end = w
        
        # Draw markers for each detection
        for class_name, (cx, cy) in detections.items():
            color = self.marker_colors.get(class_name, (255, 255, 255))
            
            # Draw horizontal line
            cv2.line(
                frame,
                (x_start, cy),
                (x_end, cy),
                color,
                2
            )
            
            # Draw small circles at endpoints
            cv2.circle(frame, (x_start, cy), 5, color, -1)
            cv2.circle(frame, (x_end, cy), 5, color, -1)
            
            # Add label
            label = class_name.replace('_', ' ').title()
            
            # Position label to the right of ROI
            label_x = x_end + 10
            label_y = cy + 5
            
            # Make sure label doesn't go off screen
            if label_x + 100 > w:
                label_x = x_start - 150
            
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                self.text_font,
                self.text_scale,
                color,
                self.text_thickness
            )
        
        return frame
    
    def draw_measurement_lines(
        self,
        frame: np.ndarray,
        detections: Dict[str, Tuple[int, int]],
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw vertical lines showing measurements between detections.
        
        Args:
            frame: Input frame
            detections: Dictionary mapping class names to (x, y) centroids
            roi: Optional ROI for positioning
        
        Returns:
            Frame with measurement lines drawn
        """
        if not self.show_detection_markers or not detections:
            return frame
        
        # Check if we have the key detections
        if 'vial_base' in detections and 'bed_height' in detections:
            base_x, base_y = detections['vial_base']
            bed_x, bed_y = detections['bed_height']
            
            # Draw vertical line showing bed height
            line_x = base_x + 20  # Offset to the right of centroid
            
            cv2.line(
                frame,
                (line_x, base_y),
                (line_x, bed_y),
                (0, 255, 255),  # Yellow
                3
            )
            
            # Draw arrow heads
            cv2.arrowedLine(
                frame,
                (line_x, base_y),
                (line_x, base_y + 20),
                (0, 255, 255),
                2,
                tipLength=0.3
            )
            cv2.arrowedLine(
                frame,
                (line_x, bed_y),
                (line_x, bed_y - 20),
                (0, 255, 255),
                2,
                tipLength=0.3
            )
            
            # Add measurement label
            mid_y = (base_y + bed_y) // 2
            height_px = abs(base_y - bed_y)
            
            cv2.putText(
                frame,
                f"{height_px:.0f} px",
                (line_x + 10, mid_y),
                self.text_font,
                self.text_scale,
                (0, 255, 255),
                self.text_thickness
            )
        
        # Draw vial total height if lid is detected
        if 'vial_base' in detections and 'vial_lid_top' in detections:
            base_x, base_y = detections['vial_base']
            lid_x, lid_y = detections['vial_lid_top']
            
            # Draw on opposite side
            line_x = base_x - 20
            
            cv2.line(
                frame,
                (line_x, base_y),
                (line_x, lid_y),
                (0, 255, 0),  # Green
                2
            )
            
            # Add measurement label
            mid_y = (base_y + lid_y) // 2
            vial_height_px = abs(base_y - lid_y)
            
            cv2.putText(
                frame,
                f"{vial_height_px:.0f} px",
                (line_x - 80, mid_y),
                self.text_font,
                self.text_scale - 0.1,
                (0, 255, 0),
                self.text_thickness - 1
            )
        
        return frame
    
    def draw_stats(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        elapsed_time: float,
        time_to_next: float,
        status: str = "Capturing"
    ) -> np.ndarray:
        """
        Draw statistics overlay on frame.
        
        Args:
            frame: Input frame
            frame_idx: Current frame index (0-based)
            total_frames: Total frames to capture
            elapsed_time: Time elapsed since start (seconds)
            time_to_next: Time until next capture (seconds)
            status: Status message
        
        Returns:
            Frame with stats overlay
        """
        if not self.show_stats:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay box
        overlay = frame.copy()
        box_height = 120
        cv2.rectangle(overlay, (0, 0), (w, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Prepare stats text
        frame_text = f"Frame: {frame_idx + 1}/{total_frames}"
        progress_pct = ((frame_idx + 1) / total_frames) * 100
        progress_text = f"Progress: {progress_pct:.1f}%"
        
        elapsed_min = int(elapsed_time // 60)
        elapsed_sec = int(elapsed_time % 60)
        elapsed_text = f"Elapsed: {elapsed_min}m {elapsed_sec}s"
        
        next_text = f"Next capture: {time_to_next:.1f}s"
        status_text = f"Status: {status}"
        quit_text = "Press 'q' to quit"
        
        # Draw text lines
        y_offset = 20
        line_spacing = 20
        
        texts = [frame_text, progress_text, elapsed_text, next_text, status_text, quit_text]
        
        for i, text in enumerate(texts):
            y_pos = y_offset + i * line_spacing
            cv2.putText(
                frame,
                text,
                (10, y_pos),
                self.text_font,
                self.text_scale,
                self.text_color,
                self.text_thickness
            )
        
        return frame
    
    def show_frame_with_detections(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        detections: Optional[Dict[str, Tuple[int, int]]] = None,
        frame_idx: Optional[int] = None,
        total_frames: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        time_to_next: Optional[float] = None,
        status: str = "Capturing"
    ) -> bool:
        """
        Display frame with overlays including detection markers.
        
        Args:
            frame: Frame to display
            roi: ROI coordinates (optional)
            detections: Dictionary of detected positions (optional)
            frame_idx: Current frame index (optional)
            total_frames: Total frames (optional)
            elapsed_time: Elapsed time (optional)
            time_to_next: Time to next capture (optional)
            status: Status message
        
        Returns:
            True if should continue, False if user pressed 'q'
        """
        display_frame = frame.copy()
        
        # Draw ROI if provided
        if roi is not None:
            display_frame = self.draw_roi(display_frame, roi)
        
        # Draw detection markers if provided
        if detections is not None:
            display_frame = self.draw_detection_markers(display_frame, detections, roi)
            display_frame = self.draw_measurement_lines(display_frame, detections, roi)
        
        # Draw stats if all info provided
        if all(x is not None for x in [frame_idx, total_frames, elapsed_time, time_to_next]):
            display_frame = self.draw_stats(
                display_frame,
                frame_idx,
                total_frames,
                elapsed_time,
                time_to_next,
                status
            )
        
        # Show frame
        self.create_window()
        cv2.imshow(self.window_name, display_frame)
        
        # Check for quit command (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        
        return True
    
    # Keep the original show_frame for backward compatibility
    def show_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: Optional[int] = None,
        total_frames: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        time_to_next: Optional[float] = None,
        status: str = "Capturing"
    ) -> bool:
        """Original show_frame method (without detections)."""
        return self.show_frame_with_detections(
            frame, roi, None, frame_idx, total_frames, elapsed_time, time_to_next, status
        )
    
    def update_display(
        self,
        cap: cv2.VideoCapture,
        roi: Tuple[int, int, int, int],
        frame_idx: int,
        total_frames: int,
        start_time: float,
        time_to_next: float
    ) -> bool:
        """
        Update display with current camera frame.
        
        Args:
            cap: OpenCV VideoCapture object
            roi: ROI coordinates
            frame_idx: Current frame index
            total_frames: Total frames
            start_time: Experiment start time
            time_to_next: Seconds until next capture
        
        Returns:
            True if should continue, False if user quit
        """
        ret, frame = cap.read()
        if not ret:
            return True  # Continue even if frame read fails
        
        elapsed_time = time.time() - start_time
        
        return self.show_frame(
            frame=frame,
            roi=roi,
            frame_idx=frame_idx,
            total_frames=total_frames,
            elapsed_time=elapsed_time,
            time_to_next=time_to_next,
            status="Capturing"
        )
