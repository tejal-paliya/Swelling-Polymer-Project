"""Custom exceptions for the polymer bed analysis project."""

class PolymerBedError(Exception):
    """Base exception for polymer bed analysis."""
    pass

class CameraError(PolymerBedError):
    """Raised when camera-related operations fail."""
    pass

class ROIError(PolymerBedError):
    """Raised when ROI selection or loading fails."""
    pass

class ModelError(PolymerBedError):
    """Raised when YOLO model operations fail."""
    pass

class DetectionError(PolymerBedError):
    """Raised when object detection fails."""
    pass

class DataError(PolymerBedError):
    """Raised when data handling operations fail."""
    pass
