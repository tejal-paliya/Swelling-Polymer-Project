"""
Smoothing filters for time-series data.

Provides multiple smoothing algorithms for reducing noise in
bed height measurements.
"""

import numpy as np
from typing import List, Optional
from scipy.signal import savgol_filter
from collections import deque


class SmootherBase:
    """Base class for all smoothing filters."""
    
    def __init__(self):
        """Initialize the smoother."""
        self.values: List[float] = []
    
    def update(self, value: float) -> float:
        """
        Add a new measurement and return smoothed value.
        
        Args:
            value: New measurement
        
        Returns:
            Smoothed value
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def reset(self) -> None:
        """Clear all stored values."""
        self.values = []
    
    def get_all_values(self) -> List[float]:
        """Get all stored values."""
        return self.values.copy()


class RunningMeanSmoother(SmootherBase):
    """
    Simple running mean (moving average) smoother.
    
    Averages the last N measurements.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize running mean smoother.
        
        Args:
            window_size: Number of values to average
        """
        super().__init__()
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
    
    def update(self, value: float) -> float:
        """Add value and return running mean."""
        self.window.append(value)
        self.values.append(value)
        return np.mean(self.window)


class KalmanSmoother(SmootherBase):
    """
    1D Kalman filter for smoothing measurements.
    
    Optimal filter that balances between measurements and predictions.
    Works well for tracking smooth, continuous changes.
    """
    
    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
        initial_estimate: float = 10.0,
        initial_error: float = 1.0
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_variance: How much we expect true value to change (Q)
            measurement_variance: How noisy measurements are (R)
            initial_estimate: Initial state estimate
            initial_error: Initial estimation error covariance (P)
        """
        super().__init__()
        
        self.Q = process_variance  # Process variance
        self.R = measurement_variance  # Measurement variance
        
        # State variables
        self.x = initial_estimate  # Estimated state
        self.P = initial_error  # Estimation error covariance
        
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """
        Update Kalman filter with new measurement.
        
        Args:
            measurement: New measurement value
        
        Returns:
            Filtered (smoothed) estimate
        """
        if not self.initialized:
            # Use first measurement to initialize
            self.x = measurement
            self.initialized = True
            self.values.append(measurement)
            return measurement
        
        # Prediction step
        x_pred = self.x  # State prediction (assumes constant)
        P_pred = self.P + self.Q  # Error covariance prediction
        
        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)  # State update
        self.P = (1 - K) * P_pred  # Error covariance update
        
        self.values.append(measurement)
        return self.x


class SavgolSmoother(SmootherBase):
    """
    Savitzky-Golay filter smoother.
    
    Fits a polynomial to the data and uses it for smoothing.
    Preserves peaks and features better than simple averaging.
    Requires buffering data before smoothing.
    """
    
    def __init__(self, window_length: int = 11, polyorder: int = 3):
        """
        Initialize Savitzky-Golay smoother.
        
        Args:
            window_length: Window size (must be odd)
            polyorder: Polynomial order (must be less than window_length)
        """
        super().__init__()
        
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.buffer = []
    
    def update(self, value: float) -> float:
        """
        Add value and return smoothed value.
        
        Note: Returns raw value until enough data is buffered.
        """
        self.buffer.append(value)
        self.values.append(value)
        
        # Need at least window_length points to smooth
        if len(self.buffer) < self.window_length:
            return value
        
        # Apply Savitzky-Golay filter to entire buffer
        smoothed = savgol_filter(
            self.buffer,
            window_length=self.window_length,
            polyorder=self.polyorder
        )
        
        # Return the most recent smoothed value
        return smoothed[-1]


class EMASmoother(SmootherBase):
    """
    Exponential Moving Average (EMA) smoother.
    
    Gives exponentially decreasing weights to older measurements.
    More responsive to recent changes than simple moving average.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA smoother.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  Higher alpha = more weight to recent values
                  Lower alpha = smoother but more lag
        """
        super().__init__()
        
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.ema = None
    
    def update(self, value: float) -> float:
        """Add value and return EMA."""
        self.values.append(value)
        
        if self.ema is None:
            # Initialize with first value
            self.ema = value
        else:
            # EMA formula: EMA_t = alpha * value + (1 - alpha) * EMA_(t-1)
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        
        return self.ema


class MedianSmoother(SmootherBase):
    """
    Median filter smoother.
    
    Uses median of last N values instead of mean.
    Very robust against outliers and spikes.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize median smoother.
        
        Args:
            window_size: Number of values to use for median
        """
        super().__init__()
        
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
    
    def update(self, value: float) -> float:
        """Add value and return median."""
        self.window.append(value)
        self.values.append(value)
        return np.median(self.window)


def create_smoother(method: str, config: dict) -> SmootherBase:
    """
    Factory function to create a smoother based on method name.
    
    Args:
        method: Smoothing method name
        config: Configuration dictionary with method-specific parameters
    
    Returns:
        Initialized smoother instance
    
    Raises:
        ValueError: If method is unknown
    """
    method = method.lower()
    
    if method == 'running_mean':
        return RunningMeanSmoother(
            window_size=config.get('window_size', 5)
        )
    
    elif method == 'kalman':
        return KalmanSmoother(
            process_variance=config.get('process_variance', 0.01),
            measurement_variance=config.get('measurement_variance', 0.1),
            initial_estimate=config.get('initial_estimate', 10.0),
            initial_error=config.get('initial_error', 1.0)
        )
    
    elif method == 'savgol':
        return SavgolSmoother(
            window_length=config.get('window_length', 11),
            polyorder=config.get('polyorder', 3)
        )
    
    elif method == 'ema':
        return EMASmoother(
            alpha=config.get('alpha', 0.3)
        )
    
    elif method == 'median':
        return MedianSmoother(
            window_size=config.get('window_size', 5)
        )
    
    else:
        raise ValueError(
            f"Unknown smoothing method: {method}. "
            f"Available: running_mean, kalman, savgol, ema, median"
        )


def apply_all_smoothers(
    values: List[float],
    config_dict: dict
) -> dict:
    """
    Apply all available smoothing methods to a list of values.
    
    Useful for comparing different smoothing approaches.
    
    Args:
        values: List of raw measurements
        config_dict: Configuration dictionary with settings for each method
    
    Returns:
        Dictionary mapping method names to smoothed value lists
    """
    methods = ['running_mean', 'kalman', 'savgol', 'ema', 'median']
    results = {'raw': values.copy()}
    
    for method in methods:
        try:
            # Get method-specific config
            method_config = config_dict.get(method, {})
            
            # Create smoother
            smoother = create_smoother(method, method_config)
            
            # Apply to all values
            smoothed = []
            for value in values:
                smoothed_value = smoother.update(value)
                smoothed.append(smoothed_value)
            
            results[method] = smoothed
            
        except Exception as e:
            print(f"Warning: Failed to apply {method} smoothing: {e}")
            results[method] = values.copy()  # Fallback to raw values
    
    return results
