import yaml
from pathlib import Path
from typing import Dict, Any, List

class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        Example: config.get('camera.index') returns 0
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def camera_index(self) -> int:
        return self.get('camera.index', 0)
    
    @property
    def base_folder(self) -> str:
        return self.get('paths.base_folder', 'data/raw')
    
    @property
    def model_path(self) -> str:
        return self.get('paths.model_path', 'src/analysis/model/best.pt')
    
    @property
    def smoothing_window(self) -> int:
        return self.get('analysis.smoothing_window', 5)
    
    @property
    def classes(self) -> list:
        return self.get('analysis.classes', ['bed_height', 'vial_base', 'vial_lid_top'])

    # Smoothing configuration properties
    
    @property
    def smoothing_method(self) -> str:
        """Get the smoothing method to use."""
        return self.get('analysis.smoothing.method', 'running_mean')
    
    @property
    def running_mean_window(self) -> int:
        return self.get('analysis.smoothing.running_mean.window_size', 5)
    
    @property
    def kalman_process_variance(self) -> float:
        return self.get('analysis.smoothing.kalman.process_variance', 0.01)
    
    @property
    def kalman_measurement_variance(self) -> float:
        return self.get('analysis.smoothing.kalman.measurement_variance', 0.1)
    
    @property
    def kalman_initial_estimate(self) -> float:
        return self.get('analysis.smoothing.kalman.initial_estimate', 10.0)
    
    @property
    def kalman_initial_error(self) -> float:
        return self.get('analysis.smoothing.kalman.initial_error', 1.0)
    
    @property
    def savgol_window_length(self) -> int:
        return self.get('analysis.smoothing.savgol.window_length', 11)
    
    @property
    def savgol_polyorder(self) -> int:
        return self.get('analysis.smoothing.savgol.polyorder', 3)
    
    @property
    def ema_alpha(self) -> float:
        return self.get('analysis.smoothing.ema.alpha', 0.3)
    
    @property
    def median_window_size(self) -> int:
        return self.get('analysis.smoothing.median.window_size', 5)
    
    @property
    def smoothing_comparison_enabled(self) -> bool:
        return self.get('analysis.smoothing.comparison.enabled', True)
    
    @property
    def smoothing_save_all_results(self) -> bool:
        return self.get('analysis.smoothing.comparison.save_all_results', True)

    # Real time display properties
    @property
    def realtime_display_enabled(self) -> bool:
        return self.get('realtime_display.enabled', True)
    
    @property
    def display_window_name(self) -> str:
        return self.get('realtime_display.window_name', 'Live Capture')
    
    @property
    def display_show_roi(self) -> bool:
        return self.get('realtime_display.show_roi', True)
    
    @property
    def display_show_stats(self) -> bool:
        return self.get('realtime_display.show_stats', True)
    
    @property
    def display_roi_color(self) -> List[int]:
        return self.get('realtime_display.roi_color', [0, 255, 0])
    
    @property
    def display_roi_thickness(self) -> int:
        return self.get('realtime_display.roi_thickness', 2)
    
    @property
    def display_text_color(self) -> List[int]:
        return self.get('realtime_display.text_color', [0, 255, 0])
    
    @property
    def display_text_font(self) -> int:
        return self.get('realtime_display.text_font', 0)
    
    @property
    def display_text_scale(self) -> float:
        return self.get('realtime_display.text_scale', 0.6)
    
    @property
    def display_text_thickness(self) -> int:
        return self.get('realtime_display.text_thickness', 2)
    
    @property
    def display_fps_target(self) -> int:
        return self.get('realtime_display.fps_target', 30)

    @property
    def display_show_detection_markers(self) -> bool:
        return self.get('realtime_display.show_detection_markers', True)

    @property
    def kinetic_analysis_enabled(self) -> bool:
        return self.get('analysis.kinetic_analysis.enabled', True)

    @property
    def kinetic_analysis_min_points(self) -> int:
        return self.get('analysis.kinetic_analysis.min_points', 10)

    @property
    def solvent_molar_volume(self) -> float:
        """
        Molar volume of the solvent in m3/mol, used in the Flory-Rehner equation.

        Set in config.yaml under:
            analysis.kinetic_analysis.solvent_molar_volume_m3_per_mol

        Defaults to water (1.8e-5 m3/mol) if not specified.

        Common values:
            Water   : 1.8e-5  m3/mol
            Ethanol : 5.84e-5 m3/mol
            Methanol: 4.07e-5 m3/mol
        """
        return self.get(
            'analysis.kinetic_analysis.solvent_molar_volume_m3_per_mol',
            1.8e-5
        )
    
    @property
    def use_flir(self) -> bool:
        return self.get('camera.use_flir', False)

    @property
    def flir_frame_rate(self) -> float:
        return self.get('flir.frame_rate', 10.0)

    @property
    def flir_exposure_us(self):          # float or None
        return self.get('flir.exposure_us', None)

    @property
    def flir_gain_db(self):              # float or None
        return self.get('flir.gain_db', None)

    @property
    def flir_width(self):                # int or None
        return self.get('flir.width', None)

    @property
    def flir_height(self):               # int or None
        return self.get('flir.height', None)
    
    @property
    def flir_gamma(self):        # float or None
        return self.get('flir.gamma', None)

    @property
    def flir_polarization_color(self) -> bool:
        return self.get('flir.polarization_color', True)

    @property
    def temporal_prediction_enabled(self) -> bool:
        return self.get('temporal_prediction.enabled', False)

    @property
    def temporal_model_path(self) -> str:
        return self.get('temporal_prediction.model_path', 'src/analysis/model/temporal_predictor.pt')

    @property
    def temporal_frame_interval_s(self) -> float:
        return self.get('temporal_prediction.frame_interval_s', 2.0)
