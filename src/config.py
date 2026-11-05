"""Central configuration module for the EV Motor Fault Detection System."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Ensure directories exist
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, SAVED_MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Model artifact paths
RF_MODEL_PATH = SAVED_MODELS_DIR / "random_forest_model.joblib"
ISOLATION_FOREST_PATH = SAVED_MODELS_DIR / "isolation_forest_model.joblib"
SCALER_PATH = SAVED_MODELS_DIR / "feature_scaler.joblib"
FEATURE_METADATA_PATH = SAVED_MODELS_DIR / "feature_metadata.json"
EVALUATION_METRICS_PATH = SAVED_MODELS_DIR / "evaluation_metrics.json"

# Operating States / Fault Classes
CLASS_HEALTHY = "HEALTHY"
CLASS_ROTOR_IMBALANCE = "ROTOR_IMBALANCE"
CLASS_MECHANICAL_STALL = "MECHANICAL_STALL"

FAULT_CLASSES: List[str] = [
    CLASS_HEALTHY,
    CLASS_ROTOR_IMBALANCE,
    CLASS_MECHANICAL_STALL,
]

CLASS_LABELS_MAP: Dict[str, int] = {
    CLASS_HEALTHY: 0,
    CLASS_ROTOR_IMBALANCE: 1,
    CLASS_MECHANICAL_STALL: 2,
}

INV_CLASS_LABELS_MAP: Dict[int, str] = {v: k for k, v in CLASS_LABELS_MAP.items()}

# Canonical ordered feature list for training and inference
FEATURE_COLUMNS: List[str] = [
    # Vibration features (9)
    "vib_rms",
    "vib_peak_to_peak",
    "vib_mean",
    "vib_std",
    "vib_variance",
    "vib_kurtosis",
    "vib_dominant_freq_hz",
    "vib_spectral_energy",
    "vib_band_energy_50_1000hz",
    # Current features (5)
    "current_mean",
    "current_rms",
    "current_peak",
    "current_variation",
    "current_fluctuation",
    # Temperature features (4)
    "temp_current_c",
    "temp_change_c",
    "temp_moving_avg_c",
    "temp_rise_rate_c_per_s",
]

FEATURE_UNITS: Dict[str, str] = {
    "vib_rms": "g",
    "vib_peak_to_peak": "g",
    "vib_mean": "g",
    "vib_std": "g",
    "vib_variance": "g²",
    "vib_kurtosis": "-",
    "vib_dominant_freq_hz": "Hz",
    "vib_spectral_energy": "g²/Hz",
    "vib_band_energy_50_1000hz": "g²",
    "current_mean": "A",
    "current_rms": "A",
    "current_peak": "A",
    "current_variation": "A",
    "current_fluctuation": "%",
    "temp_current_c": "°C",
    "temp_change_c": "°C",
    "temp_moving_avg_c": "°C",
    "temp_rise_rate_c_per_s": "°C/s",
}


@dataclass
class SignalConfig:
    """Configuration for signal acquisition and digital filtering."""

    sampling_rate_hz: float = 1000.0  # Default 1 kHz sampling for full 50-450Hz+ Nyquist support; adjustable
    window_size: int = 256  # Samples per window (~0.256s at 1kHz)
    bandpass_low_hz: float = 50.0  # Lower cutoff of target bandpass filter (Hz)
    bandpass_high_hz: float = 1000.0  # Target upper cutoff (clamped automatically below Nyquist)
    bandpass_order: int = 4
    outlier_z_threshold: float = 4.5
    moving_avg_window: int = 5


@dataclass
class SerialConfig:
    """Configuration for ESP32 USB Serial communication."""

    default_port: str = "COM3"
    default_baud_rate: int = 115200
    timeout_s: float = 1.0
    reconnect_delay_s: float = 2.0
    max_buffer_records: int = 2000


@dataclass
class ThresholdConfig:
    """Domain thresholds used alongside ML predictions for multi-modal sensor fusion."""

    # Vibration thresholds (in g, dynamic acceleration magnitude / axis)
    vib_rms_warning_g: float = 0.35
    vib_rms_critical_g: float = 0.65
    vib_p2p_warning_g: float = 1.60
    vib_band_energy_warning: float = 8.0

    # Motor Current thresholds (Amperes)
    nominal_current_a: float = 2.2
    current_rms_warning_a: float = 3.5
    current_rms_stall_a: float = 5.5
    current_fluctuation_warning_pct: float = 18.0

    # Temperature thresholds (Celsius)
    nominal_temp_c: float = 38.0
    temp_warning_c: float = 55.0
    temp_critical_c: float = 70.0
    temp_rise_rate_warning_c_per_s: float = 0.25


SIGNAL_CONFIG = SignalConfig()
SERIAL_CONFIG = SerialConfig()
THRESHOLD_CONFIG = ThresholdConfig()

