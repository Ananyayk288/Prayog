"""Digital signal processing and filtering utilities for EV Motor Fault Detection.

Implements:
- Configurable 50 Hz – 1 kHz Butterworth bandpass filter for vibration signals (with automatic
  Nyquist safety clamping and short-window fallbacks so it never crashes).
- Missing-value interpolation/imputation.
- Outlier clipping via robust median/MAD or Z-score bounds.
- Rolling-window smoothing and signal normalization.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal

from src.config import SIGNAL_CONFIG


def handle_missing_values(arr: np.ndarray, default_value: float = 0.0) -> np.ndarray:
    """Replace NaN and infinite values using linear interpolation across valid indices."""
    clean = np.asarray(arr, dtype=float).copy()
    if clean.ndim == 0:
        clean = clean.reshape(1)
    if clean.size == 0:
        return np.array([default_value], dtype=float)

    finite_mask = np.isfinite(clean)
    if not np.any(finite_mask):
        return np.full_like(clean, fill_value=default_value, dtype=float)

    if not np.all(finite_mask):
        indices = np.arange(clean.size)
        clean[~finite_mask] = np.interp(
            indices[~finite_mask],
            indices[finite_mask],
            clean[finite_mask],
        )
    return clean


def clip_outliers(
    arr: np.ndarray,
    z_threshold: float = SIGNAL_CONFIG.outlier_z_threshold,
) -> np.ndarray:
    """Clip extreme spikes using robust Median Absolute Deviation (MAD) with std fallback."""
    clean = handle_missing_values(arr)
    if clean.size < 4:
        return clean

    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    robust_sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(clean))

    if robust_sigma <= 1e-9:
        return clean

    lower = median - z_threshold * robust_sigma
    upper = median + z_threshold * robust_sigma
    return np.clip(clean, lower, upper)


def remove_dc_offset(arr: np.ndarray) -> np.ndarray:
    """Remove static DC offset (such as 1g gravity on Z-axis) from a vibration signal."""
    clean = handle_missing_values(arr)
    return clean - float(np.mean(clean))


def compute_effective_bandpass_bounds(
    sampling_rate_hz: float,
    low_hz: float = SIGNAL_CONFIG.bandpass_low_hz,
    high_hz: float = SIGNAL_CONFIG.bandpass_high_hz,
) -> Optional[Tuple[float, float, str]]:
    """Determine valid Butterworth filter cutoff frequencies given the sampling rate.

    Returns (eff_low_hz, eff_high_hz, filter_mode) or None if filtering is not possible.
    Wait-free and safe against Nyquist violations (where high_hz >= sampling_rate_hz / 2).
    """
    if sampling_rate_hz <= 0:
        return None

    nyquist = 0.5 * sampling_rate_hz
    # Leave a 2% guard band below Nyquist to avoid digital filter instability at Wn=1.0
    max_allowable_high = 0.96 * nyquist

    if low_hz >= max_allowable_high:
        # Sampling rate is too low even for the 50 Hz high-pass edge; fallback to detrend only
        return None

    eff_low = max(0.5, float(low_hz))
    if high_hz >= max_allowable_high:
        # Target high cutoff (e.g. 1000 Hz at fs=1000 Hz where Nyquist=500 Hz) exceeds Nyquist.
        # Clamp upper cutoff to 0.96 * Nyquist (e.g. 480 Hz) or use highpass above 50 Hz.
        eff_high = max_allowable_high
    else:
        eff_high = float(high_hz)

    if eff_high <= eff_low + 1.0:
        return None

    return (eff_low, eff_high, "bandpass")


def apply_vibration_bandpass_filter(
    vibration_signal: np.ndarray,
    sampling_rate_hz: float = SIGNAL_CONFIG.sampling_rate_hz,
    low_hz: float = SIGNAL_CONFIG.bandpass_low_hz,
    high_hz: float = SIGNAL_CONFIG.bandpass_high_hz,
    order: int = SIGNAL_CONFIG.bandpass_order,
) -> np.ndarray:
    """Apply a configurable 50 Hz – 1 kHz zero-phase Butterworth bandpass filter to vibration data.

    Safeguards:
    - Handles NaNs/Infs and removes DC offset first.
    - Automatically clamps `high_hz` below the Nyquist frequency (`0.5 * sampling_rate_hz`)
      when sampling at e.g. 1000 Hz (Nyquist = 500 Hz).
    - Gracefully falls back to DC removal if the window length or sampling rate is too small
      for `scipy.signal.sosfiltfilt`.
    """
    centered = remove_dc_offset(vibration_signal)
    if centered.size == 0:
        return centered

    bounds = compute_effective_bandpass_bounds(sampling_rate_hz, low_hz, high_hz)
    if bounds is None:
        return centered

    eff_low, eff_high, _ = bounds
    nyquist = 0.5 * sampling_rate_hz
    wn = [eff_low / nyquist, eff_high / nyquist]

    # Reduce filter order if signal window is very short
    effective_order = int(max(1, min(order, (centered.size - 2) // 6)))
    if centered.size < 16 or effective_order < 1:
        return centered

    try:
        sos = signal.butter(effective_order, wn, btype="bandpass", output="sos")
        padlen = min(3 * (2 * len(sos) + 1), centered.size - 1)
        filtered = signal.sosfiltfilt(sos, centered, padlen=padlen)
        return np.asarray(filtered, dtype=float)
    except Exception:
        # Fallback to first-order forward SOS filter or DC-removed signal if numerical edge case occurs
        try:
            sos = signal.butter(1, wn, btype="bandpass", output="sos")
            return np.asarray(signal.sosfilt(sos, centered), dtype=float)
        except Exception:
            return centered


def rolling_mean_filter(
    arr: np.ndarray,
    window_len: int = SIGNAL_CONFIG.moving_avg_window,
) -> np.ndarray:
    """Apply a simple causal rolling-window moving average for noise reduction on slow signals."""
    clean = handle_missing_values(arr)
    w = max(1, min(int(window_len), clean.size))
    if w <= 1:
        return clean
    kernel = np.ones(w, dtype=float) / float(w)
    # Pad left edge with first value to avoid boundary drop-off
    padded = np.pad(clean, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def normalize_signal(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization with zero-division protection."""
    clean = handle_missing_values(arr)
    mean = float(np.mean(clean))
    std = float(np.std(clean))
    if std <= 1e-9:
        return np.zeros_like(clean)
    return (clean - mean) / std

