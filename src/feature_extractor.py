"""Window-based multi-modal feature extraction for EV Motor Fault Detection.

Extracts 18 canonical features across:
- Vibration (RMS, Peak-to-Peak, Mean, Std, Variance, Kurtosis, FFT Dominant Frequency,
  FFT Spectral Energy, Frequency-Band Energy 50Hz-1kHz)
- Current (Mean, RMS, Peak, Variation, Fluctuation %)
- Temperature (Current Temp, Temp Change, Moving Average, Rise Rate °C/s)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats

from src.config import FEATURE_COLUMNS, SIGNAL_CONFIG
from src.signal_filters import (
    handle_missing_values,
    clip_outliers,
    remove_dc_offset,
    apply_vibration_bandpass_filter,
    rolling_mean_filter,
)


def compute_fft_spectrum(
    signal_window: np.ndarray,
    sampling_rate_hz: float = SIGNAL_CONFIG.sampling_rate_hz,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided Hann-windowed FFT amplitude spectrum for a real signal."""
    clean = remove_dc_offset(signal_window)
    n = clean.size
    if n < 2 or sampling_rate_hz <= 0:
        return np.array([0.0]), np.array([0.0])

    window = np.hanning(n)
    # Coherent gain compensation for Hann window
    scale = 2.0 / max(float(np.sum(window)), 1e-9)
    fft_vals = np.fft.rfft(clean * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    magnitudes = np.abs(fft_vals) * scale
    magnitudes[0] = 0.0  # Suppress residual DC bin
    return freqs, magnitudes


def extract_vibration_features(
    vib_x: np.ndarray,
    vib_y: Optional[np.ndarray] = None,
    vib_z: Optional[np.ndarray] = None,
    sampling_rate_hz: float = SIGNAL_CONFIG.sampling_rate_hz,
    band_low_hz: float = SIGNAL_CONFIG.bandpass_low_hz,
    band_high_hz: float = SIGNAL_CONFIG.bandpass_high_hz,
) -> Dict[str, Any]:
    """Extract time-domain and frequency-domain vibration features.

    Combines available triaxial channels into a dynamic vibration signal, applies the
    50 Hz - 1 kHz bandpass filter, and computes all required statistical & spectral metrics.
    """
    vx = clip_outliers(handle_missing_values(vib_x))
    dx = remove_dc_offset(vx)

    if vib_y is not None and vib_z is not None:
        vy = clip_outliers(handle_missing_values(vib_y))
        vz = clip_outliers(handle_missing_values(vib_z))
        dy = remove_dc_offset(vy)
        dz = remove_dc_offset(vz)
        # Dynamic triaxial vector sum (signed radial-dominant composite + magnitude)
        composite_raw = dx + 0.7 * dy + 0.5 * dz
    else:
        composite_raw = dx

    filtered_vib = apply_vibration_bandpass_filter(
        composite_raw,
        sampling_rate_hz=sampling_rate_hz,
        low_hz=band_low_hz,
        high_hz=band_high_hz,
    )

    vib_rms = float(np.sqrt(np.mean(np.square(filtered_vib))))
    vib_p2p = float(np.ptp(filtered_vib)) if filtered_vib.size > 0 else 0.0
    vib_mean = float(np.mean( np.abs(filtered_vib) ))  # Mean rectified dynamic vibration amplitude
    vib_std = float(np.std(filtered_vib))
    vib_var = float(np.var(filtered_vib))

    if filtered_vib.size >= 4 and vib_std > 1e-8:
        vib_kurt = float(stats.kurtosis(filtered_vib, fisher=False, bias=False))
        if not np.isfinite(vib_kurt):
            vib_kurt = 3.0
    else:
        vib_kurt = 3.0

    freqs, magnitudes = compute_fft_spectrum(filtered_vib, sampling_rate_hz=sampling_rate_hz)
    psd_energy = np.square(magnitudes)

    if freqs.size > 1 and np.max(magnitudes) > 1e-9:
        dom_idx = int(np.argmax(magnitudes[1:]) + 1)
        dominant_freq_hz = float(freqs[dom_idx])
    else:
        dominant_freq_hz = 0.0

    spectral_energy = float(np.sum(psd_energy))

    eff_high = min(float(band_high_hz), 0.5 * sampling_rate_hz)
    band_mask = (freqs >= float(band_low_hz)) & (freqs <= eff_high)
    band_energy = float(np.sum(psd_energy[band_mask])) if np.any(band_mask) else spectral_energy

    return {
        "vib_rms": vib_rms,
        "vib_peak_to_peak": vib_p2p,
        "vib_mean": vib_mean,
        "vib_std": vib_std,
        "vib_variance": vib_var,
        "vib_kurtosis": vib_kurt,
        "vib_dominant_freq_hz": dominant_freq_hz,
        "vib_spectral_energy": spectral_energy,
        "vib_band_energy_50_1000hz": band_energy,
        # Auxiliary waveform & spectrum arrays for dashboard visualization
        "_filtered_waveform": filtered_vib,
        "_raw_waveform": composite_raw,
        "_fft_freqs": freqs,
        "_fft_magnitudes": magnitudes,
    }


def extract_current_features(current_signal: np.ndarray) -> Dict[str, float]:
    """Extract electrical load/current features from ACS712 window."""
    clean = clip_outliers(handle_missing_values(current_signal, default_value=0.0))
    smoothed = rolling_mean_filter(clean, window_len=3)

    c_mean = float(np.mean(smoothed))
    c_rms = float(np.sqrt(np.mean(np.square(clean))))
    c_peak = float(np.max(np.abs(clean))) if clean.size > 0 else 0.0
    c_variation = float(np.ptp(clean)) if clean.size > 0 else 0.0

    denom = max(abs(c_mean), 0.1)
    c_fluctuation = float((np.std(clean) / denom) * 100.0)

    return {
        "current_mean": c_mean,
        "current_rms": c_rms,
        "current_peak": c_peak,
        "current_variation": c_variation,
        "current_fluctuation": c_fluctuation,
    }


def extract_temperature_features(
    temp_series_or_val: Any,
    temp_history: Optional[List[float]] = None,
    window_duration_s: float = 0.256,
) -> Dict[str, float]:
    """Extract thermal features from DS18B20 temperature readings."""
    arr = np.asarray(temp_series_or_val, dtype=float).reshape(-1)
    clean_arr = handle_missing_values(arr, default_value=38.0)
    current_temp = float(clean_arr[-1])

    if temp_history and len(temp_history) >= 2:
        hist_clean = handle_missing_values(np.asarray(temp_history, dtype=float), default_value=current_temp)
        prev_temp = float(hist_clean[-2])
        recent_slice = hist_clean[-min(10, hist_clean.size):]
        moving_avg = float(np.mean(recent_slice))
    elif clean_arr.size >= 2:
        prev_temp = float(clean_arr[0])
        moving_avg = float(np.mean(clean_arr))
    else:
        prev_temp = current_temp
        moving_avg = current_temp

    temp_change = float(current_temp - prev_temp)
    dt = max(float(window_duration_s), 0.05)
    temp_rise_rate = float(temp_change / dt)

    return {
        "temp_current_c": current_temp,
        "temp_change_c": temp_change,
        "temp_moving_avg_c": moving_avg,
        "temp_rise_rate_c_per_s": temp_rise_rate,
    }


def extract_unified_features(window_data: Dict[str, Any]) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """Extract all 18 canonical features and return (feature_dict, feature_vector, diagnostics).

    Guarantees consistent feature names, units, and exact ordering defined in `FEATURE_COLUMNS`.
    """
    fs = float(window_data.get("sampling_rate_hz", SIGNAL_CONFIG.sampling_rate_hz))
    vib_x = np.asarray(window_data.get("vibration_x", [0.0]), dtype=float)
    vib_y = window_data.get("vibration_y")
    vib_z = window_data.get("vibration_z")
    if vib_y is not None:
        vib_y = np.asarray(vib_y, dtype=float)
    if vib_z is not None:
        vib_z = np.asarray(vib_z, dtype=float)

    current_arr = np.asarray(window_data.get("current", [0.0]), dtype=float)
    temp_input = window_data.get("temperature_series", window_data.get("temperature", 38.0))
    temp_history = window_data.get("temp_history")
    dt_window = float(window_data.get("window_duration_s", max(vib_x.size / max(fs, 1.0), 0.256)))

    vib_res = extract_vibration_features(vib_x, vib_y, vib_z, sampling_rate_hz=fs)
    cur_res = extract_current_features(current_arr)
    tmp_res = extract_temperature_features(temp_input, temp_history=temp_history, window_duration_s=dt_window)

    combined: Dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col in vib_res:
            val = float(vib_res[col])
        elif col in cur_res:
            val = float(cur_res[col])
        elif col in tmp_res:
            val = float(tmp_res[col])
        else:
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        combined[col] = val

    vector = np.array([combined[col] for col in FEATURE_COLUMNS], dtype=float)
    diagnostics = {
        "filtered_vibration": vib_res["_filtered_waveform"],
        "raw_vibration": vib_res["_raw_waveform"],
        "fft_freqs": vib_res["_fft_freqs"],
        "fft_magnitudes": vib_res["_fft_magnitudes"],
    }
    return combined, vector, diagnostics

