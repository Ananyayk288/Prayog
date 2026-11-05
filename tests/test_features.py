"""Automated tests for window-based feature extraction and FFT processing."""

import numpy as np
from src.config import FEATURE_COLUMNS, CLASS_HEALTHY, CLASS_ROTOR_IMBALANCE, CLASS_MECHANICAL_STALL
from src.data_generator import SensorSimulator
from src.feature_extractor import (
    compute_fft_spectrum,
    extract_vibration_features,
    extract_current_features,
    extract_temperature_features,
    extract_unified_features,
)


def test_fft_identifies_known_dominant_frequency():
    fs = 1000.0
    n = 512
    t = np.arange(n) / fs
    target_freq = 140.0
    sig = 0.75 * np.sin(2 * np.pi * target_freq * t)

    freqs, mags = compute_fft_spectrum(sig, sampling_rate_hz=fs)
    dom_freq = freqs[np.argmax(mags)]
    assert abs(dom_freq - target_freq) <= (fs / n) * 1.5


def test_extract_unified_features_returns_exact_canonical_columns():
    sim = SensorSimulator(seed=123)
    window = sim.generate_window(state=CLASS_ROTOR_IMBALANCE)
    feat_dict, feat_vec, diag = extract_unified_features(window)

    assert list(feat_dict.keys()) == FEATURE_COLUMNS
    assert feat_vec.shape == (len(FEATURE_COLUMNS),)
    assert np.all(np.isfinite(feat_vec))
    assert "filtered_vibration" in diag
    assert "fft_freqs" in diag
    assert "fft_magnitudes" in diag


def test_distinct_operating_states_produce_physically_distinct_features():
    sim = SensorSimulator(seed=99)

    w_healthy = sim.generate_window(state=CLASS_HEALTHY)
    f_healthy, _, _ = extract_unified_features(w_healthy)

    w_imb = sim.generate_window(state=CLASS_ROTOR_IMBALANCE)
    f_imb, _, _ = extract_unified_features(w_imb)

    w_stall = sim.generate_window(state=CLASS_MECHANICAL_STALL)
    f_stall, _, _ = extract_unified_features(w_stall)

    # Rotor imbalance must have higher vibration RMS & spectral energy than Healthy
    assert f_imb["vib_rms"] > f_healthy["vib_rms"] * 2.5
    assert f_imb["vib_band_energy_50_1000hz"] > f_healthy["vib_band_energy_50_1000hz"] * 4.0

    # Mechanical stall must have significantly higher current RMS than Healthy and Rotor Imbalance
    assert f_stall["current_rms"] > f_healthy["current_rms"] * 2.5
    assert f_stall["current_rms"] > f_imb["current_rms"] * 1.8

