"""Automated tests for signal filtering, missing values, and Nyquist edge-case handling."""

import numpy as np
from src.signal_filters import (
    handle_missing_values,
    clip_outliers,
    remove_dc_offset,
    compute_effective_bandpass_bounds,
    apply_vibration_bandpass_filter,
    rolling_mean_filter,
    normalize_signal,
)


def test_handle_missing_values_interpolates_nans_and_infs():
    arr = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    cleaned = handle_missing_values(arr)
    assert np.all(np.isfinite(cleaned))
    assert np.isclose(cleaned[1], 2.0)
    assert np.isclose(cleaned[3], 4.0)


def test_handle_all_nan_array_uses_default():
    arr = np.array([np.nan, np.nan, -np.inf])
    cleaned = handle_missing_values(arr, default_value=1.25)
    assert np.allclose(cleaned, 1.25)


def test_clip_outliers_suppresses_extreme_spike():
    rng = np.random.default_rng(10)
    normal_sig = rng.normal(0.0, 0.1, size=100)
    normal_sig[50] = 25.0  # Extreme spike
    clipped = clip_outliers(normal_sig, z_threshold=4.0)
    assert clipped[50] < 2.0
    assert np.all(np.isfinite(clipped))


def test_bandpass_filter_attenuates_dc_and_low_freq_drift():
    fs = 1000.0
    t = np.arange(512) / fs
    # 5 Hz slow drift + 1.0g DC offset + 150 Hz in-band motor vibration
    raw = 1.0 + 0.8 * np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 150.0 * t)
    filtered = apply_vibration_bandpass_filter(raw, sampling_rate_hz=fs, low_hz=50.0, high_hz=1000.0)

    assert filtered.shape == raw.shape
    assert abs(np.mean(filtered)) < 0.02
    # RMS of 0.5 amplitude 150 Hz sine wave is ~0.3535g; ensure 5 Hz drift (0.8 amp) was rejected
    rms = np.sqrt(np.mean(filtered**2))
    assert 0.28 < rms < 0.42


def test_bandpass_filter_does_not_crash_on_low_sampling_rate_or_short_window():
    # Sampling rate below 100 Hz (Nyquist < 50 Hz) -> should safely return DC-removed signal
    short_sig = np.array([1.0, 1.2, 0.8, 1.1])
    res_short = apply_vibration_bandpass_filter(short_sig, sampling_rate_hz=50.0, low_hz=50.0, high_hz=1000.0)
    assert res_short.shape == short_sig.shape
    assert np.all(np.isfinite(res_short))

    # Empty array safety
    res_empty = apply_vibration_bandpass_filter(np.array([]), sampling_rate_hz=1000.0)
    assert res_empty.size == 1


def test_rolling_mean_and_normalization():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = rolling_mean_filter(arr, window_len=3)
    assert smoothed.shape == arr.shape
    normed = normalize_signal(arr)
    assert np.isclose(np.mean(normed), 0.0)
    assert np.isclose(np.std(normed), 1.0)

