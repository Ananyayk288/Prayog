#include "signal_processing.h"
#include <math.h>

EdgeSignalProcessor::EdgeSignalProcessor(
    float sampling_rate_hz,
    float low_cutoff_hz,
    float high_cutoff_hz
)
    : _fs(sampling_rate_hz),
      _eff_low_hz(low_cutoff_hz),
      _eff_high_hz(high_cutoff_hz),
      _b0(1.0f), _b1(0.0f), _b2(0.0f),
      _a1(0.0f), _a2(0.0f),
      _z1(0.0f), _z2(0.0f),
      _dc_estimate(0.0f) {
    configureBandpass(sampling_rate_hz, low_cutoff_hz, high_cutoff_hz);
}

void EdgeSignalProcessor::resetState() {
    _z1 = 0.0f;
    _z2 = 0.0f;
    _dc_estimate = 0.0f;
}

void EdgeSignalProcessor::configureBandpass(
    float sampling_rate_hz,
    float low_cutoff_hz,
    float high_cutoff_hz
) {
    _fs = (sampling_rate_hz > 20.0f) ? sampling_rate_hz : 500.0f;
    float nyquist = 0.5f * _fs;
    float max_safe_high = 0.45f * _fs; // Guard band below Nyquist

    _eff_low_hz = (low_cutoff_hz > 1.0f) ? low_cutoff_hz : 50.0f;
    if (_eff_low_hz >= max_safe_high) {
        _eff_low_hz = 0.15f * _fs;
    }

    _eff_high_hz = (high_cutoff_hz < max_safe_high) ? high_cutoff_hz : max_safe_high;
    if (_eff_high_hz <= _eff_low_hz + 5.0f) {
        _eff_high_hz = nyquist * 0.90f;
    }

    // RBJ Audio EQ Cookbook 2nd-order Butterworth Bandpass design
    float f0 = sqrtf(_eff_low_hz * _eff_high_hz);
    float bw = _eff_high_hz - _eff_low_hz;
    float q = (bw > 1.0f) ? (f0 / bw) : 0.7071f;
    if (q < 0.2f) q = 0.2f;

    float omega0 = 2.0f * PI * f0 / _fs;
    float sin_w0 = sinf(omega0);
    float cos_w0 = cosf(omega0);
    float alpha = sin_w0 / (2.0f * q);

    float a0 = 1.0f + alpha;
    _b0 = (alpha) / a0;
    _b1 = 0.0f;
    _b2 = (-alpha) / a0;
    _a1 = (-2.0f * cos_w0) / a0;
    _a2 = (1.0f - alpha) / a0;
}

float EdgeSignalProcessor::processVibrationSample(float raw_accel_g) {
    // Single-pole DC blocker before bandpass biquad
    _dc_estimate = 0.985f * _dc_estimate + 0.015f * raw_accel_g;
    float x = raw_accel_g - _dc_estimate;

    // Direct Form II Transposed Biquad step
    float y = _b0 * x + _z1;
    _z1 = _b1 * x - _a1 * y + _z2;
    _z2 = _b2 * x - _a2 * y;
    return y;
}

EdgeWindowFeatures EdgeSignalProcessor::computeWindowFeatures(
    const float* filtered_vib,
    const float* current_buf,
    uint16_t length,
    float latest_temp_c
) {
    EdgeWindowFeatures feat = {0, 0, 0, 0, 0, 0, 0, latest_temp_c};
    if (length == 0 || filtered_vib == nullptr || current_buf == nullptr) {
        return feat;
    }

    float v_sum = 0.0f;
    float v_abs_sum = 0.0f;
    float v_sq_sum = 0.0f;
    float v_min = filtered_vib[0];
    float v_max = filtered_vib[0];

    float c_sum = 0.0f;
    float c_sq_sum = 0.0f;
    float c_peak = 0.0f;

    for (uint16_t i = 0; i < length; i++) {
        float v = filtered_vib[i];
        v_sum += v;
        v_abs_sum += fabs(v);
        v_sq_sum += (v * v);
        if (v < v_min) v_min = v;
        if (v > v_max) v_max = v;

        float c = current_buf[i];
        c_sum += c;
        c_sq_sum += (c * c);
        if (fabs(c) > c_peak) c_peak = fabs(c);
    }

    float n = static_cast<float>(length);
    float v_signed_mean = v_sum / n;
    feat.vib_rms = sqrtf(v_sq_sum / n);
    feat.vib_peak_to_peak = v_max - v_min;
    feat.vib_mean = v_abs_sum / n;

    float var_sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float diff = filtered_vib[i] - v_signed_mean;
        var_sum += diff * diff;
    }
    feat.vib_std = sqrtf(var_sum / n);

    feat.current_mean = c_sum / n;
    feat.current_rms = sqrtf(c_sq_sum / n);
    feat.current_peak = c_peak;
    return feat;
}

