#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include <Arduino.h>

/**
 * Configurable IIR Biquad Bandpass Filter & Window Feature Extractor for ESP32.
 *
 * Target Frequency Range: 50 Hz - 1000 Hz (configurable).
 * IMPORTANT NYQUIST NOTE:
 * A digital filter cannot have a cutoff frequency >= 0.5 * sampling_rate_hz.
 * When the ESP32 samples the MPU6050 over I2C at e.g. 500 Hz (Nyquist = 250 Hz) or
 * 1000 Hz (Nyquist = 500 Hz), this filter automatically clamps the upper passband edge
 * to 0.45 * sampling_rate_hz so the biquad coefficients remain stable and well-behaved.
 */
struct EdgeWindowFeatures {
    float vib_rms;
    float vib_peak_to_peak;
    float vib_mean;
    float vib_std;
    float current_mean;
    float current_rms;
    float current_peak;
    float temperature_c;
};

class EdgeSignalProcessor {
public:
    EdgeSignalProcessor(
        float sampling_rate_hz = 500.0f,
        float low_cutoff_hz = 50.0f,
        float high_cutoff_hz = 1000.0f
    );

    void configureBandpass(float sampling_rate_hz, float low_cutoff_hz, float high_cutoff_hz);
    void resetState();
    float processVibrationSample(float raw_accel_g);

    static EdgeWindowFeatures computeWindowFeatures(
        const float* filtered_vib,
        const float* current_buf,
        uint16_t length,
        float latest_temp_c
    );

    float getEffectiveLowHz() const { return _eff_low_hz; }
    float getEffectiveHighHz() const { return _eff_high_hz; }

private:
    float _fs;
    float _eff_low_hz;
    float _eff_high_hz;

    // Direct Form II Transposed Biquad coefficients
    float _b0, _b1, _b2;
    float _a1, _a2;
    float _z1, _z2;
    float _dc_estimate;
};

#endif // SIGNAL_PROCESSING_H

