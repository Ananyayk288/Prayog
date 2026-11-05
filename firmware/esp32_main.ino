/**
 * ESP32 Firmware: Edge-AI Multi-Modal Sensor Fusion System for EV Motor Fault Detection
 *
 * Sensors:
 * 1. MPU6050 (I2C: SDA=21, SCL=22) -> Mechanical Vibration (X, Y, Z in g)
 * 2. ACS712  (ADC1: GPIO 34)       -> Motor Current (Amperes)
 * 3. DS18B20 (1-Wire: GPIO 4)      -> Motor Surface/Winding Temperature (Celsius)
 *
 * Outputs structured JSON records over USB Serial at 115200 baud.
 */

#include <Arduino.h>
#include "sensor_manager.h"
#include "signal_processing.h"

#define SERIAL_BAUD_RATE 115200

// Target bandpass filter bounds (50 Hz - 1000 Hz, clamped automatically below Nyquist)
#define FILTER_LOW_HZ    50.0f
#define FILTER_HIGH_HZ   1000.0f

SensorManager sensorManager;
EdgeSignalProcessor signalProcessor(
    static_cast<float>(TARGET_SAMPLE_RATE_HZ),
    FILTER_LOW_HZ,
    FILTER_HIGH_HZ
);

static float filteredVibWindow[WINDOW_SIZE_SAMPLES];
static float currentWindow[WINDOW_SIZE_SAMPLES];
static uint16_t sampleIndex = 0;

static unsigned long sampleIntervalUs = 1000000UL / TARGET_SAMPLE_RATE_HZ;
static unsigned long lastSampleMicros = 0;

void emitJsonRecord(const RawSensorSample& latest, const EdgeWindowFeatures* edgeFeat) {
    // Emit compact single-line JSON record compatible with Python `serial_reader.py`
    Serial.print("{\"timestamp\":");
    Serial.print(latest.timestamp_ms);
    Serial.print(",\"vibration_x\":");
    Serial.print(latest.vibration_x_g, 4);
    Serial.print(",\"vibration_y\":");
    Serial.print(latest.vibration_y_g, 4);
    Serial.print(",\"vibration_z\":");
    Serial.print(latest.vibration_z_g, 4);
    Serial.print(",\"current\":");
    Serial.print(latest.current_amps, 3);
    Serial.print(",\"temperature\":");
    if (latest.temp_valid) {
        Serial.print(latest.temperature_c, 2);
    } else {
        Serial.print("null");
    }

    if (edgeFeat != nullptr) {
        Serial.print(",\"features\":{");
        Serial.print("\"vib_rms\":");
        Serial.print(edgeFeat->vib_rms, 4);
        Serial.print(",\"vib_peak_to_peak\":");
        Serial.print(edgeFeat->vib_peak_to_peak, 4);
        Serial.print(",\"vib_mean\":");
        Serial.print(edgeFeat->vib_mean, 4);
        Serial.print(",\"vib_std\":");
        Serial.print(edgeFeat->vib_std, 4);
        Serial.print(",\"current_mean\":");
        Serial.print(edgeFeat->current_mean, 3);
        Serial.print(",\"current_rms\":");
        Serial.print(edgeFeat->current_rms, 3);
        Serial.print(",\"current_peak\":");
        Serial.print(edgeFeat->current_peak, 3);
        Serial.print("}");
    }

    Serial.println("}");
}

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(300);

    bool mpuOk = sensorManager.begin();
    signalProcessor.configureBandpass(
        static_cast<float>(TARGET_SAMPLE_RATE_HZ),
        FILTER_LOW_HZ,
        FILTER_HIGH_HZ
    );

    // Emit initial diagnostic status comment (ignored by JSON line parser, helpful in serial monitor)
    Serial.print("# INFO: ESP32 EV Motor Node Initialized | MPU6050=");
    Serial.print(mpuOk ? "OK" : "FAIL");
    Serial.print(" | DS18B20=");
    Serial.print(sensorManager.isTempConnected() ? "OK" : "FAIL");
    Serial.print(" | Bandpass_Effective_Hz=[");
    Serial.print(signalProcessor.getEffectiveLowHz(), 1);
    Serial.print(",");
    Serial.print(signalProcessor.getEffectiveHighHz(), 1);
    Serial.println("]");

    lastSampleMicros = micros();
}

void loop() {
    unsigned long nowUs = micros();
    if (nowUs - lastSampleMicros < sampleIntervalUs) {
        return;
    }
    lastSampleMicros = nowUs;

    RawSensorSample sample = sensorManager.readSynchronizedSample();

    // Composite radial/axial dynamic acceleration for on-device bandpass filtering
    float compositeAccel = sample.vibration_x_g + 0.7f * sample.vibration_y_g + 0.5f * (sample.vibration_z_g - 1.0f);
    float filteredVib = signalProcessor.processVibrationSample(compositeAccel);

    filteredVibWindow[sampleIndex] = filteredVib;
    currentWindow[sampleIndex] = sample.current_amps;
    sampleIndex++;

    // Stream high-rate telemetry every 8 samples (~62.5 Hz record rate at 500 Hz sampling)
    // and attach full window features every WINDOW_SIZE_SAMPLES (128 samples).
    if (sampleIndex >= WINDOW_SIZE_SAMPLES) {
        EdgeWindowFeatures edgeFeat = EdgeSignalProcessor::computeWindowFeatures(
            filteredVibWindow,
            currentWindow,
            WINDOW_SIZE_SAMPLES,
            sample.temperature_c
        );
        emitJsonRecord(sample, &edgeFeat);
        sampleIndex = 0;
    } else if ((sampleIndex % 8) == 0) {
        emitJsonRecord(sample, nullptr);
    }
}

