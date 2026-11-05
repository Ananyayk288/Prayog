#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================================
// HARDWARE PIN & CALIBRATION CONFIGURATION (ADJUSTABLE)
// ============================================================================
#define PIN_I2C_SDA          21
#define PIN_I2C_SCL          22
#define PIN_ACS712_ADC       34      // ADC1_CH6 (Input-only, safe with Wi-Fi/ADC1)
#define PIN_DS18B20_ONEWIRE  4       // Digital GPIO for 1-Wire bus (with 4.7k pull-up)

// ACS712 Calibration Constants
// Note: ESP32 ADC is 12-bit (0-4095) with 3.3V reference (at 11dB attenuation).
// If using a 5V ACS712 through a 2:3 voltage divider, adjust VOLTAGE_DIVIDER_RATIO.
#define ESP32_ADC_MAX        4095.0f
#define ESP32_VREF_VOLTS     3.30f
#define VOLTAGE_DIVIDER_RATIO 1.50f  // Set to 1.0f if already level-shifted to 3.3V midpoint
#define ACS712_SENSITIVITY   0.100f  // 0.185f for 5A, 0.100f for 20A, 0.066f for 30A module
#define ACS712_DEFAULT_ZERO  2.50f   // Nominal zero-current output voltage (auto-calibrated at boot)

// Sampling & Windowing Configuration
// NOTE ON MPU6050 SAMPLING & NYQUIST:
// Standard I2C (400 kHz Fast Mode) reading 6-axis + ADC allows up to ~500-1000 Hz loop rate.
// By Nyquist's theorem (f_max < fs / 2), a 1000 Hz sampling rate supports bandpass filtering
// up to ~480 Hz, whereas 500 Hz sampling supports 50-240 Hz.
#define TARGET_SAMPLE_RATE_HZ 500
#define WINDOW_SIZE_SAMPLES   128

struct RawSensorSample {
    unsigned long timestamp_ms;
    float vibration_x_g;
    float vibration_y_g;
    float vibration_z_g;
    float gyro_x_rad_s;
    float gyro_y_rad_s;
    float gyro_z_rad_s;
    float current_amps;
    float temperature_c;
    bool mpu_valid;
    bool acs_valid;
    bool temp_valid;
};

class SensorManager {
public:
    SensorManager();
    bool begin();
    void calibrateCurrentZero(uint16_t num_samples = 200);
    RawSensorSample readSynchronizedSample();
    void updateTemperatureAsync();

    bool isMpuConnected() const { return _mpu_ok; }
    bool isTempConnected() const { return _temp_ok; }
    float getZeroCurrentVoltage() const { return _acs_zero_voltage; }

private:
    Adafruit_MPU6050 _mpu;
    OneWire _oneWire;
    DallasTemperature _dallasTemp;

    bool _mpu_ok;
    bool _temp_ok;
    float _acs_zero_voltage;
    float _last_valid_temp_c;
    unsigned long _last_temp_request_ms;
    uint16_t _temp_conversion_delay_ms;

    float readCurrentAmps();
};

#endif // SENSOR_MANAGER_H

