#include "sensor_manager.h"

static const float GRAVITY_MSS = 9.80665f;

SensorManager::SensorManager()
    : _oneWire(PIN_DS18B20_ONEWIRE),
      _dallasTemp(&_oneWire),
      _mpu_ok(false),
      _temp_ok(false),
      _acs_zero_voltage(ACS712_DEFAULT_ZERO),
      _last_valid_temp_c(38.0f),
      _last_temp_request_ms(0),
      _temp_conversion_delay_ms(750) {}

bool SensorManager::begin() {
    // Initialize I2C at 400 kHz Fast Mode to minimize MPU6050 read latency
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Wire.setClock(400000);

    // 1. Initialize MPU6050
    if (_mpu.begin()) {
        _mpu_ok = true;
        _mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
        _mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        // Set internal Digital Low Pass Filter (DLPF) to 260 Hz (highest bandwidth mode)
        _mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
    } else {
        _mpu_ok = false;
    }

    // 2. Configure ESP32 ADC for ACS712
    analogReadResolution(12);
    analogSetPinAttenuation(PIN_ACS712_ADC, ADC_11db);
    calibrateCurrentZero(150);

    // 3. Initialize DS18B20 Temperature Sensor (non-blocking async conversion mode)
    _dallasTemp.begin();
    if (_dallasTemp.getDeviceCount() > 0) {
        _temp_ok = true;
        _dallasTemp.setResolution(10); // 10-bit resolution (~187ms conversion time)
        _temp_conversion_delay_ms = 200;
        _dallasTemp.setWaitForConversion(false);
        _dallasTemp.requestTemperatures();
        _last_temp_request_ms = millis();
    } else {
        _temp_ok = false;
    }

    return _mpu_ok;
}

void SensorManager::calibrateCurrentZero(uint16_t num_samples) {
    double sum_v = 0.0;
    for (uint16_t i = 0; i < num_samples; i++) {
        int raw = analogRead(PIN_ACS712_ADC);
        float v_pin = (static_cast<float>(raw) / ESP32_ADC_MAX) * ESP32_VREF_VOLTS;
        sum_v += (v_pin * VOLTAGE_DIVIDER_RATIO);
        delayMicroseconds(500);
    }
    float measured_zero = static_cast<float>(sum_v / num_samples);
    if (measured_zero > 1.0f && measured_zero < 4.0f) {
        _acs_zero_voltage = measured_zero;
    } else {
        _acs_zero_voltage = ACS712_DEFAULT_ZERO;
    }
}

float SensorManager::readCurrentAmps() {
    // Oversample 4x to reduce ESP32 ADC thermal noise
    long adc_sum = 0;
    for (int i = 0; i < 4; i++) {
        adc_sum += analogRead(PIN_ACS712_ADC);
    }
    float adc_avg = static_cast<float>(adc_sum) * 0.25f;
    float v_sensor = (adc_avg / ESP32_ADC_MAX) * ESP32_VREF_VOLTS * VOLTAGE_DIVIDER_RATIO;
    float current_a = (v_sensor - _acs_zero_voltage) / ACS712_SENSITIVITY;
    return fabs(current_a);
}

void SensorManager::updateTemperatureAsync() {
    unsigned long now = millis();
    if (now - _last_temp_request_ms >= _temp_conversion_delay_ms) {
        float temp_c = _dallasTemp.getTempCByIndex(0);
        // Handle DS18B20 disconnection (-127.0C) and power-on reset register value (85.0C)
        if (temp_c != DEVICE_DISCONNECTED_C && temp_c > -50.0f && temp_c < 140.0f && temp_c != 85.0f) {
            _last_valid_temp_c = temp_c;
            _temp_ok = true;
        } else {
            _temp_ok = false;
        }
        _dallasTemp.requestTemperatures();
        _last_temp_request_ms = now;
    }
}

RawSensorSample SensorManager::readSynchronizedSample() {
    RawSensorSample sample;
    sample.timestamp_ms = millis();
    sample.mpu_valid = _mpu_ok;
    sample.acs_valid = true;

    if (_mpu_ok) {
        sensors_event_t a, g, temp;
        if (_mpu.getEvent(&a, &g, &temp)) {
            sample.vibration_x_g = a.acceleration.x / GRAVITY_MSS;
            sample.vibration_y_g = a.acceleration.y / GRAVITY_MSS;
            sample.vibration_z_g = a.acceleration.z / GRAVITY_MSS;
            sample.gyro_x_rad_s = g.gyro.x;
            sample.gyro_y_rad_s = g.gyro.y;
            sample.gyro_z_rad_s = g.gyro.z;
        } else {
            sample.mpu_valid = false;
            sample.vibration_x_g = 0.0f;
            sample.vibration_y_g = 0.0f;
            sample.vibration_z_g = 1.0f;
            sample.gyro_x_rad_s = 0.0f;
            sample.gyro_y_rad_s = 0.0f;
            sample.gyro_z_rad_s = 0.0f;
        }
    } else {
        sample.vibration_x_g = 0.0f;
        sample.vibration_y_g = 0.0f;
        sample.vibration_z_g = 1.0f;
        sample.gyro_x_rad_s = 0.0f;
        sample.gyro_y_rad_s = 0.0f;
        sample.gyro_z_rad_s = 0.0f;
    }

    sample.current_amps = readCurrentAmps();
    updateTemperatureAsync();
    sample.temperature_c = _last_valid_temp_c;
    sample.temp_valid = _temp_ok;

    return sample;
}
