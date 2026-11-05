# ESP32 Sensor Acquisition & Edge Filtering Firmware

This folder contains the Arduino/ESP32 C++ firmware for the **Edge-AI Multi-Modal Sensor Fusion System for EV Motor Fault Detection**.

## 1. Supported Sensors & Pin Mapping

| Sensor | Measured Quantity | Interface | Default ESP32 Pin | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MPU6050** | 3-Axis Vibration & Gyro | I2C (400 kHz) | `SDA = GPIO 21`, `SCL = GPIO 22` | Configured to `±8g` range, `260 Hz` hardware DLPF |
| **ACS712** | Motor Phase/Supply Current | Analog (ADC1) | `OUT = GPIO 34` | 12-bit ADC (`0–4095`), auto-calibrated zero offset at boot |
| **DS18B20** | Motor Surface Temperature | 1-Wire Digital | `DQ = GPIO 4` | Non-blocking async 10-bit conversion (`~187 ms`), `4.7 kΩ` pull-up required |

All pins, ADC divider ratios (`VOLTAGE_DIVIDER_RATIO`), and ACS712 sensitivity (`ACS712_SENSITIVITY`: `0.185` V/A for 5A, `0.100` V/A for 20A, `0.066` V/A for 30A) are configurable in [`sensor_manager.h`](sensor_manager.h).

## 2. Sampling Rate, Nyquist Limit & Bandpass Filter Notes

- **Target Filter Range**: `50 Hz – 1000 Hz`.
- **MPU6050 & I2C Constraint**: Over a 400 kHz I2C bus alongside ADC oversampling and UART JSON serialization, an ESP32 loop typically achieves **500 Hz to 1000 Hz** synchronized sampling (`TARGET_SAMPLE_RATE_HZ = 500` by default, adjustable up to `1000` Hz).
- **Nyquist Theorem Safeguard**: A digital filter cannot operate at or above the Nyquist frequency ($f_{\text{Nyquist}} = f_s / 2$). In [`signal_processing.cpp`](signal_processing.cpp), `EdgeSignalProcessor::configureBandpass()` automatically clamps the upper cutoff frequency to $0.45 \times f_s$ (e.g., `225 Hz` at `500 Hz` sampling, or `450 Hz` at `1000 Hz` sampling) while keeping the `50 Hz` high-pass cutoff active to reject DC gravity and low-frequency frame sway.

## 3. Required Arduino Libraries

Install the following libraries via the **Arduino IDE Library Manager** or **PlatformIO**:
1. **Adafruit MPU6050** (by Adafruit)
2. **Adafruit Unified Sensor** (by Adafruit)
3. **OneWire** (by Paul Stoffregen)
4. **DallasTemperature** (by Miles Burton)

## 4. Serial JSON Output Format

At `115200` baud, the ESP32 outputs newline-delimited JSON records:

```json
{"timestamp":123456,"vibration_x":0.1200,"vibration_y":0.0800,"vibration_z":1.0300,"current":1.250,"temperature":38.50}
```

At the end of every synchronized window (`128` samples), on-device statistical features are appended under `"features"`:

```json
{"timestamp":123712,"vibration_x":0.1140,"vibration_y":0.0790,"vibration_z":1.0210,"current":1.260,"temperature":38.50,"features":{"vib_rms":0.0982,"vib_peak_to_peak":0.3410,"vib_mean":0.0781,"vib_std":0.0980,"current_mean":1.248,"current_rms":1.251,"current_peak":1.310}}
```

