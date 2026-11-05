# Edge-AI Multi-Modal Sensor Fusion System for EV Motor Fault Detection

An end-to-end predictive maintenance and real-time fault diagnostics system for Electric Vehicle (EV) traction/auxiliary motors. Built for edge-assisted telemetry and real-time analytics using **MPU6050** (triaxial mechanical vibration), **ACS712** (motor supply/phase current), and **DS18B20** (motor surface/winding temperature).

---

## 1. Project Overview

Electric motors subjected to mechanical wear, eccentric loading, or rotor jamming exhibit coupled electromechanical and thermal signatures prior to catastrophic failure. This system acquires synchronized multi-sensor streams from an **ESP32** node (or an integrated physics-informed **Simulation Engine** when hardware is detached), applies digital bandpass filtering (`50 Hz – 1 kHz` target passband with automatic Nyquist protection), extracts **18 time- and frequency-domain features**, and classifies operating health via a **Random Forest + Isolation Forest** machine learning pipeline paired with an explainable **Multi-Modal Sensor Fusion** engine.

---

## 2. Problem Statement

Single-modality threshold alarms (e.g., triggering solely on vibration spikes or temperature limits) suffer from high false-positive rates due to road bumps, transient load changes, or ambient temperature shifts. Conversely, waiting for thermal cutoffs alone reacts too slowly to sudden mechanical stalls. A reliable EV motor health monitor must fuse **mechanical vibration spectra**, **electrical load current dynamics**, and **thermal rise rates** to distinguish normal operation from developing mechanical faults.

---

## 3. Proposed Solution

1. **Multi-Sensor Acquisition (ESP32 / Simulator)**:
   - **MPU6050**: Triaxial acceleration ($a_x, a_y, a_z$) capturing rotational harmonics and structural chatter.
   - **ACS712**: Instantaneous motor current draw capturing load ripple and stall overcurrent surges.
   - **DS18B20**: Digital 1-Wire temperature tracking $I^2R$ and friction heating trends.
2. **Digital Signal Processing (SciPy & On-Device Biquad)**:
   - DC gravity removal, robust MAD outlier clipping, missing-value interpolation, and a configurable **50 Hz – 1 kHz Butterworth bandpass filter** applied specifically to vibration channels.
3. **18-Feature Window Extraction**:
   - Unified feature vector combining 9 vibration statistics/FFT metrics, 5 electrical current metrics, and 4 thermal trend metrics.
4. **Hybrid ML + Multi-Modal Sensor Fusion**:
   - **Random Forest Classifier** predicts `HEALTHY`, `ROTOR_IMBALANCE`, or `MECHANICAL_STALL`.
   - **Isolation Forest** flags novel out-of-distribution anomalies.
   - **Cross-Modal Fusion Rules** corroborate ML predictions against physical sensor envelopes and generate human-readable root-cause explanations.
5. **Real-Time Engineering Dashboard (Streamlit + Plotly)**:
   - Dark technical UI featuring live waveforms, FFT spectrum analysis, KPI cards, Green/Yellow/Red alert center, session CSV export, and confusion matrix evaluation.

---

## 4. System Architecture

```
+-------------------------------------------------------------------+
|                   PHYSICAL / SIMULATED SENSORS                    |
|   [MPU6050 Vibration]   [ACS712 Current]   [DS18B20 Temperature]  |
+-------------------+--------------+----------------+---------------+
                    |              |                |
                    v              v                v
+-------------------------------------------------------------------+
|  ESP32 EDGE NODE (firmware/esp32_main.ino)                        |
|  - Synchronized Sampling & ADC Zero Calibration                   |
|  - On-Device Biquad Bandpass Filter & Window Feature Calculation  |
|  - USB Serial JSON Telemetry Stream (115200 baud)                 |
+----------------------------------+--------------------------------+
                                   |  (Or SensorSimulator in Demo Mode)
                                   v
+-------------------------------------------------------------------+
|  PYTHON PROCESSING & FUSION PIPELINE (src/ & models/)             |
|  1. Serial JSON Parser & Rolling Buffer (src/serial_reader.py)    |
|  2. SciPy 50Hz-1kHz Bandpass & Cleaning (src/signal_filters.py)   |
|  3. 18-Feature Extractor + Hann FFT     (src/feature_extractor.py)|
|  4. Random Forest + Isolation Forest    (models/inference.py)     |
|  5. Multi-Modal Sensor Fusion & Alerts  (src/sensor_fusion.py)    |
+----------------------------------+--------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------+
|  STREAMLIT REAL-TIME DASHBOARD (dashboard/app.py)                 |
|  - KPI Overview Cards, Live Waveforms, Filtered FFT Spectrum,     |
|    Explainable AI Diagnostics, Alert Center & CSV History Export  |
+-------------------------------------------------------------------+
```

---

## 5. Software Architecture

- **Configuration Layer (`src/config.py`)**: Single source of truth for file paths, canonical feature names (`FEATURE_COLUMNS`), physical units (`FEATURE_UNITS`), signal filter settings (`SignalConfig`), serial port defaults (`SerialConfig`), and domain fusion thresholds (`ThresholdConfig`).
- **Acquisition & Simulation Layer (`src/serial_reader.py`, `src/data_generator.py`)**: Decouples hardware serial I/O from downstream processing. Both live ESP32 records and simulated windows feed the exact same processing pipeline.
- **DSP & Feature Layer (`src/signal_filters.py`, `src/feature_extractor.py`, `src/data_processor.py`)**: Stateless, unit-tested mathematical functions for filtering, FFT computation, and feature extraction.
- **ML & Fusion Layer (`models/preprocessing.py`, `models/train_model.py`, `models/inference.py`, `src/sensor_fusion.py`, `src/fault_detection.py`)**: Handles dataset generation, `StandardScaler` normalization, `RandomForestClassifier` + `IsolationForest` training/persistence, and multi-modal fusion logic.
- **Presentation Layer (`dashboard/app.py`, `dashboard/components.py`, `dashboard/charts.py`)**: Modular Streamlit view components and Plotly chart builders.

---

## 6. Technology Stack

- **Embedded Firmware**: C++ / Arduino Framework for ESP32 (`Wire.h`, `Adafruit_MPU6050`, `Adafruit_Sensor`, `OneWire`, `DallasTemperature`)
- **Backend & DSP**: Python 3.11+, `NumPy`, `SciPy`, `Pandas`, `PySerial`
- **Machine Learning**: `scikit-learn` (Random Forest & Isolation Forest), `joblib`
- **Dashboard & Visualization**: `Streamlit`, `Plotly`, `Matplotlib`
- **Automated Testing**: `pytest`

---

## 7. Project Folder Structure

```text
ev-motor-fault-detection/
│
├── firmware/
│   ├── esp32_main.ino          # Main Arduino ESP32 firmware sketch
│   ├── sensor_manager.h        # Pinout, calibration & sensor driver declarations
│   ├── sensor_manager.cpp      # MPU6050, ACS712, DS18B20 initialization & reading
│   ├── signal_processing.h     # Edge biquad filter & window feature declarations
│   ├── signal_processing.cpp   # On-device bandpass filter & statistical computation
│   └── README.md               # Hardware wiring, Nyquist notes & library guide
│
├── data/
│   ├── raw/                    # Captured live serial CSV logs
│   ├── processed/              # Full extracted feature training dataset CSV
│   └── sample/                 # Sample feature CSV for quick inspection
│
├── models/
│   ├── train_model.py          # RF + IsolationForest training & evaluation pipeline
│   ├── inference.py            # Model loader & prediction wrapper
│   ├── preprocessing.py        # Dataset schema validation & StandardScaler
│   └── saved/                  # Persisted .joblib models & JSON metadata/metrics
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Central system & threshold configuration
│   ├── data_generator.py       # Physics-inspired 3-state multi-sensor simulator
│   ├── serial_reader.py        # PySerial COM discovery, JSON parser & auto-reconnect
│   ├── data_processor.py       # Rolling window buffer & missing-sensor handler
│   ├── signal_filters.py       # SciPy 50Hz-1kHz bandpass filter & outlier clipping
│   ├── feature_extractor.py    # 18-feature time/FFT window extraction
│   ├── sensor_fusion.py        # Cross-modal physical & ML decision fusion
│   └── fault_detection.py      # End-to-end analysis facade & alert logger
│
├── dashboard/
│   ├── app.py                  # Main Streamlit application entrypoint
│   ├── components.py           # Dark engineering theme CSS, KPI & alert cards
│   └── charts.py               # Interactive Plotly waveform, FFT & evaluation charts
│
├── tests/
│   ├── test_features.py        # Feature extraction & FFT unit tests
│   ├── test_filters.py         # Bandpass filter, NaN & Nyquist edge-case tests
│   ├── test_inference.py       # Model loading, inference & fusion tests
│   └── test_data.py            # Serial JSON parsing & degraded sensor tests
│
├── scripts/
│   ├── generate_dataset.py     # CLI script to generate synthetic dataset CSVs
│   └── train.py                # CLI script to train & evaluate ML models
│
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Ignored virtualenvs, caches & runtime logs
├── .env.example                # Example environment variables
├── README.md                   # Complete documentation
└── run.bat                     # Windows batch launcher for the Streamlit dashboard
```

---

## 8. Environment Setup

Open **Windows PowerShell** in the project root directory (`d:\Prayog`) and create the isolated virtual environment `.venv`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted on your system, you can run commands directly via `.\.venv\Scripts\python.exe` or use `run.bat`).*

---

## 9. Dependency Installation

With `.venv` active, install all required packages from `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 10. Dataset Generation

To generate the synthetic multi-modal demonstration dataset (`720` labeled windows across `HEALTHY`, `ROTOR_IMBALANCE`, and `MECHANICAL_STALL`):

```powershell
.\.venv\Scripts\python.exe scripts/generate_dataset.py --samples-per-class 240
```

This writes:
- `data/processed/synthetic_training_features.csv`
- `data/sample/sample_sensor_features.csv`

---

## 11. ML Model Training

To train and evaluate the **Random Forest Classifier** (supervised fault classification) and **Isolation Forest** (unsupervised anomaly detection):

```powershell
.\.venv\Scripts\python.exe scripts/train.py
```

To train on a custom CSV recorded from actual hardware:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --csv data/raw/my_recorded_hardware.csv --source-label RECORDED_HARDWARE
```

Saved artifacts are written to `models/saved/`:
- `random_forest_model.joblib`
- `isolation_forest_model.joblib`
- `feature_scaler.joblib`
- `feature_metadata.json`
- `evaluation_metrics.json`

> **Important Note on Synthetic Data vs. Real Hardware Validation:**  
> By default, the model is trained on synthetic demonstration data so the full software stack can be demonstrated without physical hardware. The dashboard and metadata explicitly label this as `SYNTHETIC_DEMONSTRATION`, and Random Forest class vote shares are presented as uncalibrated ensemble agreement scores rather than guaranteed physical fault probabilities.

---

## 12. ESP32 Firmware Upload Instructions

1. Open `firmware/esp32_main.ino` in the **Arduino IDE** (with ESP32 board support installed).
2. Install the required libraries via **Sketch -> Include Library -> Manage Libraries**:
   - `Adafruit MPU6050`
   - `Adafruit Unified Sensor`
   - `OneWire`
   - `DallasTemperature`
3. Verify pin assignments in `firmware/sensor_manager.h` (`SDA=21`, `SCL=22`, `ACS712=34`, `DS18B20=4`).
4. Select your ESP32 board and COM port, then click **Upload**.

---

## 13. Serial Communication Protocol

The ESP32 transmits newline-delimited JSON packets at `115200` baud:

```json
{"timestamp":123456,"vibration_x":0.1200,"vibration_y":0.0800,"vibration_z":1.0300,"current":1.250,"temperature":38.50}
```

Every completed window (`128` samples), on-device features are included under `"features"`. The Python module `src/serial_reader.py` automatically detects available COM ports, ignores malformed/partial lines, handles sensor dropouts (`null` or `-127.0°C`), and attempts graceful reconnection if unplugged.

---

## 14. Dashboard Execution

You can launch the Streamlit dashboard either via the one-click batch file:

```powershell
.\run.bat
```

Or directly via the Streamlit CLI:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

---

## 15. Simulation Mode (No Hardware Required)

1. Start the dashboard (`streamlit run dashboard/app.py`).
2. In the left sidebar under **Operating Mode**, keep **SIMULATION MODE** selected.
3. Under **Simulation Fault Injector**, switch between:
   - **`HEALTHY`**: Low vibration RMS (`~0.08 g`), steady nominal current (`~2.15 A`), stable temperature (`~39 °C`), **GREEN** status.
   - **`ROTOR_IMBALANCE`**: Strong `135 Hz` / `270 Hz` rotational harmonics in the FFT chart, elevated vibration RMS (`>0.50 g`), moderate current modulation (`~3.0 A`), **YELLOW / RED** alert with spectral energy explanation.
   - **`MECHANICAL_STALL`**: High stall current surge (`~7.2 A`), impulsive mechanical chattering (high kurtosis), rapid thermal rise (`>0.35 °C/s`), **RED CRITICAL FAULT** alert.

---

## 16. Live Hardware Mode

1. Connect the programmed ESP32 via USB.
2. In the dashboard sidebar, switch **Operating Mode** to **LIVE HARDWARE MODE**.
3. Select the detected **COM Port** and **Baud Rate (`115200`)**, then click **Connect ESP32**.
4. Live sensor packets will populate the rolling buffers and flow through the exact same SciPy filter, FFT, ML inference, and sensor-fusion pipeline.

---

## 17. Testing Instructions

Run the automated test suite (`14` unit and integration tests covering signal filtering, Nyquist clamping, FFT accuracy, feature extraction, serial JSON parsing, missing sensors, and ML inference):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 18. Troubleshooting

- **COM Port Not Listed or Access Denied**: Ensure the Arduino IDE Serial Monitor is closed (only one program can hold a Windows COM port open at a time) and click **Connect ESP32** again.
- **DS18B20 Reads `null` or Unavailable**: Verify the `4.7 kΩ` pull-up resistor between `DQ` (`GPIO 4`) and `3.3V`. The software will automatically continue operating in degraded sensor mode using the last valid thermal state.
- **Bandpass High Cutoff Warning**: When sampling at `1000 Hz`, the Nyquist frequency is `500 Hz`. The filter automatically clamps upper cutoffs at `0.96 * Nyquist` (`480 Hz`) to guarantee numerical stability.

