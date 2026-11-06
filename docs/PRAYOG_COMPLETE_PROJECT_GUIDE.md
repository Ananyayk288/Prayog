# PRAYOG — COMPLETE PROJECT EXPLANATION AND TECHNICAL DEEP DIVE

**Project:** Prayog — Edge-AI Multi-Modal Sensor Fusion System for EV Motor Fault Detection  
**Workspace:** `D:\Prayog`  
**Audience:** CSE Engineering Student & Technical Review / Hackathon Jury Reference

---

## TABLE OF CONTENTS

1. [Part 1 — Understand the Entire Project](#part-1--understand-the-entire-project)
2. [Part 2 — Complete Website Walkthrough](#part-2--complete-website-walkthrough)
3. [Part 3 — Dashboard Metrics & Charts Explanation](#part-3--dashboard-metrics--charts-explanation)
4. [Part 4 — Hardware and Sensor Explanation](#part-4--hardware-and-sensor-explanation)
5. [Part 5 — ESP32 Firmware Deep Dive](#part-5--esp32-firmware-deep-dive)
6. [Part 6 — Signal Processing Pipeline](#part-6--signal-processing-pipeline)
7. [Part 7 — Feature Extraction Mathematics & Code](#part-7--feature-extraction-mathematics--code)
8. [Part 8 — Machine Learning Pipeline](#part-8--machine-learning-pipeline)
9. [Part 9 — Multi-Modal Sensor Fusion & Fault Detection](#part-9--multi-modal-sensor-fusion--fault-detection)
10. [Part 10 — Python Backend & Data Flow Architecture](#part-10--python-backend--data-flow-architecture)
11. [Part 11 — Technology Stack Breakdown](#part-11--technology-stack-breakdown)
12. [Part 12 — Complete File-by-File Inventory](#part-12--complete-file-by-file-inventory)
13. [Part 13 — Running the Project (Step-by-Step PowerShell)](#part-13--running-the-project-step-by-step-powershell)
14. [Part 14 — Hardware Connection & Real-World Demonstration](#part-14--hardware-connection--real-world-demonstration)
15. [Part 15 — Honest Project Limitations & Future Improvements](#part-15--honest-project-limitations--future-improvements)
16. [Part 16 — Judge & Technical Interview Q&A Preparation](#part-16--judge--technical-interview-qa-preparation)

---

## PART 1 — UNDERSTAND THE ENTIRE PROJECT

### 1.1 Implementation Reality Audit (Implemented vs. Simulated vs. Hardware-Dependent)

Before diving into theory, here is the exact engineering truth of what exists inside `D:\Prayog`:

| Subsystem | Implementation Status | Reality in Code |
| :--- | :--- | :--- |
| **ESP32 C++ Firmware** | **Implemented (Source Ready)** | Complete Arduino C++ code in `firmware/esp32_main.ino`, `sensor_manager.cpp`, and `signal_processing.cpp`. Requires physical ESP32 + sensors to be wired and flashed. |
| **Multi-Sensor Simulator** | **Implemented & Active** | `src/data_generator.py` (`SensorSimulator`) generates realistic 256-sample waveforms for `HEALTHY`, `ROTOR_IMBALANCE`, and `MECHANICAL_STALL`. |
| **SciPy Digital Signal Processing** | **Implemented & Active** | `src/signal_filters.py` performs NaN interpolation, MAD outlier clipping, DC gravity removal, and 4th-order Butterworth bandpass filtering (`50 Hz – 1 kHz` target, clamped safely below Nyquist). |
| **18-Feature Window Extractor** | **Implemented & Active** | `src/feature_extractor.py` computes 9 vibration (time + Hann FFT), 5 current, and 4 temperature features per window. |
| **ML Training & Inference** | **Implemented & Active** | `models/train_model.py` and `models/inference.py` train/load `RandomForestClassifier` (150 trees) + `IsolationForest` (120 trees) + `StandardScaler`. Currently trained on 720 synthetic windows (`SYNTHETIC_DEMONSTRATION`). |
| **Multi-Modal Sensor Fusion** | **Implemented & Active** | `src/sensor_fusion.py` (`MultiModalSensorFusion`) cross-checks ML predictions against physical thresholds across vibration, current, and temperature. |
| **USB Serial Hardware Reader** | **Implemented & Active** | `src/serial_reader.py` (`ESP32SerialReader`) scans COM ports, parses newline-delimited JSON at `115200` baud, handles disconnects, and exports CSVs. |
| **Obsidian Sage Streamlit UI** | **Implemented & Active** | `dashboard/app.py`, `dashboard/components.py`, `dashboard/charts.py` implement a 6-section interface (`Overview`, `Live Monitoring`, `Fault Detection`, `Analytics & History`, `Device Connection`, `Settings`). |

---

### 1.2 What Exactly is Prayog?

- **Simple English**: **Prayog** is a smart health-monitoring system for Electric Vehicle (EV) motors. Just like a smartwatch measures your heart rate, skin temperature, and body movement to warn you before a health issue becomes serious, Prayog listens to an electric motor's **mechanical vibration**, **electrical current**, and **temperature** to detect mechanical faults early.
- **Real-World Analogy**: Imagine an experienced car mechanic who puts their hand on an engine to feel how hard it is shaking (vibration), watches the battery ammeter to see if the motor is struggling under load (current), and checks the thermometer to see if it is overheating (temperature). Prayog automates that mechanic using an ESP32 microcontroller, digital signal processing, and Machine Learning.
- **Technical Explanation**: Prayog is an **Edge-AI Multi-Modal Predictive Maintenance System**. It acquires synchronized time-series windows from three physical sensor modalities—a 3-axis MEMS accelerometer/gyroscope (**MPU6050**), a Hall-effect current sensor (**ACS712**), and a 1-Wire digital thermometer (**DS18B20**). It filters out environmental noise using a Butterworth bandpass filter, transforms time-series windows into an **18-dimensional feature vector** (including Fast Fourier Transform spectral metrics), classifies motor state using a **Random Forest + Isolation Forest** ensemble, validates predictions via **rule-assisted multi-modal sensor fusion**, and renders real-time diagnostics in a **Streamlit** web application.

### 1.3 What Problem Does It Solve & Why is EV Motor Fault Detection Important?

- **The Problem**: Electric motors in EVs spin at thousands of RPM under varying torque loads. Two common mechanical failures are:
  1. **Rotor Imbalance**: Uneven mass distribution on the spinning rotor creates centrifugal forces that Hammer the motor bearings at every rotation.
  2. **Mechanical Stall / Jamming**: A seized bearing or jammed drivetrain locks the rotor while electrical current still flows through the stator windings, causing massive overcurrent ($I^2R$ heating) that can melt winding insulation or damage the inverter within seconds.
- **Why Traditional Monitoring Fails**:
  - Waiting until a motor smells burnt or trips a thermal fuse means damage has already occurred.
  - Relying on a **single sensor threshold** (for example, triggering an alarm whenever vibration exceeds $0.5\text{ g}$) causes constant **false alarms** whenever an EV hits a pothole or speed bump.

### 1.4 What is Predictive Maintenance?

1. **Reactive Maintenance ("Run-to-Failure")**: Fix the motor after it breaks down on the highway. (Highest cost, safety hazard).
2. **Preventive Maintenance ("Calendar-Based")**: Replace bearings every 20,000 km whether they need it or not. (Wastes money and healthy parts).
3. **Predictive Maintenance ("Condition-Based" — Used in Prayog)**: Continuously monitor live sensor signatures (vibration spectrum, current ripple, thermal slope) to detect early degradation and schedule maintenance *only when a fault pattern actually begins to form*.

### 1.5 Why Edge AI & Why Multi-Modal Sensor Fusion?

- **Why Edge AI?**: Raw vibration signals must be sampled hundreds or thousands of times per second ($500\text{ Hz} - 1000\text{ Hz}$). Streaming raw high-frequency arrays from millions of vehicles over 4G/5G to a cloud server wastes bandwidth and introduces network latency. By performing signal conditioning and window feature extraction on the local edge node (ESP32 + local Python gateway), the system works **100% offline** with sub-second response times.
- **Why Multi-Modal Sensor Fusion?**:
  - **Vibration alone** cannot easily distinguish a pothole shock from a sudden load jam.
  - **Current alone** cannot distinguish normal uphill acceleration (high current, smooth vibration) from a mechanical stall (high current + impulsive mechanical chattering + rapid temperature rise).
  - **Temperature alone** reacts too slowly (due to **thermal inertia**) to catch an instant stall in its first second, though it confirms sustained electrical overload.
  - By fusing all three, Prayog cross-verifies every fault before raising an alarm.

### 1.6 End-to-End Journey: From Motor Start to Fault Detection

Here is what happens step by step in the actual codebase (`src/fault_detection.py` $\rightarrow$ `FaultDetectionSystem.analyze_window()`):

1. **Signal Acquisition**: Either `SensorSimulator.generate_window()` (in Simulation Mode) or `ESP32SerialReader.read_available_records()` (in Live Hardware Mode) collects a synchronized window of $N = 256$ samples (at $f_s = 1000\text{ Hz}$, covering $0.256\text{ seconds}$).
2. **Sanitization & Filtering (`src/signal_filters.py`)**:
   - `handle_missing_values()` linearly interpolates any `NaN`/`Inf` samples.
   - `clip_outliers()` clamps extreme single-sample glitches using Median Absolute Deviation (MAD).
   - `remove_dc_offset()` subtracts static $1.0\text{ g}$ gravity from the accelerometer.
   - `apply_vibration_bandpass_filter()` applies a 4th-order zero-phase Butterworth bandpass filter (`scipy.signal.sosfiltfilt`) to isolate motor rotational frequencies ($50\text{ Hz}$ to $480\text{ Hz}$ at $1\text{ kHz}$ sampling).
3. **Feature Extraction (`src/feature_extractor.py`)**:
   - `extract_unified_features()` computes 18 numbers summarizing the window: 9 vibration metrics (including Hann-windowed FFT dominant frequency and band energy), 5 current metrics, and 4 temperature metrics.
4. **ML Inference (`models/inference.py`)**:
   - `FaultModelInference.predict()` scales the 18 features using `StandardScaler`, runs `RandomForestClassifier.predict_proba()` to get class vote proportions across `HEALTHY`, `ROTOR_IMBALANCE`, and `MECHANICAL_STALL`, and runs `IsolationForest` to check if the window is an out-of-distribution anomaly.
5. **Multi-Modal Fusion (`src/sensor_fusion.py`)**:
   - `MultiModalSensorFusion.fuse()` checks individual modality severities (`evaluate_modalities()`) and combines them with the Random Forest prediction to output a final fault class, an alert level (`GREEN` / `YELLOW` / `RED`), and plain-English diagnostic reasons.
6. **UI Rendering (`dashboard/app.py`)**:
   - Streamlit updates the KPI cards, Plotly waveforms, FFT spectrum, and Alert Center.

---

## PART 2 — COMPLETE WEBSITE WALKTHROUGH

The web interface is implemented in three files inside `dashboard/`:
- [`dashboard/app.py`](../dashboard/app.py): Main controller, sidebar, session state, and 6-section router.
- [`dashboard/components.py`](../dashboard/components.py): Custom Obsidian Sage CSS (`#171A17`, `#A8B58A`, `#F1EBDD`), header bar, hero banner, KPI cards, and alert rows.
- [`dashboard/charts.py`](../dashboard/charts.py): Interactive Plotly figures.

### 2.1 Global Elements (Present on Every Screen)

1. **Left Sidebar (`main()` in `dashboard/app.py`, lines 494–552)**:
   - **Navigation Radio (`active_section`)**: Lets the user switch between the 6 sections (`Overview`, `Live Monitoring`, `Fault Detection`, `Analytics & History`, `Device Connection`, `Settings`) with a single click.
   - **Data Source Mode Radio (`chosen_mode_label`)**: Switches `st.session_state.operating_mode` between `"SIMULATION MODE"` and `"LIVE HARDWARE MODE"`.
   - **Simulated Motor Condition Dropdown (`chosen_friendly`)**: Visible when in Simulation Mode. Maps friendly names (`"Healthy (Normal Operation)"`, `"Rotor Imbalance"`, `"Mechanical Stall"`) to internal constants (`CLASS_HEALTHY`, `CLASS_ROTOR_IMBALANCE`, `CLASS_MECHANICAL_STALL`) and calls `sim.set_operating_state(selected_code)`.
   - **"Capture Next Reading" Primary Button (`step_now`)**: Immediately triggers `acquire_and_process_step()` for manual step-by-step inspection.
   - **"Live Auto-Update" Toggle (`st.session_state.auto_refresh`)**: Enables or pauses automatic periodic refresh (`time.sleep(refresh_interval_s)` followed by `st.rerun()`).
2. **Top Status Bar (`render_header()` in `dashboard/components.py`, lines 211–236)**:
   - Displays four status pills at the top of the page:
     - `Mode: Simulation Mode` or `Live Hardware Mode`
     - `Device: Simulator Active` or `ESP32 Connected (COMx)` / `ESP32 Disconnected (COMx)`
     - `AI Engine: Ready`
     - `Updated: HH:MM:SS` (timestamp of the latest processed window)

---

### 2.2 Section 1: Overview (`render_overview_section()` in `dashboard/app.py`, lines 142–174)

- **Purpose**: Designed so a first-time user immediately understands whether the motor is safe or failing without reading complex engineering plots.
- **Components Displayed**:
  1. **Hero Health Banner (`render_overview_hero()` in `dashboard/components.py`)**:
     - Shows a pill badge (`System Status · Healthy`, `Warning`, or `Critical`), a large headline (`Motor Condition: Normal Operation` / `Rotor Imbalance` / `Mechanical Stall`), and the primary plain-English explanation from `fusion_result["explanations"][0]`.
  2. **8 Metric Cards (`render_kpi_overview()` in `dashboard/components.py`, lines 265–338)**:
     - **Row 1 (Physical Health & Sensors)**:
       - *Overall Motor Health*: `Healthy` / `Warning` / `Critical` + condition subtitle.
       - *Vibration Level (MPU6050)*: `vib_rms` in `g` + Peak-to-Peak subtitle.
       - *Motor Current (ACS712)*: `current_rms` in `A` + Peak current subtitle.
       - *Motor Temperature (DS18B20)*: `temp_current_c` in `°C` + rise rate in `°C/s`.
     - **Row 2 (AI & System State)**:
       - *Latest AI Prediction*: Friendly fault name + `confidence_score * 100`% agreement.
       - *Current Operating Status*: Status word + Isolation Forest Anomaly check (`Normal` vs. `Flagged`).
       - *ESP32 Connection Status*: `Connected`, `Standby (Sim)`, or `Disconnected`.
       - *Current Operating Mode*: `Simulation` or `Live ESP32`.
  3. **System Insights & Quick Guidance**:
     - Left column calls `render_alert_center()` showing the active diagnostic notice and the 3 most recent alerts.
     - Right column renders the *"How to Use Prayog"* quick-start card.

---

### 2.3 Section 2: Live Monitoring (`render_live_monitoring_section()` in `dashboard/app.py`, lines 177–234)

- **Purpose**: Displays real-time time-series waveforms for the three sensors while keeping advanced frequency-domain DSP plots inside an optional drawer.
- **Components Displayed**:
  1. **Mechanical Vibration Waveform (`create_vibration_waveform_chart()` in `dashboard/charts.py`)**: Plots `vibration_x`, `vibration_y`, and gravity-compensated `vibration_z - 1.0` over the 256-sample time axis (`0.256 s`).
  2. **Motor Current Draw (`create_current_waveform_chart()` in `dashboard/charts.py`)**: Plots `current` (Amperes) with a dashed horizontal stall threshold line at `5.5 A`.
  3. **Motor Temperature Trend (`create_temperature_trend_chart()` in `dashboard/charts.py`)**: Plots historical `temp_current_c` and `temp_moving_avg_c` across recent windows, with reference lines at `55 °C` (Warning) and `70 °C` (Critical).
  4. **Expandable Drawer — *"Advanced Signal & Frequency Analysis"***:
     - **Filtered Vibration Signal (`create_filtered_vibration_chart()`)**: Overlays raw composite vibration (`diagnostics["raw_vibration"]`) against the `50 Hz – 1 kHz` Butterworth bandpass-filtered waveform (`diagnostics["filtered_vibration"]`).
     - **Vibration Frequency Spectrum (`create_fft_spectrum_chart()`)**: Plots FFT frequency bins (`diagnostics["fft_freqs"]`) vs. spectral magnitude (`diagnostics["fft_magnitudes"]`) and marks the dominant peak (`vib_dominant_freq_hz`) with a vertical dashed line.
     - **4 Summary Metrics**: Dominant Vibration Frequency (`Hz`), Passband Energy (`g²`), Filtered Vibration RMS (`g`), and Vibration Peak-to-Peak (`g`).

---

### 2.4 Section 3: Fault Detection (`render_fault_detection_section()` in `dashboard/app.py`, lines 237–318)

- **Purpose**: Explains *what* fault the AI detected and *why* the multi-modal sensor fusion engine agreed with it.
- **Components Displayed**:
  1. **3 Summary Cards**: Detected Motor Condition, AI Model Agreement (`%`), and Isolation Forest Anomaly Verification (`Normal Pattern` vs. `Anomaly Detected` + decision score).
  2. **Explainable Reasoning List**: Bullet points from `fusion_result["explanations"]` detailing exact physical measurements that crossed thresholds.
  3. **Individual Sensor Health Check**: Three `st.metric` indicators showing modality severity (`Normal`, `Elevated`, `High`, or `Critical Overcurrent`) for Vibration, Current, and Thermal channels.
  4. **AI Model Agreement Bar Chart (`create_class_scores_chart()`)**: Horizontal bar chart of Random Forest tree-vote percentages across `Healthy`, `Rotor Imbalance`, and `Mechanical Stall`.
  5. **Diagnostic Alert Log (`render_alert_center()`)**: Lists up to 10 timestamped warning/critical events stored in `det_sys.alert_events`.
  6. **Expandable Drawer — *"Inspect All 18 Extracted Engineering Features"***: Displays a dataframe of all 18 canonical features in `FEATURE_COLUMNS`, their current numerical values, and physical units from `FEATURE_UNITS`.

---

### 2.5 Section 4: Analytics & History (`render_analytics_history_section()` in `dashboard/app.py`, lines 321–391)

- **Purpose**: Provides session logging, data export for offline study, and ML validation transparency.
- **Components Displayed**:
  1. **Session Telemetry Table**: Reverse-chronological dataframe of up to 300 windows stored in `det_sys.history_records`.
  2. **Session Condition Summary**: Counts how many windows were classified into each condition during the session.
  3. **"Download Session Data (CSV)" Button**: Exports the session dataframe as `prayog_motor_telemetry_<timestamp>.csv`.
  4. **ML Model Evaluation Summary**: Reads `models/saved/evaluation_metrics.json` (loaded via `det_sys.inference_engine.evaluation_metrics`) to show:
     - Dataset origin disclaimer (`SYNTHETIC_DEMONSTRATION` vs. recorded hardware).
     - Test Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-Score.
     - **Validation Confusion Matrix (`create_confusion_matrix_chart()`)**: Heatmap of actual vs. predicted classes on the 180-sample test split.
     - **Most Influential Sensor Features**: Top 10 features ranked by `rf_model.feature_importances_`.

---

### 2.6 Section 5: Device Connection (`render_device_connection_section()` in `dashboard/app.py`, lines 394–537)

- **Purpose**: Dedicated hardware communication interface for linking the physical ESP32 over USB Serial (`src/serial_reader.py`).
- **Components Displayed**:
  1. **3 Hardware Status Cards**:
     - *ESP32 Connection Status*: `Connected` or `Disconnected` (never faked).
     - *Serial Communication Status*: `reader.total_valid_records` packets received and `reader.malformed_count` ignored lines.
     - *Available USB COM Ports*: Number of ports returned by `list_available_com_ports()` via `serial.tools.list_ports.comports()`.
  2. **Port & Baud Controls**:
     - *USB Serial Port Selectbox*: Lists detected COM ports (with hardware descriptions) or fallback `COM3`–`COM6`.
     - *Baud Rate Selectbox*: `9600`, `57600`, `115200` (default), `230400`.
     - **Connect ESP32**, **Disconnect**, and **Rescan Ports** buttons.
     - Connection error banner displaying `reader.last_error` if opening the port fails.
  3. **Individual Sensor Availability Indicators**:
     - Shows status for **MPU6050 (`I2C GPIO 21/22`)**, **ACS712 (`ADC1 GPIO 34`)**, and **DS18B20 (`1-Wire GPIO 4`)**.
     - Honesty safeguard: Only shows `"Online"` when `reader.is_connected and reader.total_valid_records > 0`; otherwise displays `"Awaiting ESP32 Link"`.
  4. **Incoming Sensor-Data Preview**: Displays the last 8 raw JSON packets from `reader.received_records` and a button to save them to `data/raw/esp32_live_capture.csv`.
  5. **Hardware Connection & Troubleshooting Guide**: Expandable checklist covering USB cables, closing Arduino Serial Monitor, baud rate matching, and the DS18B20 $4.7\text{ k}\Omega$ pull-up resistor.

---

### 2.7 Section 6: Settings (`render_settings_section()` in `dashboard/app.py`, lines 540–596)

- **Purpose**: Central place for adjusting digital filter cutoffs, refresh speed, and retraining the ML model.
- **Controls**:
  - *Low Cutoff Frequency (`Hz`)* and *High Cutoff Frequency (`Hz`)* number inputs (updates `SIGNAL_CONFIG.bandpass_low_hz` and `bandpass_high_hz`).
  - *Enable Automatic Live Updates* toggle and *Update Interval (`seconds`)* slider (`0.5 s` to `5.0 s`).
  - **Retrain AI Model** button (invokes `train_and_evaluate_models()` and reloads `.joblib` files) and **Reset Simulator Thermal State** button (resets motor temperature to `38.2 °C`).

---

## PART 3 — DASHBOARD METRICS & CHARTS EXPLANATION

### 3.1 Complete Metric Reference Table

| Metric Name | Unit | Normal Value (Healthy) | Abnormal Value & Meaning | Where Calculated |
| :--- | :--- | :--- | :--- | :--- |
| **Motor Health Status** | Text (`Healthy` / `Warning` / `Critical`) | `Healthy` (`GREEN`) | `Warning` (early deviation / rotor imbalance) or `Critical` (mechanical stall or severe multi-sensor fault) | `MultiModalSensorFusion.fuse()` in `src/sensor_fusion.py` |
| **Vibration RMS (`vib_rms`)** | $\text{g}$ ($1\text{ g} \approx 9.81\text{ m/s}^2$) | $\approx 0.06 - 0.12\text{ g}$ | $\ge 0.35\text{ g}$ (Warning), $\ge 0.65\text{ g}$ (Critical — strong centrifugal rotor imbalance) | `extract_vibration_features()` in `src/feature_extractor.py` |
| **Peak-to-Peak (`vib_peak_to_peak`)** | $\text{g}$ | $\approx 0.25 - 0.55\text{ g}$ | $\ge 1.60\text{ g}$ (severe mechanical excursion or impulsive bearing shock) | `extract_vibration_features()` via `np.ptp(filtered_vib)` |
| **Motor Current RMS (`current_rms`)** | Amperes ($\text{A}$) | $\approx 2.0 - 2.3\text{ A}$ | $3.0 - 3.6\text{ A}$ (eccentric load ripple) or $\ge 5.5\text{ A}$ (locked-rotor mechanical stall overcurrent) | `extract_current_features()` in `src/feature_extractor.py` |
| **Motor Temperature (`temp_current_c`)** | Celsius ($^\circ\text{C}$) | $\approx 37 - 41^\circ\text{C}$ | $\ge 55^\circ\text{C}$ (Warning) or $\ge 70^\circ\text{C}$ (Critical winding overheating) | `extract_temperature_features()` in `src/feature_extractor.py` |
| **Latest AI Prediction** | Class Name | `Normal Operation` (`HEALTHY`) | `Rotor Imbalance` or `Mechanical Stall` | `FaultModelInference.predict()` + `MultiModalSensorFusion.fuse()` |
| **AI Model Agreement (`confidence_score`)** | Percentage ($0 - 100\%$) | $85\% - 100\%$ | Fraction of the 150 trees in the Random Forest voting for the top class (uncalibrated vote share) | `FaultModelInference.predict()` in `models/inference.py` |
| **Anomaly Status (`is_anomaly`)** | Boolean + Score | `False` (Score $> 0$) | `True` (Isolation Forest decision score $< 0$ or non-healthy classification) | `FaultModelInference.predict()` in `models/inference.py` |

### 3.2 How Each Chart Works (`dashboard/charts.py`)

1. **Mechanical Vibration Waveform (`create_vibration_waveform_chart`)**:
   - **X-Axis**: Time in seconds (`time_axis_s`, $256\text{ samples}$ spanning $0.256\text{ s}$ per window).
   - **Y-Axis**: Acceleration in $\text{g}$.
   - **Data Source**: Raw `vibration_x`, `vibration_y`, and `vibration_z - 1.0` (static $1\text{ g}$ gravity subtracted so all three axes oscillate around $0\text{ g}$).
   - **Interpretation**: Smooth low-amplitude waves indicate a balanced rotor; tall periodic sine waves indicate `ROTOR_IMBALANCE`; jagged random spikes indicate `MECHANICAL_STALL` chattering.
2. **Motor Current Draw (`create_current_waveform_chart`)**:
   - **X-Axis**: Time in seconds.
   - **Y-Axis**: Electrical current in Amperes ($\text{A}$).
   - **Interpretation**: Flat line around $2.15\text{ A}$ is nominal; oscillation around $3.0\text{ A}$ shows eccentric torque ripple; surge above the $5.5\text{ A}$ dashed threshold line indicates a stalled rotor drawing locked-rotor current.
3. **Motor Temperature Trend (`create_temperature_trend_chart`)**:
   - **X-Axis**: Window timestamps (`HH:MM:SS`).
   - **Y-Axis**: Temperature in $^\circ\text{C}$.
   - **Interpretation**: Shows both instantaneous temperature (`temp_current_c`) and 10-window moving average (`temp_moving_avg_c`). A steep upward slope ($> +0.25^\circ\text{C/s}$) warns of rapid $I^2R$ Joule heating during a stall.
4. **Filtered Vibration Signal (`create_filtered_vibration_chart`)**:
   - Compares the raw composite vibration signal (`dx + 0.7*dy + 0.5*dz`) against the output of the 4th-order Butterworth bandpass filter (`50 Hz – 1 kHz`). Demonstrates how DC offset and low-frequency road sway ($< 50\text{ Hz}$) are stripped away.
5. **Vibration Frequency Spectrum (`create_fft_spectrum_chart`)**:
   - **X-Axis**: Frequency in Hertz ($\text{Hz}$), from $0\text{ Hz}$ up to Nyquist ($500\text{ Hz}$ at $f_s = 1000\text{ Hz}$).
   - **Y-Axis**: Spectral Magnitude in $\text{g}$.
   - **Interpretation**: Converts time waves into frequency peaks using the Fast Fourier Transform. In `HEALTHY` mode, there is a tiny peak near $85\text{ Hz}$. In `ROTOR_IMBALANCE`, huge spikes appear at $135\text{ Hz}$ ($1\times$ rotational speed), $270\text{ Hz}$ ($2\times$), and $405\text{ Hz}$ ($3\times$).

---

## PART 4 — HARDWARE AND SENSOR EXPLANATION

### 4.1 MPU6050 — 6-DOF MEMS Accelerometer & Gyroscope

- **What is an Accelerometer?**: An electromechanical sensor that measures linear acceleration forces (in $\text{m/s}^2$ or $\text{g}$) along three mutually perpendicular axes ($X, Y, Z$). Inside the chip, microscopic silicon proof masses are suspended by tiny springs; when the motor vibrates, the mass deflects and changes the capacitance between fixed plates.
- **What is a Gyroscope?**: A sensor that measures angular velocity (rotational twist rate in $\text{deg/s}$ or $\text{rad/s}$) around the $X, Y, Z$ axes using the Coriolis effect.
- **What does 6-DOF mean?**: **6 Degrees of Freedom** — 3 linear acceleration axes ($a_x, a_y, a_z$) + 3 rotational angular velocity axes ($\omega_x, \omega_y, \omega_z$).
- **Why Vibration Detects Rotor Imbalance**: When a rotor has an uneven mass $m$ at radius $r$ spinning at angular velocity $\omega = 2\pi f$, it generates a centrifugal force:
  $$F_c = m \cdot r \cdot \omega^2$$
  Because this force rotates with the shaft, the MPU6050 sees a strong sinusoidal acceleration at the exact rotational frequency $f$ ($1\times$ harmonic) and its multiples ($2\times, 3\times$).
- **Limitations**:
  - The MPU6050's internal accelerometer hardware Digital Low-Pass Filter (DLPF) has a maximum bandwidth of $260\text{ Hz}$ (configured via `MPU6050_BAND_260_HZ` in `firmware/sensor_manager.cpp`), and reading it over standard $400\text{ kHz}$ I2C limits practical multi-sensor loop rates on an ESP32 to $\approx 500\text{ Hz} - 1000\text{ Hz}$.

### 4.2 ACS712 — Hall-Effect Current Sensor

- **How it Works**: When motor supply current flows through the internal copper conduction path of the ACS712, it generates a magnetic field proportional to the current ($B \propto I$). An integrated linear **Hall-effect sensor** measures this magnetic field and outputs a proportional analog voltage ($V_{\text{out}}$).
- **Zero-Current Offset & Sensitivity**:
  - At $0\text{ A}$, the ACS712 outputs half of its supply voltage ($V_{\text{zero}} \approx 2.50\text{ V}$ on a $5\text{ V}$ supply).
  - For a $20\text{ A}$ ACS712 module, the sensitivity is $100\text{ mV/A}$ ($0.100\text{ V/A}$).
  - Formula implemented in `SensorManager::readCurrentAmps()` (`firmware/sensor_manager.cpp`, lines 66–76):
    $$I = \frac{|V_{\text{sensor}} - V_{\text{zero}}|}{\text{Sensitivity}}$$
- **Why Current Detects Mechanical Stall**: An electric motor spinning normally generates **Back-EMF** ($V_{\text{bemf}} = k_e \cdot \omega$), which opposes the supply voltage $V_s$ and keeps current low:
  $$I = \frac{V_s - k_e \cdot \omega}{R_{\text{winding}}}$$
  When the rotor stalls ($\omega \to 0$), Back-EMF drops to **zero**! The current is then limited only by the tiny winding resistance $R_{\text{winding}}$, causing an immediate, massive current surge (**stall current**).

### 4.3 DS18B20 — 1-Wire Digital Temperature Sensor

- **How it Works**: Unlike analog thermistors, the DS18B20 has an on-board temperature-sensing element and analog-to-digital converter that communicates over a digital **1-Wire protocol** (requiring only one data pin, `GPIO 4`, plus a $4.7\text{ k}\Omega$ pull-up resistor).
- **What is Thermal Inertia & Why Monitor Trends?**:
  - **Thermal Inertia**: A motor's metal housing has significant heat capacity ($m \cdot c_p$). When a stall happens, the copper windings heat up at rate $P_{\text{heat}} = I^2 R$, but it takes several seconds for that heat to conduct to the outer casing.
  - Therefore, looking only at whether $T > 70^\circ\text{C}$ is too slow. By measuring the **Temperature Rise Rate** ($\frac{\Delta T}{\Delta t}$ in $^\circ\text{C/s}$), Prayog spots rapid abnormal heating while the absolute temperature is still only $48^\circ\text{C}$!

---

## PART 5 — ESP32 FIRMWARE DEEP DIVE

All embedded code resides in `firmware/`:
1. [`firmware/sensor_manager.h`](../firmware/sensor_manager.h) & [`firmware/sensor_manager.cpp`](../firmware/sensor_manager.cpp)
2. [`firmware/signal_processing.h`](../firmware/signal_processing.h) & [`firmware/signal_processing.cpp`](../firmware/signal_processing.cpp)
3. [`firmware/esp32_main.ino`](../firmware/esp32_main.ino)

### 5.1 Key Embedded Concepts

- **ESP32**: A dual-core 32-bit Tensilica Xtensa LX6 microcontroller running at $240\text{ MHz}$ with hardware floating-point support (FPU), 12-bit ADCs, I2C, and high-speed UART serial.
- **ADC (Analog-to-Digital Converter)**: Converts continuous analog voltage ($0\text{ V}$ to $3.3\text{ V}$ on `GPIO 34`) into a 12-bit integer between `0` and `4095`.
- **Non-Blocking DS18B20 Reads (`updateTemperatureAsync`)**: Standard `DallasTemperature` reads block the CPU for up to $750\text{ ms}$, which would destroy $500\text{ Hz}$ vibration sampling! In `sensor_manager.cpp` (lines 38–42), we set `_dallasTemp.setWaitForConversion(false)` and 10-bit resolution ($187\text{ ms}$ conversion time) so the ESP32 never blocks while waiting for temperature updates.

### 5.2 Firmware Walkthrough (`firmware/sensor_manager.cpp`)

```cpp
// Oversample 4x to reduce ESP32 ADC thermal noise
long adc_sum = 0;
for (int i = 0; i < 4; i++) {
    adc_sum += analogRead(PIN_ACS712_ADC);
}
float adc_avg = static_cast<float>(adc_sum) * 0.25f;
float v_sensor = (adc_avg / ESP32_ADC_MAX) * ESP32_VREF_VOLTS * VOLTAGE_DIVIDER_RATIO;
float current_a = (v_sensor - _acs_zero_voltage) / ACS712_SENSITIVITY;
return fabs(current_a);
```
- **Why 4x Oversampling?**: The ESP32's internal SAR ADC exhibits non-linearity and thermal noise. Averaging 4 consecutive reads reduces random white noise by $\sqrt{4} = 2\times$.
- **Boot Calibration (`calibrateCurrentZero`)**: At startup, the firmware averages 150 ADC readings to measure the exact zero-current midpoint voltage (`_acs_zero_voltage`), compensating for supply voltage drift.
- **Error Handling**:
  - If `_mpu.begin()` fails, `_mpu_ok` is set to `false` and safe fallback values (`0.0, 0.0, 1.0`) are returned instead of crashing/rebooting the ESP32.
  - If the DS18B20 returns `DEVICE_DISCONNECTED_C` (`-127.0°C`) or the power-on reset value (`85.0°C`), `_temp_ok` becomes `false` and `esp32_main.ino` emits `"temperature":null` in the JSON packet so Python knows the temperature sensor is offline.

### 5.3 On-Device Biquad Filter & Windowing (`firmware/signal_processing.cpp` & `esp32_main.ino`)

- In `esp32_main.ino`, `loop()` runs on a microsecond timer (`micros() - lastSampleMicros >= sampleIntervalUs`, where `sampleIntervalUs = 2000 us` for `TARGET_SAMPLE_RATE_HZ = 500`).
- Every sample is passed through `EdgeSignalProcessor::processVibrationSample()`, which runs a single-pole DC blocker (`_dc_estimate = 0.985f * _dc_estimate + 0.015f * raw_accel_g`) followed by a Direct Form II Transposed IIR Biquad bandpass filter.
- Every 8 samples ($\approx 62.5\text{ Hz}$), the ESP32 prints a JSON line over Serial (`115200` baud), and every `WINDOW_SIZE_SAMPLES` (`128` samples) it appends on-device computed window statistics (`vib_rms`, `vib_peak_to_peak`, `current_rms`, etc.).

---

## PART 6 — SIGNAL PROCESSING PIPELINE

All Python signal processing lives in [`src/signal_filters.py`](../src/signal_filters.py).

### 6.1 Why Filter Vibration Signals?

A raw accelerometer mounted on an EV motor captures three things mixed together:
1. **Static Gravity ($0\text{ Hz}$ DC)**: Always pulls at $1.0\text{ g}$ downward.
2. **Low-Frequency Vehicle Motion ($0.5\text{ Hz} - 30\text{ Hz}$)**: Road bumps, suspension bounce, chassis pitch/roll.
3. **Motor Mechanical Harmonics ($50\text{ Hz} - 1000\text{ Hz}$)**: Rotor rotation ($1\times, 2\times, 3\times$), electromagnetic cogging, and bearing vibrations.

If we calculated RMS on the raw signal, the $1.0\text{ g}$ gravity offset and chassis bumps would drown out the motor's actual mechanical signature! Therefore, we apply a **Digital Bandpass Filter** that blocks frequencies below $50\text{ Hz}$ and above the upper passband limit. *(Importantly, we apply this bandpass filter **only** to vibration—never to temperature or DC motor current, where the mean level is the actual signal!)*

### 6.2 The Nyquist Theorem & Our Safety Clamping (`compute_effective_bandpass_bounds`)

- **Nyquist-Shannon Sampling Theorem**: To accurately capture a signal of frequency $f_{\max}$ without aliasing, the sampling rate $f_s$ must be strictly greater than twice the maximum frequency:
  $$f_s > 2 \cdot f_{\max} \quad \Longleftrightarrow \quad f_{\text{Nyquist}} = \frac{f_s}{2}$$
- **Critical Engineering Reality Check in Prayog**:
  - Our target filter specification is `50 Hz – 1000 Hz`.
  - However, if the system samples at $f_s = 1000\text{ Hz}$ (`SIGNAL_CONFIG.sampling_rate_hz = 1000.0`), the Nyquist frequency is $f_{\text{Nyquist}} = 500\text{ Hz}$.
  - If you pass a $1000\text{ Hz}$ cutoff into `scipy.signal.butter()` when $f_s = 1000\text{ Hz}$, the normalized frequency is $W_n = \frac{1000}{500} = 2.0 \ge 1.0$, which causes SciPy to crash with `ValueError: Digital filter critical frequencies must be 0 < Wn < 1`!
- **How `src/signal_filters.py` Solves This (Lines 59–91)**:
  ```python
  nyquist = 0.5 * sampling_rate_hz
  max_allowable_high = 0.96 * nyquist  # 480 Hz when fs = 1000 Hz
  if high_hz >= max_allowable_high:
      eff_high = max_allowable_high
  ```
  - When $f_s = 1000\text{ Hz}$, `compute_effective_bandpass_bounds()` automatically clamps the upper cutoff to $0.96 \times 500 = 480\text{ Hz}$ (and if you increase $f_s$ to $2500\text{ Hz}$, it uses the full $1000\text{ Hz}$ cutoff).
  - Furthermore, if the window is shorter than 16 samples or $f_s \le 104\text{ Hz}$, `apply_vibration_bandpass_filter()` gracefully falls back to DC removal (`remove_dc_offset`) so the pipeline **never crashes**.

---

## PART 7 — FEATURE EXTRACTION MATHEMATICS & CODE

Implemented in [`src/feature_extractor.py`](../src/feature_extractor.py). Every window of $N = 256$ samples is distilled into **18 canonical features** defined in `FEATURE_COLUMNS` (`src/config.py`).

### 7.1 Vibration Features (9 Features — Computed in `extract_vibration_features()`)

First, a triaxial dynamic composite signal is formed from the outlier-clipped, DC-removed axes:
$$v[n] = \text{BandpassFilter}\Big(a_{x,\text{dyn}}[n] + 0.7\,a_{y,\text{dyn}}[n] + 0.5\,a_{z,\text{dyn}}[n]\Big)$$

1. **Vibration RMS (`vib_rms`, unit: $\text{g}$)**:
   - *Meaning*: Root-Mean-Square measures the overall continuous energy/power of the vibration wave.
   - *Formula*:
     $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} v[n]^2}$$
   - *Small Example*: For values $[-0.4, 0.4, -0.4, 0.4]\text{ g}$, $\text{RMS} = \sqrt{\frac{0.16+0.16+0.16+0.16}{4}} = 0.4\text{ g}$.
   - *Code*: `vib_rms = float(np.sqrt(np.mean(np.square(filtered_vib))))`
2. **Vibration Peak-to-Peak (`vib_peak_to_peak`, unit: $\text{g}$)**:
   - *Meaning*: Total excursion between the highest positive peak and lowest negative trough in the window.
   - *Formula*: $\text{P2P} = \max(v) - \min(v)$.
   - *Example*: For $[-0.4, 0.4]\text{ g}$, $\text{P2P} = 0.4 - (-0.4) = 0.8\text{ g}$.
   - *Code*: `vib_p2p = float(np.ptp(filtered_vib))`
3. **Vibration Mean Rectified Amplitude (`vib_mean`, unit: $\text{g}$)**:
   - *Meaning*: Average absolute deflection from zero (since a bandpass-filtered wave has a signed mean of $0$, we take the mean of $|v[n]|$).
   - *Formula*: $\mu_{|v|} = \frac{1}{N}\sum_{n=0}^{N-1} |v[n]|$.
   - *Code*: `vib_mean = float(np.mean(np.abs(filtered_vib)))`
4. **Vibration Standard Deviation (`vib_std`, unit: $\text{g}$)**:
   - *Meaning*: Spread of vibration samples around their mean.
   - *Formula*: $\sigma_v = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} (v[n] - \bar{v})^2}$.
   - *Code*: `vib_std = float(np.std(filtered_vib))`
5. **Vibration Variance (`vib_variance`, unit: $\text{g}^2$)**:
   - *Meaning*: Squared standard deviation ($\sigma_v^2$), directly proportional to dynamic mechanical power.
   - *Code*: `vib_var = float(np.var(filtered_vib))`
6. **Vibration Kurtosis (`vib_kurtosis`, unit: dimensionless `-`)**:
   - *Meaning*: Fourth standardized moment measuring "spikiness" or impulsiveness of the waveform.
   - *Formula*:
     $$K = \frac{\frac{1}{N}\sum_{n=0}^{N-1}(v[n] - \bar{v})^4}{\sigma_v^4}$$
   - *Why it Matters*: A pure sine wave or Gaussian noise has Kurtosis $\approx 1.5 - 3.0$. When a motor stalls or a bearing cracks, sudden mechanical impacts/chattering create sharp spikes that drive Kurtosis above $4.2$!
   - *Code*: `stats.kurtosis(filtered_vib, fisher=False, bias=False)`
7. **FFT Dominant Frequency (`vib_dominant_freq_hz`, unit: $\text{Hz}$)**:
   - *Meaning*: The specific frequency bin (in Cycles per Second) carrying the largest vibration amplitude after applying a Hann window (`np.hanning(n)`) and Real Fast Fourier Transform (`np.fft.rfft`).
   - *Why it Matters*: Identifies the exact rotational harmonic ($135\text{ Hz}$ in `ROTOR_IMBALANCE` vs. $65\text{ Hz}$ cogging hum in `MECHANICAL_STALL`).
   - *Code*: `compute_fft_spectrum()` $\rightarrow$ `dom_idx = int(np.argmax(magnitudes[1:]) + 1); dominant_freq_hz = float(freqs[dom_idx])`.
8. **FFT Spectral Energy (`vib_spectral_energy`, unit: $\text{g}^2/\text{Hz}$)**:
   - *Meaning*: Sum of squared FFT magnitudes across all frequency bins: $\sum_k |X[k]|^2$.
9. **Frequency-Band Energy (`vib_band_energy_50_1000hz`, unit: $\text{g}^2$)**:
   - *Meaning*: Spectral energy concentrated specifically within the $50\text{ Hz} - 1000\text{ Hz}$ fault band. Rises dramatically during `ROTOR_IMBALANCE`.

---

### 7.2 Current Features (5 Features — Computed in `extract_current_features()`)

Given current window $i[n]$ (in Amperes):
10. **Mean Current (`current_mean`, $\text{A}$)**: Average DC current draw ($\bar{i} = \frac{1}{N}\sum i[n]$) after 3-sample rolling smoothing.
11. **RMS Current (`current_rms`, $\text{A}$)**: Effective heating current ($I_{\text{rms}} = \sqrt{\frac{1}{N}\sum i[n]^2}$). Jumps from $\approx 2.15\text{ A}$ in `HEALTHY` to $> 7.2\text{ A}$ in `MECHANICAL_STALL`.
12. **Peak Current (`current_peak`, $\text{A}$)**: Maximum instantaneous current spike ($\max |i[n]|$).
13. **Current Variation (`current_variation`, $\text{A}$)**: Peak-to-peak current swing ($\max(i) - \min(i)$).
14. **Current Fluctuation (`current_fluctuation`, `%`)**: Coefficient of variation expressed as a percentage:
    $$\text{Fluctuation (\%)} = \frac{\sigma_i}{\max(|\bar{i}|, 0.1)} \times 100\%$$

---

### 7.3 Temperature Features (4 Features — Computed in `extract_temperature_features()`)

15. **Current Temperature (`temp_current_c`, $^\circ\text{C}$)**: Latest DS18B20 temperature reading $T_{\text{now}}$.
16. **Temperature Change (`temp_change_c`, $^\circ\text{C}$)**: Difference between current window and previous window ($\Delta T = T_{\text{now}} - T_{\text{prev}}$).
17. **Temperature Moving Average (`temp_moving_avg_c`, $^\circ\text{C}$)**: Average of the last up to 10 temperature readings ($\frac{1}{M}\sum_{k=1}^M T_k$).
18. **Temperature Rise Rate (`temp_rise_rate_c_per_s`, $^\circ\text{C/s}$)**: Thermal derivative with respect to window duration $\Delta t$ ($0.256\text{ s}$):
    $$\text{Rise Rate} = \frac{T_{\text{now}} - T_{\text{prev}}}{\Delta t}$$
    *(For example, a $+0.12^\circ\text{C}$ rise over $0.256\text{ s}$ equals $+0.47^\circ\text{C/s}$, signaling rapid stall heating).*

---

## PART 8 — MACHINE LEARNING PIPELINE

All Machine Learning code is inside `models/`:
- [`models/preprocessing.py`](../models/preprocessing.py)
- [`models/train_model.py`](../models/train_model.py)
- [`models/inference.py`](../models/inference.py)

### 8.1 Why Two Models? (Random Forest + Isolation Forest)

1. **Random Forest Classifier (`RandomForestClassifier`, Supervised)**:
   - **What it is**: An ensemble of `150` decision trees (`n_estimators=150`, `max_depth=10`, `min_samples_leaf=2`, `class_weight="balanced"`). Each tree is trained on a bootstrap sample of labeled data and splits on random subsets of the 18 features.
   - **How it classifies**: When a new 18-feature vector arrives, all 150 trees cast a vote for one of the 3 known classes:
     - `HEALTHY`
     - `ROTOR_IMBALANCE`
     - `MECHANICAL_STALL`
   - **Why Random Forest over Deep Learning?**: For tabular engineering features (18 numbers per window), Random Forests train in less than 2 seconds on a CPU, require no GPU, do not suffer from vanishing gradients, provide direct `feature_importances_`, and run inference in $< 2\text{ ms}$.
2. **Isolation Forest (`IsolationForest`, Unsupervised Anomaly Detection)**:
   - **What it is**: Fitted exclusively on `HEALTHY` training windows (`X_train_healthy` in `models/train_model.py`, lines 113–120, `n_estimators=120`, `contamination=0.06`).
   - **Why we include it**: Supervised classifiers only know the fault classes they were trained on. If a completely new fault occurs (e.g., a loose mounting bolt) that doesn't look exactly like `ROTOR_IMBALANCE` or `MECHANICAL_STALL`, Isolation Forest flags it as an out-of-distribution anomaly (`iso_pred == -1`).

### 8.2 Training & Preprocessing Workflow (`models/train_model.py` & `models/preprocessing.py`)

1. **Dataset Generation (`generate_synthetic_dataset`)**:
   - Generates `240` windows per class (`720` total windows) using `SensorSimulator` and `extract_unified_features()`.
   - Periodically varies initial thermal baselines (`lines 58–65`) so the model doesn't lazily memorize absolute temperature alone, and injects $4\%$ random cross-condition boundary noise (`lines 70–72`).
   - Saves `data/processed/synthetic_training_features.csv` and `data/sample/sample_sensor_features.csv`.
2. **Validation & Scaling (`prepare_features_and_labels` in `models/preprocessing.py`)**:
   - Verifies all 18 `FEATURE_COLUMNS` exist, replaces any `Inf`/`NaN` with column medians, and normalizes features to zero mean and unit variance using `StandardScaler`:
     $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
3. **Train/Test Split**:
   - Stratified 75% / 25% split (`train_test_split(..., test_size=0.25, stratify=y)`): **540 training windows** (`180` per class) and **180 test windows** (`60` per class).
4. **Artifact Persistence**:
   - Saves `random_forest_model.joblib`, `isolation_forest_model.joblib`, `feature_scaler.joblib`, `feature_metadata.json`, and `evaluation_metrics.json` inside `models/saved/`.

### 8.3 Understanding the Evaluation Metrics & Synthetic Data Honesty

In `models/saved/evaluation_metrics.json`:
- **Accuracy, Precision, Recall, F1-Score**: Currently `1.0000` (`100%`) on the held-out synthetic test set (`60/60` per class in the confusion matrix).
- **Top Feature Importances Learned by the Forest**:
  1. `vib_band_energy_50_1000hz` (`9.98%`)
  2. `current_rms` (`9.63%`)
  3. `vib_rms` (`8.69%`)
  4. `current_mean` (`8.07%`)
  5. `vib_mean` (`8.05%`)
- **Critical Engineering Honesty (Why 100% Synthetic Accuracy $\neq$ 100% Real-World Accuracy)**:
  - Synthetic waveforms generated from mathematical equations have clean class separation, which is why the test set achieves `1.00` accuracy.
  - Real EV motors experience road vibration, inverter PWM switching noise, and load transients. That is why our code explicitly records `"data_source": "SYNTHETIC_DEMONSTRATION"` and `"confidence_note": "Uncalibrated Random Forest ensemble tree-vote agreement (not a calibrated probability)"`.

---

## PART 9 — MULTI-MODAL SENSOR FUSION & FAULT DETECTION

Implemented in [`src/sensor_fusion.py`](../src/sensor_fusion.py) (`MultiModalSensorFusion`) with thresholds configured in `ThresholdConfig` ([`src/config.py`](../src/config.py), lines 118–138).

### 9.1 How `MultiModalSensorFusion` Combines All Three Sensors

Step 1 (`evaluate_modalities()`): Evaluates each physical sensor independently against domain thresholds:
- **Vibration (`vib_severity`)**:
  - `HIGH` if `vib_rms >= 0.65 g` or `vib_peak_to_peak >= 2.16 g` (`1.60 * 1.35`).
  - `ELEVATED` if `vib_rms >= 0.35 g` or `vib_peak_to_peak >= 1.60 g`.
  - Also checks `vib_band_energy_50_1000hz >= 8.0 g²` and impulsive `vib_kurtosis >= 4.2`.
- **Current (`cur_severity`)**:
  - `CRITICAL_OVERCURRENT` if `current_rms >= 5.5 A`.
  - `ELEVATED` if `current_rms >= 3.5 A` or `current_fluctuation >= 18.0%`.
- **Temperature (`temp_severity`)**:
  - `HIGH` if `temp_current_c >= 70.0 °C` or `temp_rise_rate_c_per_s >= 0.40 °C/s`.
  - `ELEVATED` if `temp_current_c >= 55.0 °C` or `temp_rise_rate_c_per_s >= 0.25 °C/s`.

Step 2 (`fuse()`): Cross-verifies the ML prediction (`ml_class`, `ml_conf`) with the three modality severities so a single noisy sensor threshold cannot falsely trigger a critical fault.

---

### 9.2 Three Concrete Walkthrough Examples

#### Example 1: Healthy Motor (`HEALTHY`)
- **What Happens Physically**: The rotor is balanced and spinning smoothly at $\approx 85\text{ Hz}$ ($5100\text{ RPM}$) under nominal load.
- **What the Sensors Observe**:
  - MPU6050: Low sinusoidal vibration ($\pm 0.09\text{ g}$ on X/Y, $1.0\text{ g}$ gravity on Z).
  - ACS712: Steady current around $2.15\text{ A}$.
  - DS18B20: Stable temperature near $39.0^\circ\text{C}$ ($\text{rise rate} \approx 0.00^\circ\text{C/s}$).
- **Features Extracted**: `vib_rms = 0.082 g`, `vib_band_energy = 0.45 g²`, `current_rms = 2.15 A`, `temp_current_c = 39.0 °C`.
- **ML & Fusion Decision**: Random Forest votes `100% HEALTHY`; all 3 modality severities are `NORMAL`.
- **Dashboard Output**: Status `Healthy` (`GREEN`), condition `Normal Operation`.

#### Example 2: Rotor Imbalance (`ROTOR_IMBALANCE`)
- **What Happens Physically**: An eccentric weight or worn rotor alignment creates strong centrifugal shaking at the rotational frequency ($135\text{ Hz}$) and its $2\times$ ($270\text{ Hz}$) and $3\times$ ($405\text{ Hz}$) harmonics.
- **What the Sensors Observe**:
  - MPU6050: Strong periodic oscillation ($\pm 0.68\text{ g}$ on X/Y).
  - ACS712: Moderate current modulation ($2.95\text{ A}$ with $0.42\text{ A}$ harmonic ripple as torque load fluctuates each revolution).
  - DS18B20: Mild bearing friction warming toward $45.5^\circ\text{C}$.
- **Features Extracted**: `vib_rms = 0.58 g`, `vib_dominant_freq_hz = 136.7 Hz`, `vib_band_energy_50_1000hz = 22.4 g²`, `current_rms = 2.98 A`.
- **ML & Fusion Decision**: Random Forest predicts `ROTOR_IMBALANCE`; `MultiModalSensorFusion` confirms elevated vibration RMS and high in-band spectral energy at $136.7\text{ Hz}$.
- **Dashboard Output**: Status `Warning` (`YELLOW`) or `Critical` (`RED`), condition `Rotor Imbalance`, with explanation: *"Elevated vibration RMS (0.580 g) and Peak-to-Peak (2.150 g) | Increased dominant-frequency spectral energy (22.40 g² at 136.7 Hz)"*.

#### Example 3: Mechanical Stall (`MECHANICAL_STALL`)
- **What Happens Physically**: The rotor is mechanically jammed or overloaded. Rotation stops ($\text{Back-EMF} \to 0$), causing stator current to surge above $7.2\text{ A}$, accompanied by $65\text{ Hz}$ electromagnetic hum, mechanical chattering impulses, and rapid $I^2R$ winding heating.
- **What the Sensors Observe**:
  - MPU6050: Impulsive chattering bursts (high Kurtosis $> 4.2$).
  - ACS712: Massive current surge (`current_rms = 7.35 A`, `current_peak = 9.10 A`).
  - DS18B20: Temperature climbing rapidly (`+0.48 °C` per window $\rightarrow \text{rise rate} = +1.85^\circ\text{C/s}$).
- **ML & Fusion Decision**: Random Forest predicts `MECHANICAL_STALL`; `cur_severity == "CRITICAL_OVERCURRENT"` and `temp_severity == "HIGH"` corroborate the stall.
- **Dashboard Output**: Status `Critical` (`RED`), condition `Mechanical Stall`, logging an immediate timestamped event in the Alert Center.

---

## PART 10 — PYTHON BACKEND & DATA FLOW ARCHITECTURE

### 10.1 Complete Data-Flow Diagram (Actual Execution Order in Code)

```text
+-----------------------------------------------------------------------------------+
| MODE A: SIMULATION MODE                   MODE B: LIVE HARDWARE MODE              |
| src/data_generator.py                     firmware/esp32_main.ino                 |
| SensorSimulator.generate_window()         ESP32 reads MPU6050, ACS712, DS18B20    |
| Produces 256-sample dict                  Sends JSON lines over USB @ 115200 baud |
+--------------------+-----------------------------------------+--------------------+
                     |                                         |
                     |                                         v
                     |                      src/serial_reader.py
                     |                      ESP32SerialReader.read_available_records()
                     |                      parse_serial_json_line()
                     |                                         |
                     |                                         v
                     |                      src/data_processor.py
                     |                      DataProcessor.ingest_record()
                     |                      Validates sensors, handles NaN/-127C,
                     |                      appends to 256-sample rolling deques
                     +--------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 1. WINDOW PROCESSING: src/data_processor.py -> DataProcessor.process_window()     |
|    Calls src/feature_extractor.py -> extract_unified_features(window_data)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. DIGITAL SIGNAL FILTERING: src/signal_filters.py                                |
|    - handle_missing_values()  (Linear interpolation over NaNs/Infs)               |
|    - clip_outliers()          (Robust MAD outlier clamping at z = 4.5)            |
|    - remove_dc_offset()       (Subtracts static 1g gravity)                       |
|    - apply_vibration_bandpass_filter() (4th-order Butterworth 50Hz - 480Hz/1kHz)  |
|    - rolling_mean_filter()    (3-sample smoothing on ACS712 current)              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. FEATURE EXTRACTION: src/feature_extractor.py                                   |
|    - extract_vibration_features() + compute_fft_spectrum() -> 9 features          |
|    - extract_current_features()                            -> 5 features          |
|    - extract_temperature_features()                        -> 4 features          |
|    Outputs ordered 18-element vector matching src/config.py::FEATURE_COLUMNS      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. ML INFERENCE: models/inference.py -> FaultModelInference.predict()             |
|    - Scales 18 features with models/saved/feature_scaler.joblib (StandardScaler)  |
|    - Predicts class & tree-vote share via random_forest_model.joblib              |
|    - Predicts anomaly status & score via isolation_forest_model.joblib            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. MULTI-MODAL SENSOR FUSION: src/sensor_fusion.py -> MultiModalSensorFusion.fuse |
|    - evaluate_modalities() checks physical thresholds across all 3 sensors        |
|    - Cross-verifies ML prediction with multi-sensor evidence                      |
|    - Checks sensor_availability for degraded mode                                 |
|    - Produces final_fault_class, alert_level (GREEN/YELLOW/RED), explanations     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. ORCHESTRATION & UI: src/fault_detection.py -> dashboard/app.py                 |
|    - FaultDetectionSystem.analyze_window() logs to history_records & alert_events |
|    - dashboard/app.py renders active section using components.py and charts.py    |
+-----------------------------------------------------------------------------------+
```

---

## PART 11 — TECHNOLOGY STACK BREAKDOWN

Every library listed below is actively imported and used in `D:\Prayog`:

| Technology / Library | Where Used | Why We Use It | What Happens If Removed |
| :--- | :--- | :--- | :--- |
| **Python (3.11+ / 3.14)** | Entire backend, ML, and dashboard | Primary programming language for DSP, ML, serial I/O, and web UI | Application cannot run. |
| **Streamlit (`streamlit`)** | `dashboard/app.py`, `dashboard/components.py` | Renders the interactive 6-section web application and manages session state without needing a separate React/FastAPI stack | Web UI and interactive controls disappear. |
| **NumPy (`numpy`)** | `src/data_generator.py`, `src/signal_filters.py`, `src/feature_extractor.py`, `models/` | Fast vectorized math, array operations, RMS/Peak-to-Peak calculation, and `np.fft.rfft` Fast Fourier Transform | All waveform generation, FFT, and feature math would break. |
| **SciPy (`scipy`)** | `src/signal_filters.py` (`scipy.signal`), `src/feature_extractor.py` (`scipy.stats`) | Designs the Butterworth bandpass filter (`butter`, `sosfiltfilt`) and computes 4th-moment statistical `kurtosis` | Bandpass vibration filtering and kurtosis calculation would fail. |
| **Pandas (`pandas`)** | `models/preprocessing.py`, `models/train_model.py`, `src/fault_detection.py`, `dashboard/` | Tabular dataset cleaning, median imputation, session history DataFrames, and CSV import/export | Dataset loading, history tables, and CSV downloads would break. |
| **Scikit-Learn (`scikit-learn`)** | `models/preprocessing.py`, `models/train_model.py` | Provides `StandardScaler`, `RandomForestClassifier`, `IsolationForest`, `train_test_split`, and evaluation metrics | Machine learning training and scaling would be unavailable. |
| **Joblib (`joblib`)** | `models/train_model.py`, `models/inference.py` | Serializes and loads trained scikit-learn models (`.joblib` files) to/from disk | Trained models could not be saved or loaded for real-time inference. |
| **Plotly (`plotly`)** | `dashboard/charts.py` | Builds interactive, zoomable waveforms, FFT spectra, bar charts, and confusion matrices styled in Obsidian Sage | All charts in the dashboard would fail to render. |
| **PySerial (`pyserial`)** | `src/serial_reader.py` | Enumerates Windows COM ports (`list_ports.comports()`) and reads UART serial streams (`serial.Serial`) from the ESP32 | Live Hardware Mode and COM port scanning would be disabled (note: `src/serial_reader.py` wraps the import in a `try/except` so Simulation Mode still runs even if `pyserial` is missing). |
| **Pytest (`pytest`)** | `tests/test_*.py` | Executes the 14 automated unit and integration tests | Automated verification suite cannot run. |
| **Arduino C++ & Libraries (`Wire`, `Adafruit_MPU6050`, `OneWire`, `DallasTemperature`)** | `firmware/*` | Communicates with physical sensors over I2C, ADC1, and 1-Wire on the ESP32 | Physical ESP32 microcontroller cannot read the sensors. |

---

## PART 12 — COMPLETE FILE-BY-FILE INVENTORY

### 12.1 Firmware (`firmware/`)
1. **`firmware/esp32_main.ino`**:
   - *Purpose*: Main Arduino sketch (`setup()`, `loop()`, `emitJsonRecord()`).
   - *Inputs*: Synchronized readings from `SensorManager` and filtered window stats from `EdgeSignalProcessor`.
   - *Outputs*: Newline-delimited JSON strings over `Serial` at `115200` baud.
2. **`firmware/sensor_manager.h` & `firmware/sensor_manager.cpp`**:
   - *Purpose*: Hardware driver layer encapsulating MPU6050 (`I2C`), ACS712 (`ADC1 GPIO 34`), and DS18B20 (`1-Wire GPIO 4`).
   - *Key Functions*: `begin()`, `calibrateCurrentZero()`, `readCurrentAmps()`, `updateTemperatureAsync()`, `readSynchronizedSample()`.
3. **`firmware/signal_processing.h` & `firmware/signal_processing.cpp`**:
   - *Purpose*: On-device Direct Form II Transposed Biquad bandpass filter and window statistics calculator.
   - *Key Functions*: `configureBandpass()`, `processVibrationSample()`, `computeWindowFeatures()`.
4. **`firmware/README.md`**: Pinout table, Nyquist notes, required Arduino libraries, and JSON payload specification.

### 12.2 Core Backend (`src/`)
5. **`src/__init__.py`**: Package initializer defining `__version__ = "1.0.0"`.
6. **`src/config.py`**:
   - *Purpose*: Central configuration for paths, class constants (`FAULT_CLASSES`), `FEATURE_COLUMNS` (18 features), `FEATURE_UNITS`, `SignalConfig`, `SerialConfig`, and `ThresholdConfig`.
   - *Used By*: Almost every file in `src/`, `models/`, `dashboard/`, and `tests/`.
7. **`src/data_generator.py`**:
   - *Purpose*: Implements `SensorSimulator` to generate physics-inspired 256-sample multi-sensor windows for `HEALTHY`, `ROTOR_IMBALANCE`, and `MECHANICAL_STALL`.
   - *Key Methods*: `set_operating_state()`, `reset_thermal_state()`, `generate_window()`.
8. **`src/signal_filters.py`**:
   - *Purpose*: Reusable SciPy signal-conditioning functions.
   - *Key Functions*: `handle_missing_values()`, `clip_outliers()`, `remove_dc_offset()`, `compute_effective_bandpass_bounds()`, `apply_vibration_bandpass_filter()`, `rolling_mean_filter()`, `normalize_signal()`.
9. **`src/feature_extractor.py`**:
   - *Purpose*: Window-based time and FFT spectral feature extractor.
   - *Key Functions*: `compute_fft_spectrum()`, `extract_vibration_features()`, `extract_current_features()`, `extract_temperature_features()`, `extract_unified_features()`.
10. **`src/data_processor.py`**:
    - *Purpose*: Implements `DataProcessor` for validating incoming serial records, tracking `sensor_availability`, maintaining `deque` rolling buffers, and running `process_window()`.
11. **`src/sensor_fusion.py`**:
    - *Purpose*: Implements `MultiModalSensorFusion` (`evaluate_modalities()` and `fuse()`).
12. **`src/fault_detection.py`**:
    - *Purpose*: Implements `FaultDetectionSystem` facade (`analyze_window()`, `get_history_dataframe()`, `get_recent_alerts()`).
13. **`src/serial_reader.py`**:
    - *Purpose*: Implements `list_available_com_ports()`, `parse_serial_json_line()`, and `ESP32SerialReader` (`configure()`, `connect()`, `disconnect()`, `read_available_records()`, `export_received_to_csv()`).

### 12.3 Machine Learning (`models/` & `scripts/`)
14. **`models/preprocessing.py`**: `validate_and_clean_dataset()`, `prepare_features_and_labels()`, `load_csv_dataset()`.
15. **`models/train_model.py`**: `generate_synthetic_dataset()` and `train_and_evaluate_models()`.
16. **`models/inference.py`**: Implements `FaultModelInference` (`load_models()`, `_prepare_vector()`, `predict()`).
17. **`models/saved/*`**:
    - `random_forest_model.joblib`, `isolation_forest_model.joblib`, `feature_scaler.joblib`
    - `feature_metadata.json`, `evaluation_metrics.json`
18. **`scripts/generate_dataset.py` & `scripts/train.py`**: CLI entrypoints for dataset generation and model training.

### 12.4 Dashboard (`dashboard/` & `.streamlit/`)
19. **`.streamlit/config.toml`**: Configures Streamlit's native dark base theme to Obsidian Sage (`#171A17`, `#1E221E`, `#A8B58A`, `#F1EBDD`).
20. **`dashboard/app.py`**: Main Streamlit application with 6 navigation sections and session state management.
21. **`dashboard/components.py`**: Custom Obsidian Sage CSS, top status bar, hero health card, 8-card KPI overview, and alert log renderer.
22. **`dashboard/charts.py`**: 7 Plotly chart builders styled in the Obsidian Sage palette.

### 12.5 Data, Tests, and Root Config (`data/`, `tests/`, Root)
23. **`data/processed/synthetic_training_features.csv`** (720 rows) & **`data/sample/sample_sensor_features.csv`** (90 rows).
24. **`tests/test_filters.py`, `tests/test_features.py`, `tests/test_data.py`, `tests/test_inference.py`**: 14 automated `pytest` unit and integration tests.
25. **`requirements.txt`, `run.bat`, `package.json`, `.env.example`, `.gitignore`, `README.md`**.

---

## PART 13 — RUNNING THE PROJECT (STEP-BY-STEP POWERSHELL)

Open **Windows PowerShell** and run the following commands inside `D:\Prayog`:

### 1. Navigate to the Workspace
```powershell
cd D:\Prayog
```

### 2. Activate the Virtual Environment (`.venv`)
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Note: If PowerShell execution policy blocks `.ps1` scripts, you can run every command directly with `.\.venv\Scripts\python.exe` or `.\run.bat` without needing `Activate.ps1`.)*

### 3. Install / Verify Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Generate Demonstration Dataset (Optional — Already Generated)
```powershell
.\.venv\Scripts\python.exe scripts/generate_dataset.py --samples-per-class 240
```

### 5. Train & Evaluate the ML Models (Optional — Already Trained & Saved)
```powershell
.\.venv\Scripts\python.exe scripts/train.py
```

### 6. Run the Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 7. Start the Dashboard
You can use any of these three commands:
```powershell
.\run.bat
```
or
```powershell
npm run dev
```
or
```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

---

## PART 14 — HARDWARE CONNECTION & REAL-WORLD DEMONSTRATION

### 14.1 Checklist for the Hardware Team Before Connecting to Prayog

1. **Wiring Verification**:
   - **MPU6050**: `VCC -> 3.3V`, `GND -> GND`, `SDA -> GPIO 21`, `SCL -> GPIO 22`.Rigidly mount the sensor to the motor housing (loose tape damps high-frequency vibrations).
   - **ACS712**: Connect in series with the motor power supply. Connect `OUT` to `GPIO 34` through a $3.3\text{ V}$-safe voltage divider (`VOLTAGE_DIVIDER_RATIO = 1.50f` in `firmware/sensor_manager.h`).
   - **DS18B20**: `VCC -> 3.3V`, `GND -> GND`, `DQ -> GPIO 4` with a **$4.7\text{ k}\Omega$ pull-up resistor** between `GPIO 4` and `3.3V`.
2. **Flash the Firmware**:
   - Open `firmware/esp32_main.ino` in Arduino IDE, install `Adafruit MPU6050`, `Adafruit Unified Sensor`, `OneWire`, and `DallasTemperature`, select your ESP32 board, and upload.
3. **Close Arduino Serial Monitor**:
   - Ensure the Arduino IDE Serial Monitor is **closed** so Windows releases the COM port lock.
4. **Connect Inside Prayog**:
   - Open the Prayog dashboard $\rightarrow$ click **Device Connection** in the left sidebar.
   - Click **Rescan Ports**, select the detected `COMx` port, keep Baud Rate at `115200`, and click **Connect ESP32**.
   - Verify that:
     - **ESP32 Connection Status** changes to `Connected`.
     - **Serial Communication Status** increments packet count (`1, 2, 3...`).
     - **Individual Sensor Availability** shows `Online` for MPU6050, ACS712, and DS18B20.
     - **Incoming Sensor-Data Preview** displays live JSON rows.

---

## PART 15 — HONEST PROJECT LIMITATIONS & FUTURE IMPROVEMENTS

1. **Synthetic Training Data vs. Real Motor Dynamics**:
   - The current `.joblib` models in `models/saved/` are trained on the `SensorSimulator` dataset (`SYNTHETIC_DEMONSTRATION`). Once physical motor recordings are captured using the *"Save Captured Hardware Records to CSV"* button on the Device Connection page, the model should be retrained on real labeled data via `python scripts/train.py --csv data/raw/esp32_live_capture.csv --source-label RECORDED_HARDWARE`.
2. **MPU6050 Bandwidth & I2C Sampling Ceiling**:
   - The MPU6050's internal accelerometer low-pass filter tops out at $260\text{ Hz}$, and I2C polling + UART serialization on a single loop limits sampling to $\approx 500\text{ Hz} - 1000\text{ Hz}$ (Nyquist limit $250\text{ Hz} - 500\text{ Hz}$). To capture full $1\text{ kHz}$ bearing fault harmonics on hardware, a future upgrade can use an SPI accelerometer (such as ADXL345 or ADXL355) read via ESP32 DMA/FreeRTOS dual-core tasks at $\ge 2500\text{ Hz}$.
3. **ACS712 Zero-Offset & Motor Rating Calibration**:
   - Thresholds in `ThresholdConfig` (`nominal_current_a = 2.2 A`, `current_rms_stall_a = 5.5 A`) are tuned for a small demonstration EV motor testbench. For a different motor rating (e.g., a $250\text{ W}$ or $1\text{ kW}$ BLDC motor), update `ThresholdConfig` in `src/config.py` and `ACS712_SENSITIVITY` in `firmware/sensor_manager.h`.
4. **Host-Assisted Inference vs. Pure On-Chip TinyML**:
   - Currently, the ESP32 performs synchronized sampling, IIR biquad bandpass filtering, and basic window feature extraction (`EdgeWindowFeatures`), while full FFT + Random Forest inference runs in Python on the connected laptop/edge gateway. A future enhancement can export the Random Forest trees to C arrays using `emlearn` or `micromlgen` to run classification directly inside the ESP32 flash memory.

---

## PART 16 — JUDGE & TECHNICAL INTERVIEW Q&A PREPARATION

### Q1: Why did you choose this project and what is the problem statement?
> **Answer**: Electric motors are the heart of EVs, but mechanical faults like rotor imbalance and rotor stalls often go unnoticed until catastrophic bearing failure or winding burnout occurs. Single-sensor threshold alarms fail in vehicles because road bumps trigger false vibration alarms, while temperature sensors react too slowly due to thermal inertia. We built **Prayog** to fuse three complementary modalities—mechanical vibration (MPU6050), motor current (ACS712), and temperature trends (DS18B20)—using digital signal processing and Machine Learning for reliable, explainable early fault detection.

### Q2: Why did you choose Random Forest instead of Deep Learning (CNN/LSTM)?
> **Answer**: After extracting 18 domain-specific physics and spectral features (RMS, Kurtosis, FFT Dominant Frequency, Band Energy, Current Ripple, Thermal Rise Rate), our data is structured tabular data. On 18 tabular features, a 150-tree `RandomForestClassifier` trains in under 2 seconds on CPU, executes inference in under 2 milliseconds, works on small datasets without overfitting, and provides direct feature importance rankings (`vib_band_energy_50_1000hz` and `current_rms` are our top predictors). We also paired it with an unsupervised `IsolationForest` trained on healthy operation to catch novel anomalies.

### Q3: What is FFT and why do you use a Hann window before computing it?
> **Answer**: The **Fast Fourier Transform (FFT)** converts a time-domain vibration signal $v[n]$ into its frequency components $X[k]$, allowing us to see exact rotational harmonics (such as a $135\text{ Hz}$ rotor imbalance peak) that are invisible in raw time waveforms. Because our 256-sample window cuts off abruptly at the edges, taking a raw FFT causes **spectral leakage** (energy smearing across neighboring frequencies). In `compute_fft_spectrum()` (`src/feature_extractor.py`), we multiply the signal by a bell-shaped **Hann window** (`np.hanning(n)`) and apply a coherent gain scale factor (`2.0 / np.sum(window)`) to suppress spectral leakage and preserve accurate peak amplitudes.

### Q4: What is RMS and why don't you just use the arithmetic mean of vibration?
> **Answer**: A bandpass-filtered vibration wave oscillates symmetrically between positive and negative acceleration (e.g., $+0.5\text{ g}$ and $-0.5\text{ g}$), so its simple arithmetic mean is always **zero**! **RMS (Root-Mean-Square)** squares every sample first, averages the squares, and takes the square root ($\sqrt{\frac{1}{N}\sum v[n]^2}$), which measures the true continuous mechanical energy of the vibration regardless of sign.

### Q5: How do you prevent false positives when an EV hits a pothole?
> **Answer**: Two ways:
> 1. **DSP Bandpass Filtering**: In `src/signal_filters.py`, our 4th-order Butterworth bandpass filter strips out low-frequency chassis bounce and road shocks below $50\text{ Hz}$.
> 2. **Multi-Modal Sensor Fusion**: In `MultiModalSensorFusion.fuse()` (`src/sensor_fusion.py`), the system cross-checks vibration against electrical current (`ACS712`) and thermal rate (`DS18B20`). A single road bump does not cause sustained rotational FFT harmonic energy or an electrical overcurrent surge, so the fusion engine prevents false critical alarms.

### Q6: What happens if a sensor wire comes loose during operation?
> **Answer**: In `DataProcessor.validate_and_parse_record()` (`src/data_processor.py`), every sensor is validated independently. For example, if the DS18B20 temperature sensor disconnects and reports `null` or `-127.0°C`, `sensor_availability["ds18b20"]` is marked `False`, the processor substitutes the last known valid temperature from `self.temp_history` so feature extraction doesn't crash, and `MultiModalSensorFusion` appends a transparent warning: *"Degraded sensor mode: DS18B20 unavailable (using fallback state)"*.

### Q7: Does your `50 Hz – 1 kHz` bandpass filter violate the Nyquist theorem when sampling at `1000 Hz`?
> **Answer**: That is a classic DSP trap that we specifically engineered around! At a sampling rate of $f_s = 1000\text{ Hz}$, the Nyquist frequency is $f_{\text{Nyquist}} = 500\text{ Hz}$. Passing a $1000\text{ Hz}$ cutoff into a digital filter at $f_s = 1000\text{ Hz}$ would violate Nyquist ($W_n \ge 1.0$). In both `compute_effective_bandpass_bounds()` (`src/signal_filters.py`) and `EdgeSignalProcessor::configureBandpass()` (`firmware/signal_processing.cpp`), the code dynamically computes the Nyquist frequency and clamps the upper passband edge below Nyquist ($0.96 \times f_{\text{Nyquist}} = 480\text{ Hz}$ in Python, $0.45 \times f_s$ in C++) unless the sampling rate is raised above $2000\text{ Hz}$.

