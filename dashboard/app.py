"""Prayog — EV Motor Health Monitoring & Predictive Maintenance Dashboard.

Visual Identity: OBSIDIAN SAGE — ENGINEERING INTELLIGENCE
- PRIMARY   (Obsidian Black): #171A17
- SECONDARY (Muted Sage):     #A8B58A
- TERTIARY  (Warm Ivory):     #F1EBDD

Simplified 6-Section Navigation:
1. Overview
2. Live Monitoring
3. Fault Detection
4. Analytics & History
5. Device Connection
6. Settings
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLASS_HEALTHY,
    CLASS_ROTOR_IMBALANCE,
    CLASS_MECHANICAL_STALL,
    FAULT_CLASSES,
    FEATURE_COLUMNS,
    FEATURE_UNITS,
    SIGNAL_CONFIG,
    SERIAL_CONFIG,
)
from src.data_generator import SensorSimulator
from src.fault_detection import FaultDetectionSystem
from src.serial_reader import ESP32SerialReader, list_available_com_ports
from dashboard.components import (
    inject_engineering_theme_css,
    render_header,
    render_overview_hero,
    render_kpi_overview,
    render_alert_center,
    format_fault_name,
    format_status_label,
)
from dashboard.charts import (
    create_vibration_waveform_chart,
    create_current_waveform_chart,
    create_temperature_trend_chart,
    create_filtered_vibration_chart,
    create_fft_spectrum_chart,
    create_class_scores_chart,
    create_confusion_matrix_chart,
)

NAVIGATION_SECTIONS = [
    "Overview",
    "Live Monitoring",
    "Fault Detection",
    "Analytics & History",
    "Device Connection",
    "Settings",
]

FRIENDLY_STATE_TO_CODE = {
    "Healthy (Normal Operation)": CLASS_HEALTHY,
    "Rotor Imbalance": CLASS_ROTOR_IMBALANCE,
    "Mechanical Stall": CLASS_MECHANICAL_STALL,
}
CODE_TO_FRIENDLY_STATE = {v: k for k, v in FRIENDLY_STATE_TO_CODE.items()}

st.set_page_config(
    page_title="Prayog | EV Motor Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize persistent state across Streamlit reruns."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator(
            sampling_rate_hz=SIGNAL_CONFIG.sampling_rate_hz,
            window_size=SIGNAL_CONFIG.window_size,
        )
    if "detection_system" not in st.session_state:
        st.session_state.detection_system = FaultDetectionSystem()
    if "serial_reader" not in st.session_state:
        st.session_state.serial_reader = ESP32SerialReader()
    if "operating_mode" not in st.session_state:
        st.session_state.operating_mode = "SIMULATION MODE"
    if "sim_fault_state" not in st.session_state:
        st.session_state.sim_fault_state = CLASS_HEALTHY
    if "sim_condition_selector" not in st.session_state:
        st.session_state.sim_condition_selector = CODE_TO_FRIENDLY_STATE[st.session_state.sim_fault_state]
    if "band_low_hz" not in st.session_state:
        st.session_state.band_low_hz = float(SIGNAL_CONFIG.bandpass_low_hz)
    if "band_high_hz" not in st.session_state:
        st.session_state.band_high_hz = float(SIGNAL_CONFIG.bandpass_high_hz)
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "refresh_interval_s" not in st.session_state:
        st.session_state.refresh_interval_s = 1.5
    if "latest_analysis" not in st.session_state:
        init_win = st.session_state.simulator.generate_window(state=CLASS_HEALTHY)
        st.session_state.latest_analysis = st.session_state.detection_system.analyze_window(
            init_win, operating_mode="SIMULATION MODE"
        )


def acquire_and_process_step(
    operating_mode: str,
    sim_state: str,
    band_low_hz: float,
    band_high_hz: float,
) -> Optional[Dict[str, Any]]:
    """Acquire one window from either Simulation Mode or Live ESP32 Hardware Mode and analyze it."""
    SIGNAL_CONFIG.bandpass_low_hz = float(band_low_hz)
    SIGNAL_CONFIG.bandpass_high_hz = float(band_high_hz)

    det_sys: FaultDetectionSystem = st.session_state.detection_system

    if operating_mode == "SIMULATION MODE":
        sim: SensorSimulator = st.session_state.simulator
        sim.set_operating_state(sim_state)
        window = sim.generate_window(state=sim_state, add_variation=True)
        analysis = det_sys.analyze_window(window, operating_mode="SIMULATION MODE")
        st.session_state.latest_analysis = analysis
        return analysis

    # LIVE HARDWARE MODE: Read strictly from ESP32 serial connection; NEVER fabricate live data
    reader: ESP32SerialReader = st.session_state.serial_reader
    records = reader.read_available_records(max_lines=64)
    if not records:
        return None

    for rec in records:
        det_sys.processor.ingest_record(rec)

    n = max(len(det_sys.processor.vib_x_buf), 1)
    dt = 1.0 / max(det_sys.processor.sampling_rate_hz, 1.0)
    live_window = {
        "timestamp_ms": det_sys.processor.last_timestamp_ms,
        "sampling_rate_hz": det_sys.processor.sampling_rate_hz,
        "time_axis_s": np.arange(n) * dt,
        "vibration_x": np.array(det_sys.processor.vib_x_buf, dtype=float),
        "vibration_y": np.array(det_sys.processor.vib_y_buf, dtype=float),
        "vibration_z": np.array(det_sys.processor.vib_z_buf, dtype=float),
        "current": np.array(det_sys.processor.current_buf, dtype=float),
        "temperature": float(det_sys.processor.temp_history[-1]) if det_sys.processor.temp_history else 38.0,
        "temp_history": list(det_sys.processor.temp_history),
        "window_duration_s": max(n * dt, 0.25),
    }
    analysis = det_sys.analyze_window(live_window, operating_mode="LIVE HARDWARE MODE")
    st.session_state.latest_analysis = analysis
    return analysis


def render_overview_section(
    analysis: Dict[str, Any],
    det_sys: FaultDetectionSystem,
    conn_status: str,
    operating_mode: str,
) -> None:
    """Section 1: Overview — Immediate, uncluttered summary for first-time and daily users."""
    render_overview_hero(analysis, connection_status=conn_status, operating_mode=operating_mode)
    render_kpi_overview(analysis, connection_status=conn_status, operating_mode=operating_mode)

    st.markdown('<div class="os-section-title">System Insights &amp; Quick Guidance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    fusion = analysis["fusion_result"]

    with c1:
        render_alert_center(det_sys.get_recent_alerts(limit=3), fusion)

    with c2:
        st.markdown(
            """
            <div class="os-card os-card-accent">
                <div class="os-card-label">How to Use Prayog</div>
                <div style="font-size:0.88rem; color:#F1EBDD; line-height:1.65;">
                    <b>1. Test Operating Conditions:</b> Use the sidebar selector to switch between
                    <i>Healthy</i>, <i>Rotor Imbalance</i>, and <i>Mechanical Stall</i> in Simulation Mode.<br>
                    <b>2. Inspect Sensor Waveforms:</b> Open <b>Live Monitoring</b> to view real-time
                    vibration, current, and temperature signals.<br>
                    <b>3. Connect Physical Hardware:</b> Open <b>Device Connection</b> to link your ESP32
                    over USB Serial when the hardware is ready.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_monitoring_section(
    analysis: Dict[str, Any],
    det_sys: FaultDetectionSystem,
) -> None:
    """Section 2: Live Monitoring — Real-time sensor waveforms and optional advanced spectral view."""
    raw_window = analysis["raw_window"]
    features = analysis["features"]
    diagnostics = analysis["diagnostics"]
    history_df = det_sys.get_history_dataframe()

    st.markdown('<div class="os-section-title">Real-Time Sensor Streams</div>', unsafe_allow_html=True)
    col_v, col_c = st.columns(2)
    with col_v:
        st.plotly_chart(
            create_vibration_waveform_chart(raw_window),
            use_container_width=True,
            key="live_vib_waveform",
        )
    with col_c:
        st.plotly_chart(
            create_current_waveform_chart(raw_window),
            use_container_width=True,
            key="live_cur_waveform",
        )

    st.plotly_chart(
        create_temperature_trend_chart(history_df),
        use_container_width=True,
        key="live_temp_trend",
    )

    with st.expander("Advanced Signal & Frequency Analysis (Bandpass Filter & FFT Spectrum)", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                create_filtered_vibration_chart(
                    time_axis_s=raw_window.get("time_axis_s", np.arange(256) * 0.001),
                    raw_vib=diagnostics["raw_vibration"],
                    filtered_vib=diagnostics["filtered_vibration"],
                ),
                use_container_width=True,
                key="live_vib_filtered",
            )
        with d2:
            st.plotly_chart(
                create_fft_spectrum_chart(
                    freqs=diagnostics["fft_freqs"],
                    magnitudes=diagnostics["fft_magnitudes"],
                    dominant_freq_hz=features["vib_dominant_freq_hz"],
                ),
                use_container_width=True,
                key="live_vib_fft",
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dominant Vibration Frequency", f"{features['vib_dominant_freq_hz']:.1f} Hz")
        m2.metric("Passband Energy (50 Hz – 1 kHz)", f"{features['vib_band_energy_50_1000hz']:.3f} g²")
        m3.metric("Filtered Vibration RMS", f"{features['vib_rms']:.4f} g")
        m4.metric("Vibration Peak-to-Peak", f"{features['vib_peak_to_peak']:.4f} g")


def render_fault_detection_section(
    analysis: Dict[str, Any],
    det_sys: FaultDetectionSystem,
) -> None:
    """Section 3: Fault Detection — Clear AI diagnosis, multi-sensor evidence, and alert log."""
    features = analysis["features"]
    fusion = analysis["fusion_result"]
    sev = fusion["modality_severities"]

    st.markdown('<div class="os-section-title">AI Fault Diagnosis &amp; Multi-Sensor Fusion</div>', unsafe_allow_html=True)

    top_c1, top_c2, top_c3 = st.columns(3)
    with top_c1:
        st.markdown(
            f"""
            <div class="os-card os-card-accent">
                <div class="os-card-label">Detected Motor Condition</div>
                <div class="os-card-value">{format_fault_name(fusion['final_fault_class'])}</div>
                <div class="os-card-sub">Overall Status: {format_status_label(fusion['alert_level'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_c2:
        st.markdown(
            f"""
            <div class="os-card">
                <div class="os-card-label">AI Model Agreement</div>
                <div class="os-card-value">{fusion['confidence_score']*100:.1f}%</div>
                <div class="os-card-sub">Random Forest Ensemble Vote Share</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_c3:
        anom_text = "Anomaly Detected" if fusion["is_anomaly"] else "Normal Pattern"
        st.markdown(
            f"""
            <div class="os-card">
                <div class="os-card-label">Anomaly Verification</div>
                <div class="os-card-value" style="font-size:1.25rem;">{anom_text}</div>
                <div class="os-card-sub">Isolation Forest Score: {fusion['anomaly_score']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    mid_l, mid_r = st.columns([3, 2])
    with mid_l:
        st.markdown("##### Why did the system make this decision?")
        for reason in fusion["explanations"]:
            st.markdown(f"- {reason}")

        st.markdown("##### Individual Sensor Health Check")
        s1, s2, s3 = st.columns(3)
        s1.metric("Vibration (MPU6050)", sev["vibration"].replace("_", " ").title())
        s2.metric("Current (ACS712)", sev["current"].replace("_", " ").title())
        s3.metric("Thermal (DS18B20)", sev["temperature"].replace("_", " ").title())

    with mid_r:
        st.plotly_chart(
            create_class_scores_chart(fusion["class_scores"]),
            use_container_width=True,
            key="fault_class_scores",
        )

    st.markdown('<div class="os-section-title">Diagnostic Alert Log</div>', unsafe_allow_html=True)
    render_alert_center(det_sys.get_recent_alerts(limit=10), fusion)

    with st.expander("Inspect All 18 Extracted Engineering Features", expanded=False):
        feat_rows = [
            {
                "Feature Name": col,
                "Measured Value": round(float(features.get(col, 0.0)), 4),
                "Unit": FEATURE_UNITS.get(col, ""),
            }
            for col in FEATURE_COLUMNS
        ]
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True, height=260)


def render_analytics_history_section(det_sys: FaultDetectionSystem) -> None:
    """Section 4: Analytics & History — Historical trends, CSV download, and model validation summary."""
    history_df = det_sys.get_history_dataframe()

    st.markdown('<div class="os-section-title">Session Telemetry History &amp; Export</div>', unsafe_allow_html=True)
    if history_df.empty:
        st.caption("No historical sensor windows recorded yet.")
    else:
        col_table, col_summary = st.columns([3, 1])
        with col_table:
            display_cols = [
                "time",
                "operating_mode",
                "predicted_fault",
                "health_status",
                "confidence",
                "vib_rms",
                "vib_dominant_freq_hz",
                "current_rms",
                "temp_current_c",
                "explanation",
            ]
            avail_cols = [c for c in display_cols if c in history_df.columns]
            st.dataframe(
                history_df[avail_cols].iloc[::-1],
                use_container_width=True,
                hide_index=True,
                height=280,
            )
        with col_summary:
            st.markdown("**Session Condition Summary**")
            fault_counts = history_df["predicted_fault"].map(format_fault_name).value_counts().reset_index()
            fault_counts.columns = ["Condition", "Windows"]
            st.dataframe(fault_counts, use_container_width=True, hide_index=True)

            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Session Data (CSV)",
                data=csv_bytes,
                file_name=f"prayog_motor_telemetry_{int(time.time())}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    st.markdown('<div class="os-section-title">Machine Learning Model Evaluation Summary</div>', unsafe_allow_html=True)
    eval_metrics = det_sys.inference_engine.evaluation_metrics
    if not eval_metrics:
        st.caption("Evaluation metrics are not available yet.")
    else:
        st.caption(f"**Dataset Note:** {eval_metrics.get('validation_disclaimer', '')}")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Test Accuracy", f"{eval_metrics.get('accuracy', 0.0) * 100:.1f}%")
        e2.metric("Precision", f"{eval_metrics.get('precision_weighted', 0.0) * 100:.1f}%")
        e3.metric("Recall", f"{eval_metrics.get('recall_weighted', 0.0) * 100:.1f}%")
        e4.metric("F1-Score", f"{eval_metrics.get('f1_weighted', 0.0) * 100:.1f}%")

        cm_col, imp_col = st.columns(2)
        with cm_col:
            if "confusion_matrix" in eval_metrics and "labels" in eval_metrics:
                st.plotly_chart(
                    create_confusion_matrix_chart(
                        eval_metrics["confusion_matrix"],
                        eval_metrics["labels"],
                    ),
                    use_container_width=True,
                    key="hist_conf_matrix",
                )
        with imp_col:
            st.markdown("**Most Influential Sensor Features**")
            imp_dict = eval_metrics.get("feature_importances", {})
            imp_df = (
                pd.DataFrame(
                    [{"Feature": k, "Relative Importance": v} for k, v in imp_dict.items()]
                )
                .sort_values("Relative Importance", ascending=False)
                .head(10)
            )
            st.dataframe(imp_df, use_container_width=True, hide_index=True, height=245)


def render_device_connection_section(
    reader: ESP32SerialReader,
    det_sys: FaultDetectionSystem,
) -> None:
    """Section 5: Device Connection — Dedicated hardware interface for connecting the physical ESP32."""
    st.markdown('<div class="os-section-title">ESP32 Hardware Communication Interface</div>', unsafe_allow_html=True)

    # Scan available COM ports
    detected_ports = list_available_com_ports()
    conn_state_label = "Connected" if reader.is_connected else "Disconnected"
    serial_health_label = (
        f"Streaming on {reader.port} @ {reader.baud_rate} baud"
        if reader.is_connected
        else "No active serial link"
    )

    # Top status cards (strictly honest — never fake connected statuses)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="os-card {'os-card-accent' if reader.is_connected else ''}">
                <div class="os-card-label">ESP32 Connection Status</div>
                <div class="os-card-value">{conn_state_label}</div>
                <div class="os-card-sub">{serial_health_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="os-card">
                <div class="os-card-label">Serial Communication Status</div>
                <div class="os-card-value">{reader.total_valid_records} Packets</div>
                <div class="os-card-sub">Malformed / Ignored Lines: {reader.malformed_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        port_count_txt = f"{len(detected_ports)} Detected" if detected_ports else "0 Detected"
        st.markdown(
            f"""
            <div class="os-card">
                <div class="os-card-label">Available USB COM Ports</div>
                <div class="os-card-value">{port_count_txt}</div>
                <div class="os-card-sub">Default Baud Rate: {SERIAL_CONFIG.default_baud_rate}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    ctrl_col, sensor_col = st.columns([3, 2])
    with ctrl_col:
        st.markdown("##### 1. Select Port &amp; Connect Hardware")
        if detected_ports:
            port_options = [p["device"] for p in detected_ports]
            port_Captions = {p["device"]: f"{p['device']} — {p['description']}" for p in detected_ports}
        else:
            port_options = [SERIAL_CONFIG.default_port, "COM4", "COM5", "COM6"]
            port_Captions = {p: f"{p} (Manual selection — no USB serial device detected)" for p in port_options}

        p_col, b_col = st.columns(2)
        with p_col:
            chosen_port = st.selectbox(
                "USB Serial Port",
                options=port_options,
                format_func=lambda x: port_Captions.get(x, x),
            )
        with b_col:
            baud_options = [9600, 57600, 115200, 230400]
            chosen_baud = st.selectbox(
                "Baud Rate",
                options=baud_options,
                index=baud_options.index(115200),
            )

        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            if st.button("Connect ESP32", type="primary", use_container_width=True):
                reader.configure(port=chosen_port, baud_rate=int(chosen_baud))
                if reader.connect():
                    st.session_state.operating_mode = "LIVE HARDWARE MODE"
                    st.success(f"Connected to {chosen_port} at {chosen_baud} baud.")
                else:
                    st.error(reader.last_error or f"Could not connect to {chosen_port}.")
        with btn2:
            if st.button("Disconnect", use_container_width=True):
                reader.disconnect()
                st.info("Serial connection closed.")
        with btn3:
            if st.button("Rescan Ports", use_container_width=True):
                st.rerun()

        if reader.last_error and not reader.is_connected:
            st.markdown(
                f"""
                <div class="os-notice" style="margin-top:12px;">
                    <b>Connection Diagnostic Message:</b><br>
                    <code>{reader.last_error}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sensor_col:
        st.markdown("##### 2. Individual Sensor Availability")
        avail = det_sys.processor.sensor_availability
        # Only report hardware sensors as Online if ESP32 is actually connected and receiving packets
        hw_active = reader.is_connected and reader.total_valid_records > 0

        sensors_meta = [
            ("MPU6050 — 3-Axis Vibration", "mpu6050", "I2C (GPIO 21 / 22)"),
            ("ACS712 — Motor Current", "acs712", "ADC1 (GPIO 34)"),
            ("DS18B20 — Temperature", "ds18b20", "1-Wire (GPIO 4)"),
        ]
        for title, key, pin_info in sensors_meta:
            if hw_active:
                s_status = "Online" if avail.get(key, False) else "Signal Missing"
            else:
                s_status = "Awaiting ESP32 Link"
            st.markdown(
                f"""
                <div class="os-event-row" style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;">
                        <b>{title}</b>
                        <span style="color:#A8B58A; font-weight:600;">{s_status}</span>
                    </div>
                    <div style="font-size:0.78rem; color:rgba(241,235,221,0.60); margin-top:2px;">
                        Interface: {pin_info}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="os-section-title">Incoming Sensor-Data Preview</div>', unsafe_allow_html=True)
    if reader.received_records:
        recent_list = list(reader.received_records)[-8:]
        st.dataframe(pd.DataFrame(recent_list[::-1]), use_container_width=True, hide_index=True)
        if st.button("Save Captured Hardware Records to CSV"):
            saved_path = reader.export_received_to_csv()
            if saved_path:
                st.success(f"Saved hardware capture to `{saved_path}`")
    else:
        st.markdown(
            """
            <div class="os-card">
                <div class="os-card-label">Live Serial Buffer Empty</div>
                <div style="font-size:0.88rem; color:rgba(241,235,221,0.75);">
                    No live JSON packets have been received from an ESP32 yet. Once connected, incoming records move here automatically:<br>
                    <code>{"timestamp": 123456, "vibration_x": 0.12, "vibration_y": 0.08, "vibration_z": 1.03, "current": 1.25, "temperature": 38.5}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Hardware Connection & Troubleshooting Guide", expanded=False):
        st.markdown(
            """
            1. **USB Cable & Driver Check**: Use a data-capable USB cable (not a charge-only cable). Verify that a `CP210x` or `CH340` COM port appears when you click **Rescan Ports**.
            2. **Close Competing Serial Monitors**: Windows allows only one program to open a COM port at a time. Close the Arduino IDE Serial Monitor before clicking **Connect ESP32** here.
            3. **Match the Baud Rate**: The ESP32 firmware (`firmware/esp32_main.ino`) defaults to `115200` baud.
            4. **DS18B20 Temperature Sensor Reads Missing**: Confirm the `4.7 kΩ` pull-up resistor is connected between `GPIO 4` (`DQ`) and `3.3V`.
            5. **No Hardware Available Right Now?**: You can switch to **Simulation Mode** in the left sidebar at any time to test all features without an ESP32 connected.
            """
        )


def render_settings_section(
    det_sys: FaultDetectionSystem,
    sim: SensorSimulator,
) -> None:
    """Section 6: Settings — Signal filtering, refresh speed, and AI model management."""
    st.markdown('<div class="os-section-title">Application &amp; Signal Processing Settings</div>', unsafe_allow_html=True)

    col_dsp, col_app = st.columns(2)
    with col_dsp:
        st.markdown("##### Digital Bandpass Filter (Vibration)")
        st.caption(
            f"Current Sampling Rate: `{SIGNAL_CONFIG.sampling_rate_hz:.0f} Hz` "
            f"(Nyquist Limit: `{0.5 * SIGNAL_CONFIG.sampling_rate_hz:.0f} Hz`). "
            "Upper cutoff automatically clamps below Nyquist for filter stability."
        )
        new_low = st.number_input(
            "Low Cutoff Frequency (Hz)",
            min_value=5.0,
            max_value=400.0,
            value=float(st.session_state.band_low_hz),
            step=5.0,
        )
        new_high = st.number_input(
            "High Cutoff Frequency (Hz)",
            min_value=100.0,
            max_value=2000.0,
            value=float(st.session_state.band_high_hz),
            step=50.0,
        )
        st.session_state.band_low_hz = float(new_low)
        st.session_state.band_high_hz = float(new_high)

    with col_app:
        st.markdown("##### Dashboard Update Preferences")
        st.session_state.auto_refresh = st.toggle(
            "Enable Automatic Live Updates",
            value=bool(st.session_state.auto_refresh),
            help="Automatically steps the sensor window at the selected interval.",
        )
        st.session_state.refresh_interval_s = st.slider(
            "Update Interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=float(st.session_state.refresh_interval_s),
            step=0.1,
        )

        st.markdown("##### Model &amp; Session Maintenance")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Retrain AI Model", type="primary", use_container_width=True):
                with st.spinner("Training Random Forest & Isolation Forest models..."):
                    from models.train_model import train_and_evaluate_models
                    train_and_evaluate_models()
                    det_sys.inference_engine.load_models()
                st.success("AI models retrained and reloaded.")
        with b2:
            if st.button("Reset Simulator Thermal State", use_container_width=True):
                sim.reset_thermal_state(38.2)
                st.info("Simulated motor temperature reset to 38.2 °C.")


def main() -> None:
    inject_engineering_theme_css()
    init_session_state()

    det_sys: FaultDetectionSystem = st.session_state.detection_system
    reader: ESP32SerialReader = st.session_state.serial_reader
    sim: SensorSimulator = st.session_state.simulator

    # =========================================================================
    # CLEAN, USER-FRIENDLY SIDEBAR NAVIGATION & QUICK CONTROLS
    # =========================================================================
    with st.sidebar:
        st.markdown("### PRAYOG")
        st.caption("Obsidian Sage · Engineering Intelligence")

        active_section = st.radio(
            "Navigation",
            options=NAVIGATION_SECTIONS,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("##### Data Source Mode")
        mode_labels = ["Simulation Mode", "Live ESP32 Hardware"]
        current_idx = 0 if st.session_state.operating_mode == "SIMULATION MODE" else 1
        chosen_mode_label = st.radio(
            "Select Operating Mode",
            options=mode_labels,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.operating_mode = (
            "SIMULATION MODE" if chosen_mode_label == "Simulation Mode" else "LIVE HARDWARE MODE"
        )

        step_now = False
        if st.session_state.operating_mode == "SIMULATION MODE":
            st.markdown("##### Simulated Motor Condition")
            friendly_options = list(FRIENDLY_STATE_TO_CODE.keys())
            
            def _on_simulation_condition_change() -> None:
                new_friendly = st.session_state.sim_condition_selector
                new_code = FRIENDLY_STATE_TO_CODE.get(new_friendly, CLASS_HEALTHY)
                st.session_state.sim_fault_state = new_code
                sim.set_operating_state(new_code)

            # Ensure session state selector is valid
            if st.session_state.sim_condition_selector not in friendly_options:
                st.session_state.sim_condition_selector = CODE_TO_FRIENDLY_STATE.get(
                    st.session_state.sim_fault_state, friendly_options[0]
                )

            chosen_friendly = st.selectbox(
                "Choose Condition to Demonstrate",
                options=friendly_options,
                key="sim_condition_selector",
                on_change=_on_simulation_condition_change,
                label_visibility="collapsed",
            )
            # Sync code with current selection
            selected_code = FRIENDLY_STATE_TO_CODE.get(chosen_friendly, CLASS_HEALTHY)
            st.session_state.sim_fault_state = selected_code
            sim.set_operating_state(selected_code)

            step_now = st.button("Capture Next Reading", type="primary", use_container_width=True)
        else:
            conn_txt = f"Connected ({reader.port})" if reader.is_connected else "Not Connected"
            st.caption(f"ESP32 Status: **{conn_txt}**")
            if not reader.is_connected:
                st.caption("Open **Device Connection** above to link your ESP32 over USB.")

        st.divider()
        st.session_state.auto_refresh = st.toggle(
            "Live Auto-Update",
            value=bool(st.session_state.auto_refresh),
        )

    # =========================================================================
    # ACQUIRE & PROCESS WINDOW
    # =========================================================================
    if step_now or st.session_state.auto_refresh:
        acquire_and_process_step(
            operating_mode=st.session_state.operating_mode,
            sim_state=st.session_state.sim_fault_state,
            band_low_hz=st.session_state.band_low_hz,
            band_high_hz=st.session_state.band_high_hz,
        )

    analysis: Dict[str, Any] = st.session_state.latest_analysis
    ml_res = analysis["ml_result"]

    if st.session_state.operating_mode == "SIMULATION MODE":
        conn_status = "Simulator Active"
    else:
        conn_status = (
            f"ESP32 Connected ({reader.port})"
            if reader.is_connected
            else f"ESP32 Disconnected ({reader.port})"
        )

    model_status_str = "Ready" if det_sys.inference_engine.is_loaded else "Offline"
    data_source_badge = ml_res.get("data_source_trained_on", "SYNTHETIC_DEMONSTRATION")

    # =========================================================================
    # TOP BAR & ACTIVE SECTION VIEW
    # =========================================================================
    render_header(
        connection_status=conn_status,
        operating_mode=st.session_state.operating_mode,
        last_update_str=analysis.get("time_str", "--:--:--"),
        model_status=model_status_str,
        data_source_badge=data_source_badge,
    )

    if st.session_state.operating_mode == "LIVE HARDWARE MODE" and not reader.is_connected and active_section != "Device Connection":
        st.markdown(
            f"""
            <div class="os-notice">
                <b>ESP32 Hardware Not Connected:</b> No serial stream detected on <code>{reader.port}</code>.
                Go to <b>Device Connection</b> in the sidebar to connect your ESP32, or switch to
                <b>Simulation Mode</b> for immediate demonstration.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if active_section == "Overview":
        render_overview_section(analysis, det_sys, conn_status, st.session_state.operating_mode)
    elif active_section == "Live Monitoring":
        render_live_monitoring_section(analysis, det_sys)
    elif active_section == "Fault Detection":
        render_fault_detection_section(analysis, det_sys)
    elif active_section == "Analytics & History":
        render_analytics_history_section(det_sys)
    elif active_section == "Device Connection":
        render_device_connection_section(reader, det_sys)
    elif active_section == "Settings":
        render_settings_section(det_sys, sim)

    # Controlled auto-refresh when enabled
    if st.session_state.auto_refresh:
        time.sleep(float(st.session_state.refresh_interval_s))
        st.rerun()


if __name__ == "__main__":
    main()
