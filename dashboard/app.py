"""Streamlit Real-Time Dashboard for EV Motor Health Monitoring & Predictive Maintenance."""

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
    render_kpi_overview,
    render_alert_center,
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

st.set_page_config(
    page_title="EV Motor Health Monitoring | Edge-AI Sensor Fusion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize persistent objects across Streamlit reruns."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator(
            sampling_rate_hz=SIGNAL_CONFIG.sampling_rate_hz,
            window_size=SIGNAL_CONFIG.window_size,
        )
    if "detection_system" not in st.session_state:
        st.session_state.detection_system = FaultDetectionSystem()
    if "serial_reader" not in st.session_state:
        st.session_state.serial_reader = ESP32SerialReader()
    if "latest_analysis" not in st.session_state:
        # Run one initial window in simulation mode so dashboard starts populated
        init_win = st.session_state.simulator.generate_window(state=CLASS_HEALTHY)
        st.session_state.latest_analysis = st.session_state.detection_system.analyze_window(
            init_win, operating_mode="SIMULATION MODE"
        )
    if "sim_fault_state" not in st.session_state:
        st.session_state.sim_fault_state = CLASS_HEALTHY


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
        analysis = det_sys.analyze_window(window, operating_mode="SIMULATION MODE (SYNTHETIC DEMO)")
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
    analysis = det_sys.analyze_window(live_window, operating_mode="LIVE HARDWARE MODE (ESP32)")
    st.session_state.latest_analysis = analysis
    return analysis


def main() -> None:
    inject_engineering_theme_css()
    init_session_state()

    det_sys: FaultDetectionSystem = st.session_state.detection_system
    reader: ESP32SerialReader = st.session_state.serial_reader
    sim: SensorSimulator = st.session_state.simulator

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.markdown("### ⚙️ System Control Panel")
        operating_mode = st.radio(
            "Operating Mode",
            options=["SIMULATION MODE", "LIVE HARDWARE MODE"],
            index=0,
            help="Switch between synthetic physics-inspired simulation and live ESP32 USB serial telemetry.",
        )

        st.divider()
        if operating_mode == "SIMULATION MODE":
            st.markdown("#### 🧪 Simulation Fault Injector")
            selected_state = st.selectbox(
                "Injected Motor Operating State",
                options=FAULT_CLASSES,
                index=FAULT_CLASSES.index(st.session_state.sim_fault_state),
                help="Select an operating state to generate real-time multi-sensor waveforms.",
            )
            if selected_state != st.session_state.sim_fault_state:
                st.session_state.sim_fault_state = selected_state
                sim.set_operating_state(selected_state)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                step_now = st.button("Step Window", use_container_width=True, type="primary")
            with col_s2:
                if st.button("Reset Thermal", use_container_width=True):
                    sim.reset_thermal_state(38.2)
        else:
            selected_state = st.session_state.sim_fault_state
            step_now = False
            st.markdown("#### 🔌 ESP32 Serial Configuration")
            ports = list_available_com_ports()
            port_options = [p["device"] for p in ports] if ports else [SERIAL_CONFIG.default_port, "COM4", "COM5"]
            selected_port = st.selectbox("Serial COM Port", options=port_options)
            selected_baud = st.selectbox(
                "Baud Rate",
                options=[9600, 57600, 115200, 230400],
                index=2,
            )
            reader.configure(port=selected_port, baud_rate=int(selected_baud))

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Connect ESP32", use_container_width=True, type="primary"):
                    reader.connect()
            with c_btn2:
                if st.button("Disconnect", use_container_width=True):
                    reader.disconnect()

            st.caption(
                f"Valid Packets: `{reader.total_valid_records}` | Malformed Ignored: `{reader.malformed_count}`"
            )

        st.divider()
        st.markdown("#### 🎛️ Signal Filter Settings")
        band_low = st.number_input("Bandpass Low Cutoff (Hz)", min_value=5.0, max_value=400.0, value=50.0, step=5.0)
        band_high = st.number_input(
            "Bandpass High Cutoff (Hz)", min_value=100.0, max_value=2000.0, value=1000.0, step=50.0
        )
        st.caption(
            f"Sampling Rate: `{SIGNAL_CONFIG.sampling_rate_hz:.0f} Hz` (Nyquist: `{0.5*SIGNAL_CONFIG.sampling_rate_hz:.0f} Hz`). "
            "Upper cutoff automatically clamps below Nyquist if needed."
        )

        st.divider()
        st.markdown("#### 🔄 Live Stream Refresh")
        auto_refresh = st.toggle("Continuous Auto-Refresh", value=True)
        refresh_interval_s = st.slider("Refresh Interval (seconds)", min_value=0.5, max_value=5.0, value=1.2, step=0.1)

        if st.button("Retrain ML Model Now", use_container_width=True):
            with st.spinner("Training Random Forest & Isolation Forest..."):
                from models.train_model import train_and_evaluate_models
                train_and_evaluate_models()
                det_sys.inference_engine.load_models()
            st.success("Models retrained and reloaded.")

    # =========================================================================
    # EXECUTE PIPELINE STEP IF REQUESTED OR AUTO-REFRESHING
    # =========================================================================
    if step_now or auto_refresh:
        acquire_and_process_step(
            operating_mode=operating_mode,
            sim_state=selected_state,
            band_low_hz=band_low,
            band_high_hz=band_high,
        )

    analysis: Dict[str, Any] = st.session_state.latest_analysis
    features = analysis["features"]
    diagnostics = analysis["diagnostics"]
    ml_res = analysis["ml_result"]
    fusion_res = analysis["fusion_result"]
    raw_window = analysis["raw_window"]

    # Determine connection status string
    if operating_mode == "SIMULATION MODE":
        conn_status = "SIMULATOR ACTIVE (NO HARDWARE REQUIRED)"
    else:
        conn_status = (
            f"ESP32 CONNECTED ({reader.port} @ {reader.baud_rate})"
            if reader.is_connected
            else f"DISCONNECTED ({reader.port})"
        )

    model_status_str = "RandomForest + IsolationForest READY" if det_sys.inference_engine.is_loaded else "NOT LOADED"
    data_source_badge = ml_res.get("data_source_trained_on", "SYNTHETIC_DEMONSTRATION")

    # =========================================================================
    # A. HEADER
    # =========================================================================
    render_header(
        connection_status=conn_status,
        operating_mode=operating_mode,
        last_update_str=analysis.get("time_str", "--:--:--"),
        model_status=model_status_str,
        data_source_badge=data_source_badge,
    )

    if "SYNTHETIC" in data_source_badge.upper():
        st.info(
            "**Demonstration Mode Notice:** The active ML model was trained on synthetic demonstration data "
            "representing physical motor fault signatures (`HEALTHY`, `ROTOR_IMBALANCE`, `MECHANICAL_STALL`). "
            "It has not yet been validated on physical EV motor testbench faults.",
            icon="ℹ️",
        )

    if operating_mode == "LIVE HARDWARE MODE" and not reader.is_connected:
        st.warning(
            f"**ESP32 Disconnected:** No active serial stream detected on `{reader.port}`. "
            f"{reader.last_error or 'Connect the ESP32 via USB and click Connect ESP32 in the sidebar, or switch to SIMULATION MODE for immediate demonstration.'}",
            icon="⚠️",
        )

    # =========================================================================
    # B. MOTOR HEALTH OVERVIEW (KPI CARDS)
    # =========================================================================
    st.markdown('<div class="section-heading">B. MOTOR HEALTH OVERVIEW</div>', unsafe_allow_html=True)
    render_kpi_overview(analysis)

    # =========================================================================
    # C. LIVE SENSOR MONITORING
    # =========================================================================
    st.markdown('<div class="section-heading">C. LIVE MULTI-MODAL SENSOR MONITORING</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    history_df = det_sys.get_history_dataframe()

    with c1:
        st.plotly_chart(
            create_vibration_waveform_chart(raw_window),
            use_container_width=True,
            key="chart_vib_raw",
        )
    with c2:
        st.plotly_chart(
            create_current_waveform_chart(raw_window),
            use_container_width=True,
            key="chart_cur_raw",
        )
    with c3:
        st.plotly_chart(
            create_temperature_trend_chart(history_df),
            use_container_width=True,
            key="chart_temp_trend",
        )

    # =========================================================================
    # D. SIGNAL ANALYSIS (BANDPASS FILTER & FFT SPECTRUM)
    # =========================================================================
    st.markdown('<div class="section-heading">D. DIGITAL SIGNAL PROCESSING & SPECTRAL ANALYSIS</div>', unsafe_allow_html=True)
    d_col1, d_col2 = st.columns([3, 3])
    with d_col1:
        st.plotly_chart(
            create_filtered_vibration_chart(
                time_axis_s=raw_window.get("time_axis_s", np.arange(256) * 0.001),
                raw_vib=diagnostics["raw_vibration"],
                filtered_vib=diagnostics["filtered_vibration"],
            ),
            use_container_width=True,
            key="chart_vib_filtered",
        )
    with d_col2:
        st.plotly_chart(
            create_fft_spectrum_chart(
                freqs=diagnostics["fft_freqs"],
                magnitudes=diagnostics["fft_magnitudes"],
                dominant_freq_hz=features["vib_dominant_freq_hz"],
            ),
            use_container_width=True,
            key="chart_vib_fft",
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dominant Frequency", f"{features['vib_dominant_freq_hz']:.1f} Hz")
    m2.metric("Band Energy (50Hz–1kHz)", f"{features['vib_band_energy_50_1000hz']:.3f} g²")
    m3.metric("Total Spectral Energy", f"{features['vib_spectral_energy']:.3f} g²")
    m4.metric("Filtered Vibration RMS", f"{features['vib_rms']:.4f} g")
    m5.metric("Vibration Peak-to-Peak", f"{features['vib_peak_to_peak']:.4f} g")

    # =========================================================================
    # E. AI FAULT DETECTION & MULTI-MODAL FUSION
    # =========================================================================
    st.markdown('<div class="section-heading">E. EDGE-AI FAULT DETECTION & MULTI-MODAL SENSOR FUSION</div>', unsafe_allow_html=True)
    e_col1, e_col2 = st.columns([3, 2])

    with e_col1:
        st.markdown("##### Explainable Multi-Sensor Fusion Diagnosis")
        st.write(f"**Fused Operating Classification:** `{fusion_res['final_fault_class']}`")
        st.write(f"**Supervised RF Prediction:** `{fusion_res['ml_predicted_class']}`")
        st.write(
            f"**Ensemble Tree-Vote Agreement:** `{fusion_res['confidence_score'] * 100:.1f}%` "
            f"*(Note: {fusion_res['confidence_note']})*"
        )
        st.write(
            f"**Isolation Forest Anomaly Status:** `{'ANOMALY DETECTED' if fusion_res['is_anomaly'] else 'NOMINAL INLIER'}` "
            f"(Score: `{fusion_res['anomaly_score']:.3f}`)"
        )

        st.markdown("**Diagnostic Evidence Across Modalities:**")
        for reason in fusion_res["explanations"]:
            st.markdown(f"- {reason}")

        sev = fusion_res["modality_severities"]
        s_c1, s_c2, s_c3 = st.columns(3)
        s_c1.info(f"**MPU6050 Vibration:** {sev['vibration']}")
        s_c2.info(f"**ACS712 Current:** {sev['current']}")
        s_c3.info(f"**DS18B20 Thermal:** {sev['temperature']}")

    with e_col2:
        st.plotly_chart(
            create_class_scores_chart(fusion_res["class_scores"]),
            use_container_width=True,
            key="chart_rf_votes",
        )
        with st.expander("View All 18 Extracted Window Features"):
            feat_rows = [
                {
                    "Feature": col,
                    "Value": round(float(features.get(col, 0.0)), 4),
                    "Unit": FEATURE_UNITS.get(col, ""),
                }
                for col in FEATURE_COLUMNS
            ]
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True, height=240)

    # =========================================================================
    # F. ALERT CENTER
    # =========================================================================
    st.markdown('<div class="section-heading">F. PREDICTIVE MAINTENANCE ALERT CENTER</div>', unsafe_allow_html=True)
    render_alert_center(det_sys.get_recent_alerts(limit=10), fusion_res)

    # =========================================================================
    # G. HISTORICAL ANALYSIS & MODEL EVALUATION SUMMARY
    # =========================================================================
    st.markdown('<div class="section-heading">G. HISTORICAL TELEMETRY & MODEL EVALUATION SUMMARY</div>', unsafe_allow_html=True)
    tab_hist, tab_eval = st.tabs(["📈 Sensor & Fault History + CSV Export", "🧠 ML Model Evaluation Metrics"])

    with tab_hist:
        if history_df.empty:
            st.caption("No historical windows recorded yet.")
        else:
            h_col1, h_col2 = st.columns([2, 1])
            with h_col1:
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
                    height=260,
                )
            with h_col2:
                st.markdown("**Fault Occurrence Breakdown (Session)**")
                fault_counts = history_df["predicted_fault"].value_counts().reset_index()
                fault_counts.columns = ["Fault Class", "Windows Count"]
                st.dataframe(fault_counts, use_container_width=True, hide_index=True)

                csv_bytes = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Session Telemetry CSV",
                    data=csv_bytes,
                    file_name=f"ev_motor_session_history_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab_eval:
        eval_metrics = det_sys.inference_engine.evaluation_metrics
        if not eval_metrics:
            st.caption("Model evaluation metrics file not found.")
        else:
            st.caption(f"**Disclaimer:** {eval_metrics.get('validation_disclaimer', '')}")
            ev1, ev2, ev3, ev4 = st.columns(4)
            ev1.metric("Test Accuracy", f"{eval_metrics.get('accuracy', 0.0) * 100:.2f}%")
            ev2.metric("Weighted Precision", f"{eval_metrics.get('precision_weighted', 0.0) * 100:.2f}%")
            ev3.metric("Weighted Recall", f"{eval_metrics.get('recall_weighted', 0.0) * 100:.2f}%")
            ev4.metric("Weighted F1-Score", f"{eval_metrics.get('f1_weighted', 0.0) * 100:.2f}%")

            cm_col, imp_col = st.columns(2)
            with cm_col:
                if "confusion_matrix" in eval_metrics and "labels" in eval_metrics:
                    st.plotly_chart(
                        create_confusion_matrix_chart(
                            eval_metrics["confusion_matrix"],
                            eval_metrics["labels"],
                        ),
                        use_container_width=True,
                        key="chart_conf_matrix",
                    )
            with imp_col:
                st.markdown("**Top Random Forest Feature Importances**")
                imp_dict = eval_metrics.get("feature_importances", {})
                imp_df = (
                    pd.DataFrame(
                        [{"Feature": k, "Importance": v} for k, v in imp_dict.items()]
                    )
                    .sort_values("Importance", ascending=False)
                    .head(10)
                )
                st.dataframe(imp_df, use_container_width=True, hide_index=True, height=240)

    # Controlled non-jarring auto-refresh loop
    if auto_refresh:
        time.sleep(float(refresh_interval_s))
        st.rerun()


if __name__ == "__main__":
    main()

