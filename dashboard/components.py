"""Reusable UI components and CSS styling for the EV Motor Health Monitoring Dashboard."""

from typing import Dict, Any, List
import streamlit as st


def inject_engineering_theme_css() -> None:
    """Apply clean dark technical theme styling for cards, status pills, and typography."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #E6EDF3;
        }
        .eng-header-box {
            background: linear-gradient(90deg, #161B22 0%, #1F2937 100%);
            border: 1px solid #30363D;
            border-left: 4px solid #38BDF8;
            border-radius: 8px;
            padding: 16px 22px;
            margin-bottom: 16px;
        }
        .eng-title {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #F0F6FC;
            margin: 0;
        }
        .eng-subtitle {
            font-size: 0.92rem;
            color: #8B949E;
            margin-top: 2px;
            margin-bottom: 10px;
        }
        .meta-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 18px;
            font-size: 0.83rem;
            color: #C9D1D9;
            border-top: 1px solid #21262D;
            padding-top: 10px;
        }
        .kpi-card {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 14px 16px;
            height: 100%;
        }
        .kpi-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #8B949E;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F0F6FC;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        .kpi-sub {
            font-size: 0.76rem;
            color: #8B949E;
            margin-top: 4px;
        }
        .status-green {
            color: #10B981 !important;
            border-left: 4px solid #10B981;
        }
        .status-yellow {
            color: #F59E0B !important;
            border-left: 4px solid #F59E0B;
        }
        .status-red {
            color: #EF4444 !important;
            border-left: 4px solid #EF4444;
        }
        .alert-item {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.86rem;
        }
        .section-heading {
            font-size: 1.05rem;
            font-weight: 600;
            color: #E6EDF3;
            border-bottom: 1px solid #21262D;
            padding-bottom: 6px;
            margin-top: 14px;
            margin-bottom: 12px;
            letter-spacing: 0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    connection_status: str,
    operating_mode: str,
    last_update_str: str,
    model_status: str,
    data_source_badge: str,
) -> None:
    """Render Section A: Engineering Header with system telemetry pills."""
    conn_color = "#10B981" if "CONNECTED" in connection_status or "SIMULATION" in operating_mode else "#F59E0B"
    st.markdown(
        f"""
        <div class="eng-header-box">
            <div class="eng-title">EV MOTOR HEALTH MONITORING</div>
            <div class="eng-subtitle">Edge-AI Multi-Modal Predictive Maintenance System (MPU6050 · ACS712 · DS18B20)</div>
            <div class="meta-bar">
                <span><b>System Connection:</b> <span style="color:{conn_color};">{connection_status}</span></span>
                <span><b>Operating Mode:</b> <code>{operating_mode}</code></span>
                <span><b>Last Data Update:</b> <code>{last_update_str}</code></span>
                <span><b>Model Status:</b> <code>{model_status}</code></span>
                <span><b>Training Data Origin:</b> <code>{data_source_badge}</code></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_overview(analysis: Dict[str, Any]) -> None:
    """Render Section B: 6 Motor Health Overview KPI Cards."""
    features = analysis["features"]
    fusion = analysis["fusion_result"]

    alert_level = fusion["alert_level"]
    status_class = {
        "GREEN": "status-green",
        "YELLOW": "status-yellow",
        "RED": "status-red",
    }.get(alert_level, "status-green")

    health_label = fusion["health_status_label"]
    fault_class = fusion["final_fault_class"]
    conf_pct = fusion["confidence_score"] * 100.0
    vib_rms = features.get("vib_rms", 0.0)
    cur_rms = features.get("current_rms", 0.0)
    temp_c = features.get("temp_current_c", 0.0)
    temp_rate = features.get("temp_rise_rate_c_per_s", 0.0)

    cols = st.columns(6)
    cards = [
        ("Motor Health Status", health_label, f"Alert Level: {alert_level}", status_class),
        ("Vibration RMS", f"{vib_rms:.3f} g", f"P2P: {features.get('vib_peak_to_peak', 0.0):.3f} g", ""),
        ("Motor Current", f"{cur_rms:.2f} A", f"Peak: {features.get('current_peak', 0.0):.2f} A", ""),
        ("Motor Temperature", f"{temp_c:.1f} °C", f"Rate: {temp_rate:+.2f} °C/s", ""),
        ("Detected Fault", fault_class, f"Anomaly: {'YES' if fusion['is_anomaly'] else 'NO'}", status_class),
        ("Model Agreement", f"{conf_pct:.1f}%", "RF Ensemble Tree Votes", ""),
    ]

    for col, (label, val, sub, extra_cls) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {extra_cls}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_alert_center(alerts: List[Dict[str, Any]], current_fusion: Dict[str, Any]) -> None:
    """Render Section F: Alert Center with Green / Yellow / Red status indicators and history."""
    level = current_fusion["alert_level"]
    badge_map = {
        "GREEN": ("#10B981", "GREEN — HEALTHY OPERATING ENVELOPE"),
        "YELLOW": ("#F59E0B", "YELLOW — EARLY WARNING / ELEVATED SENSOR SIGNATURES"),
        "RED": ("#EF4444", "RED — CRITICAL MOTOR FAULT DETECTED"),
    }
    color, banner_title = badge_map.get(level, ("#10B981", "HEALTHY"))

    st.markdown(
        f"""
        <div class="alert-item" style="border-left: 5px solid {color};">
            <div style="font-weight:700; color:{color};">{banner_title}</div>
            <div style="margin-top:4px; color:#C9D1D9;">
                <b>Active Condition:</b> <code>{current_fusion['final_fault_class']}</code> —
                {current_fusion['summary_explanation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not alerts:
        st.caption("No warning or critical fault events recorded in this session yet.")
        return

    st.markdown("**Recent Timestamped Fault & Warning Events**")
    for ev in alerts[:8]:
        ev_color = "#EF4444" if ev["alert_level"] == "RED" else "#F59E0B"
        st.markdown(
            f"""
            <div class="alert-item" style="border-left: 4px solid {ev_color}; padding: 8px 12px;">
                <span style="color:#8B949E; font-family:monospace;">[{ev['time']}]</span>
                <b style="color:{ev_color}; margin-left:6px;">{ev['status']} ({ev['fault_class']})</b>
                <span style="color:#8B949E; margin-left:8px;">Tree Vote Agreement: {ev['confidence']*100:.1f}%</span>
                <div style="color:#C9D1D9; font-size:0.82rem; margin-top:3px;">{ev['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

