"""Reusable UI components and CSS styling for Prayog — Obsidian Sage Engineering Intelligence UI.

Core Three-Colour Identity:
- PRIMARY   (Obsidian Black): #171A17
- SECONDARY (Muted Sage):     #A8B58A
- TERTIARY  (Warm Ivory):     #F1EBDD
"""

from typing import Dict, Any, List
import streamlit as st

FRIENDLY_FAULT_NAMES: Dict[str, str] = {
    "HEALTHY": "Normal Operation",
    "ROTOR_IMBALANCE": "Rotor Imbalance",
    "MECHANICAL_STALL": "Mechanical Stall",
}

FRIENDLY_STATUS_LABELS: Dict[str, str] = {
    "GREEN": "Healthy",
    "YELLOW": "Warning",
    "RED": "Critical",
}


def format_fault_name(raw_name: str) -> str:
    """Convert internal class codes into clear, human-friendly labels."""
    return FRIENDLY_FAULT_NAMES.get(str(raw_name).upper(), str(raw_name).replace("_", " ").title())


def format_status_label(alert_level: str) -> str:
    """Convert internal alert codes into clear Healthy / Warning / Critical labels."""
    return FRIENDLY_STATUS_LABELS.get(str(alert_level).upper(), "Healthy")


def inject_engineering_theme_css() -> None:
    """Apply the Obsidian Sage (#171A17 / #A8B58A / #F1EBDD) visual identity."""
    st.markdown(
        """
        <style>
        /* Main application canvas — Obsidian Black */
        .stApp {
            background-color: #171A17;
            color: #F1EBDD;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #1C201C;
            border-right: 1px solid rgba(168, 181, 138, 0.16);
        }
        [data-testid="stHeader"] {
            background-color: rgba(23, 26, 23, 0.85);
        }

        /* Clean global top header */
        .os-topbar {
            background-color: #1E221E;
            border: 1px solid rgba(241, 235, 221, 0.10);
            border-left: 4px solid #A8B58A;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
        }
        .os-brand-title {
            font-size: 1.38rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: #F1EBDD;
            margin: 0;
        }
        .os-brand-sub {
            font-size: 0.86rem;
            color: rgba(241, 235, 221, 0.65);
            margin-top: 3px;
        }
        .os-pill-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        .os-pill {
            background-color: #171A17;
            border: 1px solid rgba(168, 181, 138, 0.28);
            border-radius: 999px;
            padding: 6px 14px;
            font-size: 0.78rem;
            color: #F1EBDD;
            letter-spacing: 0.02em;
        }
        .os-pill-sage {
            background-color: rgba(168, 181, 138, 0.16);
            border: 1px solid #A8B58A;
            color: #F1EBDD;
            font-weight: 600;
        }

        /* Hero Health Banner on Overview */
        .os-hero-card {
            background-color: #1E221E;
            border: 1px solid rgba(168, 181, 138, 0.25);
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 22px;
        }
        .os-hero-badge {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .badge-healthy {
            background-color: rgba(168, 181, 138, 0.20);
            border: 1px solid #A8B58A;
            color: #A8B58A;
        }
        .badge-warning {
            background-color: rgba(241, 235, 221, 0.12);
            border: 1px solid #F1EBDD;
            color: #F1EBDD;
        }
        .badge-critical {
            background-color: #F1EBDD;
            border: 1px solid #F1EBDD;
            color: #171A17;
        }
        .os-hero-headline {
            font-size: 1.55rem;
            font-weight: 600;
            color: #F1EBDD;
            margin: 0 0 6px 0;
        }
        .os-hero-desc {
            font-size: 0.94rem;
            color: rgba(241, 235, 221, 0.78);
            line-height: 1.5;
            margin: 0;
        }

        /* Metric Cards */
        .os-card {
            background-color: #1E221E;
            border: 1px solid rgba(241, 235, 221, 0.09);
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            transition: border-color 0.2s ease, transform 0.15s ease;
        }
        .os-card:hover {
            border-color: rgba(168, 181, 138, 0.45);
        }
        .os-card-accent {
            border-left: 3px solid #A8B58A;
        }
        .os-card-alert {
            border-left: 3px solid #F1EBDD;
            background-color: rgba(241, 235, 221, 0.04);
        }
        .os-card-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: rgba(241, 235, 221, 0.62);
            margin-bottom: 8px;
        }
        .os-card-value {
            font-size: 1.45rem;
            font-weight: 600;
            color: #F1EBDD;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            margin-bottom: 6px;
        }
        .os-card-sub {
            font-size: 0.80rem;
            color: #A8B58A;
        }

        /* Section Titles */
        .os-section-title {
            font-size: 1.08rem;
            font-weight: 600;
            color: #F1EBDD;
            border-bottom: 1px solid rgba(241, 235, 221, 0.10);
            padding-bottom: 8px;
            margin-top: 18px;
            margin-bottom: 16px;
            letter-spacing: 0.02em;
        }

        /* Notice & Event Cards */
        .os-notice {
            background-color: #1E221E;
            border: 1px solid rgba(168, 181, 138, 0.28);
            border-left: 4px solid #A8B58A;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 14px;
            font-size: 0.88rem;
            color: #F1EBDD;
        }
        .os-event-row {
            background-color: #1E221E;
            border: 1px solid rgba(241, 235, 221, 0.09);
            border-left: 3px solid #A8B58A;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 0.86rem;
        }

        /* Streamlit native widget overrides for Obsidian Sage */
        .stButton > button {
            background-color: #1E221E;
            color: #F1EBDD;
            border: 1px solid rgba(168, 181, 138, 0.40);
            border-radius: 8px;
            font-weight: 500;
        }
        .stButton > button:hover {
            border-color: #A8B58A;
            color: #A8B58A;
            background-color: rgba(168, 181, 138, 0.10);
        }
        .stButton > button[kind="primary"] {
            background-color: #A8B58A;
            color: #171A17;
            border: 1px solid #A8B58A;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #BAC69E;
            color: #171A17;
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
    data_source_badge: str = "",
) -> None:
    """Render the clean top status bar with clear mode and connection indicators."""
    mode_short = "Simulation Mode" if "SIMULATION" in operating_mode.upper() else "Live Hardware Mode"
    st.markdown(
        f"""
        <div class="os-topbar">
            <div>
                <div class="os-brand-title">PRAYOG · EV MOTOR INTELLIGENCE</div>
                <div class="os-brand-sub">Multi-Modal Predictive Maintenance &amp; Sensor Fusion System</div>
            </div>
            <div class="os-pill-group">
                <span class="os-pill os-pill-sage">Mode: {mode_short}</span>
                <span class="os-pill">Device: {connection_status}</span>
                <span class="os-pill">AI Engine: {model_status}</span>
                <span class="os-pill">Updated: {last_update_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_hero(analysis: Dict[str, Any], connection_status: str, operating_mode: str) -> None:
    """Render the top-level Motor Health Summary banner on the Overview screen."""
    fusion = analysis["fusion_result"]
    alert_level = fusion["alert_level"]
    status_word = format_status_label(alert_level)
    condition_name = format_fault_name(fusion["final_fault_class"])

    badge_cls = {
        "GREEN": "badge-healthy",
        "YELLOW": "badge-warning",
        "RED": "badge-critical",
    }.get(alert_level, "badge-healthy")

    reasons = fusion.get("explanations", [])
    primary_summary = reasons[0] if reasons else "All monitored motor channels are operating within normal limits."

    st.markdown(
        f"""
        <div class="os-hero-card">
            <span class="os-hero-badge {badge_cls}">System Status · {status_word}</span>
            <div class="os-hero-headline">Motor Condition: {condition_name}</div>
            <p class="os-hero-desc">{primary_summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_overview(
    analysis: Dict[str, Any],
    connection_status: str = "Ready",
    operating_mode: str = "Simulation Mode",
) -> None:
    """Render clean, readable Overview metric cards required on the first screen."""
    features = analysis["features"]
    fusion = analysis["fusion_result"]

    alert_level = fusion["alert_level"]
    status_word = format_status_label(alert_level)
    condition_name = format_fault_name(fusion["final_fault_class"])
    conf_pct = fusion["confidence_score"] * 100.0

    vib_rms = features.get("vib_rms", 0.0)
    vib_p2p = features.get("vib_peak_to_peak", 0.0)
    cur_rms = features.get("current_rms", 0.0)
    cur_peak = features.get("current_peak", 0.0)
    temp_c = features.get("temp_current_c", 0.0)
    temp_rate = features.get("temp_rise_rate_c_per_s", 0.0)

    sev = fusion.get("modality_severities", {})
    vib_state = "Normal" if sev.get("vibration") == "NORMAL" else "Elevated"
    cur_state = "Normal" if sev.get("current") == "NORMAL" else "Elevated"
    temp_state = "Stable" if sev.get("temperature") == "NORMAL" else "Rising"

    card_accent = "os-card-accent" if alert_level == "GREEN" else "os-card-alert"
    mode_short = "Simulation" if "SIMULATION" in operating_mode.upper() else "Live ESP32"
    esp_short = "Connected" if ("CONNECTED" in connection_status.upper() and "DISCONNECTED" not in connection_status.upper()) else (
        "Standby (Sim)" if "SIMULAT" in connection_status.upper() else "Disconnected"
    )

    row1 = st.columns(4)
    row1_cards = [
        ("Overall Motor Health", status_word, f"Condition: {condition_name}", card_accent),
        ("Vibration Level (MPU6050)", f"{vib_rms:.3f} g", f"{vib_state} · Peak-to-Peak {vib_p2p:.2f} g", "os-card-accent"),
        ("Motor Current (ACS712)", f"{cur_rms:.2f} A", f"{cur_state} · Peak {cur_peak:.2f} A", "os-card-accent"),
        ("Motor Temperature (DS18B20)", f"{temp_c:.1f} °C", f"{temp_state} · Rate {temp_rate:+.2f} °C/s", "os-card-accent"),
    ]
    for col, (label, val, sub, cls) in zip(row1, row1_cards):
        with col:
            st.markdown(
                f"""
                <div class="os-card {cls}">
                    <div class="os-card-label">{label}</div>
                    <div class="os-card-value">{val}</div>
                    <div class="os-card-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    row2 = st.columns(4)
    row2_cards = [
        ("Latest AI Prediction", condition_name, f"Model Agreement: {conf_pct:.1f}%", card_accent),
        ("Current Operating Status", status_word, f"Anomaly Check: {'Flagged' if fusion['is_anomaly'] else 'Normal'}", ""),
        ("ESP32 Connection Status", esp_short, connection_status, ""),
        ("Current Operating Mode", mode_short, "Switch anytime in sidebar", ""),
    ]
    for col, (label, val, sub, cls) in zip(row2, row2_cards):
        with col:
            st.markdown(
                f"""
                <div class="os-card {cls}">
                    <div class="os-card-label">{label}</div>
                    <div class="os-card-value" style="font-size:1.22rem;">{val}</div>
                    <div class="os-card-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_alert_center(alerts: List[Dict[str, Any]], current_fusion: Dict[str, Any]) -> None:
    """Render status alerts and recent events using clear text labels alongside Obsidian Sage tones."""
    level = current_fusion["alert_level"]
    status_word = format_status_label(level)
    cond_name = format_fault_name(current_fusion["final_fault_class"])

    st.markdown(
        f"""
        <div class="os-notice">
            <div style="font-weight:600; color:#A8B58A; text-transform:uppercase; font-size:0.78rem; letter-spacing:0.06em;">
                Active Diagnostic Status · {status_word}
            </div>
            <div style="margin-top:6px; font-size:0.95rem; color:#F1EBDD;">
                <b>{cond_name}:</b> {current_fusion['summary_explanation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not alerts:
        st.caption("No warning or critical fault events have been logged during this session.")
        return

    st.markdown("**Recent Diagnostic Events**")
    for ev in alerts[:8]:
        ev_status = format_status_label(ev["alert_level"])
        ev_cond = format_fault_name(ev["fault_class"])
        st.markdown(
            f"""
            <div class="os-event-row">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span>
                        <b style="color:#A8B58A;">[{ev_status}]</b>
                        <b style="color:#F1EBDD; margin-left:6px;">{ev_cond}</b>
                    </span>
                    <span style="color:rgba(241,235,221,0.6); font-family:monospace; font-size:0.80rem;">
                        {ev['time']} · Agreement {ev['confidence']*100:.1f}%
                    </span>
                </div>
                <div style="color:rgba(241,235,221,0.78); font-size:0.83rem; margin-top:4px;">
                    {ev['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
