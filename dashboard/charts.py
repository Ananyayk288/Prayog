"""Interactive Plotly chart builders for the EV Motor Health Monitoring Streamlit Dashboard."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Restrained technical dark-theme color palette
BG_COLOR = "#0E1117"
CARD_BG = "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"

COLOR_CYAN = "#38BDF8"
COLOR_AMBER = "#F59E0B"
COLOR_RED = "#EF4444"
COLOR_GREEN = "#10B981"
COLOR_PURPLE = "#A855F7"


def _apply_dark_engineering_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 260,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_COLOR), x=0.01, y=0.95),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, Segoe UI, Roboto, monospace", size=12, color=TEXT_COLOR),
        margin=dict(l=48, r=20, t=42, b=38),
        height=height,
        xaxis=dict(
            title=xaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT, size=11),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT, size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=MUTED_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def create_vibration_waveform_chart(window_data: Dict[str, Any]) -> go.Figure:
    """Create 3-axis raw vibration waveform chart (MPU6050)."""
    t = np.asarray(window_data.get("time_axis_s", []), dtype=float)
    vx = np.asarray(window_data.get("vibration_x", []), dtype=float)
    vy = np.asarray(window_data.get("vibration_y", []), dtype=float)
    vz = np.asarray(window_data.get("vibration_z", []), dtype=float) - 1.0  # Dynamic Z-axis (gravity removed)

    if t.size != vx.size:
        t = np.arange(vx.size) * 0.001

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=vx, mode="lines", name="Vib X (g)", line=dict(color=COLOR_CYAN, width=1.6)))
    if vy.size == t.size:
        fig.add_trace(go.Scatter(x=t, y=vy, mode="lines", name="Vib Y (g)", line=dict(color=COLOR_PURPLE, width=1.3)))
    if vz.size == t.size:
        fig.add_trace(go.Scatter(x=t, y=vz, mode="lines", name="Vib Z-dyn (g)", line=dict(color=COLOR_GREEN, width=1.2)))

    return _apply_dark_engineering_layout(
        fig,
        title="MPU6050 Triaxial Mechanical Vibration Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (g)",
        height=265,
    )


def create_current_waveform_chart(window_data: Dict[str, Any]) -> go.Figure:
    """Create ACS712 motor current waveform chart."""
    t = np.asarray(window_data.get("time_axis_s", []), dtype=float)
    cur = np.asarray(window_data.get("current", []), dtype=float)
    if t.size != cur.size:
        t = np.arange(cur.size) * 0.001

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cur,
            mode="lines",
            name="Motor Current (A)",
            line=dict(color=COLOR_AMBER, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.10)",
        )
    )
    fig.add_hline(
        y=5.5,
        line_dash="dash",
        line_color=COLOR_RED,
        annotation_text="Stall Threshold (5.5A)",
        annotation_font_color=COLOR_RED,
    )
    return _apply_dark_engineering_layout(
        fig,
        title="ACS712 Motor Phase/Supply Current Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
        height=265,
    )


def create_temperature_trend_chart(history_df: pd.DataFrame) -> go.Figure:
    """Create DS18B20 temperature trend chart across windows."""
    fig = go.Figure()
    if not history_df.empty and "temp_current_c" in history_df.columns:
        x_vals = history_df["time"] if "time" in history_df.columns else np.arange(len(history_df))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=history_df["temp_current_c"],
                mode="lines+markers",
                name="Temp (°C)",
                line=dict(color=COLOR_RED, width=2.0),
                marker=dict(size=5),
            )
        )
        if "temp_moving_avg_c" in history_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=history_df["temp_moving_avg_c"],
                    mode="lines",
                    name="Moving Avg (°C)",
                    line=dict(color=COLOR_CYAN, width=1.5, dash="dot"),
                )
            )

    fig.add_hline(
        y=55.0,
        line_dash="dot",
        line_color=COLOR_AMBER,
        annotation_text="Warning (55°C)",
        annotation_font_color=COLOR_AMBER,
    )
    fig.add_hline(
        y=70.0,
        line_dash="dash",
        line_color=COLOR_RED,
        annotation_text="Critical (70°C)",
        annotation_font_color=COLOR_RED,
    )
    return _apply_dark_engineering_layout(
        fig,
        title="DS18B20 Motor Thermal Trend",
        xaxis_title="Timestamp",
        yaxis_title="Temperature (°C)",
        height=265,
    )


def create_filtered_vibration_chart(
    time_axis_s: np.ndarray,
    raw_vib: np.ndarray,
    filtered_vib: np.ndarray,
) -> go.Figure:
    """Create comparison chart of raw vs 50 Hz – 1 kHz bandpass-filtered vibration."""
    t = np.asarray(time_axis_s, dtype=float)
    raw = np.asarray(raw_vib, dtype=float)
    filt = np.asarray(filtered_vib, dtype=float)
    if t.size != filt.size:
        t = np.arange(filt.size) * 0.001

    fig = go.Figure()
    if raw.size == t.size:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=raw,
                mode="lines",
                name="Raw Composite (DC removed)",
                line=dict(color=MUTED_TEXT, width=1.1),
                opacity=0.55,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=filt,
            mode="lines",
            name="Bandpass Filtered (50 Hz – 1 kHz)",
            line=dict(color=COLOR_CYAN, width=1.8),
        )
    )
    return _apply_dark_engineering_layout(
        fig,
        title="Bandpass Filtered Vibration Signal (50 Hz – 1 kHz Target Passband)",
        xaxis_title="Time (s)",
        yaxis_title="Filtered Acceleration (g)",
        height=275,
    )


def create_fft_spectrum_chart(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    dominant_freq_hz: float,
) -> go.Figure:
    """Create interactive single-sided FFT amplitude spectrum chart."""
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(magnitudes, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f,
            y=m,
            mode="lines",
            name="FFT Amplitude",
            line=dict(color=COLOR_PURPLE, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(168, 85, 247, 0.15)",
        )
    )
    if dominant_freq_hz > 0:
        fig.add_vline(
            x=dominant_freq_hz,
            line_dash="dash",
            line_color=COLOR_AMBER,
            annotation_text=f"Dominant: {dominant_freq_hz:.1f} Hz",
            annotation_font_color=COLOR_AMBER,
        )
    return _apply_dark_engineering_layout(
        fig,
        title="Vibration FFT Frequency Spectrum (Hann Windowed)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (g)",
        height=275,
    )


def create_class_scores_chart(class_scores: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart showing Random Forest ensemble tree-vote distribution."""
    labels = list(class_scores.keys())
    values = [float(class_scores[k]) * 100.0 for k in labels]
    colors = [
        COLOR_GREEN if k == "HEALTHY" else (COLOR_AMBER if k == "ROTOR_IMBALANCE" else COLOR_RED)
        for k in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig = _apply_dark_engineering_layout(
        fig,
        title="Random Forest Tree-Vote Agreement by Operating Class (Uncalibrated)",
        xaxis_title="Ensemble Tree Vote Share (%)",
        yaxis_title="Fault Class",
        height=220,
    )
    fig.update_xaxes(range=[0, 105])
    return fig


def create_confusion_matrix_chart(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """Create annotated confusion matrix heatmap for model evaluation summary."""
    z = np.asarray(cm, dtype=int)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            textfont={"size": 13},
            showscale=False,
        )
    )
    return _apply_dark_engineering_layout(
        fig,
        title="Held-Out Test Set Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=260,
    )

