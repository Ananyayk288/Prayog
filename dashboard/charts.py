"""Interactive Plotly chart builders for Prayog — Obsidian Sage Engineering Intelligence UI.

Strictly uses the 3-colour palette and its shades/opacity variations:
- PRIMARY   (Obsidian Black): #171A17  (Surfaces: #1E221E, Grid: rgba(241, 235, 221, 0.08))
- SECONDARY (Muted Sage):     #A8B58A  (Shades: #87946B, #C2CEA7)
- TERTIARY  (Warm Ivory):     #F1EBDD  (Shades: #D6CEBF, #9E9A90)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Core Three-Colour Palette & Tonal Variations
OBSIDIAN_BG = "#171A17"
OBSIDIAN_SURFACE = "#1E221E"
OBSIDIAN_GRID = "rgba(241, 235, 221, 0.08)"

SAGE_MAIN = "#A8B58A"
SAGE_LIGHT = "#C4D0AA"
SAGE_DEEP = "#7B8862"
SAGE_FILL = "rgba(168, 181, 138, 0.14)"

IVORY_MAIN = "#F1EBDD"
IVORY_SOFT = "#CFC8B8"
IVORY_MUTED = "#9B968B"
IVORY_FILL = "rgba(241, 235, 221, 0.10)"


def _apply_obsidian_sage_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 280,
) -> go.Figure:
    """Apply clean, calm Obsidian Sage styling to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, Segoe UI, sans-serif", size=14, color=IVORY_MAIN),
            x=0.01,
            y=0.95,
        ),
        paper_bgcolor=OBSIDIAN_SURFACE,
        plot_bgcolor=OBSIDIAN_SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=IVORY_MAIN),
        margin=dict(l=50, r=22, t=46, b=40),
        height=height,
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=IVORY_MUTED, size=11)),
            gridcolor=OBSIDIAN_GRID,
            zerolinecolor=OBSIDIAN_GRID,
            tickfont=dict(color=IVORY_MUTED, size=11),
            showline=True,
            linecolor=OBSIDIAN_GRID,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=IVORY_MUTED, size=11)),
            gridcolor=OBSIDIAN_GRID,
            zerolinecolor=OBSIDIAN_GRID,
            tickfont=dict(color=IVORY_MUTED, size=11),
            showline=True,
            linecolor=OBSIDIAN_GRID,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=IVORY_SOFT),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=OBSIDIAN_BG,
            bordercolor=SAGE_MAIN,
            font=dict(color=IVORY_MAIN, size=12),
        ),
    )
    return fig


def create_vibration_waveform_chart(window_data: Dict[str, Any]) -> go.Figure:
    """Create 3-axis mechanical vibration waveform chart (MPU6050)."""
    t = np.asarray(window_data.get("time_axis_s", []), dtype=float)
    vx = np.asarray(window_data.get("vibration_x", []), dtype=float)
    vy = np.asarray(window_data.get("vibration_y", []), dtype=float)
    vz = np.asarray(window_data.get("vibration_z", []), dtype=float) - 1.0  # Gravity removed

    if t.size != vx.size:
        t = np.arange(vx.size) * 0.001

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=vx,
            mode="lines",
            name="Axis X (g)",
            line=dict(color=SAGE_MAIN, width=1.8),
        )
    )
    if vy.size == t.size:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vy,
                mode="lines",
                name="Axis Y (g)",
                line=dict(color=IVORY_MAIN, width=1.3),
            )
        )
    if vz.size == t.size:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vz,
                mode="lines",
                name="Axis Z Dynamic (g)",
                line=dict(color=SAGE_DEEP, width=1.2, dash="dot"),
            )
        )

    return _apply_obsidian_sage_layout(
        fig,
        title="Mechanical Vibration Waveform (MPU6050)",
        xaxis_title="Time (seconds)",
        yaxis_title="Acceleration (g)",
        height=280,
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
            line=dict(color=IVORY_MAIN, width=1.9),
            fill="tozeroy",
            fillcolor=IVORY_FILL,
        )
    )
    fig.add_hline(
        y=5.5,
        line_dash="dash",
        line_color=SAGE_MAIN,
        annotation_text="Stall Limit (5.5 A)",
        annotation_font_color=SAGE_LIGHT,
    )
    return _apply_obsidian_sage_layout(
        fig,
        title="Motor Current Draw (ACS712)",
        xaxis_title="Time (seconds)",
        yaxis_title="Current (Amperes)",
        height=280,
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
                name="Temperature (°C)",
                line=dict(color=SAGE_MAIN, width=2.0),
                marker=dict(size=5, color=IVORY_MAIN),
            )
        )
        if "temp_moving_avg_c" in history_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=history_df["temp_moving_avg_c"],
                    mode="lines",
                    name="Moving Average (°C)",
                    line=dict(color=IVORY_MUTED, width=1.5, dash="dot"),
                )
            )

    fig.add_hline(
        y=55.0,
        line_dash="dot",
        line_color=IVORY_SOFT,
        annotation_text="Warning (55 °C)",
        annotation_font_color=IVORY_SOFT,
    )
    fig.add_hline(
        y=70.0,
        line_dash="dash",
        line_color=SAGE_MAIN,
        annotation_text="Critical (70 °C)",
        annotation_font_color=SAGE_LIGHT,
    )
    return _apply_obsidian_sage_layout(
        fig,
        title="Motor Temperature Trend (DS18B20)",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=280,
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
                name="Raw Unfiltered (DC Removed)",
                line=dict(color=IVORY_MUTED, width=1.1),
                opacity=0.45,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=filt,
            mode="lines",
            name="Bandpass Filtered (50 Hz – 1 kHz)",
            line=dict(color=SAGE_MAIN, width=1.8),
        )
    )
    return _apply_obsidian_sage_layout(
        fig,
        title="Filtered Vibration Signal (50 Hz – 1 kHz Bandpass)",
        xaxis_title="Time (seconds)",
        yaxis_title="Filtered Acceleration (g)",
        height=285,
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
            name="Spectral Amplitude",
            line=dict(color=SAGE_MAIN, width=1.8),
            fill="tozeroy",
            fillcolor=SAGE_FILL,
        )
    )
    if dominant_freq_hz > 0:
        fig.add_vline(
            x=dominant_freq_hz,
            line_dash="dash",
            line_color=IVORY_MAIN,
            annotation_text=f"Peak: {dominant_freq_hz:.1f} Hz",
            annotation_font_color=IVORY_MAIN,
        )
    return _apply_obsidian_sage_layout(
        fig,
        title="Vibration Frequency Spectrum (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (g)",
        height=285,
    )


def create_class_scores_chart(class_scores: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart showing Random Forest ensemble agreement."""
    friendly_names = {
        "HEALTHY": "Healthy",
        "ROTOR_IMBALANCE": "Rotor Imbalance",
        "MECHANICAL_STALL": "Mechanical Stall",
    }
    raw_keys = list(class_scores.keys())
    labels = [friendly_names.get(k, k) for k in raw_keys]
    values = [float(class_scores[k]) * 100.0 for k in raw_keys]
    colors = [
        SAGE_MAIN if k == "HEALTHY" else (IVORY_SOFT if k == "ROTOR_IMBALANCE" else IVORY_MAIN)
        for k in raw_keys
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=OBSIDIAN_BG, width=1),
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(color=OBSIDIAN_BG, weight="bold"),
        )
    )
    fig = _apply_obsidian_sage_layout(
        fig,
        title="AI Model Agreement by Condition",
        xaxis_title="Ensemble Tree Agreement (%)",
        yaxis_title="",
        height=230,
    )
    fig.update_xaxes(range=[0, 105])
    return fig


def create_confusion_matrix_chart(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """Create annotated confusion matrix heatmap using Obsidian-Sage tonal scale."""
    z = np.asarray(cm, dtype=int)
    friendly_labels = [lbl.replace("_", " ").title() for lbl in labels]
    custom_scale = [
        [0.0, "#1E221E"],
        [0.5, "#606B4B"],
        [1.0, "#A8B58A"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=friendly_labels,
            y=friendly_labels,
            colorscale=custom_scale,
            text=z,
            texttemplate="%{text}",
            textfont={"size": 13, "color": IVORY_MAIN},
            showscale=False,
        )
    )
    return _apply_obsidian_sage_layout(
        fig,
        title="Validation Confusion Matrix",
        xaxis_title="Predicted Condition",
        yaxis_title="Actual Condition",
        height=265,
    )
