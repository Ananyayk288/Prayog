"""Realistic multi-modal sensor data simulator for EV Motor Fault Detection.

Generates synchronized time-series windows and individual sample records for:
1. HEALTHY: Nominal motor rotation (~80 Hz fundamental), low vibration, ~2.2A steady current, ~38-41°C stable temperature.
2. ROTOR_IMBALANCE: Elevated 1X/2X/3X rotational harmonics (e.g., 120 Hz & 240 Hz in-band components), high vibration RMS & peak-to-peak, moderate periodic current ripple (~2.8-3.4A), slight temperature elevation (~44-48°C).
3. MECHANICAL_STALL: Locked/stalled rotor or heavy overload with high current surge (~6.5-8.5A), high low-frequency chattering/broadband structural rattle, and rapid thermal rise (>0.4 °C/s toward 60-78°C).
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from src.config import (
    CLASS_HEALTHY,
    CLASS_ROTOR_IMBALANCE,
    CLASS_MECHANICAL_STALL,
    FAULT_CLASSES,
    SIGNAL_CONFIG,
)


class SensorSimulator:
    """Stateful multi-modal sensor simulator producing realistic waveforms and JSON records."""

    def __init__(
        self,
        sampling_rate_hz: float = SIGNAL_CONFIG.sampling_rate_hz,
        window_size: int = SIGNAL_CONFIG.window_size,
        seed: Optional[int] = None,
    ) -> None:
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.window_size = int(window_size)
        self.rng = np.random.default_rng(seed)
        self.current_state: str = CLASS_HEALTHY
        self.elapsed_time_s: float = 0.0
        self.current_temp_c: float = 38.2
        self.prev_temp_c: float = 38.2
        self.temp_history: List[float] = [38.2]

    def set_operating_state(self, state: str) -> None:
        """Switch the simulated motor operating state."""
        normalized = state.strip().upper()
        if normalized not in FAULT_CLASSES:
            raise ValueError(f"Unsupported operating state '{state}'. Expected one of {FAULT_CLASSES}")
        self.current_state = normalized

    def reset_thermal_state(self, initial_temp_c: float = 38.2) -> None:
        """Reset thermal memory to nominal ambient/operating equilibrium."""
        self.current_temp_c = float(initial_temp_c)
        self.prev_temp_c = float(initial_temp_c)
        self.temp_history = [self.current_temp_c]

    def generate_window(
        self,
        state: Optional[str] = None,
        add_variation: bool = True,
    ) -> Dict[str, Any]:
        """Generate a synchronized multi-channel sensor window for the active state.

        Returns a dictionary containing:
            - timestamp_ms: int
            - state: str
            - sampling_rate_hz: float
            - time_axis_s: np.ndarray
            - vibration_x: np.ndarray (in g)
            - vibration_y: np.ndarray (in g)
            - vibration_z: np.ndarray (in g)
            - current: np.ndarray (in Amperes)
            - temperature: float (current DS18B20 reading in °C)
            - temperature_series: np.ndarray (per-sample interpolated temperature in °C)
            - latest_record: dict matching ESP32 JSON format
        """
        active_state = (state or self.current_state).strip().upper()
        if active_state not in FAULT_CLASSES:
            raise ValueError(f"Invalid state: {active_state}")

        dt = 1.0 / self.sampling_rate_hz
        window_duration_s = self.window_size * dt
        t = self.elapsed_time_s + np.arange(self.window_size) * dt
        self.elapsed_time_s += window_duration_s

        # Jitter factor so consecutive windows have natural physical variation
        jitter = self.rng.uniform(0.90, 1.10) if add_variation else 1.0
        phase_x = self.rng.uniform(0, 2 * np.pi)
        phase_y = phase_x + np.pi / 2.0

        if active_state == CLASS_HEALTHY:
            # Nominal smooth motor rotation: small 85 Hz fundamental + minor 170 Hz harmonic + low noise
            f_rot = 85.0 + self.rng.normal(0.0, 1.2)
            vib_x = (
                0.09 * jitter * np.sin(2 * np.pi * f_rot * t + phase_x)
                + 0.03 * np.sin(2 * np.pi * (2 * f_rot) * t)
                + self.rng.normal(0.0, 0.035, size=self.window_size)
            )
            vib_y = (
                0.08 * jitter * np.sin(2 * np.pi * f_rot * t + phase_y)
                + 0.025 * np.cos(2 * np.pi * (2 * f_rot) * t)
                + self.rng.normal(0.0, 0.035, size=self.window_size)
            )
            # Z-axis includes 1.0g static gravity + small axial vibration
            vib_z = (
                1.0
                + 0.05 * jitter * np.sin(2 * np.pi * f_rot * t)
                + self.rng.normal(0.0, 0.03, size=self.window_size)
            )

            # Normal steady EV motor current under nominal load (~2.15 A with minor ripple)
            base_current = 2.15 * jitter
            current = (
                base_current
                + 0.10 * np.sin(2 * np.pi * 50.0 * t)
                + self.rng.normal(0.0, 0.06, size=self.window_size)
            )

            # Thermal behaviour: relaxes gently toward ~39.0°C equilibrium
            target_temp = 39.0 + self.rng.normal(0.0, 0.3)
            temp_step = (target_temp - self.current_temp_c) * 0.18 + self.rng.normal(0.0, 0.04)

        elif active_state == CLASS_ROTOR_IMBALANCE:
            # Centrifugal mass imbalance: strong 1X (135 Hz) and 2X (270 Hz) + 3X (405 Hz) radial vibration
            f_imb = 135.0 + self.rng.normal(0.0, 3.0)
            vib_x = (
                0.68 * jitter * np.sin(2 * np.pi * f_imb * t + phase_x)
                + 0.34 * jitter * np.sin(2 * np.pi * (2 * f_imb) * t + 0.4)
                + 0.16 * np.sin(2 * np.pi * (3 * f_imb) * t)
                + self.rng.normal(0.0, 0.08, size=self.window_size)
            )
            vib_y = (
                0.64 * jitter * np.sin(2 * np.pi * f_imb * t + phase_y)
                + 0.30 * jitter * np.cos(2 * np.pi * (2 * f_imb) * t)
                + 0.14 * np.cos(2 * np.pi * (3 * f_imb) * t)
                + self.rng.normal(0.0, 0.08, size=self.window_size)
            )
            vib_z = (
                1.0
                + 0.28 * jitter * np.sin(2 * np.pi * f_imb * t)
                + 0.15 * np.sin(2 * np.pi * (2 * f_imb) * t)
                + self.rng.normal(0.0, 0.06, size=self.window_size)
            )

            # Eccentric load causes moderate current modulation at rotational frequency
            base_current = 2.95 * jitter
            current = (
                base_current
                + 0.42 * np.sin(2 * np.pi * f_imb * t)
                + 0.18 * np.sin(2 * np.pi * 30.0 * t)
                + self.rng.normal(0.0, 0.11, size=self.window_size)
            )

            # Mild friction/iron loss heating toward ~45.5°C
            target_temp = 45.5 + self.rng.normal(0.0, 0.5)
            temp_step = (target_temp - self.current_temp_c) * 0.16 + self.rng.normal(0.03, 0.05)

        else:  # CLASS_MECHANICAL_STALL
            # Mechanical stall / jammed rotor: high current draw, electromagnetic cogging/hum (65 Hz + 195 Hz),
            # plus impulsive mechanical chattering bursts
            f_cog = 65.0 + self.rng.normal(0.0, 2.5)
            impulses = self.rng.choice([0.0, 1.0], size=self.window_size, p=[0.92, 0.08]) * self.rng.normal(
                0.0, 0.55, size=self.window_size
            )
            vib_x = (
                0.28 * jitter * np.sin(2 * np.pi * f_cog * t + phase_x)
                + 0.22 * np.sin(2 * np.pi * 195.0 * t)
                + impulses
                + self.rng.normal(0.0, 0.12, size=self.window_size)
            )
            vib_y = (
                0.25 * jitter * np.cos(2 * np.pi * f_cog * t + phase_y)
                + 0.20 * np.cos(2 * np.pi * 195.0 * t)
                + impulses * 0.85
                + self.rng.normal(0.0, 0.12, size=self.window_size)
            )
            vib_z = (
                1.0
                + 0.22 * np.sin(2 * np.pi * f_cog * t)
                + impulses * 0.7
                + self.rng.normal(0.0, 0.10, size=self.window_size)
            )

            # Stall current spike: high DC draw (6.8 - 8.4 A) with strong current fluctuation/ripple
            base_current = 7.25 * jitter
            current = (
                base_current
                + 1.15 * np.sin(2 * np.pi * f_cog * t)
                + 0.65 * np.cos(2 * np.pi * 18.0 * t)
                + self.rng.normal(0.0, 0.32, size=self.window_size)
            )

            # Rapid I^2*R winding heating during stall
            temp_step = self.rng.uniform(0.32, 0.65)
            if self.current_temp_c > 78.0:
                temp_step = self.rng.uniform(-0.05, 0.10)  # saturate near thermal limit

        self.prev_temp_c = self.current_temp_c
        self.current_temp_c = float(np.clip(self.current_temp_c + temp_step, 22.0, 95.0))
        self.temp_history.append(self.current_temp_c)
        if len(self.temp_history) > 100:
            self.temp_history = self.temp_history[-100:]

        temp_series = np.linspace(self.prev_temp_c, self.current_temp_c, self.window_size)
        timestamp_ms = int(time.time() * 1000)

        latest_record = {
            "timestamp": timestamp_ms,
            "vibration_x": round(float(vib_x[-1]), 4),
            "vibration_y": round(float(vib_y[-1]), 4),
            "vibration_z": round(float(vib_z[-1]), 4),
            "current": round(float(current[-1]), 4),
            "temperature": round(float(self.current_temp_c), 2),
            "operating_state": active_state,
        }

        return {
            "timestamp_ms": timestamp_ms,
            "state": active_state,
            "sampling_rate_hz": self.sampling_rate_hz,
            "time_axis_s": t,
            "vibration_x": vib_x,
            "vibration_y": vib_y,
            "vibration_z": vib_z,
            "current": current,
            "temperature": self.current_temp_c,
            "temperature_series": temp_series,
            "temp_history": list(self.temp_history),
            "window_duration_s": window_duration_s,
            "latest_record": latest_record,
        }

