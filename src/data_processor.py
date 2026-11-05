"""Sensor data validation, rolling-window buffering, and preprocessing pipeline.

Supports:
1. Streaming individual ESP32 JSON records into a synchronized rolling window.
2. Direct window processing from `SensorSimulator` or pre-batched ESP32 payloads.
3. Graceful degradation if a sensor channel (e.g. DS18B20 or ACS712) drops out or reports NaN.
"""

from collections import deque
import time
from typing import Dict, Any, List, Optional
import numpy as np

from src.config import SIGNAL_CONFIG
from src.feature_extractor import extract_unified_features


class DataProcessor:
    """Maintains synchronized sensor rolling windows and orchestrates feature extraction."""

    def __init__(
        self,
        window_size: int = SIGNAL_CONFIG.window_size,
        sampling_rate_hz: float = SIGNAL_CONFIG.sampling_rate_hz,
    ) -> None:
        self.window_size = int(window_size)
        self.sampling_rate_hz = float(sampling_rate_hz)

        self.vib_x_buf: deque = deque(maxlen=self.window_size)
        self.vib_y_buf: deque = deque(maxlen=self.window_size)
        self.vib_z_buf: deque = deque(maxlen=self.window_size)
        self.current_buf: deque = deque(maxlen=self.window_size)
        self.temp_history: deque = deque(maxlen=60)
        self.last_timestamp_ms: int = int(time.time() * 1000)
        self.sensor_availability: Dict[str, bool] = {
            "mpu6050": True,
            "acs712": True,
            "ds18b20": True,
        }

    def validate_and_parse_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming JSON sensor record and handle missing/disconnected sensors safely."""
        if not isinstance(record, dict):
            raise ValueError("Sensor record must be a dictionary.")

        ts = record.get("timestamp", int(time.time() * 1000))
        try:
            ts_int = int(float(ts))
        except (TypeError, ValueError):
            ts_int = int(time.time() * 1000)

        # Check MPU6050 vibration channels
        vx = record.get("vibration_x")
        vy = record.get("vibration_y")
        vz = record.get("vibration_z")
        mpu_ok = vx is not None or vy is not None or vz is not None
        self.sensor_availability["mpu6050"] = bool(mpu_ok)

        def _safe_float(val: Any, fallback: float) -> float:
            if val is None:
                return fallback
            try:
                f = float(val)
                return f if np.isfinite(f) else fallback
            except (TypeError, ValueError):
                return fallback

        vx_f = _safe_float(vx, self.vib_x_buf[-1] if self.vib_x_buf else 0.0)
        vy_f = _safe_float(vy, self.vib_y_buf[-1] if self.vib_y_buf else 0.0)
        vz_f = _safe_float(vz, self.vib_z_buf[-1] if self.vib_z_buf else 1.0)

        # Check ACS712 current sensor
        cur_raw = record.get("current")
        self.sensor_availability["acs712"] = cur_raw is not None
        cur_f = _safe_float(cur_raw, self.current_buf[-1] if self.current_buf else 0.0)

        # Check DS18B20 temperature sensor (-127.0 and 85.0 reset code handling)
        temp_raw = record.get("temperature")
        temp_valid = True
        if temp_raw is None:
            temp_valid = False
        else:
            try:
                tf = float(temp_raw)
                if not np.isfinite(tf) or tf <= -55.0 or tf >= 150.0:
                    temp_valid = False
            except (TypeError, ValueError):
                temp_valid = False

        self.sensor_availability["ds18b20"] = temp_valid
        fallback_temp = self.temp_history[-1] if self.temp_history else 38.0
        temp_f = float(temp_raw) if temp_valid else fallback_temp

        return {
            "timestamp": ts_int,
            "vibration_x": vx_f,
            "vibration_y": vy_f,
            "vibration_z": vz_f,
            "current": cur_f,
            "temperature": temp_f,
            "sensor_availability": dict(self.sensor_availability),
            "edge_features": record.get("features") if isinstance(record.get("features"), dict) else None,
        }

    def ingest_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append a single validated record into rolling buffers and return current window snapshot."""
        clean = self.validate_and_parse_record(record)
        self.last_timestamp_ms = clean["timestamp"]

        # If record also includes array buffers from ESP32 burst packet, ingest them
        if isinstance(record.get("vibration_x_window"), list):
            for v in record["vibration_x_window"]:
                self.vib_x_buf.append(float(v))
        else:
            self.vib_x_buf.append(clean["vibration_x"])

        if isinstance(record.get("vibration_y_window"), list):
            for v in record["vibration_y_window"]:
                self.vib_y_buf.append(float(v))
        else:
            self.vib_y_buf.append(clean["vibration_y"])

        if isinstance(record.get("vibration_z_window"), list):
            for v in record["vibration_z_window"]:
                self.vib_z_buf.append(float(v))
        else:
            self.vib_z_buf.append(clean["vibration_z"])

        if isinstance(record.get("current_window"), list):
            for c in record["current_window"]:
                self.current_buf.append(float(c))
        else:
            self.current_buf.append(clean["current"])

        self.temp_history.append(clean["temperature"])
        return clean

    def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a full sensor window and return extracted features, vectors, and waveforms."""
        features_dict, feature_vector, diagnostics = extract_unified_features(window_data)
        return {
            "timestamp_ms": window_data.get("timestamp_ms", int(time.time() * 1000)),
            "features": features_dict,
            "feature_vector": feature_vector,
            "diagnostics": diagnostics,
            "sensor_availability": dict(self.sensor_availability),
            "raw_window": window_data,
        }

    def process_rolling_buffer(self) -> Dict[str, Any]:
        """Build a window from the internal rolling buffers (used in live serial streaming)."""
        n = max(len(self.vib_x_buf), 1)
        dt = 1.0 / max(self.sampling_rate_hz, 1.0)
        window_data = {
            "timestamp_ms": self.last_timestamp_ms,
            "sampling_rate_hz": self.sampling_rate_hz,
            "time_axis_s": np.arange(n) * dt,
            "vibration_x": np.array(self.vib_x_buf if self.vib_x_buf else [0.0], dtype=float),
            "vibration_y": np.array(self.vib_y_buf if self.vib_y_buf else [0.0], dtype=float),
            "vibration_z": np.array(self.vib_z_buf if self.vib_z_buf else [1.0], dtype=float),
            "current": np.array(self.current_buf if self.current_buf else [0.0], dtype=float),
            "temperature": float(self.temp_history[-1]) if self.temp_history else 38.0,
            "temp_history": list(self.temp_history) if self.temp_history else [38.0],
            "window_duration_s": max(n * dt, 0.25),
        }
        return self.process_window(window_data)

