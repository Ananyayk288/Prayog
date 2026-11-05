"""ESP32 USB Serial Communication Module for EV Motor Fault Detection.

Features:
- Automatic discovery of available COM ports via `pyserial`.
- Configurable port and baud rate (default 115200).
- Robust JSON line parsing and schema validation.
- Graceful handling of malformed JSON, partial lines, and USB cable disconnections.
- Automatic reconnection attempts without crashing the dashboard.
- Strictly separates LIVE HARDWARE MODE from SIMULATION MODE (never fabricates live readings).
"""

from collections import deque
import json
import time
from typing import Dict, Any, List, Optional

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

from src.config import SERIAL_CONFIG, RAW_DATA_DIR


def list_available_com_ports() -> List[Dict[str, str]]:
    """Return a list of detected serial COM ports on the host machine."""
    if not PYSERIAL_AVAILABLE:
        return []
    ports_info = []
    try:
        for port in serial.tools.list_ports.comports():
            ports_info.append(
                {
                    "device": port.device,
                    "description": port.description or "Serial Device",
                    "hwid": port.hwid or "",
                }
            )
    except Exception:
        pass
    return ports_info


def parse_serial_json_line(raw_line: str) -> Optional[Dict[str, Any]]:
    """Parse and validate a single JSON line emitted by the ESP32 over USB serial.

    Returns a normalized dict if valid, or None if the line is empty, non-JSON telemetry,
    or malformed.
    """
    if not raw_line or not isinstance(raw_line, str):
        return None

    stripped = raw_line.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    # Require at least one expected sensor key or timestamp
    expected_keys = {"vibration_x", "vibration_y", "vibration_z", "current", "temperature", "timestamp"}
    if not any(k in payload for k in expected_keys):
        return None

    return payload


class ESP32SerialReader:
    """Manages non-blocking serial connection to the ESP32 sensor node."""

    def __init__(
        self,
        port: str = SERIAL_CONFIG.default_port,
        baud_rate: int = SERIAL_CONFIG.default_baud_rate,
        timeout_s: float = SERIAL_CONFIG.timeout_s,
    ) -> None:
        self.port = port
        self.baud_rate = int(baud_rate)
        self.timeout_s = float(timeout_s)
        self.serial_conn: Optional[Any] = None
        self.is_connected: bool = False
        self.last_error: str = ""
        self.last_reconnect_attempt: float = 0.0
        self.received_records: deque = deque(maxlen=SERIAL_CONFIG.max_buffer_records)
        self.malformed_count: int = 0
        self.total_valid_records: int = 0

    def configure(self, port: str, baud_rate: int = 115200) -> None:
        """Update target serial port or baud rate, closing any previous connection if changed."""
        if port != self.port or int(baud_rate) != self.baud_rate:
            self.disconnect()
            self.port = port
            self.baud_rate = int(baud_rate)

    def connect(self) -> bool:
        """Attempt to open the configured COM port."""
        if not PYSERIAL_AVAILABLE:
            self.is_connected = False
            self.last_error = "pyserial package is not installed."
            return False

        if self.is_connected and self.serial_conn is not None:
            try:
                if self.serial_conn.is_open:
                    return True
            except Exception:
                pass

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout_s,
            )
            self.is_connected = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.serial_conn = None
            self.is_connected = False
            self.last_error = f"Unable to open {self.port} @ {self.baud_rate} baud: {exc}"
            return False

    def disconnect(self) -> None:
        """Safely close the active serial connection."""
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.is_connected = False

    def read_available_records(self, max_lines: int = 64) -> List[Dict[str, Any]]:
        """Read up to `max_lines` available JSON records from serial buffer with auto-reconnect."""
        if not self.is_connected or self.serial_conn is None:
            now = time.time()
            if now - self.last_reconnect_attempt >= SERIAL_CONFIG.reconnect_delay_s:
                self.last_reconnect_attempt = now
                if not self.connect():
                    return []
            else:
                return []

        records: List[Dict[str, Any]] = []
        try:
            lines_read = 0
            while (
                self.serial_conn is not None
                and self.serial_conn.in_waiting > 0
                and lines_read < max_lines
            ):
                raw_bytes = self.serial_conn.readline()
                lines_read += 1
                try:
                    line_str = raw_bytes.decode("utf-8", errors="replace")
                except Exception:
                    self.malformed_count += 1
                    continue

                parsed = parse_serial_json_line(line_str)
                if parsed is not None:
                    if "timestamp" not in parsed:
                        parsed["timestamp"] = int(time.time() * 1000)
                    self.received_records.append(parsed)
                    self.total_valid_records += 1
                    records.append(parsed)
                else:
                    if line_str.strip():
                        self.malformed_count += 1

        except Exception as exc:
            self.last_error = f"Serial connection lost on {self.port}: {exc}"
            self.disconnect()

        return records

    def export_received_to_csv(self, filename: str = "esp32_live_capture.csv") -> Optional[str]:
        """Save buffered live ESP32 records to `data/raw/` for offline analysis and model training."""
        if not self.received_records:
            return None
        import pandas as pd

        out_path = RAW_DATA_DIR / filename
        df = pd.DataFrame(list(self.received_records))
        df.to_csv(out_path, index=False)
        return str(out_path)

