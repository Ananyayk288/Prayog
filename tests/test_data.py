"""Automated tests for serial JSON parsing, missing sensor values, and simulation generation."""

import pytest
from src.config import FAULT_CLASSES, CLASS_HEALTHY, CLASS_ROTOR_IMBALANCE, CLASS_MECHANICAL_STALL
from src.data_generator import SensorSimulator
from src.data_processor import DataProcessor
from src.serial_reader import parse_serial_json_line


def test_serial_json_parser_handles_valid_and_malformed_inputs():
    valid_line = '{"timestamp": 123456, "vibration_x": 0.12, "vibration_y": 0.08, "vibration_z": 1.03, "current": 1.25, "temperature": 38.5}\r\n'
    parsed = parse_serial_json_line(valid_line)
    assert parsed is not None
    assert parsed["timestamp"] == 123456
    assert parsed["vibration_x"] == 0.12
    assert parsed["temperature"] == 38.5

    # Malformed / truncated / comment lines should return None without raising exceptions
    assert parse_serial_json_line("# INFO: ESP32 Booting...") is None
    assert parse_serial_json_line('{"timestamp": 123, "vibration_x": ') is None
    assert parse_serial_json_line("") is None
    assert parse_serial_json_line('{"unrelated_key": 42}') is None


def test_data_processor_handles_missing_and_disconnected_sensors():
    proc = DataProcessor(window_size=32)

    # Normal record first
    rec1 = {
        "timestamp": 1000,
        "vibration_x": 0.10,
        "vibration_y": 0.05,
        "vibration_z": 1.01,
        "current": 2.1,
        "temperature": 39.0,
    }
    clean1 = proc.ingest_record(rec1)
    assert clean1["sensor_availability"]["ds18b20"] is True

    # Simulate DS18B20 disconnection (-127.0 error code) and missing current sensor (None)
    rec2 = {
        "timestamp": 1010,
        "vibration_x": 0.12,
        "vibration_y": 0.06,
        "vibration_z": 1.02,
        "current": None,
        "temperature": -127.0,
    }
    clean2 = proc.ingest_record(rec2)
    assert clean2["sensor_availability"]["ds18b20"] is False
    assert clean2["sensor_availability"]["acs712"] is False
    # Should gracefully fall back to previous valid temperature (39.0) and current (2.1)
    assert clean2["temperature"] == 39.0
    assert clean2["current"] == 2.1

    # Rolling buffer processing should still succeed
    out = proc.process_rolling_buffer()
    assert "features" in out
    assert len(out["features"]) == 18


def test_simulator_generates_all_three_states_and_rejects_invalid():
    sim = SensorSimulator(seed=7)
    for state in FAULT_CLASSES:
        win = sim.generate_window(state=state)
        assert win["state"] == state
        assert win["vibration_x"].size == sim.window_size
        assert win["current"].size == sim.window_size
        assert "latest_record" in win

    with pytest.raises(ValueError):
        sim.set_operating_state("NON_EXISTENT_FAULT")

