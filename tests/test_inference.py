"""Automated tests for ML model loading, inference, invalid input handling, and sensor fusion."""

import numpy as np
import pytest
from src.config import (
    CLASS_HEALTHY,
    CLASS_ROTOR_IMBALANCE,
    CLASS_MECHANICAL_STALL,
    FEATURE_COLUMNS,
)
from src.data_generator import SensorSimulator
from src.fault_detection import FaultDetectionSystem
from models.inference import FaultModelInference


def test_model_loading_and_inference_across_all_three_states():
    det_sys = FaultDetectionSystem()
    assert det_sys.inference_engine.is_loaded is True

    sim = SensorSimulator(seed=42)

    for expected_state in (CLASS_HEALTHY, CLASS_ROTOR_IMBALANCE, CLASS_MECHANICAL_STALL):
        sim.set_operating_state(expected_state)
        if expected_state == CLASS_MECHANICAL_STALL:
            sim.reset_thermal_state(58.0)
        elif expected_state == CLASS_ROTOR_IMBALANCE:
            sim.reset_thermal_state(44.0)
        else:
            sim.reset_thermal_state(38.5)

        window = sim.generate_window(state=expected_state)
        result = det_sys.analyze_window(window, operating_mode="SIMULATION")

        assert result["fusion_result"]["final_fault_class"] == expected_state
        assert 0.0 <= result["ml_result"]["confidence_score"] <= 1.0
        assert len(result["fusion_result"]["explanations"]) >= 1


def test_inference_invalid_input_handling():
    engine = FaultModelInference()

    # Missing feature keys in dict should raise ValueError
    with pytest.raises(ValueError):
        engine.predict({"vib_rms": 0.1})

    # Wrong vector length should raise ValueError
    with pytest.raises(ValueError):
        engine.predict(np.zeros(5))

    # NaN/Inf values in valid-length vector should be sanitized without crashing
    vec_with_nan = np.zeros(len(FEATURE_COLUMNS))
    vec_with_nan[0] = np.nan
    vec_with_nan[1] = np.inf
    res = engine.predict(vec_with_nan)
    assert res["predicted_class"] in (CLASS_HEALTHY, CLASS_ROTOR_IMBALANCE, CLASS_MECHANICAL_STALL)

