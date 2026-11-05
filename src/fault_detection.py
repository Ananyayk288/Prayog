"""End-to-end Fault Detection Pipeline orchestrator.

Ties together:
- DataProcessor (windowing, validation, signal filtering, feature extraction)
- FaultModelInference (Random Forest + Isolation Forest)
- MultiModalSensorFusion (cross-sensor verification & explainable alert generation)
"""

from collections import deque
import time
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import CLASS_HEALTHY
from src.data_processor import DataProcessor
from src.sensor_fusion import MultiModalSensorFusion
from models.inference import FaultModelInference


class FaultDetectionSystem:
    """High-level facade for processing windows/records and tracking historical fault events."""

    def __init__(
        self,
        processor: Optional[DataProcessor] = None,
        inference_engine: Optional[FaultModelInference] = None,
        fusion_engine: Optional[MultiModalSensorFusion] = None,
        max_history: int = 300,
    ) -> None:
        self.processor = processor or DataProcessor()
        self.inference_engine = inference_engine or FaultModelInference(auto_train_if_missing=True)
        self.fusion_engine = fusion_engine or MultiModalSensorFusion()

        self.history_records: deque = deque(maxlen=max_history)
        self.alert_events: deque = deque(maxlen=100)

    def analyze_window(
        self,
        window_data: Dict[str, Any],
        operating_mode: str = "SIMULATION",
    ) -> Dict[str, Any]:
        """Run full processing -> feature extraction -> ML inference -> sensor fusion on a window."""
        processed = self.processor.process_window(window_data)
        features = processed["features"]
        diagnostics = processed["diagnostics"]
        sensor_avail = processed["sensor_availability"]

        ml_res = self.inference_engine.predict(features)
        fusion_res = self.fusion_engine.fuse(
            features=features,
            ml_result=ml_res,
            sensor_availability=sensor_avail,
        )

        timestamp_ms = int(processed["timestamp_ms"])
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(timestamp_ms / 1000.0))

        # Log history record
        history_entry = {
            "timestamp_ms": timestamp_ms,
            "time": timestamp_str,
            "operating_mode": operating_mode,
            "predicted_fault": fusion_res["final_fault_class"],
            "alert_level": fusion_res["alert_level"],
            "health_status": fusion_res["health_status_label"],
            "confidence": fusion_res["confidence_score"],
            "is_anomaly": fusion_res["is_anomaly"],
            "explanation": fusion_res["summary_explanation"],
            **features,
        }
        self.history_records.append(history_entry)

        # Log alert event if non-healthy or warning
        if fusion_res["alert_level"] in ("YELLOW", "RED") or fusion_res["final_fault_class"] != CLASS_HEALTHY:
            self.alert_events.appendleft(
                {
                    "timestamp_ms": timestamp_ms,
                    "time": timestamp_str,
                    "alert_level": fusion_res["alert_level"],
                    "status": fusion_res["health_status_label"],
                    "fault_class": fusion_res["final_fault_class"],
                    "confidence": fusion_res["confidence_score"],
                    "reasons": fusion_res["explanations"],
                    "summary": fusion_res["summary_explanation"],
                }
            )

        return {
            "timestamp_ms": timestamp_ms,
            "time_str": timestamp_str,
            "operating_mode": operating_mode,
            "features": features,
            "diagnostics": diagnostics,
            "ml_result": ml_res,
            "fusion_result": fusion_res,
            "raw_window": window_data,
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return recent sensor and prediction history as a pandas DataFrame for plotting and CSV export."""
        if not self.history_records:
            return pd.DataFrame()
        return pd.DataFrame(list(self.history_records))

    def get_recent_alerts(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Return most recent fault/warning events."""
        return list(self.alert_events)[:limit]

