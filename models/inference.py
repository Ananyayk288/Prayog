"""Reusable ML inference module for EV Motor Fault Detection.

Loads saved Random Forest and Isolation Forest models, validates feature ordering against
`feature_metadata.json`, computes class vote proportions (clearly documented as uncalibrated
tree-vote confidence rather than guaranteed physical fault probabilities), and returns
anomaly detection status.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import joblib
import numpy as np

from src.config import (
    CLASS_HEALTHY,
    FAULT_CLASSES,
    FEATURE_COLUMNS,
    RF_MODEL_PATH,
    ISOLATION_FOREST_PATH,
    SCALER_PATH,
    FEATURE_METADATA_PATH,
    EVALUATION_METRICS_PATH,
)


class FaultModelInference:
    """Wrapper for loading trained models and executing single-window or batch fault inference."""

    def __init__(
        self,
        rf_path: Path = RF_MODEL_PATH,
        iso_path: Path = ISOLATION_FOREST_PATH,
        scaler_path: Path = SCALER_PATH,
        metadata_path: Path = FEATURE_METADATA_PATH,
        auto_train_if_missing: bool = True,
    ) -> None:
        self.rf_path = Path(rf_path)
        self.iso_path = Path(iso_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)

        self.rf_model = None
        self.iso_model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.evaluation_metrics: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self.load_models(auto_train_if_missing=auto_train_if_missing)

    def load_models(self, auto_train_if_missing: bool = True) -> bool:
        """Load saved joblib models and metadata; optionally train if artifacts do not exist yet."""
        if not (self.rf_path.exists() and self.scaler_path.exists() and self.metadata_path.exists()):
            if auto_train_if_missing:
                from models.train_model import train_and_evaluate_models
                train_and_evaluate_models()
            else:
                self.is_loaded = False
                return False

        self.rf_model = joblib.load(self.rf_path)
        self.scaler = joblib.load(self.scaler_path)
        if self.iso_path.exists():
            self.iso_model = joblib.load(self.iso_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if EVALUATION_METRICS_PATH.exists():
            self.evaluation_metrics = json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
        self.is_loaded = True
        return True

    def _prepare_vector(self, features: Union[Dict[str, float], np.ndarray, list]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Validate input features and convert into a 2D row vector matching `FEATURE_COLUMNS`."""
        expected_cols = self.metadata.get("feature_columns", FEATURE_COLUMNS)

        if isinstance(features, dict):
            missing = [c for c in expected_cols if c not in features]
            if missing:
                raise ValueError(f"Missing required features for inference: {missing}")
            feat_dict = {}
            for col in expected_cols:
                val = float(features[col])
                if not np.isfinite(val):
                    val = 0.0
                feat_dict[col] = val
            vec = np.array([[feat_dict[col] for col in expected_cols]], dtype=float)
            return vec, feat_dict

        arr = np.asarray(features, dtype=float).reshape(-1)
        if arr.size != len(expected_cols):
            raise ValueError(
                f"Feature vector length mismatch: got {arr.size}, expected {len(expected_cols)}."
            )
        clean_arr = np.where(np.isfinite(arr), arr, 0.0)
        feat_dict = {col: float(clean_arr[i]) for i, col in enumerate(expected_cols)}
        return clean_arr.reshape(1, -1), feat_dict

    def predict(self, features: Union[Dict[str, float], np.ndarray, list]) -> Dict[str, Any]:
        """Run supervised RF classification and Isolation Forest anomaly detection on a feature vector."""
        if not self.is_loaded or self.rf_model is None or self.scaler is None:
            raise RuntimeError("ML models are not loaded. Call load_models() first.")

        X_raw, feat_dict = self._prepare_vector(features)
        X_scaled = self.scaler.transform(X_raw)

        predicted_class = str(self.rf_model.predict(X_scaled)[0])
        proba_arr = self.rf_model.predict_proba(X_scaled)[0]
        class_labels = [str(c) for c in self.rf_model.classes_]

        class_vote_scores = {
            cls: round(float(score), 4)
            for cls, score in zip(class_labels, proba_arr)
        }
        # Ensure all canonical classes appear in dictionary
        for cls in FAULT_CLASSES:
            class_vote_scores.setdefault(cls, 0.0)

        top_confidence = float(np.max(proba_arr))

        # Isolation Forest anomaly check (-1 = anomaly, 1 = normal inlier)
        is_anomaly = False
        anomaly_score = 0.0
        if self.iso_model is not None:
            iso_pred = int(self.iso_model.predict(X_scaled)[0])
            anomaly_score = float(self.iso_model.decision_function(X_scaled)[0])
            is_anomaly = (iso_pred == -1) or (predicted_class != CLASS_HEALTHY)
        else:
            is_anomaly = predicted_class != CLASS_HEALTHY

        return {
            "predicted_class": predicted_class,
            "confidence_score": round(top_confidence, 4),
            "confidence_note": "Uncalibrated Random Forest ensemble tree-vote agreement (not a calibrated probability).",
            "class_scores": class_vote_scores,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(anomaly_score, 4),
            "data_source_trained_on": self.metadata.get("data_source", "SYNTHETIC_DEMONSTRATION"),
            "extracted_features": feat_dict,
        }
