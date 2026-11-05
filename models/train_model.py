"""Training pipeline for Random Forest supervised classification & Isolation Forest anomaly detection.

NOTE ON SYNTHETIC DATA:
When trained on synthetic demonstration data, the saved metadata explicitly records
`data_source = "SYNTHETIC_DEMONSTRATION"`. Models trained on synthetic data are intended for
architectural demonstration and must not be claimed as validated on real physical EV motor faults
until retrained on labeled hardware recordings.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_HEALTHY,
    FAULT_CLASSES,
    FEATURE_COLUMNS,
    FEATURE_UNITS,
    RF_MODEL_PATH,
    ISOLATION_FOREST_PATH,
    SCALER_PATH,
    FEATURE_METADATA_PATH,
    EVALUATION_METRICS_PATH,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
)
from src.data_generator import SensorSimulator
from src.feature_extractor import extract_unified_features
from models.preprocessing import prepare_features_and_labels


def generate_synthetic_dataset(
    samples_per_class: int = 240,
    seed: int = 42,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Generate a labeled multi-modal dataset using the physical-pattern SensorSimulator."""
    rng = np.random.default_rng(seed)
    records = []

    for fault_class in FAULT_CLASSES:
        sim = SensorSimulator(seed=int(rng.integers(1, 100_000)))
        sim.set_operating_state(fault_class)

        if fault_class == CLASS_HEALTHY:
            sim.reset_thermal_state(38.0)
        elif fault_class == "ROTOR_IMBALANCE":
            sim.reset_thermal_state(42.5)
        else:
            sim.reset_thermal_state(48.0)

        for i in range(samples_per_class):
            # Periodically vary initial thermal baseline so the classifier learns across temperature ranges
            if i % 35 == 0 and i > 0:
                base_t = {
                    "HEALTHY": float(rng.uniform(36.0, 41.5)),
                    "ROTOR_IMBALANCE": float(rng.uniform(41.0, 48.0)),
                    "MECHANICAL_STALL": float(rng.uniform(46.0, 68.0)),
                }[fault_class]
                sim.reset_thermal_state(base_t)

            window = sim.generate_window(state=fault_class, add_variation=True)
            feat_dict, _, _ = extract_unified_features(window)

            # Inject mild realistic measurement noise / cross-condition boundary overlap (~4%)
            if rng.random() < 0.04:
                for k in feat_dict:
                    feat_dict[k] *= float(rng.uniform(0.85, 1.15))

            row = dict(feat_dict)
            row["fault_class"] = fault_class
            row["data_source"] = "SYNTHETIC_DEMONSTRATION"
            records.append(row)

    df = pd.DataFrame(records)
    if save_csv:
        processed_path = PROCESSED_DATA_DIR / "synthetic_training_features.csv"
        sample_path = SAMPLE_DATA_DIR / "sample_sensor_features.csv"
        df.to_csv(processed_path, index=False)
        df.sample(n=min(90, len(df)), random_state=seed).to_csv(sample_path, index=False)

    return df


def train_and_evaluate_models(
    df: Optional[pd.DataFrame] = None,
    data_source_label: str = "SYNTHETIC_DEMONSTRATION",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest classifier + Isolation Forest anomaly detector, evaluate, and save."""
    if df is None:
        df = generate_synthetic_dataset(samples_per_class=240, seed=random_state, save_csv=True)

    X_scaled, y, scaler = prepare_features_and_labels(df, target_col="fault_class", fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 1. Primary Supervised Classifier: Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )
    rf_model.fit(X_train, y_train)

    # 2. Complementary Unsupervised Anomaly Detector: Isolation Forest (fitted primarily on HEALTHY data)
    healthy_mask = y_train == CLASS_HEALTHY
    X_train_healthy = X_train[healthy_mask] if np.any(healthy_mask) else X_train
    iso_model = IsolationForest(
        n_estimators=120,
        contamination=0.06,
        random_state=random_state,
    )
    iso_model.fit(X_train_healthy)

    # Evaluate Random Forest on held-out test set
    y_pred = rf_model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_test, y_pred, labels=FAULT_CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=FAULT_CLASSES)

    importances = {
        col: round(float(imp), 5)
        for col, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)
    }

    metrics_summary: Dict[str, Any] = {
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": data_source_label,
        "validation_disclaimer": (
            "Model trained on synthetic demonstration patterns. Not validated on physical EV motor faults."
            if "SYNTHETIC" in data_source_label.upper()
            else "Model trained on recorded sensor CSV dataset."
        ),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": FAULT_CLASSES,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            cls: {
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4),
                "support": int(per_class_sup[idx]),
            }
            for idx, cls in enumerate(FAULT_CLASSES)
        },
        "feature_importances": importances,
        "classification_report_text": classification_report(
            y_test, y_pred, labels=FAULT_CLASSES, zero_division=0
        ),
    }

    # Save artifacts via joblib and JSON
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(iso_model, ISOLATION_FOREST_PATH)
    joblib.dump(scaler, SCALER_PATH)

    feature_metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_units": FEATURE_UNITS,
        "fault_classes": FAULT_CLASSES,
        "data_source": data_source_label,
        "model_type": "RandomForestClassifier + IsolationForest",
    }
    FEATURE_METADATA_PATH.write_text(json.dumps(feature_metadata, indent=2), encoding="utf-8")
    EVALUATION_METRICS_PATH.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    return metrics_summary


if __name__ == "__main__":
    summary = train_and_evaluate_models()
    print(f"Trained RF + IsolationForest on {summary['total_samples']} samples ({summary['data_source']}).")
    print(
        f"Test Accuracy: {summary['accuracy']:.4f} | "
        f"Precision: {summary['precision_weighted']:.4f} | "
        f"Recall: {summary['recall_weighted']:.4f} | "
        f"F1: {summary['f1_weighted']:.4f}"
    )

