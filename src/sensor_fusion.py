"""Multi-Modal Sensor Fusion Engine for EV Motor Fault Detection.

Combines:
1. Feature-level ML classification (Random Forest + Isolation Forest on the 18-feature unified vector).
2. Cross-modal physical consistency checks across Mechanical Vibration (MPU6050),
   Electrical Current (ACS712), and Thermal Dynamics (DS18B20).
3. Explainable root-cause reasoning so alerts never trigger from a single arbitrary threshold
   without contextualizing the other sensor modalities.
"""

from typing import Dict, Any, List, Optional
from src.config import (
    CLASS_HEALTHY,
    CLASS_ROTOR_IMBALANCE,
    CLASS_MECHANICAL_STALL,
    THRESHOLD_CONFIG,
    ThresholdConfig,
)


class MultiModalSensorFusion:
    """Evaluates multi-sensor evidence and fuses ML predictions with physical domain rules."""

    def __init__(self, thresholds: ThresholdConfig = THRESHOLD_CONFIG) -> None:
        self.thresholds = thresholds

    def evaluate_modalities(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze each individual sensor modality and generate explainable diagnostic indicators."""
        reasons: List[str] = []

        vib_rms = float(features.get("vib_rms", 0.0))
        vib_p2p = float(features.get("vib_peak_to_peak", 0.0))
        vib_dom_freq = float(features.get("vib_dominant_freq_hz", 0.0))
        vib_band_energy = float(features.get("vib_band_energy_50_1000hz", 0.0))
        vib_kurtosis = float(features.get("vib_kurtosis", 3.0))

        cur_rms = float(features.get("current_rms", 0.0))
        cur_peak = float(features.get("current_peak", 0.0))
        cur_fluct = float(features.get("current_fluctuation", 0.0))

        temp_c = float(features.get("temp_current_c", 38.0))
        temp_rate = float(features.get("temp_rise_rate_c_per_s", 0.0))

        # 1. Mechanical Vibration Assessment
        vib_severity = "NORMAL"
        if vib_rms >= self.thresholds.vib_rms_critical_g or vib_p2p >= self.thresholds.vib_p2p_warning_g * 1.35:
            vib_severity = "HIGH"
            reasons.append(
                f"Elevated vibration RMS ({vib_rms:.3f} g) and Peak-to-Peak ({vib_p2p:.3f} g) above critical limits"
            )
        elif vib_rms >= self.thresholds.vib_rms_warning_g or vib_p2p >= self.thresholds.vib_p2p_warning_g:
            vib_severity = "ELEVATED"
            reasons.append(
                f"Elevated vibration RMS ({vib_rms:.3f} g) and Peak-to-Peak ({vib_p2p:.3f} g)"
            )

        if vib_band_energy >= self.thresholds.vib_band_energy_warning:
            reasons.append(
                f"Increased dominant-frequency spectral energy ({vib_band_energy:.2f} g² at {vib_dom_freq:.1f} Hz)"
            )
        elif vib_kurtosis >= 4.2:
            reasons.append(
                f"Impulsive mechanical vibration kurtosis ({vib_kurtosis:.2f}) indicating structural chattering"
            )

        # 2. Electrical Current Assessment
        cur_severity = "NORMAL"
        if cur_rms >= self.thresholds.current_rms_stall_a:
            cur_severity = "CRITICAL_OVERCURRENT"
            reasons.append(
                f"Abnormal high stall current draw (RMS {cur_rms:.2f} A, Peak {cur_peak:.2f} A)"
            )
        elif cur_rms >= self.thresholds.current_rms_warning_a or cur_fluct >= self.thresholds.current_fluctuation_warning_pct:
            cur_severity = "ELEVATED"
            reasons.append(
                f"Abnormal current fluctuation ({cur_fluct:.1f}%) and load current (RMS {cur_rms:.2f} A)"
            )

        # 3. Thermal Behaviour Assessment
        temp_severity = "NORMAL"
        if temp_c >= self.thresholds.temp_critical_c or temp_rate >= self.thresholds.temp_rise_rate_warning_c_per_s * 1.6:
            temp_severity = "HIGH"
            reasons.append(
                f"Unusual temperature rise ({temp_c:.1f} °C, rise rate {temp_rate:+.2f} °C/s)"
            )
        elif temp_c >= self.thresholds.temp_warning_c or temp_rate >= self.thresholds.temp_rise_rate_warning_c_per_s:
            temp_severity = "ELEVATED"
            reasons.append(
                f"Elevated motor thermal trend ({temp_c:.1f} °C, rate {temp_rate:+.2f} °C/s)"
            )

        active_abnormal_modalities = sum(
            1 for s in (vib_severity, cur_severity, temp_severity) if s != "NORMAL"
        )

        return {
            "vibration_severity": vib_severity,
            "current_severity": cur_severity,
            "temperature_severity": temp_severity,
            "active_abnormal_modalities": active_abnormal_modalities,
            "reasons": reasons,
        }

    def fuse(
        self,
        features: Dict[str, float],
        ml_result: Dict[str, Any],
        sensor_availability: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Fuse ML classification with multi-sensor physical evidence to produce a final diagnosis."""
        modality_eval = self.evaluate_modalities(features)
        ml_class = str(ml_result.get("predicted_class", CLASS_HEALTHY))
        ml_conf = float(ml_result.get("confidence_score", 0.0))
        is_anomaly = bool(ml_result.get("is_anomaly", False))

        vib_sev = modality_eval["vibration_severity"]
        cur_sev = modality_eval["current_severity"]
        temp_sev = modality_eval["temperature_severity"]
        abnormal_count = modality_eval["active_abnormal_modalities"]
        reasons: List[str] = list(modality_eval["reasons"])

        # Cross-modal fusion decision logic:
        # Never classify a critical fault from a single slight threshold violation if ML + other sensors say HEALTHY.
        final_class = ml_class
        alert_level = "GREEN"  # GREEN (Healthy), YELLOW (Warning), RED (Critical Fault)

        if ml_class == CLASS_MECHANICAL_STALL:
            # Verify stall has electrical or thermal or impulsive vibration corroboration
            if cur_sev == "CRITICAL_OVERCURRENT" or (cur_sev != "NORMAL" and (temp_sev != "NORMAL" or vib_sev != "NORMAL")):
                final_class = CLASS_MECHANICAL_STALL
                alert_level = "RED"
            elif ml_conf >= 0.75:
                final_class = CLASS_MECHANICAL_STALL
                alert_level = "RED"
            else:
                final_class = CLASS_HEALTHY
                alert_level = "YELLOW"
                reasons.append("Low-confidence stall pattern without multi-sensor current/thermal confirmation.")

        elif ml_class == CLASS_ROTOR_IMBALANCE:
            if vib_sev in ("ELEVATED", "HIGH") or cur_sev != "NORMAL" or ml_conf >= 0.70:
                final_class = CLASS_ROTOR_IMBALANCE
                alert_level = "RED" if (vib_sev == "HIGH" and cur_sev != "NORMAL") else "YELLOW"
            else:
                final_class = CLASS_HEALTHY
                alert_level = "GREEN"

        else:  # ML predicted HEALTHY
            # Check if multi-modal physical indicators strongly indicate a developing condition
            if cur_sev == "CRITICAL_OVERCURRENT" and (vib_sev != "NORMAL" or temp_sev != "NORMAL"):
                final_class = CLASS_MECHANICAL_STALL
                alert_level = "RED"
                reasons.append("Multi-sensor overcurrent + mechanical/thermal stress override.")
            elif vib_sev == "HIGH" and cur_sev != "NORMAL":
                final_class = CLASS_ROTOR_IMBALANCE
                alert_level = "YELLOW"
                reasons.append("Correlated high periodic vibration and current modulation detected.")
            elif abnormal_count >= 2 or (is_anomaly and abnormal_count >= 1):
                final_class = CLASS_HEALTHY
                alert_level = "YELLOW"
                reasons.append("Early multi-sensor deviation detected; monitoring closely.")
            else:
                final_class = CLASS_HEALTHY
                alert_level = "GREEN"

        # Note any unavailable sensors if running in degraded hardware mode
        if sensor_availability:
            missing_sensors = [k.upper() for k, ok in sensor_availability.items() if not ok]
            if missing_sensors:
                reasons.append(f"Degraded sensor mode: {', '.join(missing_sensors)} unavailable (using fallback state).")
                if alert_level == "GREEN":
                    alert_level = "YELLOW"

        if not reasons:
            reasons.append(
                "All three sensor modalities (MPU6050 vibration, ACS712 current, DS18B20 temperature) "
                "are within nominal operating envelopes."
            )

        status_label_map = {
            "GREEN": "HEALTHY",
            "YELLOW": "WARNING",
            "RED": "CRITICAL FAULT",
        }

        return {
            "final_fault_class": final_class,
            "ml_predicted_class": ml_class,
            "confidence_score": ml_conf,
            "confidence_note": ml_result.get("confidence_note", ""),
            "class_scores": ml_result.get("class_scores", {}),
            "is_anomaly": bool(is_anomaly or final_class != CLASS_HEALTHY),
            "anomaly_score": ml_result.get("anomaly_score", 0.0),
            "alert_level": alert_level,
            "health_status_label": status_label_map[alert_level],
            "modality_severities": {
                "vibration": vib_sev,
                "current": cur_sev,
                "temperature": temp_sev,
            },
            "explanations": reasons,
            "summary_explanation": " | ".join(reasons),
        }

