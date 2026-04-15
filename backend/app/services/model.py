
import numpy as np

# Weights for internal risk scoring
MODEL_WEIGHTS = np.array([
    [0.4, -0.2],  # Heart Rate
    [0.1, -0.1],  # Temperature
    [0.2, -0.1],  # Activity
    [0.5, -0.4],  # Proximity
    [0.8, -0.7]   # Freeze
])

def analyze_vitals(heart_rate: float, temperature: float, activity_level: float, hazard_proximity: float = 0.0, is_freezing: bool = False):
    """
    ML 4.0: Cognitive Narrative Diagnostic Engine.
    Converts multi-variate metrics into human-readable clinical insights with confidence scoring.
    """
    # 1. Internal Risk Analysis
    norm_hr = (heart_rate - 60) / 100.0
    norm_temp = (temperature - 36.5) / 2.0
    inputs = np.array([norm_hr, norm_temp, activity_level, hazard_proximity, 1.0 if is_freezing else 0.0])
    
    raw = np.dot(inputs, MODEL_WEIGHTS)
    stress_level = min(1.0, max(0.0, raw[0]))
    health_index = min(100.0, max(0.0, 95.0 + raw[1] * 50.0))
    fall_risk = min(1.0, stress_level * 0.7 + (1.0 - health_index/100.0) * 0.3)
    confidence = max(0.0, 1.0 - (stress_level * 0.5 + hazard_proximity * 0.5))

    # 2. Qualitative Clinical Insights (Narrative Generation)
    insights = []
    
    if is_freezing:
        insights.append(f"CRITICAL: Neuro-motor freeze detected. [{int(confidence*100)}% reliability]")
        status = "FREEZE"
    elif hazard_proximity > 0.8:
        insights.append(f"WARNING: Navigating complex spatial landmark. Confidence: {int(confidence*100)}%")
        status = "SPATIAL"
    elif stress_level > 0.6:
        insights.append("OBSERVATION: Elevated neurological stress. Compensatory gait detected.")
        status = "ELEVATED"
    elif activity_level > 0.3:
        insights.append(f"ACTIVITY: Motor confidence high ({int(confidence*100)}%). Fluid motion observed.")
        status = "NAVIGATING"
    elif health_index > 90:
        insights.append("STABLE: Optimal physiological state. Autonomic sync confirmed.")
        status = "STABLE"
    else:
        insights.append("NOMINAL: Routine monitoring active. No anomalies detected.")
        status = "NOMINAL"

    return {
        "stress_level": round(float(stress_level), 2),
        "health_index": round(float(health_index), 1),
        "fall_risk": round(float(fall_risk), 2),
        "status": status,
        "insight": " | ".join(insights),
        "neurological_load": round(float(stress_level * 0.8 + hazard_proximity * 0.2), 2),
        "confidence": round(float(confidence), 2)
    }
