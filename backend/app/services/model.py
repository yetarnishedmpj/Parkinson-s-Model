"""
Diagnostic engine with temporal FOG prediction and RL intervention cues.

The preferred path uses the FogLSTM once a full temporal window is available.
Until then, the trained random-forest models are used when present. A small
deterministic heuristic remains as a final fallback so the demo stays live even
on machines without model artifacts.
"""

from collections import deque
import os

try:
    import joblib
    import pandas as pd
except Exception as e:
    print(f"RF model dependencies unavailable: {e}")
    joblib = None
    pd = None


def _clip(value, low, high):
    return max(low, min(high, value))

try:
    from .fog_lstm import SEQ_LEN, load_lstm, predict as lstm_predict
except Exception as e:
    print(f"FogLSTM dependencies unavailable: {e}")
    SEQ_LEN = 30
    load_lstm = None
    lstm_predict = None

try:
    from .rl_agent import ClinicalRLAgent
except Exception as e:
    print(f"RL agent dependencies unavailable: {e}")

    class ClinicalRLAgent:
        ACTIONS_MAP = {
            0: "No Intervention",
            1: "Rhythmic Auditory Stimulation to break freezing",
            2: "Visual Laser Cues to step over",
            3: "DBS Voltage Increase for motor control",
            4: "Suggest Rest - fatigue level high",
        }

        def observe(self, *args, **kwargs):
            return None

        def learn(self):
            return None

        def get_action(self, state_array):
            hazard = state_array[3]
            is_freezing = bool(state_array[4])
            if is_freezing:
                return 1, self.ACTIONS_MAP[1]
            if hazard > 0.75:
                return 2, self.ACTIONS_MAP[2]
            return 0, self.ACTIONS_MAP[0]

basedir = os.path.dirname(__file__)

try:
    if joblib is None or pd is None:
        raise RuntimeError("joblib/pandas unavailable")
    classifier = joblib.load(os.path.join(basedir, "parkinsons_classifier.pkl"))
    regressor = joblib.load(os.path.join(basedir, "parkinsons_regressor.pkl"))
    MODELS_LOADED = True
    print("Random Forest models loaded.")
except Exception as e:
    print(f"RF models not found: {e}")
    classifier = regressor = None
    MODELS_LOADED = False

lstm_model = load_lstm() if load_lstm else None
if lstm_model:
    print("FogLSTM loaded: temporal diagnostics active.")
else:
    print("FogLSTM not found: run train_real_data.py to train it.")

rl_agent = ClinicalRLAgent()

_temporal_buf: deque[list[float]] = deque(maxlen=SEQ_LEN)
_prev_neuro = 0.0
_prev_state = None
_prev_action = 0
_prev_freezing = False
_learn_counter = 0


def _make_feature(hr, temp, act, prox, is_frz):
    hr_n = (_clip(hr, 40, 180) - 60) / 120.0
    temp_n = (_clip(temp, 35, 40) - 36.0) / 3.0
    act_n = float(_clip(act, 0, 2.0)) / 2.0
    return [
        float(hr_n),
        float(temp_n),
        float(act_n),
        float(_clip(prox, 0, 1)),
        float(is_frz),
    ]


def _heuristic_diagnosis(heart_rate, temperature, activity_level, hazard_proximity, is_freezing):
    norm_hr = (heart_rate - 60) / 100.0
    norm_temp = (temperature - 36.5) / 2.0
    stress_level = float(_clip(
        norm_hr * 0.4 + norm_temp * 0.1 + activity_level * 0.2 + hazard_proximity * 0.5 + float(is_freezing) * 0.8,
        0.0,
        1.0,
    ))
    health_index = float(_clip(
        95.0 - stress_level * 35.0 - hazard_proximity * 10.0 - float(is_freezing) * 30.0,
        0.0,
        100.0,
    ))
    fall_risk = float(_clip(stress_level * 0.7 + (1.0 - health_index / 100.0) * 0.3, 0.0, 1.0))
    confidence = float(_clip(1.0 - (stress_level * 0.45 + hazard_proximity * 0.35), 0.35, 0.98))

    if is_freezing:
        status = "FREEZE"
    elif hazard_proximity > 0.8:
        status = "SPATIAL"
    elif stress_level > 0.6:
        status = "ELEVATED"
    elif activity_level > 0.3:
        status = "NAVIGATING"
    elif health_index > 90:
        status = "STABLE"
    else:
        status = "NOMINAL"

    return status, fall_risk, fall_risk, stress_level, confidence


def analyze_vitals(
    heart_rate: float,
    temperature: float,
    activity_level: float,
    hazard_proximity: float = 0.0,
    is_freezing: bool = False,
    fatigue: float = 0.0,
):
    global _prev_neuro, _prev_state, _prev_action, _prev_freezing, _learn_counter

    feat_vec = _make_feature(
        heart_rate,
        temperature,
        activity_level,
        hazard_proximity,
        is_freezing,
    )
    _temporal_buf.append(feat_vec)

    if lstm_model and len(_temporal_buf) == SEQ_LEN:
        predicted_state, freeze_prob, neuro_load = lstm_predict(lstm_model, list(_temporal_buf))
        confidence_score = max(0.5, 1.0 - freeze_prob * 0.5)
        predicted_fall_risk = freeze_prob
    elif MODELS_LOADED:
        fv = pd.DataFrame([{
            "heart_rate": heart_rate,
            "temperature": temperature,
            "activity_level": activity_level,
            "hazard_proximity": hazard_proximity,
            "is_freezing": int(is_freezing),
        }])
        predicted_state = classifier.predict(fv)[0]
        predicted_fall_risk = float(regressor.predict(fv)[0])
        class_probs = classifier.predict_proba(fv)[0]
        confidence_score = float(max(class_probs))
        neuro_load = predicted_fall_risk * 0.8 + hazard_proximity * 0.2
        freeze_prob = predicted_fall_risk
    else:
        (
            predicted_state,
            predicted_fall_risk,
            freeze_prob,
            neuro_load,
            confidence_score,
        ) = _heuristic_diagnosis(
            heart_rate,
            temperature,
            activity_level,
            hazard_proximity,
            is_freezing,
        )

    hr_n = (heart_rate - 60) / 100.0
    temp_n = (temperature - 36.5) / 2.0
    rl_state = [hr_n, temp_n, activity_level, hazard_proximity, float(is_freezing)]

    if _prev_state is not None:
        rl_agent.observe(
            _prev_state,
            _prev_action,
            rl_state,
            _prev_neuro,
            neuro_load,
            _prev_freezing,
            is_freezing,
        )
        _learn_counter += 1
        if _learn_counter % 50 == 0:
            rl_agent.learn()

    action_idx, intervention = rl_agent.get_action(rl_state)

    _prev_state = rl_state
    _prev_action = action_idx
    _prev_neuro = neuro_load
    _prev_freezing = is_freezing

    insights = []
    if predicted_state == "FREEZE":
        insights.append(f"CRITICAL: Neuro-motor freeze detected. [{int(confidence_score * 100)}% confidence]")
        insights.append(f"RL PRESCRIPTION: {intervention}")
    elif predicted_state == "SPATIAL":
        insights.append(f"WARNING: Complex spatial navigation. Fall risk {int(predicted_fall_risk * 100)}%.")
        if action_idx != 0:
            insights.append(f"RL SUGGESTION: {intervention}")
    elif predicted_state == "ELEVATED":
        insights.append(f"OBSERVATION: Elevated neurological stress. [{int(confidence_score * 100)}% confidence]")
    elif predicted_state == "NAVIGATING":
        insights.append(f"ACTIVITY: Motor confidence high. [{int(confidence_score * 100)}% confidence]")
    elif predicted_state == "STABLE":
        insights.append("STABLE: Optimal physiological synchronisation. Motor confidence high.")
    else:
        insights.append("NOMINAL: Routine monitoring active. Telemetry in bounds.")

    if fatigue > 0.6:
        insights.append(f"FATIGUE ALERT: Motor endurance at {int((1 - fatigue) * 100)}% capacity.")

    return {
        "stress_level": round(float(neuro_load), 2),
        "health_index": round(100.0 - float(predicted_fall_risk * 100), 1),
        "fall_risk": round(float(predicted_fall_risk), 2),
        "freeze_prob_3s": round(float(freeze_prob), 2),
        "status": predicted_state,
        "insight": " | ".join(insights),
        "neurological_load": round(float(neuro_load), 2),
        "confidence": round(float(confidence_score), 2),
        "rl_action": action_idx,
        "rl_intervention": intervention,
        "using_lstm": lstm_model is not None and len(_temporal_buf) == SEQ_LEN,
    }
