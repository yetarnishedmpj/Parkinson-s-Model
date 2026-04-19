"""
model.py — upgraded diagnostic engine with LSTM temporal buffer.
Falls back to Random Forest for the first SEQ_LEN frames; switches
to the FogLSTM once the ring buffer is full.
"""
import numpy as np, joblib, os, pandas as pd
from collections import deque
from .rl_agent import ClinicalRLAgent
from .fog_lstm import FogLSTM, SEQ_LEN, predict as lstm_predict, load_lstm

basedir = os.path.dirname(__file__)

# ── Load models ────────────────────────────────────────────────────────────
try:
    classifier = joblib.load(os.path.join(basedir, "parkinsons_classifier.pkl"))
    regressor  = joblib.load(os.path.join(basedir, "parkinsons_regressor.pkl"))
    MODELS_LOADED = True
    print("Random Forest models loaded.")
except Exception as e:
    print(f"RF models not found: {e}")
    classifier = regressor = None
    MODELS_LOADED = False

lstm_model = load_lstm()
if lstm_model:
    print("FogLSTM loaded — temporal diagnostics active.")
else:
    print("FogLSTM not found — run train_real_data.py to train it.")

rl_agent = ClinicalRLAgent()

# ── Temporal ring buffer ───────────────────────────────────────────────────
_temporal_buf: deque[list[float]] = deque(maxlen=SEQ_LEN)

# Tracks previous state for RL reward assignment
_prev_neuro   = 0.0
_prev_state   = None
_prev_action  = 0
_prev_freezing = False
_learn_counter = 0


def _make_feature(hr, temp, act, prox, is_frz):
    hr_n   = (np.clip(hr,   40, 180) - 60) / 120.0
    temp_n = (np.clip(temp, 35,  40) - 36.0) / 3.0
    act_n  = float(np.clip(act, 0, 2.0)) / 2.0
    return [float(hr_n), float(temp_n), float(act_n),
            float(np.clip(prox, 0, 1)), float(is_frz)]


def analyze_vitals(heart_rate: float, temperature: float,
                   activity_level: float, hazard_proximity: float = 0.0,
                   is_freezing: bool = False, fatigue: float = 0.0):
    """
    Cognitive Narrative Diagnostic & Intervention Engine.

    Preferred path: FogLSTM (temporal, 30-frame window).
    Fallback:       Random Forest (stateless snapshot).
    """
    global _prev_neuro, _prev_state, _prev_action, _prev_freezing, _learn_counter

    feat_vec = _make_feature(heart_rate, temperature, activity_level,
                             hazard_proximity, is_freezing)
    _temporal_buf.append(feat_vec)

    # ── Diagnosis ──────────────────────────────────────────────────────────
    if lstm_model and len(_temporal_buf) == SEQ_LEN:
        predicted_state, freeze_prob, neuro_load = lstm_predict(
            lstm_model, list(_temporal_buf))
        confidence_score = max(0.5, 1.0 - freeze_prob * 0.5)
        predicted_fall_risk = freeze_prob
    elif MODELS_LOADED:
        fv = pd.DataFrame([{
            'heart_rate': heart_rate, 'temperature': temperature,
            'activity_level': activity_level, 'hazard_proximity': hazard_proximity,
            'is_freezing': int(is_freezing)
        }])
        predicted_state      = classifier.predict(fv)[0]
        predicted_fall_risk  = float(regressor.predict(fv)[0])
        class_probs          = classifier.predict_proba(fv)[0]
        confidence_score     = float(max(class_probs))
        neuro_load           = predicted_fall_risk * 0.8 + hazard_proximity * 0.2
        freeze_prob          = predicted_fall_risk
    else:
        predicted_state = "NOMINAL"; neuro_load = 0.1
        predicted_fall_risk = 0.1; confidence_score = 0.5; freeze_prob = 0.1

    # ── RL Intervention ────────────────────────────────────────────────────
    hr_n   = (heart_rate - 60) / 100.0
    temp_n = (temperature - 36.5) / 2.0
    rl_state = [hr_n, temp_n, activity_level, hazard_proximity, float(is_freezing)]

    # Online learning: observe outcome of previous action
    if _prev_state is not None:
        rl_agent.observe(
            _prev_state, _prev_action, rl_state,
            _prev_neuro, neuro_load, _prev_freezing, is_freezing,
        )
        _learn_counter += 1
        if _learn_counter % 50 == 0:
            rl_agent.learn()

    action_idx, intervention = rl_agent.get_action(rl_state)

    # Store for next tick
    _prev_state    = rl_state
    _prev_action   = action_idx
    _prev_neuro    = neuro_load
    _prev_freezing = is_freezing

    # ── Narrative ──────────────────────────────────────────────────────────
    insights = []
    if predicted_state == "FREEZE":
        insights.append(f"CRITICAL: Neuro-motor freeze detected. [{int(confidence_score*100)}% confidence]")
        insights.append(f"RL PRESCRIPTION: {intervention}")
    elif predicted_state == "SPATIAL":
        insights.append(f"WARNING: Complex spatial navigation. (Fall Risk: {int(predicted_fall_risk*100)}%)")
        if action_idx != 0:
            insights.append(f"RL SUGGESTION: {intervention}")
    elif predicted_state == "ELEVATED":
        insights.append(f"OBSERVATION: Elevated neurological stress. ({int(confidence_score*100)}% confidence)")
    elif predicted_state == "STABLE":
        insights.append("STABLE: Optimal physiological synchronisation. Motor confidence high.")
    else:
        insights.append("NOMINAL: Routine monitoring active. Telemetry in bounds.")

    if fatigue > 0.6:
        insights.append(f"FATIGUE ALERT: Motor endurance at {int((1-fatigue)*100)}% capacity.")

    return {
        "stress_level":      round(float(neuro_load), 2),
        "health_index":      round(100.0 - float(predicted_fall_risk * 100), 1),
        "fall_risk":         round(float(predicted_fall_risk), 2),
        "freeze_prob_3s":    round(float(freeze_prob), 2),
        "status":            predicted_state,
        "insight":           " | ".join(insights),
        "neurological_load": round(float(neuro_load), 2),
        "confidence":        round(float(confidence_score), 2),
        "rl_action":         action_idx,
        "rl_intervention":   intervention,
        "using_lstm":        lstm_model is not None and len(_temporal_buf) == SEQ_LEN,
    }
