"""
Load the saved HA models and make a prediction.
Drop this file anywhere — just update SAVE_DIR to point at the ha_model folder.
"""
import pickle, json, numpy as np
from pathlib import Path

SAVE_DIR = Path(r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ha_model')

# ── Load ──────────────────────────────────────────────────────────────────────
with open(SAVE_DIR / "random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open(SAVE_DIR / "xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open(SAVE_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open(SAVE_DIR / "model_metadata.json") as f:
    meta = json.load(f)

feature_names = meta["feature_names"]
classes       = meta["classes"]

print(f"Loaded models — {len(classes)} classes, {len(feature_names)} features")
print(f"RF accuracy:  Top-1={meta['accuracy']['Random Forest']['top1']:.1%}  Top-3={meta['accuracy']['Random Forest']['top3']:.1%}")
print(f"XGB accuracy: Top-1={meta['accuracy']['XGBoost']['top1']:.1%}  Top-3={meta['accuracy']['XGBoost']['top3']:.1%}")

# ── Predict function ─────────────────────────────────────────────────────────
def predict_top3(feature_vector, model="rf"):
    """
    feature_vector: dict or list/array with values in feature_names order.
    Returns list of (action_label, probability) tuples, top-3.
    """
    if isinstance(feature_vector, dict):
        x = np.array([feature_vector.get(f, 0.0) for f in feature_names], dtype=np.float32)
    else:
        x = np.array(feature_vector, dtype=np.float32)
    x = x.reshape(1, -1)

    m = rf_model if model == "rf" else xgb_model
    proba = m.predict_proba(x)[0]
    top3i = np.argsort(proba)[::-1][:3]
    return [(classes[i], round(float(proba[i]), 4)) for i in top3i]

# ── Quick demo with a zeroed feature vector ───────────────────────────────────
if __name__ == "__main__":
    demo = {f: 0.0 for f in feature_names}
    demo["hour"] = 20          # 8pm
    demo["hour_sin"] = np.sin(2 * np.pi * 20 / 24)
    demo["hour_cos"] = np.cos(2 * np.pi * 20 / 24)
    demo["is_evening"] = 1
    demo["is_occupied"] = 1
    demo["people_count"] = 2
    demo["is_weekday"] = 1
    demo["media_player_tv_on"] = 0    # TV is off right now
    demo["light_living_room_on"] = 1  # living room light is on

    print("\nSample prediction (8pm, home occupied, TV off):")
    for action, prob in predict_top3(demo, model="rf"):
        print(f"  RF  → {action.replace('_', ' '):40s} {prob:.1%}")
    for action, prob in predict_top3(demo, model="xgb"):
        print(f"  XGB → {action.replace('_', ' '):40s} {prob:.1%}")
