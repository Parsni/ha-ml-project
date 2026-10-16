"""
Export trained models to disk so they can be reloaded without retraining.
Saves: Random Forest, XGBoost, LabelEncoder, feature names, metadata.
Also prints a quick usage example.
"""
import sys, os, importlib, json
sys.path.insert(0, r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace')

DATASET  = r'C:\Users\Shrimay Dhankar\.kiro\crew\uploads\c90946c0ea3a4b56a91dc5b12035ecf6_home_assistant_interactions_combined.jsonl'
SAVE_DIR = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ha_model'
os.makedirs(SAVE_DIR, exist_ok=True)

import numpy as np
import pickle

import ha_ml_trainer
importlib.reload(ha_ml_trainer)
from ha_ml_trainer import load_and_featurize, train_models

print("Loading dataset...")
df, labels = load_and_featurize(DATASET, max_rows=10000)
print(f"  {len(df)} samples | {df.shape[1]} features | {len(set(labels))} classes")

print("Training models (150 trees)...")
results, meta = train_models(df, labels, n_estimators=150,
                              progress_cb=lambda m,p: print(f"  [{p:3d}%] {m}"))

le = meta["le"]

# ── Save each model ────────────────────────────────────────────────────────────
for name, r in results.items():
    slug = name.lower().replace(" ", "_")
    path = os.path.join(SAVE_DIR, f"{slug}.pkl")
    with open(path, "wb") as f:
        pickle.dump(r["model"], f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {name}: {path}  ({size_mb:.1f} MB)")

# ── Save label encoder ────────────────────────────────────────────────────────
le_path = os.path.join(SAVE_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
print(f"  Saved LabelEncoder: {le_path}")

# ── Save feature names & metadata as JSON (human-readable) ───────────────────
meta_path = os.path.join(SAVE_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump({
        "feature_names": meta["feature_names"],
        "classes":       le.classes_.tolist(),
        "n_classes":     meta["n_classes"],
        "n_features":    len(meta["feature_names"]),
        "n_train":       meta["n_train"],
        "n_test":        meta["n_test"],
        "accuracy": {
            name: {"top1": round(r["acc"], 4), "top3": round(r["top3"], 4)}
            for name, r in results.items()
        },
        "trained_on":    "home_assistant_interactions_combined.jsonl",
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
    }, f, indent=2)
print(f"  Saved metadata:     {meta_path}")

# ── Write a ready-to-use loader script ───────────────────────────────────────
loader_path = os.path.join(SAVE_DIR, "load_and_predict.py")
with open(loader_path, "w") as f:
    f.write('''"""
Load the saved HA models and make a prediction.
Drop this file anywhere — just update SAVE_DIR to point at the ha_model folder.
"""
import pickle, json, numpy as np
from pathlib import Path

SAVE_DIR = Path(r\'C:\\Users\\Shrimay Dhankar\\.kiro\\crew\\workspace\\ha_model\')

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
print(f"RF accuracy:  Top-1={meta[\'accuracy\'][\'Random Forest\'][\'top1\']:.1%}  Top-3={meta[\'accuracy\'][\'Random Forest\'][\'top3\']:.1%}")
print(f"XGB accuracy: Top-1={meta[\'accuracy\'][\'XGBoost\'][\'top1\']:.1%}  Top-3={meta[\'accuracy\'][\'XGBoost\'][\'top3\']:.1%}")

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

    print("\\nSample prediction (8pm, home occupied, TV off):")
    for action, prob in predict_top3(demo, model="rf"):
        print(f"  RF  → {action.replace(\'_\', \' \'):40s} {prob:.1%}")
    for action, prob in predict_top3(demo, model="xgb"):
        print(f"  XGB → {action.replace(\'_\', \' \'):40s} {prob:.1%}")
''')
print(f"  Saved loader script: {loader_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
total_mb = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) / 1e6
               for f in os.listdir(SAVE_DIR))
print(f"\n{'='*52}")
print(f"All files saved to: {SAVE_DIR}")
print(f"Total size: {total_mb:.1f} MB")
print(f"\nTo reload and predict:")
print(f"  python \"{loader_path}\"")
print(f"\nFiles:")
for fn in sorted(os.listdir(SAVE_DIR)):
    fp = os.path.join(SAVE_DIR, fn)
    print(f"  {fn:<35} {os.path.getsize(fp)/1e6:>6.1f} MB")
print("DONE")
