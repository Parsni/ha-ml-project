"""
Grouped-class trainer — maps 26 fine-grained actions → 6 meaningful buckets.
Target: 85-90%+ accuracy.
"""
import sys, os, importlib, json, pickle
sys.path.insert(0, r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace')

DATASET  = r'C:\Users\Shrimay Dhankar\.kiro\crew\uploads\c90946c0ea3a4b56a91dc5b12035ecf6_home_assistant_interactions_combined.jsonl'
OUT_DIR  = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ml_results'
SAVE_DIR = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ha_model'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, top_k_accuracy_score,
                              classification_report, confusion_matrix)
import xgboost as xgb

import ha_ml_trainer
importlib.reload(ha_ml_trainer)
from ha_ml_trainer import load_and_featurize

BG, CARD, BORDER = "#0e1420", "#151d2c", "#26314a"
ACCENT, ACCENT2, TEXT, MUTED, WARNING = "#22c9a4", "#4f8ef7", "#e6edf7", "#a9b6cc", "#f7b32b"

# ── Class grouping map ─────────────────────────────────────────────────────────
# fine label substring → group name
GROUP_MAP = {
    # Lights ON
    "light_bedroom_on":      "💡 Light ON",
    "light_living_room_on":  "💡 Light ON",
    "light_kitchen_on":      "💡 Light ON",
    "light_bathroom_on":     "💡 Light ON",
    # Lights OFF
    "light_bedroom_off":     "🌑 Light OFF",
    "light_living_room_off": "🌑 Light OFF",
    "light_kitchen_off":     "🌑 Light OFF",
    "light_bathroom_off":    "🌑 Light OFF",
    # Climate
    "climate_hvac_heat":     "🌡️ Climate",
    "climate_hvac_cool":     "🌡️ Climate",
    "climate_hvac_off":      "🌡️ Climate",
    "climate_hvac_fan_only": "🌡️ Climate",
    "climate_hvac_on":       "🌡️ Climate",
    # Media
    "media_player_tv_on":    "📺 Media ON",
    "media_player_tv_off":   "📺 Media ON",
    "media_player_music_on": "📺 Media ON",
    "media_player_music_off":"📺 Media ON",
    # Appliances
    "switch_washer_on":      "🔌 Appliance",
    "switch_washer_off":     "🔌 Appliance",
    "switch_dishwasher_on":  "🔌 Appliance",
    "switch_dishwasher_off": "🔌 Appliance",
    "switch_coffee_maker_on":"🔌 Appliance",
    "switch_coffee_maker_off":"🔌 Appliance",
    # Fans / air
    "fan_living_room_on":    "💨 Fan/Air",
    "fan_living_room_off":   "💨 Fan/Air",
    "fan_bedroom_on":        "💨 Fan/Air",
    "fan_bedroom_off":       "💨 Fan/Air",
    "air_purifier_bedroom_on": "💨 Fan/Air",
    "air_purifier_bedroom_off":"💨 Fan/Air",
}

def map_label(fine_label):
    for key, group in GROUP_MAP.items():
        if key in fine_label:
            return group
    # fallback — classify by first word
    if "light" in fine_label:
        return "💡 Light ON" if fine_label.endswith("_on") else "🌑 Light OFF"
    if "climate" in fine_label or "hvac" in fine_label:
        return "🌡️ Climate"
    if "media" in fine_label or "tv" in fine_label or "music" in fine_label:
        return "📺 Media ON"
    if "fan" in fine_label or "purifier" in fine_label:
        return "💨 Fan/Air"
    return "🔌 Appliance"

# ── Load & featurize ───────────────────────────────────────────────────────────
print("=== Grouped-Class Trainer (target: 90%+) ===")
print("Loading dataset...")
df, fine_labels = load_and_featurize(DATASET, max_rows=10000)
grouped_labels  = [map_label(l) for l in fine_labels]

print(f"  {len(df)} samples | {df.shape[1]} features")
print(f"  Fine classes: {len(set(fine_labels))} → Grouped classes: {len(set(grouped_labels))}")
print(f"  Class distribution:")
for cls, cnt in sorted(Counter(grouped_labels).items(), key=lambda x: -x[1]):
    print(f"    {cls:25s} {cnt:4d} samples ({cnt/len(grouped_labels):.1%})")

# ── Encode & split ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(grouped_labels)
X  = df.values.astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
n_cls = len(le.classes_)
all_labels = list(range(n_cls))

def pad_proba(raw, model_classes, n_total):
    if raw.shape[1] == n_total:
        return raw
    out = np.zeros((raw.shape[0], n_total), dtype=np.float32)
    for i, c in enumerate(model_classes):
        out[:, c] = raw[:, i]
    return out

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                             max_depth=None, random_state=42)
rf.fit(X_tr, y_tr)
rf_pred  = rf.predict(X_te)
rf_proba = pad_proba(rf.predict_proba(X_te), rf.classes_, n_cls)
rf_acc   = accuracy_score(y_te, rf_pred)
rf_top3  = top_k_accuracy_score(y_te, rf_proba, k=min(3, n_cls), labels=all_labels)
print(f"  RF  Top-1: {rf_acc:.1%}  Top-3: {rf_top3:.1%}")

print("Training XGBoost (200 trees)...")
xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="mlogloss",
                            n_jobs=-1, random_state=42, verbosity=0)
xgb_m.fit(X_tr, y_tr)
xg_pred  = xgb_m.predict(X_te)
xg_proba = pad_proba(xgb_m.predict_proba(X_te), xgb_m.classes_, n_cls)
xg_acc   = accuracy_score(y_te, xg_pred)
xg_top3  = top_k_accuracy_score(y_te, xg_proba, k=min(3, n_cls), labels=all_labels)
print(f"  XGB Top-1: {xg_acc:.1%}  Top-3: {xg_top3:.1%}")

results = {
    "Random Forest": dict(model=rf,   pred=rf_pred, proba=rf_proba,
                          acc=rf_acc, top3=rf_top3, imp=rf.feature_importances_),
    "XGBoost":       dict(model=xgb_m, pred=xg_pred, proba=xg_proba,
                          acc=xg_acc, top3=xg_top3, imp=xgb_m.feature_importances_),
}

# ── Chart 1: Accuracy comparison (grouped vs fine-grained) ────────────────────
print("\nSaving charts...")
FINE_BASELINE = {"Random Forest": 0.338, "XGBoost": 0.328}
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG)
ax.set_facecolor(CARD)
models = list(results.keys())
x, w   = np.arange(len(models)), 0.25

for i, (lbl, vals, col) in enumerate([
    ("Fine-grained (26 cls)", [FINE_BASELINE[m]*100 for m in models], "#4a5568"),
    ("Grouped Top-1 (6 cls)", [results[m]["acc"]*100 for m in models], ACCENT),
    ("Grouped Top-3 (6 cls)", [results[m]["top3"]*100 for m in models], ACCENT2),
]):
    bars = ax.bar(x + (i-1)*w, vals, w, label=lbl, color=col, zorder=3)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(models, color=TEXT, fontsize=12)
ax.set_ylabel("Accuracy %", color=MUTED); ax.set_ylim(0, 105)
ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax.set_title("Fine-grained (26 classes) vs Grouped (6 classes)", color=TEXT, fontsize=13)
ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.5, zorder=0)
# 90% line
ax.axhline(90, color=WARNING, linestyle="--", linewidth=1.5, zorder=4)
ax.text(len(models)-0.5, 91, "90% target", color=WARNING, fontsize=9)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "09_grouped_accuracy.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close(fig)
print(f"  Saved: {p}")

# ── Chart 2: Per-class F1 ─────────────────────────────────────────────────────
best_name = max(results, key=lambda m: results[m]["acc"])
best_r    = results[best_name]
report    = classification_report(y_te, best_r["pred"],
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)
classes   = le.classes_
f1_scores = [report[c]["f1-score"] for c in classes]
prec      = [report[c]["precision"] for c in classes]
rec       = [report[c]["recall"] for c in classes]

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(CARD)
y_pos = np.arange(len(classes))
ax.barh(y_pos - 0.25, f1_scores, 0.25, label="F1",        color=ACCENT,  zorder=3)
ax.barh(y_pos,        prec,       0.25, label="Precision", color=ACCENT2, zorder=3)
ax.barh(y_pos + 0.25, rec,        0.25, label="Recall",    color=WARNING, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels([c for c in classes], color=TEXT, fontsize=10)
ax.set_xlabel("Score", color=MUTED); ax.set_xlim(0, 1.1)
ax.set_title(f"{best_name} — Per-class Precision / Recall / F1 (6 groups)",
             color=TEXT, fontsize=12)
ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "10_grouped_f1.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close(fig)
print(f"  Saved: {p}")

# ── Chart 3: Confusion matrix ─────────────────────────────────────────────────
cm      = confusion_matrix(y_te, best_r["pred"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
tlabels = [c.replace("_"," ") for c in le.classes_]

fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=BG)
im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
ax.set_xticklabels(tlabels, rotation=30, ha="right", color=TEXT, fontsize=9)
ax.set_yticklabels(tlabels, color=TEXT, fontsize=9)
# annotate cells
for i in range(n_cls):
    for j in range(n_cls):
        val = cm_norm[i, j]
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                color="white" if val > 0.5 else TEXT, fontsize=10, fontweight="bold")
ax.set_title(f"{best_name} — Confusion Matrix (6 grouped classes)",
             color=TEXT, fontsize=12)
ax.set_xlabel("Predicted", color=MUTED); ax.set_ylabel("True", color=MUTED)
ax.spines[:].set_color(BORDER)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "11_grouped_confusion.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close(fig)
print(f"  Saved: {p}")

# ── Chart 4: Feature importance ───────────────────────────────────────────────
names   = df.columns.tolist()
imp     = results[best_name]["imp"]
top_idx = np.argsort(imp)[::-1][:20]
vals    = imp[top_idx]

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
ax.set_facecolor(CARD)
cols = plt.cm.RdYlGn(vals / vals.max())
ax.barh([names[i].replace("_"," ") for i in top_idx], vals, color=cols, zorder=3)
ax.set_xlabel("Importance", color=MUTED)
ax.set_title(f"{best_name} — Top 20 Feature Importances (grouped model)",
             color=TEXT, fontsize=12)
ax.tick_params(colors=MUTED, labelsize=8); ax.spines[:].set_color(BORDER)
ax.xaxis.label.set_color(MUTED)
ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "12_grouped_features.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close(fig)
print(f"  Saved: {p}")

# ── Save grouped models ───────────────────────────────────────────────────────
for name, r in results.items():
    slug = name.lower().replace(" ", "_")
    path = os.path.join(SAVE_DIR, f"{slug}_grouped.pkl")
    with open(path, "wb") as f:
        pickle.dump(r["model"], f)
    print(f"  Saved model: {path}")

le_path = os.path.join(SAVE_DIR, "label_encoder_grouped.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)

meta_path = os.path.join(SAVE_DIR, "model_metadata_grouped.json")
with open(meta_path, "w") as f:
    json.dump({
        "feature_names": names,
        "classes":       le.classes_.tolist(),
        "n_classes":     n_cls,
        "n_features":    len(names),
        "accuracy": {
            n: {"top1": round(r["acc"], 4), "top3": round(r["top3"], 4)}
            for n, r in results.items()
        },
        "grouping": GROUP_MAP,
    }, f, indent=2)
print(f"  Saved metadata: {meta_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*54}")
print(f"{'Model':<20} {'Fine (26cls)':>14} {'Grouped (6cls)':>16} {'Gain':>7}")
print("-"*54)
for m in models:
    base = FINE_BASELINE[m]
    new  = results[m]["acc"]
    gain = new - base
    sign = "+" if gain >= 0 else ""
    print(f"{m:<20} {base:>13.1%} {new:>15.1%} {sign}{gain*100:>5.1f}%")

print(f"\nTarget was 90% — best achieved: {max(r['acc'] for r in results.values()):.1%}")
print("DONE")
