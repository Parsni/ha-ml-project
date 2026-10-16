"""
Full 3-model run: Random Forest + XGBoost + LightGBM
Generates all comparison charts and saves all three models.
"""
import sys, os, importlib, pickle, json
sys.path.insert(0, r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace')
import ha_ml_trainer; importlib.reload(ha_ml_trainer)
from ha_ml_trainer import load_and_featurize

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, top_k_accuracy_score,
                              classification_report, confusion_matrix)
import xgboost as xgb_lib
import lightgbm as lgb_lib
from collections import Counter

DATASET  = r'C:\Users\Shrimay Dhankar\.kiro\crew\uploads\c90946c0ea3a4b56a91dc5b12035ecf6_home_assistant_interactions_combined.jsonl'
OUT_DIR  = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ml_results'
SAVE_DIR = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ha_model'

BG, CARD, BORDER = "#0e1420", "#151d2c", "#26314a"
ACCENT, ACCENT2, TEXT, MUTED, WARN = "#22c9a4", "#4f8ef7", "#e6edf7", "#a9b6cc", "#f7b32b"
PURPLE = "#c084fc"

# ── Load ──────────────────────────────────────────────────────────────────────
print("=== Full 3-Model Run: RF + XGBoost + LightGBM ===")
print("Loading dataset...")
df, labels = load_and_featurize(DATASET, max_rows=10000)
le    = LabelEncoder()
y     = le.fit_transform(labels)
X     = df.values.astype("float32")
n_cls = len(le.classes_)
names = df.columns.tolist()

can_strat = all(v >= 2 for v in Counter(y).values())
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if can_strat else None)
print(f"  {len(df)} samples | {df.shape[1]} features | {n_cls} classes "
      f"| {len(X_tr)} train / {len(X_te)} test")

def pad(raw, mc, n):
    if raw.shape[1] == n:
        return raw
    out = np.zeros((raw.shape[0], n), "float32")
    for i, c in enumerate(mc):
        out[:, c] = raw[:, i]
    return out

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Random Forest (150 trees)...")
rf = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                             max_depth=None, random_state=42)
rf.fit(X_tr, y_tr)
rf_pred  = rf.predict(X_te)
rf_proba = pad(rf.predict_proba(X_te), rf.classes_, n_cls)
rf_acc   = accuracy_score(y_te, rf_pred)
rf_top3  = top_k_accuracy_score(y_te, rf_proba, k=min(3, n_cls),
                                 labels=list(range(n_cls)))
print(f"  RF  -> Top-1: {rf_acc:.1%}  Top-3: {rf_top3:.1%}")

print("Training XGBoost (150 trees)...")
xgbm = xgb_lib.XGBClassifier(n_estimators=150, max_depth=8,
                               learning_rate=0.1, use_label_encoder=False,
                               eval_metric="mlogloss", n_jobs=-1,
                               random_state=42, verbosity=0)
xgbm.fit(X_tr, y_tr)
xg_pred  = xgbm.predict(X_te)
xg_proba = pad(xgbm.predict_proba(X_te), xgbm.classes_, n_cls)
xg_acc   = accuracy_score(y_te, xg_pred)
xg_top3  = top_k_accuracy_score(y_te, xg_proba, k=min(3, n_cls),
                                 labels=list(range(n_cls)))
print(f"  XGB -> Top-1: {xg_acc:.1%}  Top-3: {xg_top3:.1%}")

print("Training LightGBM (150 trees)...")
lgbm = lgb_lib.LGBMClassifier(n_estimators=150, max_depth=8,
                                learning_rate=0.1, num_leaves=63,
                                n_jobs=-1, random_state=42, verbose=-1)
lgbm.fit(X_tr, y_tr)
lg_pred  = lgbm.predict(X_te)
lg_proba = pad(lgbm.predict_proba(X_te), lgbm.classes_, n_cls)
lg_acc   = accuracy_score(y_te, lg_pred)
lg_top3  = top_k_accuracy_score(y_te, lg_proba, k=min(3, n_cls),
                                 labels=list(range(n_cls)))
print(f"  LGB -> Top-1: {lg_acc:.1%}  Top-3: {lg_top3:.1%}")

results = {
    "Random Forest": dict(pred=rf_pred, proba=rf_proba, acc=rf_acc,
                          top3=rf_top3, imp=rf.feature_importances_, model=rf),
    "XGBoost":       dict(pred=xg_pred, proba=xg_proba, acc=xg_acc,
                          top3=xg_top3, imp=xgbm.feature_importances_, model=xgbm),
    "LightGBM":      dict(pred=lg_pred, proba=lg_proba, acc=lg_acc,
                          top3=lg_top3, imp=lgbm.feature_importances_, model=lgbm),
}
colors_map = {"Random Forest": ACCENT, "XGBoost": ACCENT2, "LightGBM": PURPLE}
mnames = list(results.keys())

# ── Chart 1: 3-model accuracy comparison ─────────────────────────────────────
print("\nSaving charts...")
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=BG)
ax.set_facecolor(CARD)
x, w = np.arange(3), 0.3
top1_vals = [results[m]["acc"]*100  for m in mnames]
top3_vals = [results[m]["top3"]*100 for m in mnames]
bcols = [colors_map[m] for m in mnames]
b1 = ax.bar(x - w/2, top1_vals, w, label="Top-1", color=bcols, zorder=3)
b2 = ax.bar(x + w/2, top3_vals, w, label="Top-3", color=bcols, alpha=0.45, zorder=3)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom",
            color=TEXT, fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(mnames, color=TEXT, fontsize=12)
ax.set_ylim(0, 75); ax.set_ylabel("Accuracy %", color=MUTED)
ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)
legend_patches = [Patch(color=ACCENT, label="Random Forest"),
                  Patch(color=ACCENT2, label="XGBoost"),
                  Patch(color=PURPLE,  label="LightGBM")]
ax.legend(handles=legend_patches, facecolor=CARD, edgecolor=BORDER,
          labelcolor=TEXT, fontsize=9)
ax.set_title("RF vs XGBoost vs LightGBM — Top-1 & Top-3 Accuracy",
             color=TEXT, fontsize=13)
ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.5, zorder=0)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "13_three_model_comparison.png")
fig.savefig(p, dpi=130, facecolor=BG); plt.close()
print(f"  Saved: {p}")

# ── Chart 2: Feature importance side-by-side ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
for ax, mname in zip(axes, mnames):
    imp     = results[mname]["imp"]
    top_idx = np.argsort(imp)[::-1][:15]
    vals    = imp[top_idx]
    ax.set_facecolor(CARD)
    ax.barh([names[i].replace("_", " ") for i in top_idx], vals,
            color=plt.cm.RdYlGn(vals / vals.max()), zorder=3)
    ax.set_title(mname, color=TEXT, fontsize=11)
    ax.set_xlabel("Importance", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.spines[:].set_color(BORDER)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
fig.suptitle("Feature Importances — RF vs XGBoost vs LightGBM (Top 15)",
             color=TEXT, fontsize=13)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "14_three_model_features.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close()
print(f"  Saved: {p}")

# ── Chart 3: Per-class F1 for best model ─────────────────────────────────────
best_name = max(results, key=lambda m: results[m]["acc"])
best_r    = results[best_name]
present_labels = sorted(set(y_te))
present_names  = [le.classes_[i] for i in present_labels]
report    = classification_report(y_te, best_r["pred"],
                                   labels=present_labels,
                                   target_names=present_names,
                                   output_dict=True, zero_division=0)
f1_scores  = [report[c]["f1-score"] for c in present_names]
sorted_idx = np.argsort(f1_scores)[::-1][:22]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(CARD)
cols = [ACCENT if f1_scores[i] >= 0.3 else WARN if f1_scores[i] >= 0.1 else "#e05c6e"
        for i in sorted_idx]
ax.barh([present_names[i].replace("_", " ") for i in sorted_idx],
        [f1_scores[i] for i in sorted_idx], color=cols, zorder=3)
ax.set_xlabel("F1 Score", color=MUTED)
ax.set_title(f"{best_name} — Per-class F1 Score", color=TEXT, fontsize=12)
ax.tick_params(colors=MUTED, labelsize=8)
ax.spines[:].set_color(BORDER)
ax.set_xlim(0, 1)
ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "15_best_model_f1.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close()
print(f"  Saved: {p}  (best model: {best_name})")

# ── Chart 4: Confusion matrix for best model ─────────────────────────────────
top_cls = [c for c, _ in Counter(y_te).most_common(18)]
mask    = np.isin(y_te, top_cls) & np.isin(best_r["pred"], top_cls)
y_t2    = y_te[mask]; y_p2 = best_r["pred"][mask]
mapping = {c: i for i, c in enumerate(top_cls)}
y_t2r   = np.vectorize(mapping.get)(y_t2)
y_p2r   = np.vectorize(mapping.get)(y_p2)
cm      = confusion_matrix(y_t2r, y_p2r)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
tlabels = [le.classes_[i].replace("_", " ") for i in top_cls]

fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ax.set_xticks(range(len(tlabels))); ax.set_yticks(range(len(tlabels)))
ax.set_xticklabels(tlabels, rotation=45, ha="right", color=MUTED, fontsize=7)
ax.set_yticklabels(tlabels, color=MUTED, fontsize=7)
ax.set_title(f"{best_name} — Confusion Matrix (top-18 classes)", color=TEXT, fontsize=12)
ax.set_xlabel("Predicted", color=MUTED); ax.set_ylabel("True", color=MUTED)
ax.spines[:].set_color(BORDER)
fig.tight_layout(pad=1.5)
p = os.path.join(OUT_DIR, "16_best_model_confusion.png")
fig.savefig(p, dpi=120, facecolor=BG); plt.close()
print(f"  Saved: {p}")

# ── Save all 3 models ─────────────────────────────────────────────────────────
print("\nSaving models...")
for mname, r in results.items():
    slug = mname.lower().replace(" ", "_")
    path = os.path.join(SAVE_DIR, f"{slug}_v3.pkl")
    with open(path, "wb") as f:
        pickle.dump(r["model"], f)
    size = os.path.getsize(path) / 1e6
    print(f"  {slug}_v3.pkl  ({size:.1f} MB)")

with open(os.path.join(SAVE_DIR, "label_encoder_v3.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(SAVE_DIR, "model_metadata_v3.json"), "w") as f:
    json.dump({
        "feature_names": names,
        "classes":       le.classes_.tolist(),
        "n_classes":     n_cls,
        "accuracy": {m: {"top1": round(results[m]["acc"], 4),
                         "top3": round(results[m]["top3"], 4)}
                     for m in results},
    }, f, indent=2)
print("  model_metadata_v3.json")

# ── Summary ───────────────────────────────────────────────────────────────────
sep = "=" * 50
print(f"\n{sep}")
print("  FINAL RESULTS")
print(sep)
row_fmt = "  {:<18} {:>8} {:>8}"
print(row_fmt.format("Model", "Top-1", "Top-3"))
print("  " + "-" * 36)
for m in mnames:
    r = results[m]
    print(row_fmt.format(m, f"{r['acc']:.1%}", f"{r['top3']:.1%}"))
print(f"\n  Samples: {len(X_tr)} train / {len(X_te)} test")
print(f"  Classes: {n_cls}   Features: {len(names)}")
print(f"  Best model: {best_name}")
print("  DONE")
