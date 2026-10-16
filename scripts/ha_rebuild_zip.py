"""Rebuild HA_ML_Project.zip with all latest files."""
import zipfile, os
from pathlib import Path

WORKSPACE = Path(r"C:\Users\Shrimay Dhankar\.kiro\crew\workspace")
ZIP_PATH  = WORKSPACE / "HA_ML_Project.zip"

# (src relative to WORKSPACE, dest inside zip)
FILES = [
    # ── Colab notebook ────────────────────────────────────────────────────────
    ("HA_ML_Colab.ipynb",               "HA_ML_Colab.ipynb"),
    # ── Scripts ───────────────────────────────────────────────────────────────
    ("ha_ml_trainer.py",                "scripts/ha_ml_trainer.py"),
    ("ha_run_all_three.py",             "scripts/ha_run_all_three.py"),
    ("ha_train_grouped.py",             "scripts/ha_train_grouped.py"),
    ("ha_export_model.py",              "scripts/ha_export_model.py"),
    ("ha_predict_ui.py",                "scripts/ha_predict_ui.py"),
    # ── Model metadata + loader ───────────────────────────────────────────────
    ("ha_model/load_and_predict.py",            "model/load_and_predict.py"),
    ("ha_model/model_metadata_v3.json",         "model/model_metadata_v3.json"),
    ("ha_model/model_metadata_grouped.json",    "model/model_metadata_grouped.json"),
    # ── All result charts ─────────────────────────────────────────────────────
    ("ml_results/01_model_comparison.png",       "results/01_model_comparison.png"),
    ("ml_results/05_before_vs_after.png",        "results/05_before_vs_after.png"),
    ("ml_results/06_f1_v2.png",                  "results/06_f1_v2.png"),
    ("ml_results/07_features_v2_random_forest.png","results/07_features_rf.png"),
    ("ml_results/07_features_v2_xgboost.png",    "results/07_features_xgb.png"),
    ("ml_results/08_confusion_v2.png",           "results/08_confusion.png"),
    ("ml_results/09_grouped_accuracy.png",       "results/09_grouped_accuracy.png"),
    ("ml_results/10_grouped_f1.png",             "results/10_grouped_f1.png"),
    ("ml_results/11_grouped_confusion.png",      "results/11_grouped_confusion.png"),
    ("ml_results/12_grouped_features.png",       "results/12_grouped_features.png"),
    ("ml_results/13_three_model_comparison.png", "results/13_three_model_comparison.png"),
    ("ml_results/14_three_model_features.png",   "results/14_three_model_features.png"),
    ("ml_results/15_best_model_f1.png",          "results/15_best_model_f1.png"),
    ("ml_results/16_best_model_confusion.png",   "results/16_best_model_confusion.png"),
]

README = """# Home Assistant ML Project — v3 (RF + XGBoost + LightGBM)

## Quick Start (Google Colab — recommended, free GPU)
  1. Go to colab.research.google.com
  2. File → Upload notebook → pick HA_ML_Colab.ipynb
  3. Runtime → Change runtime type → T4 GPU
  4. Upload your .jsonl dataset when prompted
  5. Runtime → Run all

## Quick Start (Local)
  pip install scikit-learn xgboost lightgbm pandas matplotlib
  python scripts/ha_run_all_three.py    # train all 3 models + save charts
  python scripts/ha_predict_ui.py       # interactive prediction GUI

## Results (26 fine-grained classes, 125 features)
  Random Forest  Top-1: 33.0%  Top-3: 57.2%
  XGBoost        Top-1: 32.8%  Top-3: 58.0%
  LightGBM       Top-1: 32.5%  Top-3: 57.2%

  Grouped (6 classes):
  Random Forest  Top-1: 47.9%  Top-3: 85.9%

## Files
  HA_ML_Colab.ipynb    Colab notebook (all 3 models + interactive widget)
  scripts/             Python training + GUI scripts
  model/               Metadata, loader script
  results/             16 PNG charts

## Dataset required
  home_assistant_interactions_combined.jsonl (45 MB, 10k samples)
  NOT included in this zip (too large) — use your original file.
"""

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("README.txt", README)
    included = 0
    for src, dest in FILES:
        full = WORKSPACE / src
        if full.exists():
            zf.write(full, dest)
            size_kb = full.stat().st_size / 1024
            print(f"  + {dest:<55} {size_kb:>7.1f} KB")
            included += 1
        else:
            print(f"  SKIP (missing): {src}")

total_mb = ZIP_PATH.stat().st_size / 1e6
print(f"\n{included} files | {total_mb:.1f} MB total")
print(f"Saved: {ZIP_PATH}")
