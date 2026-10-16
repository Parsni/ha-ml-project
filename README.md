# 🏠 Home Assistant Next-Action Predictor

ML pipeline that predicts the next smart home device action from context.

## Models
| Model | Top-1 (26 cls) | Top-3 (26 cls) | Top-1 (6 cls) | Top-3 (6 cls) |
|---|---|---|---|---|
| Random Forest | 33.0% | 57.2% | 47.9% | 85.9% |
| XGBoost | 32.8% | 58.0% | — | — |
| LightGBM | 32.5% | 57.2% | — | — |

## Dataset
10,000 synthetic Home Assistant state-transition samples, 26 action classes, 125 features.

## Quick Start

### Google Colab (recommended)
1. Open `HA_ML_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → T4 GPU
3. Upload your `.jsonl` dataset when prompted
4. Runtime → Run all

### Local
```bash
pip install scikit-learn xgboost lightgbm pandas matplotlib
python scripts/ha_run_all_three.py   # train all 3 models
python scripts/ha_predict_ui.py      # launch prediction GUI
```

## Files
```
HA_ML_Colab.ipynb      Full Colab notebook with interactive widget
scripts/               Training + prediction scripts
models/                Trained .pkl models + metadata (via Git LFS)
results/               16 PNG charts
```

## Features (125 total)
- Cyclical time encoding (hour/minute/day sin+cos)
- Device states (10 devices × on/off + attributes)
- Device usage stats (uses today, avg duration, first/last hour)
- Recent history (last 3 triggered events + elapsed time)
- Weather & occupancy context
- Derived features (HVAC delta, lights count, humidity index)
