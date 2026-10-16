"""
Home Assistant Next-Action Predictor
=====================================
Trains a Random Forest + XGBoost on the HA interaction dataset,
shows results in a clean Tkinter GUI with charts.

Requirements (auto-installed on first run):
  pip install scikit-learn xgboost pandas matplotlib
"""

import json, os, sys, threading, importlib, subprocess
from pathlib import Path

# ── Auto-install missing deps ──────────────────────────────────────────────────
REQUIRED = ["sklearn", "xgboost", "pandas", "matplotlib"]
def ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pip_name or pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg])

ensure("sklearn",     "scikit-learn")
ensure("xgboost",     "xgboost")
ensure("pandas",      "pandas")
ensure("matplotlib",  "matplotlib")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              top_k_accuracy_score, accuracy_score)
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

DEVICE_KEYS = [
    "light_bedroom", "light_living_room", "light_kitchen", "light_bathroom",
    "climate_hvac", "media_player_tv", "media_player_music",
    "fan_living_room", "fan_bedroom", "air_purifier_bedroom",
]

def load_and_featurize(jsonl_path: str, max_rows: int = 10000):
    rows, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            d = json.loads(line)

            # Skip samples with no triggered device (no label)
            td = d.get("triggered_device")
            if not td:
                continue
            label = f"{td['entity_id'].replace('.','_')}_{td['action']}"
            labels.append(label)

            tf  = d["temporal_features"]
            occ = d["occupancy"]
            ctx = d["contextual_features"]
            cf  = d["correlation_flags"]
            hs  = d["home_state_before"]
            dus = d.get("device_usage_stats", {})
            rh  = d.get("recent_history", [])

            h = tf["hour"]
            m = tf["minute"]
            dow = tf["day_of_week"]

            row = {
                # ── Cyclical time encoding (avoids 23→0 discontinuity) ──────
                "hour_sin":       np.sin(2 * np.pi * h / 24),
                "hour_cos":       np.cos(2 * np.pi * h / 24),
                "minute_sin":     np.sin(2 * np.pi * m / 60),
                "minute_cos":     np.cos(2 * np.pi * m / 60),
                "dow_sin":        np.sin(2 * np.pi * dow / 7),
                "dow_cos":        np.cos(2 * np.pi * dow / 7),
                # raw time (keep for tree splits)
                "hour":           h,
                "minute":         m,
                "dow":            dow,
                # time-of-day flags
                "is_morning":     int(tf["is_morning"]),
                "is_afternoon":   int(tf["is_afternoon"]),
                "is_evening":     int(tf["is_evening"]),
                "is_night":       int(tf["is_night"]),
                "is_weekday":     int(tf["is_weekday"]),
                # occupancy
                "is_occupied":    int(occ["is_occupied"]),
                "occ_confidence": occ["occupancy_confidence"],
                "people_count":   occ["people_count"],
                # context / weather
                "outdoor_temp":   ctx["outdoor_temperature"],
                "outdoor_hum":    ctx["outdoor_humidity"],
                "is_raining":     int(ctx["is_raining"]),
                "is_sunny":       int(ctx["is_sunny"]),
                "ambient_light":  ctx["ambient_light_level"],
                "last_away_min":  ctx["last_away_duration_minutes"],
                "days_away":      ctx["days_since_last_away"],
                # ── Derived context features ────────────────────────────────
                "temp_humidity_index": ctx["outdoor_temperature"] * ctx["outdoor_humidity"] / 100,
                "just_arrived":   int(ctx["last_away_duration_minutes"] > 0 and
                                      ctx["days_since_last_away"] == 0),
                # correlation flags
                "morning_routine":   int(cf["follows_morning_routine"]),
                "evening_routine":   int(cf["follows_evening_routine"]),
                "entertainment":     int(cf["is_entertainment_time"]),
                "away_mode":         int(cf["is_away_mode_active"]),
                "sleep_mode":        int(cf["is_sleep_mode_active"]),
                # ── How many devices are currently on ──────────────────────
                "lights_on_count":   0,
                "total_devices_on":  0,
            }

            # device states (on/off + attributes)
            lights_on = 0
            total_on  = 0
            for eid, es in hs.items():
                key    = eid.replace(".", "_").replace(" ", "_")
                is_on  = 1 if es["state"] in ("on","heating","cooling","playing") else 0
                row[f"{key}_on"] = is_on
                total_on += is_on
                if "light" in eid:
                    lights_on += is_on
                attrs = es.get("attributes", {})
                for attr, feat in [("brightness","brightness"),
                                    ("color_temp","color_temp"),
                                    ("current_temperature","cur_temp"),
                                    ("target_temperature","tgt_temp"),
                                    ("volume_level","volume"),
                                    ("speed","speed"),
                                    ("aqi","aqi"),
                                    ("humidity","humidity")]:
                    if attrs.get(attr) is not None:
                        row[f"{key}_{feat}"] = attrs[attr]
                # temp delta (how far off target the HVAC is)
                if "climate" in eid and attrs.get("current_temperature") and attrs.get("target_temperature"):
                    row["hvac_temp_delta"] = attrs["target_temperature"] - attrs["current_temperature"]
            row["lights_on_count"]  = lights_on
            row["total_devices_on"] = total_on

            # ── Device usage stats (NEW) ────────────────────────────────────
            for dk in DEVICE_KEYS:
                stats = dus.get(dk, {})
                row[f"{dk}_uses_today"]     = stats.get("times_used_today", 0) or 0
                row[f"{dk}_avg_dur"]        = stats.get("avg_duration_minutes", 0) or 0
                row[f"{dk}_first_use_hour"] = stats.get("first_use_hour") or -1
                row[f"{dk}_last_use_hour"]  = stats.get("last_use_hour") or -1

            # ── Recent history features (NEW) ───────────────────────────────
            # Last 3 events: one-hot encode which device + elapsed time
            row["rh_len"] = len(rh)
            for rank, event in enumerate(rh[:3]):
                prefix = f"rh{rank}"
                eid    = event.get("entity_id", "").replace(".", "_")
                action = event.get("action", "")
                row[f"{prefix}_elapsed_s"]  = event.get("elapsed_seconds", 9999)
                row[f"{prefix}_action_on"]  = int(action == "on")
                # which device category
                for cat in ["light","climate","media_player","fan","air_purifier","switch"]:
                    row[f"{prefix}_is_{cat}"] = int(cat in eid)

            rows.append(row)

    df = pd.DataFrame(rows).fillna(0)
    return df, labels


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_models(df, labels, n_estimators=200, progress_cb=None):
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    X  = df.values.astype(np.float32)

    # Use stratify only when every class has >=2 members
    from collections import Counter
    cls_counts = Counter(y)
    can_stratify = all(v >= 2 for v in cls_counts.values())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None)

    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────────
    if progress_cb: progress_cb("Training Random Forest…", 20)
    n_cls = len(le.classes_)
    all_labels = list(range(n_cls))

    def _pad_proba(proba_raw, model_classes, n_total):
        """Expand proba matrix to full n_total columns (some classes may be missing from train split)."""
        if proba_raw.shape[1] == n_total:
            return proba_raw
        out = np.zeros((proba_raw.shape[0], n_total), dtype=np.float32)
        for i, cls in enumerate(model_classes):
            out[:, cls] = proba_raw[:, i]
        return out

    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                 max_depth=20, random_state=42, verbose=0)
    rf.fit(X_tr, y_tr)
    rf_pred      = rf.predict(X_te)
    rf_proba     = _pad_proba(rf.predict_proba(X_te), rf.classes_, n_cls)
    rf_acc       = accuracy_score(y_te, rf_pred)
    rf_top3      = top_k_accuracy_score(y_te, rf_proba, k=min(3, n_cls), labels=all_labels)
    results["Random Forest"] = dict(
        model=rf, pred=rf_pred, proba=rf_proba,
        acc=rf_acc, top3=rf_top3,
        importances=rf.feature_importances_,
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if progress_cb: progress_cb("Training XGBoost…", 60)
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=8,
        learning_rate=0.1, use_label_encoder=False,
        eval_metric="mlogloss", n_jobs=-1, random_state=42, verbosity=0)
    xgb_model.fit(X_tr, y_tr)
    xg_pred      = xgb_model.predict(X_te)
    xg_proba     = _pad_proba(xgb_model.predict_proba(X_te), xgb_model.classes_, n_cls)
    xg_acc       = accuracy_score(y_te, xg_pred)
    xg_top3      = top_k_accuracy_score(y_te, xg_proba, k=min(3, n_cls), labels=all_labels)
    results["XGBoost"] = dict(
        model=xgb_model, pred=xg_pred, proba=xg_proba,
        acc=xg_acc, top3=xg_top3,
        importances=xgb_model.feature_importances_,
    )

    if progress_cb: progress_cb("Done!", 100)

    meta = dict(le=le, feature_names=df.columns.tolist(),
                X_te=X_te, y_te=y_te, X_tr=X_tr, y_tr=y_tr,
                n_train=len(X_tr), n_test=len(X_te),
                n_classes=len(le.classes_))
    return results, meta


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════

BG        = "#0e1420"
CARD      = "#151d2c"
BORDER    = "#26314a"
ACCENT    = "#22c9a4"
ACCENT2   = "#4f8ef7"
TEXT      = "#e6edf7"
MUTED     = "#a9b6cc"
WARNING   = "#f7b32b"

class HATrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🏠 Home Assistant ML Trainer")
        self.geometry("1180x820")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.dataset_path = tk.StringVar()
        self.n_estimators = tk.IntVar(value=150)
        self.max_rows     = tk.IntVar(value=8000)
        self.status_var   = tk.StringVar(value="Load a dataset to begin.")
        self.progress_var = tk.DoubleVar(value=0)

        self._results = None
        self._meta    = None
        self._df      = None
        self._labels  = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG, pady=14)
        top.pack(fill="x", padx=24)

        tk.Label(top, text="🏠 Home Assistant Next-Action Predictor",
                 bg=BG, fg=TEXT, font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(top, text="  ML Training & Evaluation Dashboard",
                 bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(side="left", pady=4)

        # ── control panel ─────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=CARD, bd=0, relief="flat",
                        highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill="x", padx=24, pady=(0, 14))

        inner = tk.Frame(ctrl, bg=CARD, pady=12, padx=16)
        inner.pack(fill="x")

        # Dataset path
        tk.Label(inner, text="Dataset (.jsonl):", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", padx=(0,8))
        entry = tk.Entry(inner, textvariable=self.dataset_path, bg="#1d2b45",
                         fg=TEXT, insertbackground=TEXT, relief="flat",
                         font=("Segoe UI", 10), width=60)
        entry.grid(row=0, column=1, sticky="ew", padx=(0,8))
        tk.Button(inner, text="Browse…", bg=ACCENT2, fg="#0e1420",
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  command=self._browse, padx=10).grid(row=0, column=2, padx=(0,16))

        # Hyper-params
        tk.Label(inner, text="Trees:", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).grid(row=0, column=3, padx=(0,6))
        tk.Spinbox(inner, from_=50, to=500, increment=50,
                   textvariable=self.n_estimators, bg="#1d2b45", fg=TEXT,
                   buttonbackground=BORDER, relief="flat",
                   font=("Segoe UI", 10), width=6).grid(row=0, column=4, padx=(0,16))

        tk.Label(inner, text="Max rows:", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).grid(row=0, column=5, padx=(0,6))
        tk.Spinbox(inner, from_=500, to=10000, increment=500,
                   textvariable=self.max_rows, bg="#1d2b45", fg=TEXT,
                   buttonbackground=BORDER, relief="flat",
                   font=("Segoe UI", 10), width=7).grid(row=0, column=6, padx=(0,16))

        self.train_btn = tk.Button(
            inner, text="▶  Train Models", bg=ACCENT, fg="#0e1420",
            font=("Segoe UI", 10, "bold"), relief="flat",
            command=self._start_training, padx=16, pady=4)
        self.train_btn.grid(row=0, column=7, padx=(0,8))

        inner.columnconfigure(1, weight=1)

        # Progress bar
        prog_frame = tk.Frame(ctrl, bg=CARD, padx=16, pady=6)
        prog_frame.pack(fill="x")
        self.prog_bar = ttk.Progressbar(prog_frame, variable=self.progress_var,
                                         maximum=100, length=400)
        self.prog_bar.pack(side="left", fill="x", expand=True, padx=(0,12))
        tk.Label(prog_frame, textvariable=self.status_var, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")

        # ── notebook (tabs) ───────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                        padding=[12, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        self.nb = ttk.Notebook(self, style="TNotebook")
        self.nb.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        self.tab_overview  = self._make_tab("📊 Overview")
        self.tab_accuracy  = self._make_tab("🎯 Accuracy")
        self.tab_features  = self._make_tab("🔑 Feature Importance")
        self.tab_confusion = self._make_tab("🔥 Confusion Matrix")
        self.tab_predict   = self._make_tab("🤖 Live Predict")

        self._build_overview_tab()
        self._build_accuracy_tab()
        self._build_features_tab()
        self._build_confusion_tab()
        self._build_predict_tab()

    def _make_tab(self, title):
        frame = tk.Frame(self.nb, bg=BG)
        self.nb.add(frame, text=title)
        return frame

    # ── placeholder until training done ──────────────────────────────────────
    def _placeholder(self, parent, msg="Train a model to see results here."):
        for w in parent.winfo_children():
            w.destroy()
        tk.Label(parent, text=msg, bg=BG, fg=MUTED,
                 font=("Segoe UI", 13)).pack(expand=True)

    # ── Browse ─────────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select JSONL dataset",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")])
        if path:
            self.dataset_path.set(path)

    # ── Training thread ────────────────────────────────────────────────────────
    def _start_training(self):
        path = self.dataset_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("Error", "Please select a valid .jsonl file first.")
            return
        self.train_btn.config(state="disabled")
        self.status_var.set("Loading dataset…")
        self.progress_var.set(5)
        threading.Thread(target=self._train_worker, args=(path,), daemon=True).start()

    def _train_worker(self, path):
        try:
            self._update_status("Parsing & featurizing…", 10)
            df, labels = load_and_featurize(path, max_rows=self.max_rows.get())
            self._df, self._labels = df, labels

            results, meta = train_models(
                df, labels,
                n_estimators=self.n_estimators.get(),
                progress_cb=self._update_status)

            self._results = results
            self._meta    = meta
            self.after(0, self._render_all)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.after(0, lambda: self.train_btn.config(state="normal"))

    def _update_status(self, msg, pct):
        self.after(0, lambda: self.status_var.set(msg))
        self.after(0, lambda: self.progress_var.set(pct))

    # ── Render all tabs ────────────────────────────────────────────────────────
    def _render_all(self):
        self._render_overview()
        self._render_accuracy()
        self._render_features()
        self._render_confusion()
        self._render_predict()
        self.train_btn.config(state="normal")
        self.nb.select(0)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB BUILDERS (empty state)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_overview_tab(self):
        self._placeholder(self.tab_overview)

    def _build_accuracy_tab(self):
        self._placeholder(self.tab_accuracy)

    def _build_features_tab(self):
        self._placeholder(self.tab_features)

    def _build_confusion_tab(self):
        self._placeholder(self.tab_confusion)

    def _build_predict_tab(self):
        self._placeholder(self.tab_predict,
                          "Train a model, then use controls here to predict.")

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDER: OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    def _render_overview(self):
        parent = self.tab_overview
        for w in parent.winfo_children():
            w.destroy()

        meta = self._meta
        res  = self._results

        # KPI cards row
        kpi_data = [
            ("Samples",   f"{meta['n_train']+meta['n_test']:,}"),
            ("Train set", f"{meta['n_train']:,}"),
            ("Test set",  f"{meta['n_test']:,}"),
            ("Classes",   str(meta["n_classes"])),
            ("Features",  str(len(meta["feature_names"]))),
            ("RF Top-1",  f"{res['Random Forest']['acc']:.1%}"),
            ("XGB Top-1", f"{res['XGBoost']['acc']:.1%}"),
            ("RF Top-3",  f"{res['Random Forest']['top3']:.1%}"),
            ("XGB Top-3", f"{res['XGBoost']['top3']:.1%}"),
        ]
        kpi_frame = tk.Frame(parent, bg=BG)
        kpi_frame.pack(fill="x", padx=20, pady=(16, 8))
        for label, val in kpi_data:
            card = tk.Frame(kpi_frame, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            card.pack(side="left", padx=6, pady=4, ipadx=12, ipady=8, expand=True, fill="x")
            tk.Label(card, text=val, bg=CARD, fg=ACCENT,
                     font=("Segoe UI", 18, "bold")).pack()
            tk.Label(card, text=label, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 9)).pack()

        # Bar chart comparing models
        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
        ax.set_facecolor(CARD)
        models   = list(res.keys())
        top1     = [res[m]["acc"]  for m in models]
        top3     = [res[m]["top3"] for m in models]
        x        = np.arange(len(models))
        w        = 0.35
        b1 = ax.bar(x - w/2, [v*100 for v in top1], w, label="Top-1 Accuracy",
                    color=ACCENT, zorder=3)
        b2 = ax.bar(x + w/2, [v*100 for v in top3], w, label="Top-3 Accuracy",
                    color=ACCENT2, zorder=3)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(models, color=TEXT, fontsize=11)
        ax.set_ylabel("Accuracy %", color=MUTED); ax.set_ylim(0, 100)
        ax.tick_params(colors=MUTED); ax.yaxis.label.set_color(MUTED)
        ax.spines[:].set_color(BORDER)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.set_title("Model Comparison", color=TEXT, fontsize=13)
        ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.5, zorder=0)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0,16))

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDER: ACCURACY (per-class bar chart)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_accuracy(self):
        parent = self.tab_accuracy
        for w in parent.winfo_children():
            w.destroy()

        meta = self._meta
        res  = self._results

        # model selector
        sel_frame = tk.Frame(parent, bg=BG)
        sel_frame.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(sel_frame, text="Model:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        model_var = tk.StringVar(value="Random Forest")
        combo = ttk.Combobox(sel_frame, textvariable=model_var,
                             values=list(res.keys()), state="readonly", width=18)
        combo.pack(side="left", padx=8)

        chart_frame = tk.Frame(parent, bg=BG)
        chart_frame.pack(fill="both", expand=True)
        canvas_holder = [None]

        def draw(model_name):
            for w in chart_frame.winfo_children():
                w.destroy()
            r   = res[model_name]
            le  = meta["le"]
            y_t = meta["y_te"]

            report = classification_report(y_t, r["pred"],
                                           target_names=le.classes_,
                                           output_dict=True, zero_division=0)
            classes   = le.classes_
            f1_scores = [report[c]["f1-score"] for c in classes]
            sorted_idx = np.argsort(f1_scores)[::-1][:30]   # top 30

            fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
            ax.set_facecolor(CARD)
            colors = [ACCENT if f1_scores[i] >= 0.3 else WARNING
                      if f1_scores[i] >= 0.1 else "#e05c6e"
                      for i in sorted_idx]
            ax.barh([classes[i].replace("_"," ") for i in sorted_idx],
                    [f1_scores[i] for i in sorted_idx],
                    color=colors, zorder=3)
            ax.set_xlabel("F1 Score", color=MUTED)
            ax.set_title(f"{model_name} — Per-class F1 (top 30)", color=TEXT, fontsize=12)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.spines[:].set_color(BORDER)
            ax.set_xlim(0, 1)
            ax.xaxis.label.set_color(MUTED)
            ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
            fig.tight_layout(pad=1.5)

            c = FigureCanvasTkAgg(fig, master=chart_frame)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=8)
            canvas_holder[0] = c

        combo.bind("<<ComboboxSelected>>", lambda e: draw(model_var.get()))
        draw("Random Forest")

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDER: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    def _render_features(self):
        parent = self.tab_features
        for w in parent.winfo_children():
            w.destroy()

        meta  = self._meta
        res   = self._results
        names = meta["feature_names"]

        sel_frame = tk.Frame(parent, bg=BG)
        sel_frame.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(sel_frame, text="Model:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        model_var = tk.StringVar(value="Random Forest")
        combo = ttk.Combobox(sel_frame, textvariable=model_var,
                             values=list(res.keys()), state="readonly", width=18)
        combo.pack(side="left", padx=8)

        chart_frame = tk.Frame(parent, bg=BG)
        chart_frame.pack(fill="both", expand=True)

        def draw(model_name):
            for w in chart_frame.winfo_children():
                w.destroy()
            imp     = res[model_name]["importances"]
            top_idx = np.argsort(imp)[::-1][:25]

            fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
            ax.set_facecolor(CARD)
            vals = imp[top_idx]
            cols = plt.cm.RdYlGn(vals / vals.max())
            ax.barh([names[i].replace("_", " ") for i in top_idx], vals,
                    color=cols, zorder=3)
            ax.set_xlabel("Importance", color=MUTED)
            ax.set_title(f"{model_name} — Top 25 Feature Importances",
                         color=TEXT, fontsize=12)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.spines[:].set_color(BORDER)
            ax.xaxis.label.set_color(MUTED)
            ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)
            fig.tight_layout(pad=1.5)
            c = FigureCanvasTkAgg(fig, master=chart_frame)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=8)

        combo.bind("<<ComboboxSelected>>", lambda e: draw(model_var.get()))
        draw("Random Forest")

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDER: CONFUSION MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    def _render_confusion(self):
        parent = self.tab_confusion
        for w in parent.winfo_children():
            w.destroy()

        meta = self._meta
        res  = self._results

        sel_frame = tk.Frame(parent, bg=BG)
        sel_frame.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(sel_frame, text="Model:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        model_var = tk.StringVar(value="Random Forest")
        combo = ttk.Combobox(sel_frame, textvariable=model_var,
                             values=list(res.keys()), state="readonly", width=18)
        combo.pack(side="left", padx=8)
        tk.Label(sel_frame, text="  (showing top 20 classes by frequency)",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")

        chart_frame = tk.Frame(parent, bg=BG)
        chart_frame.pack(fill="both", expand=True)

        def draw(model_name):
            for w in chart_frame.winfo_children():
                w.destroy()
            r    = res[model_name]
            le   = meta["le"]
            y_t  = meta["y_te"]
            y_p  = r["pred"]
            # pick top-20 most frequent classes in test set
            from collections import Counter
            top_cls = [c for c, _ in Counter(y_t).most_common(20)]
            mask    = np.isin(y_t, top_cls)
            y_t2    = y_t[mask]; y_p2 = y_p[mask]
            mapping = {c: i for i, c in enumerate(top_cls)}
            y_t2r   = np.vectorize(mapping.get)(y_t2)
            y_p2r   = np.vectorize(mapping.get)(y_p2)
            cm      = confusion_matrix(y_t2r, y_p2r)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            tlabels = [le.classes_[i].replace("_"," ") for i in top_cls]

            fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
            im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            ax.set_xticks(range(len(tlabels)))
            ax.set_yticks(range(len(tlabels)))
            ax.set_xticklabels(tlabels, rotation=45, ha="right",
                               color=MUTED, fontsize=7)
            ax.set_yticklabels(tlabels, color=MUTED, fontsize=7)
            ax.set_title(f"{model_name} — Confusion Matrix (top-20 classes)",
                         color=TEXT, fontsize=12)
            ax.set_xlabel("Predicted", color=MUTED)
            ax.set_ylabel("True", color=MUTED)
            ax.spines[:].set_color(BORDER)
            fig.tight_layout(pad=1.5)
            c = FigureCanvasTkAgg(fig, master=chart_frame)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=8)

        combo.bind("<<ComboboxSelected>>", lambda e: draw(model_var.get()))
        draw("Random Forest")

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDER: LIVE PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    def _render_predict(self):
        parent = self.tab_predict
        for w in parent.winfo_children():
            w.destroy()

        meta = self._meta
        res  = self._results
        le   = meta["le"]

        # ── Pick a random test sample and predict ─────────────────────────────
        info_frame = tk.Frame(parent, bg=BG)
        info_frame.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(info_frame,
                 text="Pick a random sample from the test set and see what the models predict:",
                 bg=BG, fg=TEXT, font=("Segoe UI", 11)).pack(anchor="w")

        result_frame = tk.Frame(parent, bg=BG)
        result_frame.pack(fill="both", expand=True, padx=20, pady=8)

        def predict_random():
            for w in result_frame.winfo_children():
                w.destroy()

            idx     = np.random.randint(0, len(meta["X_te"]))
            x_row   = meta["X_te"][idx:idx+1]
            true_lbl= le.classes_[meta["y_te"][idx]]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
            fig.suptitle(f"True label:  {true_lbl.replace('_',' ')}",
                         color=WARNING, fontsize=12, fontweight="bold")

            for ax, (mname, r) in zip(axes, res.items()):
                model  = r["model"]
                proba  = model.predict_proba(x_row)[0]
                top5i  = np.argsort(proba)[::-1][:8]
                top5l  = [le.classes_[i].replace("_"," ") for i in top5i]
                top5p  = proba[top5i] * 100
                colors = [ACCENT if le.classes_[top5i[i]] == true_lbl else ACCENT2
                          for i in range(len(top5i))]

                ax.set_facecolor(CARD)
                bars = ax.barh(top5l[::-1], top5p[::-1], color=colors[::-1], zorder=3)
                for bar, pct in zip(bars, top5p[::-1]):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                            f"{pct:.1f}%", va="center", color=TEXT, fontsize=8)
                ax.set_xlabel("Probability %", color=MUTED)
                ax.set_title(mname, color=TEXT, fontsize=11)
                ax.tick_params(colors=MUTED, labelsize=8)
                ax.spines[:].set_color(BORDER)
                ax.xaxis.label.set_color(MUTED)
                ax.set_xlim(0, 105)
                ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)

            fig.tight_layout(pad=1.8)
            c = FigureCanvasTkAgg(fig, master=result_frame)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True)

        tk.Button(info_frame, text="🔀  Random Sample Predict",
                  bg=ACCENT, fg="#0e1420", font=("Segoe UI", 10, "bold"),
                  relief="flat", command=predict_random, padx=14, pady=5
                  ).pack(anchor="w", pady=(8, 0))

        predict_random()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = HATrainerApp()

    # If a path was passed as CLI argument, auto-fill it
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        app.dataset_path.set(sys.argv[1])

    app.mainloop()
