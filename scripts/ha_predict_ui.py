"""
Home Assistant — Live Prediction Demo UI
=========================================
Loads the saved trained models (no retraining).
Lets you dial in home context and see what action both models predict.
"""
import sys, os, pickle, json
sys.path.insert(0, r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace')

MODEL_DIR = r'C:\Users\Shrimay Dhankar\.kiro\crew\workspace\ha_model'

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    return rf, xgb, le, meta

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#0e1420"
CARD   = "#151d2c"
BORDER = "#26314a"
ACCENT = "#22c9a4"
ACCENT2= "#4f8ef7"
TEXT   = "#e6edf7"
MUTED  = "#a9b6cc"
WARN   = "#f7b32b"
RED    = "#e05c6e"

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_feature_vector(controls, feature_names):
    h   = controls["hour"].get()
    m   = controls["minute"].get()
    dow = controls["dow"].get()
    vec = {f: 0.0 for f in feature_names}

    # time
    vec["hour"]        = h
    vec["minute"]      = m
    vec["dow"]         = dow
    vec["hour_sin"]    = np.sin(2*np.pi*h/24)
    vec["hour_cos"]    = np.cos(2*np.pi*h/24)
    vec["minute_sin"]  = np.sin(2*np.pi*m/60)
    vec["minute_cos"]  = np.cos(2*np.pi*m/60)
    vec["dow_sin"]     = np.sin(2*np.pi*dow/7)
    vec["dow_cos"]     = np.cos(2*np.pi*dow/7)
    vec["is_morning"]  = int(5  <= h < 12)
    vec["is_afternoon"]= int(12 <= h < 17)
    vec["is_evening"]  = int(17 <= h < 22)
    vec["is_night"]    = int(h >= 22 or h < 5)
    vec["is_weekday"]  = int(dow < 5)

    # occupancy
    vec["is_occupied"]    = controls["occupied"].get()
    vec["people_count"]   = controls["people"].get()
    vec["occ_confidence"] = 0.9 if controls["occupied"].get() else 0.1

    # weather
    vec["outdoor_temp"]  = controls["outdoor_temp"].get()
    vec["outdoor_hum"]   = controls["outdoor_hum"].get()
    vec["is_raining"]    = controls["is_raining"].get()
    vec["is_sunny"]      = controls["is_sunny"].get()
    vec["ambient_light"] = controls["ambient_light"].get()

    # device states
    for dev in ["light_bedroom", "light_living_room", "light_kitchen", "light_bathroom"]:
        k = f"{dev}_on"
        if k in vec:
            vec[k] = controls.get(dev, tk.IntVar(value=0)).get()

    vec["climate_hvac_on"]      = controls["hvac_on"].get()
    vec["climate_hvac_cur_temp"]= controls["hvac_cur"].get()
    vec["climate_hvac_tgt_temp"]= controls["hvac_tgt"].get()
    vec["hvac_temp_delta"]      = controls["hvac_tgt"].get() - controls["hvac_cur"].get()
    vec["media_player_tv_on"]   = controls["tv_on"].get()
    vec["media_player_music_on"]= controls["music_on"].get()
    vec["fan_living_room_on"]   = controls["fan_lr_on"].get()
    vec["fan_bedroom_on"]       = controls["fan_br_on"].get()
    vec["air_purifier_bedroom_on"] = controls["purifier_on"].get()

    # derived counts
    lights = sum(controls.get(d, tk.IntVar(value=0)).get()
                 for d in ["light_bedroom","light_living_room","light_kitchen","light_bathroom"])
    vec["lights_on_count"]  = lights
    vec["total_devices_on"] = lights + vec["climate_hvac_on"] + vec["media_player_tv_on"] + \
                               vec["media_player_music_on"] + vec["fan_living_room_on"] + \
                               vec["fan_bedroom_on"] + vec["air_purifier_bedroom_on"]

    # routine flags
    vec["morning_routine"] = int(5  <= h < 9  and controls["occupied"].get())
    vec["evening_routine"] = int(17 <= h < 22 and controls["occupied"].get())
    vec["entertainment"]   = int(controls["tv_on"].get() or controls["music_on"].get())
    vec["sleep_mode"]      = int(h >= 22 or h < 6)

    return np.array([vec.get(f, 0.0) for f in feature_names], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
class PredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🏠 HA Prediction Demo")
        self.geometry("1200x780")
        self.configure(bg=BG)
        self.resizable(True, True)

        # Load models
        try:
            self.rf, self.xgb_m, self.le, self.meta = load_models()
            self.feature_names = self.meta["feature_names"]
            self.classes       = self.meta["classes"]
        except FileNotFoundError as e:
            import tkinter.messagebox as mb
            mb.showerror("Models not found",
                f"Could not load models from:\n{MODEL_DIR}\n\nRun ha_export_model.py first.\n\n{e}")
            self.destroy(); return

        self.controls = {}
        self._canvas  = None
        self._fig     = None

        self._build_ui()
        self._predict()   # initial prediction on load

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG, pady=10)
        hdr.pack(fill="x", padx=22)
        tk.Label(hdr, text="🏠  Home Assistant — Live Next-Action Predictor",
                 bg=BG, fg=TEXT, font=("Segoe UI",17,"bold")).pack(side="left")
        acc_rf  = self.meta["accuracy"]["Random Forest"]["top1"]
        acc_xgb = self.meta["accuracy"]["XGBoost"]["top1"]
        tk.Label(hdr, text=f"  RF {acc_rf:.1%} | XGB {acc_xgb:.1%} top-1",
                 bg=BG, fg=MUTED, font=("Segoe UI",10)).pack(side="left", pady=4)

        # main split
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=22, pady=(0,16))
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── LEFT: controls ────────────────────────────────────────────────────
        left = tk.Frame(main, bg=CARD, width=320,
                        highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0,14))
        left.pack_propagate(False)

        canvas_scroll = tk.Canvas(left, bg=CARD, highlightthickness=0)
        scrollbar     = ttk.Scrollbar(left, orient="vertical",
                                       command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas_scroll, bg=CARD)
        canvas_scroll.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas_scroll.configure(
                       scrollregion=canvas_scroll.bbox("all")))

        self._add_controls(inner)

        # Predict button
        btn_frame = tk.Frame(left, bg=CARD, pady=8)
        btn_frame.pack(fill="x", padx=14, side="bottom")
        tk.Button(btn_frame, text="▶  Predict Now",
                  bg=ACCENT, fg="#0e1420", font=("Segoe UI",11,"bold"),
                  relief="flat", command=self._predict,
                  padx=16, pady=6).pack(fill="x")

        # ── RIGHT: chart ──────────────────────────────────────────────────────
        self.right = tk.Frame(main, bg=BG)
        self.right.grid(row=0, column=1, sticky="nsew")

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=CARD, fg=ACCENT,
                 font=("Segoe UI",10,"bold")).pack(anchor="w", padx=14, pady=(12,2))

    def _slider(self, parent, label, key, from_, to, default, resolution=1):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", padx=14, pady=2)
        var = tk.DoubleVar(value=default)
        self.controls[key] = var
        tk.Label(f, text=label, bg=CARD, fg=MUTED,
                 font=("Segoe UI",9), width=22, anchor="w").pack(side="left")
        val_lbl = tk.Label(f, textvariable=var, bg=CARD, fg=TEXT,
                           font=("Segoe UI",9), width=5)
        val_lbl.pack(side="right")
        tk.Scale(f, variable=var, from_=from_, to=to, orient="horizontal",
                 resolution=resolution, bg=CARD, fg=TEXT, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False, length=120,
                 command=lambda v: var.set(round(float(v), 2))
                 ).pack(side="left", fill="x", expand=True, padx=6)

    def _toggle(self, parent, label, key, default=0):
        f   = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", padx=14, pady=2)
        var = tk.IntVar(value=default)
        self.controls[key] = var
        cb = tk.Checkbutton(f, text=label, variable=var, bg=CARD, fg=TEXT,
                            selectcolor=BORDER, activebackground=CARD,
                            activeforeground=ACCENT, font=("Segoe UI",9))
        cb.pack(side="left")

    def _add_controls(self, parent):
        self._section(parent, "⏰  Time")
        self._slider(parent, "Hour (0–23)",    "hour",   0,  23, 18)
        self._slider(parent, "Minute (0–59)",  "minute", 0,  59, 30)
        self._slider(parent, "Day of week (0=Mon)", "dow", 0, 6, 4)

        self._section(parent, "👥  Occupancy")
        self._toggle(parent, "Home occupied",  "occupied", 1)
        self._slider(parent, "People count",   "people",   0, 5, 2)

        self._section(parent, "🌤  Weather / Context")
        self._slider(parent, "Outdoor temp (°C)", "outdoor_temp", 0,  45, 25)
        self._slider(parent, "Outdoor humidity %","outdoor_hum",  0, 100, 55)
        self._slider(parent, "Ambient light (0–1)","ambient_light",0, 1, 0.05, 0.01)
        self._toggle(parent, "Is raining",     "is_raining")
        self._toggle(parent, "Is sunny",       "is_sunny", 1)

        self._section(parent, "💡  Lights currently ON")
        self._toggle(parent, "Bedroom light",      "light_bedroom",     0)
        self._toggle(parent, "Living room light",  "light_living_room", 1)
        self._toggle(parent, "Kitchen light",      "light_kitchen",     1)
        self._toggle(parent, "Bathroom light",     "light_bathroom",    0)

        self._section(parent, "🌡️  Climate")
        self._toggle(parent, "HVAC on",        "hvac_on",  1)
        self._slider(parent, "Current temp °C","hvac_cur", 10, 35, 21)
        self._slider(parent, "Target temp °C", "hvac_tgt", 10, 35, 23)

        self._section(parent, "📺  Media")
        self._toggle(parent, "TV on",          "tv_on",    0)
        self._toggle(parent, "Music on",       "music_on", 0)

        self._section(parent, "💨  Fans & Air")
        self._toggle(parent, "Living room fan","fan_lr_on",0)
        self._toggle(parent, "Bedroom fan",    "fan_br_on",0)
        self._toggle(parent, "Air purifier",   "purifier_on",0)

        # spacer
        tk.Frame(parent, bg=CARD, height=20).pack()

    # ── Prediction & chart ────────────────────────────────────────────────────
    def _predict(self):
        x = build_feature_vector(self.controls, self.feature_names)

        # pad to full class space
        def pad(raw, model_classes):
            n = len(self.classes)
            if raw.shape[1] == n:
                return raw
            out = np.zeros((raw.shape[0], n), dtype=np.float32)
            for i, c in enumerate(model_classes):
                out[:, c] = raw[:, i]
            return out

        rf_proba  = pad(self.rf.predict_proba(x.reshape(1,-1)),  self.rf.classes_)[0]
        xgb_proba = pad(self.xgb_m.predict_proba(x.reshape(1,-1)), self.xgb_m.classes_)[0]

        self._draw_chart(rf_proba, xgb_proba)

    def _draw_chart(self, rf_proba, xgb_proba):
        # clear old
        for w in self.right.winfo_children():
            w.destroy()
        if self._fig:
            plt.close(self._fig)

        classes = self.classes
        top_n   = min(10, len(classes))

        # top indices for each model
        rf_top  = np.argsort(rf_proba)[::-1][:top_n]
        xg_top  = np.argsort(xgb_proba)[::-1][:top_n]

        self._fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=BG)
        self._fig.suptitle("Predicted Next Action", color=TEXT, fontsize=13,
                           fontweight="bold", y=0.98)

        for ax, top_idx, proba, model_name, color in [
            (axes[0], rf_top,  rf_proba,  "Random Forest", ACCENT),
            (axes[1], xg_top,  xgb_proba, "XGBoost",       ACCENT2),
        ]:
            ax.set_facecolor(CARD)
            labels = [classes[i].replace("_", " ") for i in top_idx]
            values = [proba[i] * 100 for i in top_idx]

            # gradient colors: brightest = most confident
            alphas = np.linspace(1.0, 0.35, len(values))
            bar_colors = []
            base = np.array(matplotlib.colors.to_rgb(color))
            for a in alphas:
                bar_colors.append(tuple(base * a + np.array([0.055,0.078,0.125]) * (1-a)))

            bars = ax.barh(labels[::-1], values[::-1],
                           color=bar_colors[::-1], zorder=3, height=0.6)

            # value labels
            for bar, val in zip(bars, values[::-1]):
                ax.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", color=TEXT, fontsize=9)

            ax.set_xlim(0, 105)
            ax.set_xlabel("Confidence %", color=MUTED)
            ax.set_title(model_name, color=TEXT, fontsize=11, pad=8)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.spines[:].set_color(BORDER)
            ax.xaxis.label.set_color(MUTED)
            ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4, zorder=0)

            # highlight top prediction
            bars[-1].set_edgecolor(color)
            bars[-1].set_linewidth(2)

        self._fig.tight_layout(rect=[0, 0, 1, 0.96], pad=1.8)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self.right)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # live summary strip at bottom
        rf_best  = classes[np.argmax(rf_proba)].replace("_"," ")
        xg_best  = classes[np.argmax(xgb_proba)].replace("_"," ")
        rf_conf  = rf_proba.max() * 100
        xg_conf  = xgb_proba.max() * 100

        strip = tk.Frame(self.right, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        strip.pack(fill="x", pady=(8,0))
        tk.Label(strip, text="🎯  RF predicts:", bg=CARD, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left", padx=(14,4), pady=6)
        tk.Label(strip, text=rf_best, bg=CARD, fg=ACCENT,
                 font=("Segoe UI",10,"bold")).pack(side="left")
        tk.Label(strip, text=f"  ({rf_conf:.1f}%)", bg=CARD, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left", padx=(0,20))

        tk.Label(strip, text="⚡  XGB predicts:", bg=CARD, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left", padx=(0,4))
        tk.Label(strip, text=xg_best, bg=CARD, fg=ACCENT2,
                 font=("Segoe UI",10,"bold")).pack(side="left")
        tk.Label(strip, text=f"  ({xg_conf:.1f}%)", bg=CARD, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left")

        agree_color = ACCENT if rf_best == xg_best else WARN
        agree_text  = "✓ Models agree" if rf_best == xg_best else "△ Models disagree"
        tk.Label(strip, text=agree_text, bg=CARD, fg=agree_color,
                 font=("Segoe UI",9,"bold")).pack(side="right", padx=14)


if __name__ == "__main__":
    import matplotlib
    app = PredictApp()
    app.mainloop()
