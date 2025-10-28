"""
Smart Irrigation GUI (single-file)
- Purpose: Demonstration GUI for a smart irrigation system using a small AI model
  (Decision Tree) trained on synthetic sensor data. The GUI allows manual input
  of sensor readings (soil moisture, temperature, humidity, rainfall chance),
  runs the model to recommend irrigation (Yes/No) and suggested water amount,
  logs history, and provides a simple simulation mode.

- Requirements:
  * Python 3.8+
  * pip install scikit-learn matplotlib
  * tkinter is included with standard Python (on some Linux you may need to install python3-tk)

- How to run in VS Code:
  1. Open this file in VS Code.
  2. (Optional) create a venv: python -m venv .venv && .\.venv\Scripts\activate (Windows) OR source .venv/bin/activate (Linux/Mac)
  3. pip install scikit-learn matplotlib
  4. Run this script (F5 or `python smart_irrigation_gui.py`)

Notes about the AI model:
- We train a small Decision Tree on synthetic data inside the script. This keeps the example
  self-contained. Replace with real sensor data and a more robust model for production.
- The model predicts a binary "Irrigate" label, and we compute water amount using a simple
  regression-like heuristic (can be replaced with a regression model).

"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import random
from datetime import datetime
import math

# AI imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Optional plotting
import matplotlib.pyplot as plt

# ------------------------
# Synthetic data + Model
# ------------------------

def generate_synthetic_data(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    X = []
    y = []
    for _ in range(n):
        # soil_moisture: 0-100 (%), temp: 5-45 C, humidity: 10-100 (%), rain_prob: 0-100 (%)
        soil = rng.uniform(5, 90)
        temp = rng.uniform(5, 45)
        hum = rng.uniform(10, 100)
        rain = rng.uniform(0, 100)

        # simple expert rule (used to label): irrigate if soil dry and low rain chance and high evap
        evap = (0.05 * temp) + (0.02 * (100 - hum))
        need_irrigation = 1 if (soil < 40 and rain < 40 and evap > 2.0) else 0

        X.append([soil, temp, hum, rain])
        y.append(need_irrigation)

    return np.array(X), np.array(y)


class SmartIrrigationModel:
    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=6, random_state=0)
        self.trained = False

    def train(self, verbose=False):
        X, y = generate_synthetic_data(n=2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.trained = True
        if verbose:
            print(f"Model trained. Test accuracy: {acc:.3f}")
        return acc

    def predict(self, soil, temp, hum, rain):
        if not self.trained:
            self.train()
        features = np.array([[soil, temp, hum, rain]])
        pred = self.clf.predict(features)[0]
        # simple water amount heuristic if pred==1
        water_amount = 0.0
        if pred == 1:
            # More water if soil is drier, less if rain chance high
            base = max(0, 50 - soil) / 50.0  # 0..1
            rain_factor = max(0, (50 - rain) / 50.0)
            evap_factor = max(0, (temp - 20) / 25.0)  # higher temp -> more evap
            water_amount = 5.0 + 10.0 * base * rain_factor * (1 + evap_factor)
            water_amount = round(water_amount, 2)  # liters per square meter (example)
        return int(pred), water_amount


# ------------------------
# GUI
# ------------------------

class SmartIrrigationGUI:
    def __init__(self, root):
        self.root = root
        root.title("Smart Irrigation System - Demo")
        root.geometry("900x650")

        self.model = SmartIrrigationModel()
        # Train model in background to avoid blocking UI
        threading.Thread(target=self.model.train, args=(False,), daemon=True).start()

        self.create_widgets()
        self.auto_running = False

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Left: Controls
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        label = ttk.Label(left, text="Sensor Inputs", font=(None, 14, "bold"))
        label.pack(pady=5)

        self.soil_var = tk.DoubleVar(value=35.0)
        self.temp_var = tk.DoubleVar(value=28.0)
        self.hum_var = tk.DoubleVar(value=55.0)
        self.rain_var = tk.DoubleVar(value=10.0)

        self.add_sensor_control(left, "Soil Moisture (%)", self.soil_var, 0, 100)
        self.add_sensor_control(left, "Temperature (°C)", self.temp_var, -10, 60)
        self.add_sensor_control(left, "Humidity (%)", self.hum_var, 0, 100)
        self.add_sensor_control(left, "Rain Chance (%)", self.rain_var, 0, 100)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=10)

        self.predict_btn = ttk.Button(btn_frame, text="Predict Recommendation", command=self.run_prediction)
        self.predict_btn.grid(row=0, column=0, padx=5)

        self.auto_btn = ttk.Button(btn_frame, text="Start Auto (Sim)", command=self.toggle_auto)
        self.auto_btn.grid(row=0, column=1, padx=5)

        save_btn = ttk.Button(btn_frame, text="Plot Model Tree", command=self.plot_tree)
        save_btn.grid(row=1, column=0, columnspan=2, pady=6)

        # Right: Output and log
        right = ttk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        out_label = ttk.Label(right, text="Recommendation", font=(None, 14, "bold"))
        out_label.pack(pady=5)

        self.reco_var = tk.StringVar(value="No prediction yet")
        self.amount_var = tk.StringVar(value="--")

        reco_frame = ttk.Frame(right, padding=8, relief=tk.RIDGE)
        reco_frame.pack(fill=tk.X, padx=5)

        ttk.Label(reco_frame, text="Irrigate?", font=(None, 12)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Label(reco_frame, textvariable=self.reco_var, font=(None, 12, "bold")).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(reco_frame, text="Suggested water (L/m²)", font=(None, 12)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Label(reco_frame, textvariable=self.amount_var, font=(None, 12, "bold")).grid(row=1, column=1, sticky=tk.W)

        # Log
        log_label = ttk.Label(right, text="Event Log", font=(None, 12, "bold"))
        log_label.pack(pady=(10,0), anchor=tk.W)

        self.log_text = scrolledtext.ScrolledText(right, width=60, height=18, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom: History table
        hist_frame = ttk.Frame(self.root, padding=8)
        hist_frame.pack(fill=tk.X)
        ttk.Label(hist_frame, text="History (last 20)", font=(None, 11, "bold")).pack(anchor=tk.W)
        self.history = []

        self.tree = ttk.Treeview(hist_frame, columns=("time","soil","temp","hum","rain","irrigate","water"), show='headings')
        for col, w in [("time",160),("soil",70),("temp",70),("hum",70),("rain",70),("irrigate",80),("water",90)]:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=w)
        self.tree.pack(fill=tk.X)

    def add_sensor_control(self, parent, label_text, var, lo, hi):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=6)
        ttk.Label(f, text=label_text).pack(anchor=tk.W)
        s = ttk.Scale(f, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL)
        s.pack(fill=tk.X)
        # current value label
        val = ttk.Label(f, textvariable=var)
        val.pack(anchor=tk.E)

    def run_prediction(self):
        soil = float(self.soil_var.get())
        temp = float(self.temp_var.get())
        hum = float(self.hum_var.get())
        rain = float(self.rain_var.get())

        pred, water = self.model.predict(soil, temp, hum, rain)
        reco = "YES" if pred == 1 else "NO"
        self.reco_var.set(reco)
        self.amount_var.set(str(water) if pred == 1 else "0")

        self.log_event(soil, temp, hum, rain, reco, water)

    def log_event(self, soil, temp, hum, rain, reco, water):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{ts} | Soil:{soil:.1f}% Temp:{temp:.1f}C Hum:{hum:.1f}% Rain:{rain:.1f}% => Irrigate: {reco} Water:{water} L/m2"
        self._append_log(msg)
        self.history.insert(0, (ts, f"{soil:.1f}", f"{temp:.1f}", f"{hum:.1f}", f"{rain:.1f}", reco, f"{water}"))
        # keep last 20
        self.history = self.history[:20]
        self._refresh_history()

    def _append_log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _refresh_history(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for row in self.history:
            self.tree.insert('', tk.END, values=row)

    def toggle_auto(self):
        if not self.auto_running:
            self.auto_running = True
            self.auto_btn.config(text="Stop Auto (Sim)")
            threading.Thread(target=self._auto_loop, daemon=True).start()
        else:
            self.auto_running = False
            self.auto_btn.config(text="Start Auto (Sim)")

    def _auto_loop(self):
        self._append_log("[Auto] Simulation started")
        while self.auto_running:
            # simulate sensors with small random walk
            soil = max(3.0, min(95.0, self.soil_var.get() + random.uniform(-3, 3)))
            temp = max(-5.0, min(50.0, self.temp_var.get() + random.uniform(-1.5, 1.5)))
            hum = max(5.0, min(100.0, self.hum_var.get() + random.uniform(-4, 4)))
            rain = max(0.0, min(100.0, self.rain_var.get() + random.uniform(-10, 10)))

            # update UI (on main thread)
            self.root.after(0, lambda s=soil, t=temp, h=hum, r=rain: self._update_sensor_vars(s, t, h, r))
            # predict and log
            pred, water = self.model.predict(soil, temp, hum, rain)
            reco = "YES" if pred == 1 else "NO"
            self.root.after(0, lambda r=reco, w=water: (self.reco_var.set(r), self.amount_var.set(str(w) if r=="YES" else "0")))
            self.log_event(soil, temp, hum, rain, reco, water)

            # sleep
            time.sleep(2.5)
        self._append_log("[Auto] Simulation stopped")

    def _update_sensor_vars(self, soil, temp, hum, rain):
        self.soil_var.set(round(soil, 2))
        self.temp_var.set(round(temp, 2))
        self.hum_var.set(round(hum, 2))
        self.rain_var.set(round(rain, 2))

    def plot_tree(self):
        # quick visualization to inspect model (feature importances)
        try:
            feat_names = ['soil','temp','humidity','rain']
            importances = self.model.clf.feature_importances_
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(feat_names, importances)
            ax.set_title('Feature importances (Decision Tree)')
            ax.set_ylabel('Importance')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror('Plot error', f'Could not plot model: {e}')


if __name__ == '__main__':
    root = tk.Tk()
    app = SmartIrrigationGUI(root)
    root.mainloop()
