# ml/generate_fake_dataset.py
# Creates a fake multi-sensor feature dataset for leak vs no-leak (CSV)
# Run: python ml/generate_fake_dataset.py

import csv
import os
import random
from pathlib import Path

SEED = 42
N_SAMPLES = 500               # total rows
LEAK_RATIO = 0.5              # 50/50 leak vs no leak
OUT_PATH = Path("ml/fake_features.csv")

# Feature columns (keep consistent with training script)
COLUMNS = [
    "thermal_mean", "thermal_max", "thermal_std", "thermal_range",
    "sound_rms", "sound_band_energy", "sound_peak_freq",
    "ultra_mean", "ultra_var", "ultra_jump",
    "label"
]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def make_row(label: int):
    """
    label: 0 = no leak, 1 = leak
    We simulate realistic-ish differences:
      - Leak: slightly cooler mean, higher thermal gradient/range, higher sound band energy, more ultrasonic instability
      - No leak: steadier ultrasonic, lower leak-band energy, smaller thermal anomalies
    """
    # Thermal (units: arbitrary "degC-like")
    if label == 1:
        thermal_mean  = random.gauss(24.0, 1.0)   # leak often slightly cooler (evaporation)
        thermal_max   = thermal_mean + abs(random.gauss(3.0, 1.0))
        thermal_std   = abs(random.gauss(1.4, 0.4))
        thermal_range = abs(random.gauss(6.0, 1.8))
    else:
        thermal_mean  = random.gauss(26.0, 1.0)
        thermal_max   = thermal_mean + abs(random.gauss(1.0, 0.5))
        thermal_std   = abs(random.gauss(0.7, 0.2))
        thermal_range = abs(random.gauss(2.5, 0.8))

    # Sound (units: arbitrary)
    # Leak: more energy in 2–8 kHz band, slightly different peak freq distribution
    if label == 1:
        sound_rms         = abs(random.gauss(0.70, 0.15))
        sound_band_energy = abs(random.gauss(0.85, 0.18))
        sound_peak_freq   = clamp(random.gauss(5000, 900), 1000, 12000)  # Hz
    else:
        sound_rms         = abs(random.gauss(0.40, 0.12))
        sound_band_energy = abs(random.gauss(0.35, 0.12))
        sound_peak_freq   = clamp(random.gauss(2500, 800), 1000, 12000)

    # Ultrasonic (units: arbitrary "cm-like")
    # Leak: reflections/instability => higher variance/jumps
    base_dist = random.gauss(60.0, 8.0)  # scanning distance
    if label == 1:
        ultra_mean = clamp(base_dist + random.gauss(0.0, 2.0), 10, 200)
        ultra_var  = abs(random.gauss(6.0, 2.0))
        ultra_jump = abs(random.gauss(10.0, 4.0))
    else:
        ultra_mean = clamp(base_dist + random.gauss(0.0, 1.0), 10, 200)
        ultra_var  = abs(random.gauss(2.0, 1.0))
        ultra_jump = abs(random.gauss(4.0, 2.0))

    return [
        thermal_mean, thermal_max, thermal_std, thermal_range,
        sound_rms, sound_band_energy, sound_peak_freq,
        ultra_mean, ultra_var, ultra_jump,
        label
    ]

def main():
    random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_leak = int(N_SAMPLES * LEAK_RATIO)
    n_noleak = N_SAMPLES - n_leak

    rows = []
    rows += [make_row(1) for _ in range(n_leak)]
    rows += [make_row(0) for _ in range(n_noleak)]
    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"Saved: {OUT_PATH}  (rows={len(rows)})")

if __name__ == "__main__":
    main()
