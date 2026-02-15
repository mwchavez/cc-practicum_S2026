# ml/train_fusion_model.py
# Trains a tiny TensorFlow fusion model on the fake dataset and exports .tflite
# Run: python ml/train_fusion_model.py

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

DATA_PATH = Path("ml/fake_features.csv")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_PATH = OUT_DIR / "leak_fusion_model.tflite"
MU_PATH = OUT_DIR / "norm_mu.csv"
SIGMA_PATH = OUT_DIR / "norm_sigma.csv"

SEED = 42
EPOCHS = 25
BATCH = 32
THRESH = 0.5

FEATURE_COLS = [
    "thermal_mean", "thermal_max", "thermal_std", "thermal_range",
    "sound_rms", "sound_band_energy", "sound_peak_freq",
    "ultra_mean", "ultra_var", "ultra_jump"
]
LABEL_COL = "label"

def load_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        X = []
        y = []
        for row in reader:
            X.append([float(row[c]) for c in FEATURE_COLS])
            y.append(float(row[LABEL_COL]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run generate_fake_dataset.py first.")

    X, y = load_csv(DATA_PATH)
    n, f = X.shape
    print(f"Loaded dataset: N={n}, features={f}")

    # Shuffle + split
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Normalize (save mu/sigma so ESP32 can use same normalization)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6

    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma

    np.savetxt(MU_PATH, mu, delimiter=",")
    np.savetxt(SIGMA_PATH, sigma, delimiter=",")
    print(f"Saved normalization: {MU_PATH}, {SIGMA_PATH}")

    # Tiny “Fusion MLP” model (good for ESP32-S3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(f,)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1
    )

    # Quick validation stats
    probs = model.predict(X_val_n, verbose=0).reshape(-1)
    preds = (probs >= THRESH).astype(np.int32)
    y_true = y_val.astype(np.int32)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))

    print("\nValidation confusion matrix:")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")
    print(f"Val accuracy (threshold {THRESH}): {acc:.3f}")

    # Export to TFLite (float model; later you can quantize)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    TFLITE_PATH.write_bytes(tflite_model)
    print(f"\nSaved TFLite model: {TFLITE_PATH}")

if __name__ == "__main__":
    main()
