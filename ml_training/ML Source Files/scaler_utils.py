from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_standard_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(features, axis=0).astype(np.float32)
    stds = np.std(features, axis=0).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


def apply_standard_scaler(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((features - means) / stds).astype(np.float32)


def save_scaler_json(feature_names: list[str], means: np.ndarray, stds: np.ndarray, output_path: Path) -> None:
    payload = {
        "feature_names": feature_names,
        "means": [float(value) for value in means],
        "stds": [float(value) for value in stds],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scaler_json(input_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    feature_names = list(payload["feature_names"])
    means = np.asarray(payload["means"], dtype=np.float32)
    stds = np.asarray(payload["stds"], dtype=np.float32)
    return feature_names, means, stds


def _format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(value):.8f}f" for value in values)


def write_scaler_header(feature_names: list[str], means: np.ndarray, stds: np.ndarray, output_path: Path) -> None:
    lines = [
        "#pragma once",
        "",
        f"#define FEATURE_COUNT {len(feature_names)}",
        "",
        "// Feature order used by the ESP32 model.",
    ]

    for index, feature_name in enumerate(feature_names):
        lines.append(f"// {index}: {feature_name}")

    lines.extend(
        [
            "",
            f"static const float FEATURE_MEANS[FEATURE_COUNT] = {{{_format_float_array(means)}}};",
            f"static const float FEATURE_STDS[FEATURE_COUNT] = {{{_format_float_array(stds)}}};",
            "",
            "inline void normalize_features(const float* input, float* output) {",
            "  for (int i = 0; i < FEATURE_COUNT; ++i) {",
            "    output[i] = (input[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];",
            "  }",
            "}",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
