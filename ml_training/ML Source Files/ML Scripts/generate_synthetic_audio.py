from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_utils import load_config, write_wav_mono
from synthetic_audio import synthesize_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic audio for pipeline smoke testing.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.json",
        help="Path to experiment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory where label folders will be created.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=40,
        help="Number of synthetic clips to create per raw label.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sample_rate = int(config["sample_rate"])
    target_samples = int(config["window_samples"])
    rng = np.random.default_rng(int(config["training"]["random_seed"]))

    total = 0
    for raw_label in config["raw_labels"]:
        if raw_label == "leak":
            continue
        for index in range(args.samples_per_class):
            audio = synthesize_clip(raw_label, sample_rate, target_samples, rng)
            output_path = args.output_dir / raw_label / f"synthetic_{raw_label}_{index:03d}.wav"
            write_wav_mono(output_path, audio, sample_rate)
            total += 1

    print(f"Generated {total} synthetic clips in {args.output_dir}")


if __name__ == "__main__":
    main()
