from __future__ import annotations

import argparse
from pathlib import Path


def format_bytes(data: bytes) -> str:
    hex_values = [f"0x{byte:02x}" for byte in data]
    lines = []
    for start in range(0, len(hex_values), 12):
        lines.append(", ".join(hex_values[start : start + 12]))
    return ",\n  ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a .tflite file into Arduino/ESP32 model source files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exports" / "leak_model_float.tflite",
        help="Input .tflite file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exports" / "firmware_model",
        help="Output directory for generated .h and .cpp files.",
    )
    parser.add_argument(
        "--variable-name",
        type=str,
        default="g_leak_model_data",
        help="C symbol name for the model bytes.",
    )
    args = parser.parse_args()

    model_bytes = args.input.read_bytes()
    byte_array = format_bytes(model_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header_path = args.output_dir / "leak_model_data.h"
    source_path = args.output_dir / "leak_model_data.cpp"
    length_symbol = f"{args.variable_name}_len"

    header_path.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {args.variable_name}[];",
                f"extern const unsigned int {length_symbol};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    source_path.write_text(
        "\n".join(
            [
                '#include "leak_model_data.h"',
                "",
                f"const unsigned char {args.variable_name}[] = {{",
                "  " + byte_array,
                "};",
                "",
                f"const unsigned int {length_symbol} = {len(model_bytes)};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {header_path}")
    print(f"Wrote {source_path}")


if __name__ == "__main__":
    main()
