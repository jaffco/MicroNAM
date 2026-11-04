#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def _to_symbol_stem(nam_path: Path) -> str:
    stem = nam_path.stem.strip()
    if not stem:
        return "NamModel"

    # Split on non-alphanumeric boundaries, then form PascalCase.
    raw_parts = [p for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    if not raw_parts:
        return "NamModel"

    parts = []
    for part in raw_parts:
        if part[0].isdigit():
            part = f"M{part}"
        parts.append(part[0].upper() + part[1:])

    return "".join(parts)


def _format_float(value: float) -> str:
    # Keep enough precision for model weights while staying compact.
    return f"{format(value, '.17g')}f"


def _extract_weights(payload: dict, target_size: int | None = None) -> list:
    """Extract weights from both A1 and A2 formats.

    Args:
        payload: Parsed JSON from .nam file
        target_size: Optional target weight count (1871 for Lite, 12146 for Full).
                    If None, returns first valid weights found.

    Returns:
        List of weights (floats)

    Raises:
        ValueError: If no valid weights found or no matching submodel
    """
    # A1 format: weights at top level
    if "weights" in payload and isinstance(payload["weights"], list) and payload["weights"]:
        weights = payload["weights"]
        if target_size is None or len(weights) == target_size:
            return weights

    # A2 SlimmableContainer format: weights in submodels
    if "config" in payload and "submodels" in payload["config"]:
        submodels = payload["config"]["submodels"]
        if submodels and isinstance(submodels, list):
            # Collect all valid submodel weights
            submodel_weights = []
            for submodel in submodels:
                if "model" not in submodel:
                    continue
                model = submodel["model"]
                if "weights" in model and isinstance(model["weights"], list):
                    submodel_weights.append(model["weights"])

            # If we have specific target size, find matching submodel
            if target_size is not None:
                for weights in submodel_weights:
                    if len(weights) == target_size:
                        return weights
                raise ValueError(
                    f"No submodel with {target_size} weights found. "
                    f"Available: {[len(w) for w in submodel_weights]}"
                )

            # No target specified - prefer Full (12146) over Lite (1871)
            for weights in submodel_weights:
                if len(weights) == 12146:
                    return weights
            for weights in submodel_weights:
                if len(weights) == 1871:
                    return weights
            if submodel_weights:
                return submodel_weights[0]  # Return first if neither size matches

    raise ValueError("No weights found in .nam file - neither top-level nor in submodels")


def convert_nam_to_header(nam_file: Path, header_file: Path) -> tuple[str, int]:
    with nam_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    weights = _extract_weights(payload)

    for idx, value in enumerate(weights):
        if not isinstance(value, (int, float)):
            raise ValueError(f"weights[{idx}] is not numeric")

    symbol_stem = _to_symbol_stem(nam_file)
    count_name = f"{symbol_stem}WeightsCount"
    weights_name = f"{symbol_stem}Weights"

    lines = [
        "#pragma once",
        "",
        f"static constexpr unsigned int {count_name} = {len(weights)};",
        f"static constexpr float {weights_name}[{count_name}] = {{",
    ]

    row = []
    for idx, value in enumerate(weights, start=1):
        row.append(_format_float(float(value)))
        if idx % 6 == 0:
            lines.append("    " + ", ".join(row) + ",")
            row = []

    if row:
        lines.append("    " + ", ".join(row) + ",")

    lines.append("};")
    lines.append("")

    header_file.parent.mkdir(parents=True, exist_ok=True)
    header_file.write_text("\n".join(lines), encoding="utf-8")

    return weights_name, len(weights)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Neural Amp Modeler .nam JSON file into a C++ header "
            "with static constexpr weight symbols."
        )
    )
    parser.add_argument("input_nam", type=Path, help="Path to input .nam file")
    parser.add_argument("output_header", type=Path, help="Path to output .h file")
    args = parser.parse_args()

    if not args.input_nam.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_nam}")

    weights_name, count = convert_nam_to_header(args.input_nam, args.output_header)
    print(f"Generated {args.output_header} with {count} weights in symbol {weights_name}.")


if __name__ == "__main__":
    main()
