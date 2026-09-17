"""Measure the shipped C++ path (preprocessing + ggml) over a stroke-width sweep.

This is the number that matters: it renders raw ink bitmaps, hands them to
hccr-verify exactly as the pad would, and scores the returned candidates against
the character that was drawn.

  python3 eval_runtime.py --count 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from hccr_model import MobileNetV2HCCR  # noqa: E402
from render_strokes import load_medians, render_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--widths", type=int, nargs="*", default=[3, 5, 8, 12, 16])
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/hccr/best.pt")
    parser.add_argument("--charset", type=Path, default=REPO / "models/hccr/ncnn/charset.json")
    parser.add_argument("--verify-bin", type=Path, default=REPO / "build/hccr-verify")
    parser.add_argument("--model", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "computing/extension/hanzi-strokes/package")
    parser.add_argument("--out", type=Path, default=REPO / "models/hccr/test/runtime")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.png"):
        stale.unlink()

    model = MobileNetV2HCCR(args.checkpoint, args.charset)
    ordered = sorted(model.index_to_char.items())
    step = max(1, len(ordered) // args.count)
    characters = [char for _, char in ordered[::step]][: args.count]

    print(f"{'width':>6} | {'top-1':>7} | {'top-5':>7} | n")
    print("-" * 32)

    worst = 1.0
    for width in args.widths:
        truth = []
        paths = []
        for character in characters:
            medians = load_medians(character, args.data_dir)
            if medians is None:
                continue
            bitmap = render_medians(medians, size=args.size, stroke_width=width)
            path = args.out / f"w{width}_{len(paths):03d}_{character}.png"
            Image.fromarray(bitmap).save(path)
            paths.append(path)
            truth.append(character)

        result = subprocess.run(
            [str(args.verify_bin), "--model", str(args.model), "--preprocess", "-k", "5", *[str(path) for path in paths]],
            check=True,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
        )

        top1 = top5 = used = 0
        rows = [line for line in result.stdout.splitlines() if "->" in line]
        for row, character in zip(rows, truth):
            candidates = [token for token in row.split("->", 1)[1].split() if not token.endswith("%")]
            used += 1
            top1 += bool(candidates) and candidates[0] == character
            top5 += character in candidates[:5]

        accuracy = top1 / used if used else 0.0
        worst = min(worst, accuracy)
        print(f"{width:>6} | {accuracy:>7.1%} | {top5 / used if used else 0:>7.1%} | {used}")

    if worst < 0.9:
        raise SystemExit(f"C++ path accuracy dropped to {worst:.1%}")
    print(f"\nC++ path holds at or above {worst:.1%} top-1 across the sweep")


if __name__ == "__main__":
    main()
