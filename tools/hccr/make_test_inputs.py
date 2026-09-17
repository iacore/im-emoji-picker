"""Emit normalized 64x64 bitmaps plus reference logits for the ggml harness.

The bitmaps are already normalized (bbox cropped, long side 56, centered), so
the C++ tool exercises exactly the same tensor it will see at runtime, and any
disagreement is a port bug rather than a preprocessing difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from hccr_model import MobileNetV2HCCR, normalize_bitmap, preprocess  # noqa: E402
from render_strokes import load_medians, render_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "models/hccr/test")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/hccr/best.pt")
    parser.add_argument("--charset", type=Path, default=REPO / "models/hccr/ncnn/charset.json")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "computing/extension/hanzi-strokes/package")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--size", type=int, default=300)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = MobileNetV2HCCR(args.checkpoint, args.charset)

    ordered = sorted(model.index_to_char.items())
    step = max(1, len(ordered) // args.count)
    characters = [char for _, char in ordered[::step]][: args.count]

    logits = []
    index = []
    for character in characters:
        medians = load_medians(character, args.data_dir)
        if medians is None:
            continue
        normalized = normalize_bitmap(render_medians(medians, size=args.size, stroke_width=args.width))
        name = f"{len(index):02d}_{character}.png"
        Image.fromarray(normalized).save(args.out / name)
        logits.append(model.forward(preprocess(normalized)))
        index.append({"file": name, "character": character})

    np.save(args.out / "logits.npy", np.stack(logits))
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(index)} bitmaps and logits.npy to {args.out}")


if __name__ == "__main__":
    main()
