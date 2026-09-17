"""Measure HCCR top-1/top-5 on rendered input, sweeping stroke width.

This answers the only question that decides whether a handwriting pad is worth
building: can the model read characters drawn as *strokes of a chosen width*
(what a mouse or stylus delivers) rather than scanned pen ink?

Usage:
  python3 eval_accuracy.py --minimal            # a few chars, visual dump
  python3 eval_accuracy.py --widths 4 8 16      # full sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hccr_model import MobileNetV2HCCR, normalize_bitmap, preprocess  # noqa: E402
from render_strokes import render_font, render_medians, load_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def pick_characters(charset: Path, count: int) -> list[str]:
    mapping = json.loads(charset.read_text(encoding="utf-8"))["char_to_idx"]
    ordered = sorted(mapping.items(), key=lambda item: item[1])
    step = max(1, len(ordered) // count)
    return [char for char, _ in ordered[::step]][:count]


def rank_of(probabilities: np.ndarray, thought: str, model: MobileNetV2HCCR) -> int:
    target = next(i for i, ch in model.index_to_char.items() if ch == thought)
    order = np.argsort(-probabilities)
    return int(np.where(order == target)[0][0])


def sweep(model, characters, data_dir, widths, size, wobble, examples, model_dir, report):
    print(f"{'width':>6} | {'top-1':>7} | {'top-5':>7} | {'mean rank':>9} | n")
    print("-" * 46)
    for width in widths:
        top1 = top5 = 0
        ranks: list[int] = []
        used = 0
        shown = 0
        for character in characters:
            bitmap = render_medians(load_medians(character, data_dir), size=size, stroke_width=width, wobble=wobble)
            if bitmap is None:
                continue
            probabilities = model.softmax(model.forward(preprocess(bitmap)))
            rank = rank_of(probabilities, character, model)
            ranks.append(rank)
            used += 1
            top1 += rank == 0
            top5 += rank < 5
            if rank >= 1 and shown < examples:
                hypotheses = [f"{ch} {p:.1%}" for ch, p in model.predict(bitmap, k=5)]
                print(f"   miss: wrote {character}, got " + ", ".join(hypotheses))
                shown += 1
            if report:
                stem = f"w{width}_{character}"
                np.save(model_dir / f"debug_{stem}.npy", bitmap)
        if used:
            print(f"{width:>6} | {top1 / used:>7.1%} | {top5 / used:>7.1%} | {np.mean(ranks):>9.1f} | {used}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/hccr/best.pt")
    parser.add_argument("--charset", type=Path, default=REPO / "models/hccr/ncnn/charset.json")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "computing/extension/hanzi-strokes/package")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--widths", type=int, nargs="*", default=[4, 8, 16])
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--wobble", type=float, default=0.0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--report", action="store_true", help="dump preprocessed bitmaps for inspection")
    parser.add_argument("--font-check", action="store_true", help="also score font glyphs as a pipeline sanity check")
    parser.add_argument("--minimal", action="store_true", help="few characters, dump the normalized bitmaps")
    args = parser.parse_args()

    model = MobileNetV2HCCR(args.checkpoint, args.charset)
    print(f"checkpoint: epoch={model.epoch} acc1={model.acc1} cfg.model={model.cfg.get('model')}")

    if args.font_check:
        characters = pick_characters(args.charset, args.count)
        top1 = top5 = used = 0
        for character in characters:
            bitmap = render_font(character, size=args.size)
            probabilities = model.softmax(model.forward(preprocess(bitmap)))
            rank = rank_of(probabilities, character, model)
            used += 1
            top1 += rank == 0
            top5 += rank < 5
            if rank >= 1 and used <= 6:
                print(f"   miss: wrote {character}, got " + ", ".join(f"{c} {p:.1%}" for c, p in model.predict(bitmap, k=5)))
        print(f"font sanity: top-1 {top1 / used:.1%}  top-5 {top5 / used:.1%}  (n={used})")
        return

    if args.minimal:
        characters = pick_characters(args.charset, 6)
        out = REPO / "build/hccr-debug"
        out.mkdir(parents=True, exist_ok=True)
        width = args.widths[0]
        for character in characters:
            bitmap = render_medians(load_medians(character, args.data_dir), size=args.size, stroke_width=width)
            if bitmap is None:
                print(f"{character}: no stroke data")
                continue
            from PIL import Image

            Image.fromarray(bitmap).save(out / f"{character}_raw.png")
            Image.fromarray(normalize_bitmap(bitmap)).resize((256, 256), Image.NEAREST).save(out / f"{character}_norm.png")
            print(f"{character} -> " + ", ".join(f"{c} {p:.1%}" for c, p in model.predict(bitmap, k=5)))
        print(f"images written to {out}")
        return

    sweep(model, pick_characters(args.charset, args.count), args.data_dir, args.widths, args.size, args.wobble, args.examples, REPO / "build/hccr-debug", args.report)


if __name__ == "__main__":
    main()
