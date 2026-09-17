"""Diff the ggml port's logits against the numpy reference.

Fails when the port disagrees about the top candidate, which is the contract the
pad depends on. Small numeric drift is expected: the file stores float16
weights and the two implementations reduce in different orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGIT_TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=REPO / "models/hccr/test")
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    dump_path = args.dump or args.inputs / "logits.bin"
    index = json.loads((args.inputs / "index.json").read_text(encoding="utf-8"))
    reference = np.load(args.inputs / "logits.npy")
    actual = np.fromfile(dump_path, dtype=np.float32)

    if actual.size != reference.size:
        raise SystemExit(f"logit count mismatch: {actual.size} from ggml vs {reference.size} from numpy")
    actual = actual.reshape(reference.shape)

    print(f"{'glyph':<8} {'max |delta|':>12} {'top-1':>7} {'top-5 overlap':>14}")
    print("-" * 46)

    failures = 0
    drift = 0.0
    for row, entry in enumerate(index):
        expected = reference[row]
        got = actual[row]
        delta = float(np.abs(expected - got).max())
        drift = max(drift, delta)

        expected_top = np.argsort(-expected)[:5]
        got_top = np.argsort(-got)[:5]
        overlap = len(set(expected_top.tolist()) & set(got_top.tolist()))
        agrees = int(expected_top[0] == got_top[0])

        if not agrees:
            failures += 1
        print(f"{entry['character']:<8} {delta:>12.3e} {'ok' if agrees else 'MISMATCH':>7} {overlap:>14}")

    print(f"\ntop-1 agreement: {len(index) - failures}/{len(index)}, worst |delta| {drift:.3e}")

    if drift > LOGIT_TOLERANCE:
        raise SystemExit(f"logits drift by {drift:.3e}, above the {LOGIT_TOLERANCE} tolerance")
    if failures:
        raise SystemExit(f"{failures} glyphs disagree on the top candidate")
    print("ggml port matches the reference")


if __name__ == "__main__":
    main()
