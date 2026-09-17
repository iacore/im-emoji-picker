"""Diff the ggml port's logits against the numpy reference.

Fails when the port disagrees about the top candidate, which is the contract the
pad depends on. Small numeric drift is expected: the file stores float16
weights and the two implementations reduce in different orders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGIT_TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=REPO / "models/hccr/test")
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--verify-bin", type=Path, default=REPO / "build/hccr-verify")
    parser.add_argument("--model", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    parser.add_argument("--preprocess", action="store_true", help="feed the raw rendered bitmaps through the C++ normalize()")
    parser.add_argument("--skip-run", action="store_true", help="compare an existing dump instead of running the tool")
    args = parser.parse_args()

    dump_path = args.dump or args.inputs / ("logits-raw.bin" if args.preprocess else "logits.bin")
    index = json.loads((args.inputs / "index.json").read_text(encoding="utf-8"))
    image_dir = args.inputs / ("raw" if args.preprocess else "")
    images = [image_dir / entry["file"] for entry in index]
    missing = [path for path in images if not path.exists()]
    if missing:
        raise SystemExit(f"missing test bitmaps such as {missing[0]}; run make_test_inputs.py first")

    reference = np.load(args.inputs / "logits.npy")

    if not args.skip_run:
        command = [str(args.verify_bin), "--model", str(args.model), "--dump", str(dump_path)]
        if args.preprocess:
            command.append("--preprocess")
        command += [str(path) for path in images]
        result = subprocess.run(command, check=True, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True)
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")

    actual = np.fromfile(dump_path, dtype=np.float32)
    if actual.size != reference.size:
        raise SystemExit(f"logit count mismatch: {actual.size} from ggml vs {reference.size} from numpy")
    actual = actual.reshape(reference.shape)

    stage = "preprocessing + ggml" if args.preprocess else "ggml"
    print(f"\n{stage}, {len(index)} glyphs")
    print(f"{'glyph':<8} {'max |delta|':>12} {'top-1':>9} {'top-5 overlap':>14}")
    print("-" * 48)

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

    if args.preprocess:
        # The runtime resamples the glyph itself, so bit-identical logits are not
        # the contract; agreeing decisions and undiminished accuracy are.
        if failures:
            raise SystemExit(f"{failures} glyphs disagree on the top candidate")
        print("C++ preprocessing + ggml agrees with the reference on every glyph")
        return

    if drift > LOGIT_TOLERANCE:
        raise SystemExit(f"logits drift by {drift:.3e}, above the {LOGIT_TOLERANCE} tolerance")
    if failures:
        raise SystemExit(f"{failures} glyphs disagree on the top candidate")
    print("ggml port matches the reference")


if __name__ == "__main__":
    main()
