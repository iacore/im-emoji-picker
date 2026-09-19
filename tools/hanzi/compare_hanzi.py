#!/usr/bin/env python3
"""Compare the ported Hanzi routes against the Python reference they came from.

The reference lives in ~/computing/lib/hanzi-handwriting and is the source of
truth for every number here; this script runs both sides on the same drawing and
reports where they agree and where they do not.

    ./compare_hanzi.py                  # both cases below
    ./compare_hanzi.py --hanzi-verify build/hanzi-verify

Two cases, because they exercise different code:

  glyph     the reference renders 谞 from a font that is not a template font and
            matches the ink (its --char path). The ink is dumped as 128x128
            float32 so both sides rank byte-identical input; any difference left
            is the port's scoring, not the rasteriser.
  strokes   a real pen trajectory of 谞 built from the 讠 and 胥 medians and
            jittered (the reference's make_test_strokes.py). This is the pad
            case: the trajectory route is exact, the component route has to name
            ⿰讠胥.

Exit status is 0 when the glyph case keeps 谞 in the field route's top three and
the stroke case puts 谞 first in the component route - the two results the
reference README reports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REFERENCE = Path(os.environ.get("HANZI_HANDWRITING_REFERENCE", "/home/user/computing/lib/hanzi-handwriting"))
GLYPH = "谞"
GLYPH_FONT = ("/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc", 2)


def reference_module(name: str):
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    return __import__(name)


def reference_routes(argv: list[str]) -> dict:
    """The reference's own output, through its command line."""
    result = subprocess.run(
        [sys.executable, "hanzi_handwriting.py", "--no-api", "--json", *argv],
        cwd=REFERENCE,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 3):
        raise SystemExit(f"reference failed ({result.returncode}): {result.stderr}")
    return json.loads(result.stdout)


def dump_glyph_ink(path: Path) -> None:
    """The reference's normalized ink for the glyph, as float32."""
    import numpy as np

    ink_module = reference_module("ink")
    gray = ink_module.render_char(GLYPH, GLYPH_FONT[0], GLYPH_FONT[1])
    ink = ink_module.normalize(gray)
    np.asarray(ink, dtype="<f4").tofile(path)


def ported_routes(binary: Path, input_flag: str, path: Path, extra: list[str], timeout: float = 3600.0) -> dict:
    result = subprocess.run(
        [str(binary), input_flag, str(path), "--json", *extra],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise SystemExit(f"hanzi-verify failed ({result.returncode}): {result.stderr}")
    if result.stderr.strip():
        print(f"    (harness said: {result.stderr.strip().splitlines()[-1]})")
    return json.loads(result.stdout)["routes"]


def characters(entries) -> list[str]:
    return [entry["char"] for entry in entries]


def rank_of(char: str, entries) -> int | None:
    chars = characters(entries)
    return chars.index(char) + 1 if char in chars else None


def report(case: str, reference: dict, ported: dict, target: str, expect: dict[str, int | None]) -> bool:
    print(f"== {case} ==")
    ok = True
    for route, bound in expect.items():
        reference_chars = characters(reference.get(route, []))
        ported_chars = characters(ported.get(route, []))
        ported_rank = rank_of(target, ported.get(route, []))
        shared = len(set(reference_chars) & set(ported_chars))
        line = f"  {route:<10} reference {' '.join(reference_chars[:10])}\n" f"  {'':<10} ported    {' '.join(ported_chars[:10])}\n"
        print(line, end="")
        print(f"  {'':<10} {target}: reference rank {rank_of(target, reference.get(route, []))}, ported rank {ported_rank}, {shared}/{len(ported_chars)} shared")
        if bound is not None and (ported_rank is None or ported_rank > bound):
            print(f"  {'':<10} FAIL: {target} must be within rank {bound} in {route}")
            ok = False
    print()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hanzi-verify", default="build/hanzi-verify", help="the harness to compare (default build/hanzi-verify)")
    parser.add_argument("--reference-strokes", default="/tmp/hw_xu.json", help="a drawn 谞, from make_test_strokes.py")
    parser.add_argument("--make-strokes", action="store_true", help="rebuild the stroke sample with the reference")
    parser.add_argument("--time", action="store_true", help="let the harness report its millisecond counts")
    args = parser.parse_args()

    binary = Path(args.hanzi_verify).resolve()
    if not binary.exists():
        raise SystemExit(f"{binary} does not exist; build it with `cmake --build build --target hanzi-verify`")

    strokes = Path(args.reference_strokes)
    if args.make_strokes or not strokes.exists():
        subprocess.run([sys.executable, "make_test_strokes.py", GLYPH, "讠", "胥", str(strokes), "--jitter", "6", "--rotate", "3"], cwd=REFERENCE, check=True)
    if not strokes.exists():
        raise SystemExit(f"{strokes} does not exist and could not be built")

    extra = ["--time"] if args.time else []
    ok = True

    with tempfile.TemporaryDirectory() as directory:
        ink = Path(directory) / "glyph.raw"
        dump_glyph_ink(ink)
        reference = reference_routes(["--char", GLYPH, "--no-compose"])
        # The glyph carries no trajectory, so only the field route can answer for
        # it; the reference's field list is the one to match.
        ported = ported_routes(binary, "--ink-raw", ink, extra)
        ok &= report("glyph (font-rendered 谞, field route)", reference, ported, GLYPH, {"field": 3})

        reference = reference_routes(["--strokes", str(strokes), "--target", GLYPH])
        ported = ported_routes(binary, "--strokes", strokes, extra)
        ok &= report("strokes (drawn 谞, all routes)", reference, ported, GLYPH, {"trajectory": None, "field": None, "composed": 1})

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
