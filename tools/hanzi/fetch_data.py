"""Download the upstream data the hanzi routes need into models/hanzi.

Two sources, both fetched verbatim and checked against a pinned digest:

  Make Me a Hanzi medians, graphics.txt, one JSON object per line, about 30 MB.
  https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt
  The raw pen medians per character; mmah_matcher.load_dictionary preprocesses
  them into the trajectory matcher's encoding. Arphic-derived data: the tool's
  own README says to check the Make Me a Hanzi project's licence before
  redistributing it, and it is not committed here.

  CJKVI IDS table, ids.txt.
  https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt
  The Unicode ideographic description per character, from the CJKVI Database,
  which is based on the CHISE IDS Database. The reference tool downloads this
  one to ~/.cache/hanzi-handwriting on first use.

Both URLs were checked on 2026-09-19 and answered with the files the digests
below name. Run this before tools/hanzi/convert_data.py.

convert_data.py also accepts the reference checkout's preprocessed
graphics.json, which is what its matcher loads by default and what the port's
dictionary is built and checked against; that file is not published upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "models/hanzi"

# name -> (url, sha256, description)
FILES = {
    "graphics.txt": (
        "https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt",
        "a28c478b5178e98f67f510b2d52fde08a69dc664654ef43498253b9b764d46ee",
        "Make Me a Hanzi medians",
    ),
    "ids.txt": (
        "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt",
        "bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6",
        "CJKVI IDS table",
    ),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"directory to write into (default {DEFAULT_OUT})")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (url, expected, description) in FILES.items():
        target = args.out / name
        # A file that is already there and hashes correctly is left alone, so
        # reruns cost nothing and a partial download (which does not hash) is
        # replaced.
        if target.exists() and digest(target) == expected:
            print(f"ok       {name} ({description}, {target.stat().st_size} bytes)")
            continue
        print(f"fetching {name} ({description}) from {url}")
        urllib.request.urlretrieve(url, target)
        actual = digest(target)
        if actual != expected:
            print(f"checksum mismatch for {name}: {actual} != {expected}", file=sys.stderr)
            return 1
        print(f"wrote    {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
