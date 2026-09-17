"""Download the released HCCR model files (weights are not committed).

Model: Ismantic/Handwritten - MobileNetV2, 3755 GB2312 level-1 classes,
trained on CASIA-HWDB1.1. Apache-2.0.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "models/hccr"
BASE = "https://huggingface.co/Ismantic/Handwritten/resolve/main"

FILES = {
    "best.pt": "151f32b03c49dbed77f3b8076035d28ce050d70a1a102e12581274a44cb10460",
    "ncnn/charset.json": "31f5700027c8692122cec5281d12c6382506b36d56075331cd73718f624d4ed2",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    for name, expected in FILES.items():
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) == expected:
            print(f"ok       {name}")
            continue
        print(f"fetching {name}")
        urllib.request.urlretrieve(f"{BASE}/{name}", target)
        actual = digest(target)
        if actual != expected:
            print(f"checksum mismatch for {name}: {actual} != {expected}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
