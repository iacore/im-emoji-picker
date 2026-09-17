"""Dump structure of best.pt so the ggml graph can be written against it."""

from __future__ import annotations

import sys
from pathlib import Path

from torch_ckpt import load_checkpoint

ckpt = load_checkpoint(Path(sys.argv[1] if len(sys.argv) > 1 else "model/best.pt"))

print("top-level keys:", list(ckpt.keys()))
for key, value in ckpt.items():
    if hasattr(value, "shape"):
        continue
    print(f"  {key} = {value!r}"[:400])

weights = ckpt["model"]
print(f"\ntensors: {len(weights)}")
total = 0
for name, tensor in weights.items():
    total += tensor.size
    print(f"  {name:45s} {str(tensor.shape):22s} {tensor.dtype}")
print(f"total params: {total:,}")
