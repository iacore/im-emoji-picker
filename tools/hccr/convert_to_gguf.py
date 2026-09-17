"""Export the folded HCCR weights to GGUF for the ggml runtime.

The charset travels inside the file, so the C++ side needs no sidecar JSON.

Usage:
  python3 convert_to_gguf.py                 # F16 weights
  python3 convert_to_gguf.py --q8            # Q8_0 weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hccr_model import CONTENT, CANVAS, FG_THRESHOLD, MobileNetV2HCCR  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARCH = "hccr"
NAME = "Ismantic/Handwritten mobilenet_v2 (CASIA-HWDB1.1)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/hccr/best.pt")
    parser.add_argument("--charset", type=Path, default=REPO / "models/hccr/ncnn/charset.json")
    parser.add_argument("--out", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    args = parser.parse_args()

    from gguf import GGMLQuantizationType as Q, GGUFWriter

    model = MobileNetV2HCCR(args.checkpoint, args.charset)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(args.out, arch=ARCH)
    writer.add_name(NAME)
    writer.add_string(f"{ARCH}.source", "https://huggingface.co/Ismantic/Handwritten")
    writer.add_string(f"{ARCH}.license", "apache-2.0")
    writer.add_uint32(f"{ARCH}.num_classes", len(model.index_to_char))
    writer.add_uint32(f"{ARCH}.canvas", CANVAS)
    writer.add_uint32(f"{ARCH}.content", CONTENT)
    writer.add_uint32(f"{ARCH}.fg_threshold", FG_THRESHOLD)
    writer.add_array(f"{ARCH}.charset", [model.index_to_char[i] for i in range(len(model.index_to_char))])
    writer.add_array(f"{ARCH}.block_expand", [block["expand"] for block in model.plan])
    writer.add_array(f"{ARCH}.block_channels", [block["out"] for block in model.plan])
    writer.add_array(f"{ARCH}.block_stride", [block["stride"] for block in model.plan])

    for name, (weight, bias) in model.tensors.items():
        # raw_dtype *declares* the buffer type, it does not convert, so the
        # float16 cast has to happen here. Q8_0 is not an option for this
        # checkpoint: classifier.weight has ne[0] = 3755, which is not a
        # multiple of the 32-element quantization block.
        writer.add_tensor(f"{name}.weight", weight.astype(np.float16), raw_dtype=Q.F16)
        writer.add_tensor(f"{name}.bias", bias, raw_dtype=Q.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1e6:.2f} MB, F16 weights + F32 biases, {len(model.tensors)} tensors)")


if __name__ == "__main__":
    main()
