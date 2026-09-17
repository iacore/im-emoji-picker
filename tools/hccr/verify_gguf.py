"""Check that the exported GGUF reproduces the reference model exactly.

Recomputes the forward pass from the tensors as stored in the GGUF file (after
dequantization) and compares logits against the checkpoint-derived reference on
real rendered input. This is what licenses the ggml port to trust the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hccr_model import MobileNetV2HCCR, forward, preprocess  # noqa: E402
from render_strokes import load_medians, render_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def load_gguf_tensors(path: Path) -> tuple[dict, dict]:
    from gguf import GGUFReader, quants

    reader = GGUFReader(path)
    tensors: dict[str, list[np.ndarray]] = {}

    for tensor in reader.tensors:
        data = tensor.data
        if data.dtype == np.float16 or data.dtype == np.float32:
            flat = np.asarray(data, dtype=np.float32).reshape(-1)
        else:
            flat = quants.dequantize(data, tensor.tensor_type).astype(np.float32).reshape(-1)

        # GGUF stores dimensions in ggml order (ne[0] varies fastest), i.e. the
        # reverse of the numpy shape.
        array = flat.reshape(tuple(reversed(tensor.shape)))

        stem, _, field = tensor.name.rpartition(".")
        slot = tensors.setdefault(stem, [None, None])
        slot[0 if field == "weight" else 1] = array

    fields = reader.fields
    metadata = {
        "charset": [str(char) for char in fields["hccr.charset"].contents()],
        "num_classes": int(fields["hccr.num_classes"].contents()),
        "block_expand": [int(value) for value in fields["hccr.block_expand"].contents()],
        "block_channels": [int(value) for value in fields["hccr.block_channels"].contents()],
        "block_stride": [int(value) for value in fields["hccr.block_stride"].contents()],
    }
    return {name: (pair[0], pair[1]) for name, pair in tensors.items()}, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/hccr/best.pt")
    parser.add_argument("--charset", type=Path, default=REPO / "models/hccr/ncnn/charset.json")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "computing/extension/hanzi-strokes/package")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    reference = MobileNetV2HCCR(args.checkpoint, args.charset)
    tensors, metadata = load_gguf_tensors(args.gguf)

    assert set(tensors) == set(reference.tensors), (
        f"tensor mismatch: missing {set(reference.tensors) - set(tensors)}, extra {set(tensors) - set(reference.tensors)}"
    )
    for name, (weight, bias) in reference.tensors.items():
        got_weight, got_bias = tensors[name]
        assert got_weight.shape == weight.shape, f"{name}: shape {got_weight.shape} != {weight.shape}"
        assert got_bias.shape == bias.shape, f"{name}: bias {got_bias.shape} != {bias.shape}"

    expected_charset = [reference.index_to_char[i] for i in range(len(reference.index_to_char))]
    assert metadata["charset"] == expected_charset, "charset mismatch"
    assert metadata["num_classes"] == len(expected_charset)
    assert metadata["block_expand"] == [b["expand"] for b in reference.plan]
    assert metadata["block_channels"] == [b["out"] for b in reference.plan]
    assert metadata["block_stride"] == [b["stride"] for b in reference.plan]

    order = sorted(reference.index_to_char.items())
    step = max(1, len(order) // args.count)
    characters = [char for _, char in order[::step]][: args.count]

    worst = 0.0
    agreements = 0
    used = 0
    for character in characters:
        medians = load_medians(character, args.data_dir)
        if medians is None:
            continue
        bitmap = render_medians(medians, size=300, stroke_width=8)
        image = preprocess(bitmap)
        expected = reference.softmax(reference.forward(image))
        actual = reference.softmax(forward(image, tensors, reference.plan))
        worst = max(worst, float(np.abs(expected - actual).max()))
        agreements += int(np.argmax(expected) == np.argmax(actual))
        used += 1

    print(f"topology + charset + {len(tensors)} tensors match")
    print(f"max |softmax delta| over {used} glyphs: {worst:.3e}")
    print(f"argmax agreement: {agreements}/{used}")
    assert worst < 1e-3, "GGUF weights do not reproduce the reference output"
    assert agreements == used, "GGUF weights disagree on argmax"
    print("GGUF export verified")


if __name__ == "__main__":
    main()
