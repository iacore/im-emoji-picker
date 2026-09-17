"""NumPy reference implementation of the HCCR MobileNetV2 classifier.

This is the oracle the GGUF export and the ggml port are checked against. It
reads the released PyTorch checkpoint directly (no torch), folds BatchNorm into
the convolutions, and reproduces the model's documented preprocessing.

Model: Ismantic/Handwritten (Apache-2.0) - MobileNetV2, 1x64x64 grayscale in,
3755 GB2312 level-1 classes out, trained on CASIA-HWDB1.1.

Preprocessing (from the release's normalize.py):
  bitmap (255=paper, 0=ink) -> bbox crop -> long side to 56 -> center in
  64x64 -> invert to floating point so ink=1.0, paper=0.0.

Layout of a folded checkpoint (canonical names used everywhere downstream):
  stem.{weight,bias}
  blk.<i>.expand.{weight,bias}      # 1x1, only when expand ratio != 1
  blk.<i>.dw.{weight,bias}          # 3x3 depthwise
  blk.<i>.project.{weight,bias}     # 1x1 linear projection
  head.{weight,bias}                # 320 -> 576, 1x1, followed by ReLU6
  classifier.{weight,bias}          # 576 -> 3755
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS = 64
CONTENT = 56
FG_THRESHOLD = 220
BN_EPS = 1e-5
STEM_CHANNELS = 32
HEAD_CHANNELS = 576
NUM_CLASSES = 3755

# torchvision mobilenet_v2 inverted_residual_setting, expanded to one entry per
# `features.<i>` module (indices 1..17). (expand_ratio, out_channels, stride)
BLOCKS = (
    (1, 16, 1),
    (6, 24, 2), (6, 24, 1),
    (6, 32, 2), (6, 32, 1), (6, 32, 1),
    (6, 64, 2), (6, 64, 1), (6, 64, 1), (6, 64, 1),
    (6, 96, 1), (6, 96, 1), (6, 96, 1),
    (6, 160, 2), (6, 160, 1), (6, 160, 1),
    (6, 320, 1),
)


# ---------------------------------------------------------------- preprocessing


def normalize_bitmap(bitmap: np.ndarray) -> np.ndarray:
    """bbox crop -> long side 56 -> centered 64x64 uint8, paper white (255)."""
    if bitmap.ndim != 2:
        raise ValueError(f"expected a 2D bitmap, got shape={bitmap.shape}")

    foreground = bitmap < FG_THRESHOLD
    if not foreground.any():
        return np.full((CANVAS, CANVAS), 255, dtype=np.uint8)

    ys, xs = np.where(foreground)
    cropped = bitmap[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    height, width = cropped.shape
    scale = CONTENT / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = np.asarray(Image.fromarray(cropped, mode="L").resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)

    canvas = np.full((CANVAS, CANVAS), 255, dtype=np.uint8)
    off_y = (CANVAS - new_h) // 2
    off_x = (CANVAS - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def preprocess(bitmap: np.ndarray) -> np.ndarray:
    """uint8 paper/ink bitmap -> float32 (1, 64, 64) with ink=1.0."""
    return ((255 - normalize_bitmap(bitmap)).astype(np.float32) / 255.0)[None, :, :]


# --------------------------------------------------------------------- kernels


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1, pad: int = 1) -> np.ndarray:
    """x:(C,H,W) weight:(O,C,kh,kw), or (C,1,kh,kw) for a depthwise conv."""
    depthwise = weight.shape[1] == 1 and weight.shape[0] == x.shape[0] and weight.shape[0] > 1
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))

    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel_h, kernel_w), axis=(1, 2))
    windows = windows[:, ::stride, ::stride]

    if depthwise:
        out = np.einsum("chwkl,ckl->chw", windows, weight[:, 0])
    else:
        out = np.einsum("chwkl,ockl->ohw", windows, weight)
    return out + bias.reshape(-1, 1, 1)


def conv1x1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    return (weight[:, :, 0, 0] @ flat).reshape(-1, *x.shape[1:]) + bias.reshape(-1, 1, 1)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0)


def fold_bn(conv_weight: np.ndarray, weights: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Absorb a BatchNorm2d that follows `conv_weight` into it."""
    gamma = weights[f"{prefix}.weight"]
    beta = weights[f"{prefix}.bias"]
    mean = weights[f"{prefix}.running_mean"]
    var = weights[f"{prefix}.running_var"]
    scale = gamma / np.sqrt(var + BN_EPS)
    spread = scale.reshape(-1, *([1] * (conv_weight.ndim - 1)))
    return conv_weight * spread, beta - mean * scale


def fold_checkpoint(weights: dict) -> tuple[dict, list[dict]]:
    """Fold BN into every conv. Returns (canonical tensors, block plan)."""
    tensors: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, weight, bias):
        tensors[name] = (weight.astype(np.float32), bias.astype(np.float32))

    add("stem", *fold_bn(weights["features.0.0.weight"], weights, "features.0.1"))

    plan = []
    in_channels = STEM_CHANNELS
    for index, (expand, out_channels, stride) in enumerate(BLOCKS):
        prefix = f"features.{index + 1}.conv"
        hidden = in_channels * expand
        if expand != 1:
            add(f"blk.{index}.expand", *fold_bn(weights[f"{prefix}.0.0.weight"], weights, f"{prefix}.0.1"))
            add(f"blk.{index}.dw", *fold_bn(weights[f"{prefix}.1.0.weight"], weights, f"{prefix}.1.1"))
            add(f"blk.{index}.project", *fold_bn(weights[f"{prefix}.2.weight"], weights, f"{prefix}.3"))
        else:
            # expand==1: depthwise straight from the input; ReLU6 sits inside a
            # nested Sequential and carries no tensors of its own.
            add(f"blk.{index}.dw", *fold_bn(weights[f"{prefix}.0.0.weight"], weights, f"{prefix}.0.1"))
            add(f"blk.{index}.project", *fold_bn(weights[f"{prefix}.1.weight"], weights, f"{prefix}.2"))

        plan.append({
            "expand": expand,
            "stride": stride,
            "in": in_channels,
            "out": out_channels,
            "hidden": hidden,
            "residual": in_channels == out_channels and stride == 1,
        })
        in_channels = out_channels

    if in_channels != 320:
        raise ValueError(f"backbone ended with {in_channels} channels, expected 320")

    add("head", *fold_bn(weights["features.18.0.weight"], weights, "features.18.1"))
    add("classifier", weights["classifier.1.weight"], weights["classifier.1.bias"])
    return tensors, plan


def forward(image: np.ndarray, tensors: dict, plan: list[dict]) -> np.ndarray:
    """image: (1, 64, 64) float32 with ink=1.0 -> logits (3755,)."""
    x = relu6(conv2d(image, *tensors["stem"], stride=2, pad=1))

    for index, block in enumerate(plan):
        y = x
        if block["expand"] != 1:
            y = relu6(conv1x1(y, *tensors[f"blk.{index}.expand"]))
        y = relu6(conv2d(y, *tensors[f"blk.{index}.dw"], stride=block["stride"], pad=1))
        y = conv1x1(y, *tensors[f"blk.{index}.project"])
        x = x + y if block["residual"] else y

    x = relu6(conv1x1(x, *tensors["head"]))
    pooled = x.mean(axis=(1, 2))
    weight, bias = tensors["classifier"]
    return weight @ pooled + bias


class MobileNetV2HCCR:
    def __init__(self, checkpoint: Path, charset: Path):
        from torch_ckpt import load_checkpoint

        raw = load_checkpoint(checkpoint)
        self.epoch = raw.get("epoch")
        self.acc1 = raw.get("acc1")
        self.cfg = raw.get("cfg")
        self.tensors, self.plan = fold_checkpoint(raw["model"])

        index = json.loads(Path(charset).read_text(encoding="utf-8"))["char_to_idx"]
        self.index_to_char = {int(value): char for char, value in index.items()}
        if len(self.index_to_char) != NUM_CLASSES:
            raise ValueError(f"charset has {len(self.index_to_char)} classes, expected {NUM_CLASSES}")

    def forward(self, image: np.ndarray) -> np.ndarray:
        return forward(image, self.tensors, self.plan)

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exponentiated = np.exp(logits - logits.max())
        return exponentiated / exponentiated.sum()

    def predict(self, bitmap: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """bitmap: uint8 paper/ink -> top-k (character, probability)."""
        image = preprocess(bitmap)
        if not image.any():
            return []
        probabilities = self.softmax(self.forward(image))
        order = np.argsort(-probabilities)[:k]
        return [(self.index_to_char[int(i)], float(probabilities[i])) for i in order]
