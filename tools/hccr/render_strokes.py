"""Render test inputs for the HCCR model.

Two sources of test glyphs:
  * stroke medians from hanzi-writer-data - these are stroke *centerlines*, so
    drawing them with a fixed width reproduces what a mouse/stylus pad produces
    far better than a font ever could;
  * a system CJK font, purely as a pipeline sanity check (font glyphs are a
    different distribution, but a correct graph must still read them).

hanzi-writer coordinates live in a 1024x1024 box with the y axis pointing up,
so rendering flips y. The caller crops/normalizes afterwards, so only the flip
direction matters here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STROKE_BOX = 1024
DEFAULT_FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"


def load_medians(character: str, data_dir: str | Path) -> list[list[tuple[float, float]]] | None:
    path = Path(data_dir) / f"{character}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["medians"]


def render_medians(
    medians: list[list[tuple[float, float]]],
    size: int = 300,
    stroke_width: int = 8,
    wobble: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Draw stroke centerlines as ink on paper. wobble simulates hand tremor (px)."""
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    scale = size / STROKE_BOX
    rng = np.random.default_rng(seed)

    for index, stroke in enumerate(medians):
        points = []
        for x, y in stroke:
            jitter = rng.normal(0.0, wobble, 2) if wobble else (0.0, 0.0)
            points.append(((x + jitter[0]) * scale, (STROKE_BOX - y + jitter[1]) * scale))
        if len(points) == 1:
            cx, cy = points[0]
            draw.ellipse((cx - stroke_width / 2, cy - stroke_width / 2, cx + stroke_width / 2, cy + stroke_width / 2), fill=0)
        else:
            draw.line(points, fill=0, width=stroke_width, joint="curve")
        if index > 500:
            break
    return np.asarray(image, dtype=np.uint8)


def render_glyph(character: str, data_dir: str | Path, size: int = 300, stroke_width: int = 8, wobble: float = 0.0, seed: int = 0):
    medians = load_medians(character, data_dir)
    if medians is None:
        return None
    return render_medians(medians, size=size, stroke_width=stroke_width, wobble=wobble, seed=seed)


def render_font(character: str, size: int = 300, font_path: str | Path = DEFAULT_FONT, fill_ratio: float = 0.72) -> np.ndarray:
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), int(size * fill_ratio))
    draw.text((size / 2, size / 2), character, font=font, fill=0, anchor="mm")
    return np.asarray(image, dtype=np.uint8)
