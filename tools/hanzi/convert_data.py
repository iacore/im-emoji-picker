"""Convert the upstream hanzi data into the compact tables the C++ routes read.

Two files, both written into --out (models/hanzi by default):

  hanzi-dictionary.bin  the Make Me a Hanzi medians in the trajectory matcher's
                  encoding, for StrokeDictionary. "HZMD", version 1, count, then
                  per character a uint32 code point, a uint8 stroke count and
                  that many encoded strokes of 10 uint8 values (four x,y pairs,
                  angle, length).
  hanzi-ids.bin         the CJKVI ideographic descriptions restricted to the two
                  operators the component route uses, for IdsTable. "HZID",
                  version 1, count, then per character a uint8 operator index
                  (0 = U+2FF0 left-right, 1 = U+2FF1 top-bottom), a uint16 left
                  code point, a uint16 right code point and a uint32 character.

The two encodings are not the same encoder, and this is the part worth reading
before changing anything.

The dictionary has to be the encoding the reference's matcher *loads*, which is
the shipped graphics.json, and that file came out of the original JavaScript
(to_js_encode below is that pipeline: juststrokes/mmah-vite/src/mmah.ts in the
reference). The reference's own Python mmah_matcher.preprocess does not
reproduce it: it encodes the y axis and the stroke lengths the other way round,
so a dictionary built with it scores differently - and worse - against the same
queries. From the raw graphics.txt this tool therefore runs the JavaScript
pipeline, which reproduces graphics.json value for value: 9,574 characters and
1,126,170 encoded values, checked against the reference checkout's copy.

The *query* is encoded by the reference's Python mmah_matcher.preprocess, and
the port does the same in src/hanzi/StrokeMatcher.cpp, because that pair is what
the reference's measurements were taken with - it is what makes a template score
0 against its own trajectory. The length unit differs between the two (the
JavaScript encodes hypot/sqrt(2), the Python hypot/2), which only scales the
angle penalty that is added to the distance; the port reproduces it rather than
inventing a third convention.

--graphics takes either file. graphics.json is read as it is; graphics.txt is
encoded by the pipeline above.

The IDS side keeps only what the component route can look up: a description of
exactly two parts, the character in U+4E00..U+9FFF and encodable in GBK, which is
the reference's own filter against its GBK template alphabet. Descriptions whose
two parts are not both in the BMP are dropped as well, because the file stores a
part as a uint16 and every alphabet the routes recognize a part from is in the
BMP; the tool prints how many that is.

Usage:
  tools/hanzi/fetch_data.py
  tools/hanzi/convert_data.py --graphics models/hanzi/graphics.txt --ids models/hanzi/ids.txt --out models/hanzi
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "models/hanzi"

DICTIONARY_MAGIC = b"HZMD"
IDS_MAGIC = b"HZID"
VERSION = 1

# mmah_matcher.NUM_POINTS points, an angle and a length per stroke.
NUM_POINTS = 4
STROKE_VALUES = 2 * NUM_POINTS + 2
NUM_VALUES = 256

# The medians' own box: their y grows upwards, the canvas' grows downwards, and
# the JavaScript pipeline flips between the two before anything else.
FLIP_HEIGHT = 1024

# compose.py: OPERATORS = ("⿰", "⿱"), left-right then top-bottom. The file stores
# the index into this tuple.
OPERATORS = ("\u2ff0", "\u2ff1")

# The file stores a part code point as a uint16, so a part past this cannot be
# represented and its description is dropped.
BMP_MAX = 0xFFFF


# ------------------------------------------------------- the dictionary encoder


def js_round(value: float) -> int:
    """JavaScript's Math.round: a half goes up rather than to even, which is what
    Python's round() does and numpy's rint() too. Every value the encoder writes
    passes through this, so the difference is worth a function of its own."""
    return math.floor(value + 0.5)


def _js_adjust_box(low: list, high: list, max_ratio: float, min_width: float):
    """do_something_to_aabb(): round the box, then pad it to min_width and to the
    aspect ratio the matcher wants. The padding is rounded too."""
    low = [js_round(low[0]), js_round(low[1])]
    high = [js_round(high[0]), js_round(high[1])]
    width = high[0] - low[0]
    height = high[1] - low[1]
    if width < 0 or height < 0:
        raise SystemExit("a character's medians do not span an ordered box")
    if width < min_width:
        pad = js_round((min_width - width) / 2)
        low[0] -= pad
        high[0] += pad
    if height < min_width:
        pad = js_round((min_width - height) / 2)
        low[1] -= pad
        high[1] += pad
    width = high[0] - low[0]
    height = high[1] - low[1]
    if width < height / max_ratio:
        pad = js_round((height / max_ratio - width) / 2)
        low[0] -= pad
        high[0] += pad
    elif height < width / max_ratio:
        pad = js_round((width / max_ratio - height) / 2)
        low[1] -= pad
        high[1] += pad
    return low, high


def _js_process_stroke(stroke: list, count: int) -> list:
    """process_stroke(): count - 1 points walked out along the polyline by arc
    length, then the stroke's own last point. The span, and with it the angle and
    the length, is taken between the first and that last point."""
    length = 0.0
    for index in range(len(stroke) - 1):
        length += math.hypot(stroke[index + 1][0] - stroke[index][0], stroke[index + 1][1] - stroke[index][1])

    sampled = []
    walked = 0.0
    index = 0
    candidate = stroke[0]
    for step in range(count - 1):
        target = (step * length) / (count - 1)
        while target > walked:
            segment = math.hypot(stroke[index + 1][0] - candidate[0], stroke[index + 1][1] - candidate[1])
            if target > walked + segment:
                index += 1
                candidate = stroke[index]
                walked += segment
            else:
                fraction = (target - walked) / segment
                candidate = (
                    (1 - fraction) * candidate[0] + fraction * stroke[index + 1][0],
                    (1 - fraction) * candidate[1] + fraction * stroke[index + 1][1],
                )
                walked = target
        sampled.append((js_round(candidate[0]), js_round(candidate[1])))
    sampled.append(stroke[-1])
    return sampled


def to_js_encode(strokes: list, count: int = NUM_POINTS, max_ratio: float = 1.0, min_width: float = 8.0) -> list:
    """One character's medians -> its ten encoded values per stroke, exactly as
    the shipped graphics.json has them: y flipped into canvas coordinates, the
    box mapped onto 0..255, NUM_POINTS points sampled, then the encoded angle
    and length of the span between the first and last sampled point."""
    flipped = [[(point[0], FLIP_HEIGHT - point[1]) for point in stroke] for stroke in strokes]
    low = [min(point[0] for stroke in flipped for point in stroke), min(point[1] for stroke in flipped for point in stroke)]
    high = [max(point[0] for stroke in flipped for point in stroke), max(point[1] for stroke in flipped for point in stroke)]
    low, high = _js_adjust_box(low, high, max_ratio, min_width)

    scale_x = (NUM_VALUES - 1) / (high[0] - low[0])
    scale_y = (NUM_VALUES - 1) / (high[1] - low[1])

    encoded = []
    for stroke in flipped:
        projected = [(js_round(scale_x * (point[0] - low[0])), js_round(scale_y * (point[1] - low[1]))) for point in stroke]
        sampled = _js_process_stroke(projected, count)
        span_x = sampled[-1][0] - sampled[0][0]
        span_y = sampled[-1][1] - sampled[0][1]
        angle = int(js_round(((math.atan2(span_y, span_x) + math.pi) * NUM_VALUES) / (2 * math.pi))) % NUM_VALUES
        length = int(js_round(math.sqrt((span_x * span_x + span_y * span_y) / 2)))
        encoded.append([int(value) for point in sampled for value in point] + [angle, length])
    return encoded


def js_dictionary(path: Path) -> dict:
    """graphics.txt, the raw Make Me a Hanzi medians -> the dictionary the
    reference's matcher loads from its shipped graphics.json."""
    dictionary = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        medians = entry.get("medians")
        if not medians:
            continue
        if len(entry["character"]) != 1 or not 1 <= len(medians) <= 0xFF:
            raise SystemExit(f"{entry['character']!r} is not a character with 1..255 strokes")
        dictionary[entry["character"]] = to_js_encode(medians)
    return dictionary


def json_dictionary(path: Path) -> dict:
    """graphics.json, the dictionary in its encoded form, read as it is."""
    dictionary = {}
    for character, strokes in json.loads(path.read_text(encoding="utf-8")):
        if len(character) != 1:
            raise SystemExit(f"the dictionary has the key {character!r}, which is not a single character")
        dictionary[character] = strokes
    return dictionary


def encoded_strokes(character: str, strokes: list) -> bytes:
    """One character's strokes as the ten bytes each the file stores."""
    if not 1 <= len(strokes) <= 0xFF:
        raise SystemExit(f"{character!r} has {len(strokes)} strokes; the file stores 1..255 per character")
    values = bytearray()
    for stroke in strokes:
        if len(stroke) != STROKE_VALUES:
            raise SystemExit(f"{character!r} has a stroke of {len(stroke)} values, expected {STROKE_VALUES}")
        for value in stroke:
            if not 0 <= value <= 0xFF:
                raise SystemExit(f"{character!r} has the encoded value {value}, which does not fit a byte")
            values.append(int(value))
    return bytes(values)


# --------------------------------------------------------------- the IDS table


def gbk_encodable(character: str) -> bool:
    """Whether the field templates can name this character at all: they are built
    from the GBK charset, and a character is in it exactly when GBK encodes it.
    This is the reference's filter, which indexes the table against its template
    alphabet."""
    try:
        character.encode("gbk")
    except UnicodeEncodeError:
        return False
    return True


def ids_rows(path: Path):
    """ids.txt -> (operator index, left, right, character) per description.

    The file is one description per line, tab separated, with '#' comments, and
    a character can appear more than once: the first description wins, which is
    what the reference's setdefault does as well."""
    rows = []
    described = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        character, description = parts[1], parts[2]
        if len(character) != 1 or character in described:
            continue
        if len(description) != 3 or description[0] not in OPERATORS:
            continue
        if not 0x4E00 <= ord(character) <= 0x9FFF or not gbk_encodable(character):
            continue
        described.add(character)
        rows.append((OPERATORS.index(description[0]), ord(description[1]), ord(description[2]), ord(character)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphics", type=Path, required=True, help="graphics.txt (the raw medians) or the preprocessed graphics.json")
    parser.add_argument("--ids", type=Path, required=True, help="ids.txt, the CJKVI ideographic description table")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"directory to write into (default {DEFAULT_OUT})")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- dictionary

    if not args.graphics.exists():
        print(f"{args.graphics} is missing; run tools/hanzi/fetch_data.py first", file=sys.stderr)
        return 1

    started = time.monotonic()
    dictionary = json_dictionary(args.graphics) if args.graphics.suffix == ".json" else js_dictionary(args.graphics)
    records = bytearray()
    # The dictionary's own order is kept: characters that score the same
    # tie-break by it, which is the reference's behaviour too.
    for character, strokes in dictionary.items():
        records += struct.pack("<IB", ord(character), len(strokes)) + encoded_strokes(character, strokes)
    dictionary_path = args.out / "hanzi-dictionary.bin"
    dictionary_path.write_bytes(DICTIONARY_MAGIC + struct.pack("<II", VERSION, len(dictionary)) + records)
    print(f"dictionary  {args.graphics} ({args.graphics.stat().st_size} bytes)")
    print(f"            {len(dictionary)} characters, {len(records)} bytes of encoded strokes")
    print(f"            wrote {dictionary_path} ({dictionary_path.stat().st_size} bytes) in {time.monotonic() - started:.1f} s")

    # ----------------------------------------------------------------- IDs

    if not args.ids.exists():
        print(f"{args.ids} is missing; run tools/hanzi/fetch_data.py first", file=sys.stderr)
        return 1

    started = time.monotonic()
    rows = sorted(ids_rows(args.ids))

    groups = 0
    dropped_groups = 0
    dropped_characters = 0
    records = bytearray()
    previous = None
    for operator, left, right, character in rows:
        if left > BMP_MAX or right > BMP_MAX:
            if (operator, left, right) != previous:
                dropped_groups += 1
            dropped_characters += 1
            previous = (operator, left, right)
            continue
        if (operator, left, right) != previous:
            groups += 1
        previous = (operator, left, right)
        records += struct.pack("<BHHI", operator, left, right, character)
    ids_path = args.out / "hanzi-ids.bin"
    ids_path.write_bytes(IDS_MAGIC + struct.pack("<II", VERSION, len(rows) - dropped_characters) + records)
    print(f"ids         {args.ids} ({args.ids.stat().st_size} bytes)")
    print(f"            {len(rows)} described characters, {len(rows) - dropped_characters} in {groups} groups kept, " f"{dropped_groups} groups / {dropped_characters} characters dropped (a part outside the BMP)")
    print(f"            wrote {ids_path} ({ids_path.stat().st_size} bytes) in {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
