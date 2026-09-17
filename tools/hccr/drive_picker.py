"""End-to-end check of the handwriting pad against the real window.

Starts an X server, runs the picker harness, switches to the handwriting view,
draws a character with actual mouse events (xdotool), and commits the top
candidate with a keystroke. Screenshots are kept for inspection.

  python3 drive_picker.py --char 休
"""

from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from render_strokes import STROKE_BOX, load_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DISPLAY = ":97"
SCREEN = "1280x1024x24"


def run(command, env, check=True):
    return subprocess.run(command, env=env, capture_output=True, text=True, check=check)


def wait_for_window(env, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run(["xdotool", "search", "--onlyvisible", "--name", "im-emoji-picker"], env, check=False)
        ids = [line for line in result.stdout.split() if line.strip()]
        if ids:
            return ids[-1]
        time.sleep(0.25)
    raise RuntimeError("the picker window never appeared")


def screenshot(env, path):
    run(["import", "-window", "root", str(path)], env)
    return path


def ink_rect(path):
    """Bounding box of the white ink pad, found by looking for white rows."""
    gray = np.asarray(Image.open(path).convert("L"))
    white = gray > 245
    rows = np.where(white.sum(axis=1) > 120)[0]
    columns = np.where(white.sum(axis=0) > 120)[0]
    if rows.size == 0 or columns.size == 0:
        raise RuntimeError("no ink pad found in the screenshot")
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def draw(env, medians, rect, inset_ratio=0.08):
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    inset_x = width * inset_ratio
    inset_y = height * inset_ratio

    for stroke in medians:
        points = []
        for x, y in stroke:
            points.append((
                left + inset_x + (x / STROKE_BOX) * (width - 2 * inset_x),
                top + inset_y + ((STROKE_BOX - y) / STROKE_BOX) * (height - 2 * inset_y),
            ))
        if not points:
            continue

        run(["xdotool", "mousemove", str(int(points[0][0])), str(int(points[0][1])), "mousedown", "1"], env)
        for x, y in points[1:]:
            run(["xdotool", "mousemove", str(int(x)), str(int(y))], env)
            time.sleep(0.004)
        run(["xdotool", "mouseup", "1"], env)
        time.sleep(0.05)


def send(app, line):
    app.stdin.write(line + "\n")
    app.stdin.flush()
    time.sleep(0.25)


def drain(app, timeout=3.0):
    lines = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([app.stdout], [], [], 0.2)
        if not ready:
            continue
        line = app.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
    return lines


def window_size(env, window):
    geometry = run(["xdotool", "getwindowgeometry", "--shell", window], env).stdout
    values = dict(line.split("=", 1) for line in geometry.strip().splitlines() if "=" in line)
    return int(values["WIDTH"]), int(values["HEIGHT"])


def wait_for_size(env, window, expected_height, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if window_size(env, window)[1] == expected_height:
            time.sleep(0.3)
            return True
        time.sleep(0.15)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default="休")
    parser.add_argument("--model", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    parser.add_argument("--picker", type=Path, default=REPO / "build/picker-gui")
    parser.add_argument("--out", type=Path, default=REPO / "build/picker-test")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "computing/extension/hanzi-strokes/package")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    medians = load_medians(args.char, args.data_dir)
    if medians is None:
        raise SystemExit(f"no stroke data for {args.char}")

    env = {**os.environ, "DISPLAY": DISPLAY, "QT_QPA_PLATFORM": "xcb", "IM_EMOJI_PICKER_HCCR_MODEL": str(args.model)}

    xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", SCREEN, "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    app = None
    try:
        time.sleep(0.7)
        app = subprocess.Popen([str(args.picker)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        window = wait_for_window(env)
        geometry = run(["xdotool", "getwindowgeometry", "--shell", window], env).stdout
        print("window geometry:")
        print(geometry.strip())

        for _ in range(3):  # MRU -> LIST -> KAOMOJI -> HANDWRITING
            send(app, "key tab")

        if not wait_for_size(env, window, 280):
            raise RuntimeError(f"the window did not switch to the handwriting view (size {window_size(env, window)})")

        screenshot(env, args.out / "1-handwriting-view.png")
        rect = ink_rect(args.out / "1-handwriting-view.png")
        print(f"ink pad at {rect} ({rect[2] - rect[0]}x{rect[3] - rect[1]})")

        draw(env, medians, rect)
        time.sleep(0.6)
        screenshot(env, args.out / "2-after-strokes.png")

        send(app, "key 1")
        time.sleep(0.4)
        screenshot(env, args.out / "3-after-commit.png")

        lines = drain(app)
        print("harness output:")
        for line in lines:
            print(f"  {line}")

        commits = [line.split(" ", 1)[1] for line in lines if line.startswith("COMMIT ")]
        print()
        if not commits:
            print(f"FAIL: nothing committed while drawing {args.char}")
            return 1
        if commits[0] != args.char:
            print(f"FAIL: drew {args.char}, pad committed {commits[0]}")
            return 1
        print(f"PASS: drew {args.char} with the mouse, pad committed {commits[0]}")

        # Second round: the pad cleared itself, so draw again and commit through
        # the keyboard (arrow selection + Enter) instead of the number key.
        draw(env, medians, rect)
        time.sleep(0.6)
        send(app, "key down")
        send(app, "key return")
        time.sleep(0.4)
        screenshot(env, args.out / "4-keyboard-commit.png")

        keyboard_lines = drain(app)
        keyboard_commits = [line.split(" ", 1)[1] for line in keyboard_lines if line.startswith("COMMIT ")]
        if not keyboard_commits:
            print("FAIL: selecting a candidate with the arrow keys and Enter committed nothing")
            return 1
        if keyboard_commits[0] == commits[0]:
            print(f"FAIL: arrow-down did not move the selection (still {keyboard_commits[0]})")
            return 1
        print(f"PASS: arrow-down + Enter committed the second candidate {keyboard_commits[0]}")
        return 0
    finally:
        if app is not None:
            app.terminate()
        xvfb.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
