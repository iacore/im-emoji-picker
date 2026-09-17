"""End-to-end check of the handwriting pad against the real window.

Starts an X server, runs the picker harness, and drives it the way a user
would: asserts the pad is what the picker opens with, draws a character with
actual mouse events, commits a candidate with a keystroke, and switches views
by clicking the status bar indicators. Screenshots are kept for inspection.

  python3 drive_picker.py --char 休
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import threading

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from render_strokes import STROKE_BOX, load_medians  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DISPLAY = ":97"
SCREEN = "1280x1024x24"
PAD_VIEW_HEIGHT = 420
EMOJI_VIEW_HEIGHT = 190


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


def window_rect(env, window):
    geometry = run(["xdotool", "getwindowgeometry", "--shell", window], env).stdout
    values = dict(line.split("=", 1) for line in geometry.strip().splitlines() if "=" in line)
    return int(values["X"]), int(values["Y"]), int(values["WIDTH"]), int(values["HEIGHT"])


def wait_for_height(env, window, expected, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if window_rect(env, window)[3] == expected:
            time.sleep(0.3)
            return True
        time.sleep(0.15)
    return False


def ink_rect(path):
    """Bounding box of the white ink pad, found by looking for white rows."""
    gray = np.asarray(Image.open(path).convert("L"))
    white = gray > 245
    rows = np.where(white.sum(axis=1) > 120)[0]
    columns = np.where(white.sum(axis=0) > 120)[0]
    if rows.size == 0 or columns.size == 0:
        raise RuntimeError("no ink pad found in the screenshot")
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def ink_pixels(path, rect):
    """Count strongly dark pixels inside the pad, i.e. actual ink."""
    left, top, right, bottom = rect
    gray = np.asarray(Image.open(path).convert("L"))
    inner = gray[top + 2:bottom - 1, left + 2:right - 1]
    return int((inner < 100).sum())


def click(env, x, y):
    run(["xdotool", "mousemove", str(int(x)), str(int(y)), "click", "1"], env)
    time.sleep(0.35)


def click_indicator(env, window, target_height):
    """Click status bar indicators until the window takes the wanted height."""
    x, y, width, height = window_rect(env, window)
    for dy in (9, 11, 13, 7, 15):
        for dx in (16, 20, 25, 30, 36, 43, 52, 62, 74, 88, 104):
            click(env, x + width - dx, y + height - dy)
            if window_rect(env, window)[3] == target_height:
                return dx, dy
    raise RuntimeError(f"no status bar indicator switched the window to {target_height}px")


def click_clear_button(env, window, rect, out):
    """Find the Clear button by probing the lower left of the panel.

    Candidate labels sit in the middle of the row above, so the probe stays left
    of them and the caller rejects any attempt that commits something.
    """
    x, y, _width, _height = window_rect(env, window)
    del y
    for offset_y in range(44, 104, 8):
        for offset_x in (24, 34, 44, 54):
            click(env, x + offset_x, rect[3] + offset_y)
            screenshot(env, out / "probe.png")
            if ink_pixels(out / "probe.png", rect) == 0:
                return offset_x, offset_y
    return None


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


def collect_output(app):
    """Read the harness output in a thread: select() on the pipe misses lines
    that Python has already buffered, which silently hides commits."""
    collected: list[str] = []

    def reader():
        for line in app.stdout:
            collected.append(line.rstrip())

    threading.Thread(target=reader, daemon=True).start()
    return collected


def drain(app, collected, timeout=2.0):
    time.sleep(timeout)
    lines = list(collected)
    collected.clear()
    return lines


def commits_in(lines):
    return [line.split(" ", 1)[1] for line in lines if line.startswith("COMMIT ")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default="休")
    parser.add_argument("--model", type=Path, default=REPO / "models/hccr/hccr-mobilenetv2.gguf")
    parser.add_argument("--picker", type=Path, default=REPO / "build-usr/picker-gui")
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
        collected = collect_output(app)

        window = wait_for_window(env)

        # 1. The picker opens on the emoji view again.
        if not wait_for_height(env, window, EMOJI_VIEW_HEIGHT):
            print(f"FAIL: picker opened at {window_rect(env, window)} instead of the emoji view")
            return 1
        print(f"window {window}: {window_rect(env, window)}")

        # 2. Clicking the pen indicator switches to the pad.
        pen_dx = click_indicator(env, window, PAD_VIEW_HEIGHT)
        screenshot(env, args.out / "1-pad-view.png")
        rect = ink_rect(args.out / "1-pad-view.png")
        print(f"PASS: clicking the pen indicator at -{pen_dx[0]}px,-{pen_dx[1]}px opened the pad, ink area {rect[2] - rect[0]}x{rect[3] - rect[1]}")

        # 2. Draw with the mouse and commit the top candidate with a digit.
        draw(env, medians, rect)
        time.sleep(0.6)
        screenshot(env, args.out / "2-after-strokes.png")

        send(app, "key 1")
        time.sleep(0.4)

        drawn = commits_in(drain(app, collected))
        if not drawn:
            print(f"FAIL: nothing committed while drawing {args.char}")
            for line in lines:
                print(f"  harness said: {line}")
            return 1
        if drawn[0] != args.char:
            print(f"FAIL: drew {args.char}, pad committed {drawn[0]}")
            return 1
        print(f"PASS: drew {args.char} with the mouse, pad committed {drawn[0]}")

        # 3. Same again, committed through arrow selection and Enter.
        draw(env, medians, rect)
        time.sleep(0.6)
        send(app, "key down")
        send(app, "key return")
        time.sleep(0.4)

        picked = commits_in(drain(app, collected))
        if not picked:
            print("FAIL: arrow keys + Enter committed nothing")
            return 1
        if picked[0] == drawn[0]:
            print(f"FAIL: arrow-down did not move the selection (still {picked[0]})")
            return 1
        print(f"PASS: arrow-down + Enter committed the second candidate {picked[0]}")

        # 4. Right click wipes the pad.
        draw(env, medians, rect)
        time.sleep(0.5)
        if ink_pixels(args.out / "2-after-strokes.png", rect) == 0:
            print("FAIL: the drawn strokes left no ink to test clearing with")
            return 1
        x, y, width, height = window_rect(env, window)
        run(["xdotool", "mousemove", str((rect[0] + rect[2]) // 2), str((rect[1] + rect[3]) // 2), "click", "3"], env)
        time.sleep(0.4)
        screenshot(env, args.out / "5-after-right-click.png")
        if ink_pixels(args.out / "5-after-right-click.png", rect) != 0:
            print("FAIL: right clicking the pad did not wipe it")
            return 1
        print("PASS: right clicking the pad wiped it")

        # 5. The Clear button wipes it too, and must not commit anything.
        drain(app, collected)
        draw(env, medians, rect)
        time.sleep(0.5)
        button = click_clear_button(env, window, rect, args.out)
        if button is None:
            print("FAIL: no Clear button found in the lower left of the panel")
            return 1
        screenshot(env, args.out / "6-after-clear-button.png")
        leftovers = drain(app, collected)
        if ink_pixels(args.out / "6-after-clear-button.png", rect) != 0:
            for line in leftovers:
                print(f"  harness said: {line}")
            print("FAIL: the Clear button did not wipe the pad")
            return 1
        if commits_in(leftovers):
            print("FAIL: the click landed on a candidate instead of the Clear button")
            return 1
        print(f"PASS: the Clear button at +{button[0]}px,+{button[1]}px (window relative) wiped the pad without committing")

        # 6. Status bar indicators switch back.
        emoji_dx = click_indicator(env, window, EMOJI_VIEW_HEIGHT)
        screenshot(env, args.out / "7-emoji-view.png")
        print(f"PASS: clicking a status indicator at -{emoji_dx[0]}px,-{emoji_dx[1]}px left the pad view")
        return 0
    finally:
        if app is not None:
            app.terminate()
        xvfb.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
