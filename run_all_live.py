#!/usr/bin/env python3
"""Run several models' live.py at the same time, side by side, on one webcam.

    python run_all_live.py

Choose the models by commenting / uncommenting lines in MODELS below.
Each model runs in its own window, with its own .venv, so set up each model
first (see its README: create .venv, pip install, download_model.py).

The windows fill the screen: 1 model = one big window, 2 = side by side,
3 or 4 = a 2x2 grid, 5 or 6 = 3x2, and so on.

Close a window with q. When all windows are closed, the script ends.
Ctrl+C in the terminal closes everything.

Note: all models share one CPU. With several models open, each one runs slower
than it would alone, so use run.py or a single live.py for fair FPS numbers.
Up to 4 at once works well on a laptop; more slows every window down.
"""
import math
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- settings
# Comment out (#) the models you do not want to run. Up to 4 at once is a good limit.
MODELS = [
    "pp-humanseg-v1",
    "pp-humanseg-v2",
    "mediapipe-selfie",
    "mediapipe-multiclass",   # slow on CPU (about 7 FPS)
    "modnet",
    "rvm",                    # GPL-3.0 license
    "sinet",
    "u2net",                  # very slow on CPU (about 2 FPS) and slows the others: comment out if needed
    "deeplabv3-mobilenet",    # general 21-class model, soft edges
    "bodypix",                # deprecated by Google, weakest mask
    "fast-person-segmentation",
    "ncnn-portrait-segmentation",
    "fast-portrait-segmentation",  # same model as ncnn-portrait-segmentation, in PyTorch
    "slimnet",
    "webinar-humanseg",       # PP-HumanSegV2 + the netesh3/webinar temporal and edge steps
    "volcomix-virtual-background",  # Google Meet model (lite) + joint bilateral filter; --model in EXTRA_ARGS
    "gregblur",               # MediaPipe selfie + gregblur pipeline (no-halo blur); --model in EXTRA_ARGS
    "linux-fake-background-webcam",  # MediaPipe selfie + lfbw mask recipe (threshold, dilate, box blur)
]

# Extra options for a model's live.py (optional).
EXTRA_ARGS = {
    "modnet": ["--ref-size", "256"],  # 512 (official) is about 5 FPS on CPU
    "rvm": ["--downsample-ratio", "0.4"],  # automatic (0.8 at 640x480) is about 10 FPS on CPU
    "deeplabv3-mobilenet": ["--input-size", "256"],  # 520 (official) is about 4 FPS on CPU
    "volcomix-virtual-background": ["--model", "meet-lite"],  # or meet-full / mlkit
    "gregblur": ["--model", "selfie-segmenter"],  # gregblur's default selfie-multiclass is about 5 FPS on CPU
}

CAMERA = 0           # webcam index
SCREEN = None        # usable screen area (x, y, width, height); None = detect it.
                     # Example: SCREEN = (0, 0, 1536, 816)
# ---------------------------------------------------------------- end of settings

ROOT = Path(__file__).resolve().parent
BAR_HEIGHT = 90      # the picker bar under the video in every live.py
# Window frame around the picture on Windows 11 (title bar + borders), in pixels.
# Measured: a 240 px wide picture gives a 256 x 252 window.
FRAME_W, FRAME_H = 16, 38


def screen_area():
    """The usable part of the screen (without the taskbar): x, y, width, height."""
    if SCREEN:
        return SCREEN
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        SPI_GETWORKAREA = 0x0030
        ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return 0, 0, w, h - 60  # leave room for a top or bottom panel
    except Exception:
        return 0, 0, 1280, 720  # safe guess; set SCREEN above if it does not fit


def grid_for(count):
    """Columns and rows: 1 -> 1x1, 2 -> 2x1, 3-4 -> 2x2, 5-6 -> 3x2, 7-9 -> 3x3, ..."""
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return columns, rows


def window_layout(count, video_w, video_h):
    """Position (x, y) and picture width for each window, as large as the grid allows."""
    left, top, width, height = screen_area()
    columns, rows = grid_for(count)
    cell_w, cell_h = width // columns, height // rows
    view_h_per_w = (video_h + BAR_HEIGHT) / video_w  # live.py shows video + picker bar
    picture_w = int(min(cell_w - FRAME_W, (cell_h - FRAME_H) / view_h_per_w))
    window_w = picture_w + FRAME_W
    window_h = int(picture_w * view_h_per_w) + FRAME_H
    layout = []
    for i in range(count):
        col, row = i % columns, i // columns
        x = left + col * cell_w + (cell_w - window_w) // 2  # centred in its cell
        y = top + row * cell_h + (cell_h - window_h) // 2
        layout.append((x, y, picture_w))
    return layout


def venv_python(folder):
    """Path to the Python inside <folder>/.venv (Windows and Linux/macOS)."""
    win = ROOT / folder / ".venv" / "Scripts" / "python.exe"
    unix = ROOT / folder / ".venv" / "bin" / "python"
    return win if win.exists() else unix


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    if not MODELS:
        raise SystemExit("No models selected. Uncomment at least one line in MODELS.")
    missing = [m for m in MODELS if not venv_python(m).exists()]
    if missing:
        raise SystemExit("These models have no .venv yet. Set them up first (see their README):\n  "
                         + "\n  ".join(missing))

    # 1. Open the webcam once and share its frames (see camera_share.py).
    #    It needs OpenCV, so we borrow the first model's .venv Python.
    print("Starting the shared camera...")
    camera = subprocess.Popen([str(venv_python(MODELS[0])), str(ROOT / "camera_share.py"), "--camera", str(CAMERA)],
                              stdout=subprocess.PIPE, text=True)
    line = camera.stdout.readline().strip()
    if "ready" not in line:
        stop(camera)
        raise SystemExit(f"The shared camera did not start: {line or 'no output'}")
    size = re.search(r"(\d+)x(\d+)", line)  # "Shared camera ready. 640x480"
    video_w, video_h = (int(size[1]), int(size[2])) if size else (640, 480)

    # 2. Start one live.py per model, each in its own window. The screen is split
    #    into a grid that depends on how many models are selected.
    if len(MODELS) > 4:
        print(f"Note: {len(MODELS)} models selected. More than 4 at once makes every window slow.")
    models = []
    for folder, (x, y, picture_w) in zip(MODELS, window_layout(len(MODELS), video_w, video_h)):
        cmd = [str(venv_python(folder)), "live.py", "--shared-camera",
               "--window-pos", str(x), str(y), "--window-width", str(picture_w)]
        cmd += EXTRA_ARGS.get(folder, [])
        print(f"Starting {folder}")
        models.append(subprocess.Popen(cmd, cwd=ROOT / folder))

    # 3. Wait until every window is closed (or Ctrl+C), then stop the camera.
    print("Running. Close each window with q, or press Ctrl+C here to close all.")
    try:
        while any(p.poll() is None for p in models):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Closing all windows...")
    finally:
        for p in models:
            stop(p)
        stop(camera)


if __name__ == "__main__":
    main()
