#!/usr/bin/env python3
"""Run several models' live.py at the same time, side by side, on one webcam.

    python run_all_live.py

Choose the models by commenting / uncommenting lines in MODELS below.
Each model runs in its own window, with its own .venv, so set up each model
first (see its README: create .venv, pip install, download_model.py).

Close a window with q. When all windows are closed, the script ends.
Ctrl+C in the terminal closes everything.

Note: all models share one CPU. With several models open, each one runs slower
than it would alone, so use run.py or a single live.py for fair FPS numbers.
"""
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------- settings
# Comment out (#) the models you do not want to run.
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
]

# Extra options for a model's live.py (optional).
EXTRA_ARGS = {
    "modnet": ["--ref-size", "256"],  # 512 (official) is about 5 FPS on CPU
    "rvm": ["--downsample-ratio", "0.4"],  # automatic (0.8 at 640x480) is about 10 FPS on CPU
    "deeplabv3-mobilenet": ["--input-size", "256"],  # 520 (official) is about 4 FPS on CPU
}

CAMERA = 0           # webcam index
COLUMNS = 6          # windows per row
WINDOW_WIDTH = 240   # pixels; the height follows the video's aspect ratio
                     # (6 x 240 fits 3 rows = 18 windows on a 1080p laptop screen at 125% scaling)
# ---------------------------------------------------------------- end of settings

ROOT = Path(__file__).resolve().parent


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

    # 2. Start one live.py per model, each in its own window, tiled on the screen.
    window_height = int(WINDOW_WIDTH * 570 / 640)  # 640x480 video + 90 px picker bar
    models = []
    for i, folder in enumerate(MODELS):
        x = (i % COLUMNS) * (WINDOW_WIDTH + 10)
        y = (i // COLUMNS) * (window_height + 40)  # + room for the window title bar
        cmd = [str(venv_python(folder)), "live.py", "--shared-camera",
               "--window-pos", str(x), str(y), "--window-width", str(WINDOW_WIDTH)]
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
