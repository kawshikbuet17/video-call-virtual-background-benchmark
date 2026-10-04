# Robust Video Matting (RVM)

> **License warning: GPL-3.0.** If you ship RVM (code or model) inside a product,
> GPL-3.0 requires you to release that product's source code under the GPL too.
> Fine for learning and benchmarking; check with your legal team before any
> commercial use. See "License note" at the end.

## What this model is

RVM (2021) is a real-time video **matting** model. It predicts a soft alpha
value (0..1) for every pixel, like MODNet, but it is *recurrent*: it carries
memory from frame to frame, so the matte stays stable over time in videos and
webcam streams. We use the official MobileNetV3 ONNX model with ONNX Runtime.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed. On an Intel i7-13620H: about 88 ms per frame with the default
  setting (about 10 FPS), or about 25-29 ms with `--downsample-ratio 0.4`
  (about 25-29 FPS).
- About 300 MB of disk space for the virtual environment (CPU)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd rvm
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd rvm
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`rvm/`).

### 1. Create a virtual environment and install packages

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download the model

```
python download_model.py
```

This downloads the official ONNX model from the RVM GitHub release into
`models/` (about 15 MB) and checks its checksum.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0). Use `--camera 1` for a second camera.

```
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
```

#### Live video call view

```
python live.py
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. Local only (it needs a webcam and a window).

Useful options for `run.py`:

| Option | What it does |
|--------|--------------|
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |
| `--downsample-ratio R` | How much RVM shrinks the frame for its first stage (0..1). Default: automatic (largest side 512 px, so 0.8 for 640x480). 0.4 is about 3x faster with slightly less detail |

## Option B: Run with Docker

Run all commands from this folder (`rvm/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-rvm .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/rvm/outputs" vbg-rvm
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/rvm/outputs" vbg-rvm --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/rvm/outputs" vbg-rvm
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/rvm/outputs" vbg-rvm --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/rvm/outputs" vbg-rvm --camera 0
xhost -local:docker
```

- `--device /dev/video0` gives the container your webcam.
- `-e DISPLAY` and `-v /tmp/.X11-unix` let the container open a window on your screen.
- `xhost +local:docker` allows that; `xhost -local:docker` removes the permission again.

TODO: not tested yet (Docker was not available, and no Linux machine with a
webcam).

### GPU

Not provided. This project runs every model on the CPU.

## Controls

`run.py`:

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit |
| `s` | Save a screenshot of the window into `outputs/` |

`live.py`:

| Key | Action |
|-----|--------|
| Click a tile, or `1`-`9` | Choose None / Blur light / Blur strong / a background |
| `s` | Save the current output image into `outputs/` |
| `q` or `Esc` | Quit |

## Expected result

A window with four panels side by side: **Original | Mask | Blur | Replace**.
The mask is white where the model sees a person. The top bar shows FPS (the
whole pipeline) and the model's inference time in ms.

Measured on an Intel i7-13620H, 640x480:

| Setting | Inference | FPS |
|---------|-----------|-----|
| default (automatic, 0.8) | about 88 ms | about 10 |
| `--downsample-ratio 0.4` | about 25-29 ms | about 25-29 |

The matte is as clean as MODNet's (soft hair edges, no background leaks) and
stays steady from frame to frame. On a single image it has no earlier frames to
learn from, so it is best on video and webcam.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `download_model.py`: "does not match the expected checksum" | The download broke. Run it again. |
| Low FPS | Use `--downsample-ratio 0.4` (or 0.3). |
| The matte flickers or lags for a moment after a big change | Normal for a recurrent model: it needs a few frames to adapt. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `rvm/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

**GPL-3.0** (stated in the official repo: "Code is re-released under GPL-3.0
license"). You may use, study and change it freely. But if you distribute
software that includes RVM's code or model, that software must also be released
under GPL-3.0 with its source code. This is usually not acceptable for
closed-source commercial products. Check with your legal team first.

- Repo: https://github.com/PeterL1n/RobustVideoMatting
- Model release: https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0
- Paper: https://arxiv.org/abs/2108.11515
