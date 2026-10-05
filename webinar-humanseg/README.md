# Webinar HumanSeg (netesh3/webinar background pipeline)

## What this model is

The virtual background of the **netesh3/webinar** browser app, redone in Python.
It was introduced in pull request #242 and changed later on the `main` branch. This
folder follows `main` as it was on 2026-10-03 (commit `d9051ea`):

1. **Model**: PP-HumanSegV2-Lite (PaddleSeg, Apache-2.0) at 256x144, as an ONNX file
   that the webinar project converted. It is the same network as `pp-humanseg-v2/`,
   but runs on ONNX Runtime instead of PaddlePaddle (about 5 ms instead of 33 ms).
2. **Temporal step**: block-matching motion between the previous and the current frame
   (grey image, 160x90). The previous mask is moved along that motion and partly kept
   where the picture did not change. Less flicker, fewer holes in a still person.
3. **Edge step**: joint bilateral upsample. The mask is brought to the frame size. At
   the person's edge, the 4x4 nearest mask texels vote, weighted by distance and by
   colour match, so the edge follows the picture instead of the coarse model grid.

What is **not** included from the app:

- The **guided filter** from PR #242. It is switched off on `main`, because on a real
  camera it made shoulders and hair translucent (commit `6d92241`).
- The **presenter lock** ("only me": keeps out people who walk behind the presenter).
  It needs a face detector and is part of the app, not of the segmentation.
- Display polish in the app's WebGL compositor (low-light lift, skin soften, edge
  decontamination, background pan).

The browser runs steps 2 and 3 on the GPU (WebGL) or in WebAssembly. Here they are
plain Python loops compiled with **numba**, so they are fast enough on the CPU
(numpy alone was about 7x slower). Add `--no-refine` to see the raw model mask.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows was tested)
- No GPU needed. On an Intel i7-13620H at 640x480: model about 5 ms, temporal step
  about 4 ms, edge step about 14 ms (4 threads). About 34 FPS for the whole pipeline.
- About 450 MB of disk space for the virtual environment
- The first run compiles the numba steps (a few seconds). Later runs use the cache.

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd webinar-humanseg
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd webinar-humanseg
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`webinar-humanseg/`).

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

This downloads `humanseg.onnx` (about 3.8 MB) from the `main` branch of the webinar
repo into `models/` and checks the checksum. If the repo has changed its model since
this folder was tested, the file is kept and a warning shows the new checksum.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0). Use `--camera 1` for a second camera.

```
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
python run.py --no-refine        # raw model mask, to compare
```

#### Live video call view

```
python live.py
python live.py --no-refine
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
| `--no-refine` | Raw model mask: no temporal step, no edge step |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

The temporal step uses the previous frame, so it only does something on video and
webcam, not on a single image.

## Option B: Run with Docker

Run all commands from this folder (`webinar-humanseg/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-webinar-humanseg .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/webinar-humanseg/outputs" vbg-webinar-humanseg
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/webinar-humanseg/outputs" vbg-webinar-humanseg --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/webinar-humanseg/outputs" vbg-webinar-humanseg
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/webinar-humanseg/outputs" vbg-webinar-humanseg --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/webinar-humanseg/outputs" vbg-webinar-humanseg --camera 0
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
whole pipeline, including the temporal and edge steps) and the model's inference
time in ms (the ONNX model only).

Measured on an Intel i7-13620H (CPU only), 640x480:

| Run | Inference | Whole pipeline |
|-----|-----------|----------------|
| Video file | about 5 ms | 33-35 FPS |
| Video file, `--no-refine` | about 4.5 ms | 86-88 FPS |
| Webcam | about 5 ms | 34 FPS (one run of five was much slower, 8 FPS) |
| Webcam, `--no-refine` | about 4.5 ms | 88-89 FPS |

Most of the time goes to the edge step (about 14 ms) and the compositing (about
5 ms). The edge step works on every pixel near the person's edge. This model's mask
is soft (much of the person is 0.9-0.98, not 1.0), so that is a large part of the
frame. On larger frames it is slower: about 50 ms at 1280x720.

The mask is the same network as `pp-humanseg-v2/`. With the refine steps its edge
follows hair and shoulders a little more closely, and it flickers less on video.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `WARNING: this is not the model file this folder was tested with` | The webinar repo has a newer model on `main`. Try it; if `run.py` fails, the model's input or output changed. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| The first start takes a few seconds | numba compiles the temporal and edge steps once, then caches them in `__pycache__/`. |
| Low FPS on a big video | The edge step grows with the frame size. Use a 640x480 or 1280x720 input, or `--no-refine`. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `webinar-humanseg/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

- **The model** is PP-HumanSegV2-Lite from PaddleSeg, **Apache-2.0**. Commercial use
  is allowed. The ONNX file is the webinar project's conversion of it.
- **The webinar repo has no license file.** Without a license, its code is not
  open for reuse. The code in this folder is our own Python version of the steps
  described in its comments (the guided-filter and bilateral methods are published
  techniques). None of its TypeScript is copied. Treat this folder as a learning and
  benchmarking copy, and ask the author before using the webinar code itself.

- Webinar repo: https://github.com/netesh3/webinar
- Pull request #242: https://github.com/netesh3/webinar/pull/242
- The pipeline on main: https://github.com/netesh3/webinar/blob/main/web/lib/humanseg.ts
- Model notes: https://github.com/netesh3/webinar/blob/main/web/public/models/README.md
- PP-HumanSeg: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg
- numba: https://pypi.org/project/numba/
