# PP-HumanSeg V2 (portrait Lite)

## What this model is

The second version (2022) of Baidu's PaddleSeg portrait segmentation model for
video calls. Compared with v1 it uses a smaller input (256x144), is a bit faster,
and gives a cleaner mask. It outputs, for every pixel, the probability that it
belongs to a person. It runs on CPU with PaddlePaddle.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed. Inference is about 33 ms per frame on an Intel i7-13620H.
- About 1 GB of disk space for the virtual environment (PaddlePaddle is large)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd pp-humanseg-v2
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd pp-humanseg-v2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`pp-humanseg-v2/`).

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

This saves the model into `models/` (about 6 MB).

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

## Option B: Run with Docker

TODO: the V2 Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work. The Dockerfile is the same as v1's (which works for
image/video), except for the model.

Run all commands from this folder (`pp-humanseg-v2/`). The model is downloaded
during the build. The image is about 1.4 GB, mostly PaddlePaddle.

### Build

```
docker build -t vbg-pp-humanseg-v2 .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/pp-humanseg-v2/outputs" vbg-pp-humanseg-v2
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/pp-humanseg-v2/outputs" vbg-pp-humanseg-v2 --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/pp-humanseg-v2/outputs" vbg-pp-humanseg-v2
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/pp-humanseg-v2/outputs" vbg-pp-humanseg-v2 --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/pp-humanseg-v2/outputs" vbg-pp-humanseg-v2 --camera 0
xhost -local:docker
```

- `--device /dev/video0` gives the container your webcam.
- `-e DISPLAY` and `-v /tmp/.X11-unix` let the container open a window on your screen.
- `xhost +local:docker` allows that; `xhost -local:docker` removes the permission again.

TODO: not verified yet. A first test of the window (WSLg on Windows, with a
video file) showed `libsm6` was missing. It is now in the Dockerfile, but the
fixed image has not been re-tested. The webcam part was not tested, because no
Linux machine with a webcam was available.

### GPU

Not provided. The model is already small and fast on CPU, and the GPU build of
PaddlePaddle is a much larger, CUDA-specific install.

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

Measured on an Intel i7-13620H (CPU only), 640x480: about 25 FPS, about 33 ms
inference on a quiet machine. Numbers vary a lot from run to run on laptops; see
"Results" in the main README. Compared with v1 the body is filled in more
solidly. Busy backgrounds (for example a flag at the edge) can still leak in a
little.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `INFO: Could not find files for the given pattern(s).` on Windows | Harmless. Paddle runs the Windows `where` command on import. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| `OneDnnContext does not have the input Filter` | You enabled oneDNN (MKL-DNN). `run.py` disables it on purpose because it does not work with this model in Paddle 3.x. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `pp-humanseg-v2/` exactly as shown. |
| Docker: `libgomp.so.1` or `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

PaddleSeg (code and the released PP-HumanSeg models) is licensed under
Apache-2.0. Commercial use is allowed. Keep the license and notices if you
redistribute it.

- Repo: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg
- Paper: https://arxiv.org/abs/2112.07146
