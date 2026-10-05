# gregblur (MediaPipe selfie segmentation + no-halo blur)

## What this model is

**gregce/gregblur** (Apache-2.0) is a browser library for background blur in video
calls (it has a LiveKit adapter). It uses MediaPipe's selfie segmentation and does
the rest in WebGL2 shaders. This folder is a Python version of it (main branch,
last changed 2026-03-26). Its `docs/pipeline.md` explains every stage.

| `--model` | Model | Speed here |
|-----------|-------|------------|
| `selfie-multiclass` (gregblur's default) | MediaPipe selfie multiclass 256x256, 16 MB | slow on CPU: about 115 ms |
| `selfie-segmenter` | MediaPipe selfie segmenter 256x256, 0.25 MB | about 6 ms |

The multiclass model is the same one as in `mediapipe-multiclass/`.

The pipeline, with gregblur's default settings:

1. **Model**: MediaPipe Tasks gives a soft person mask at the frame size.
2. **Joint bilateral filter** (11x11 samples, sigma space 4 px, sigma colour 0.1):
   the mask is smoothed, but its edge snaps to edges in the picture.
3. **Temporal blend**: mask = 76% this frame + 24% the previous mask (less flicker).
4. **Masked downsample**: the frame is shrunk to half size, leaving out the person's
   pixels, so the person is not baked into the blurred background.
5. **Mask-weighted blur**: a Gaussian blur (radius 25) where every sample is
   weighted by "how much background" it is. The person leaves no bright halo.
6. **Composite**: `alpha = smoothstep(0.26, 0.72, mask + 0.035)`. The small +0.035
   keeps ears and temples when the model is unsure.

gregblur only blurs. Here the **Replace** panel (and the background tiles in
`live.py`) use the same alpha with a background image.

The browser runs steps 2-5 as GPU shaders. Here the per-pixel ones are plain Python
loops compiled with **numba**, and the blur uses OpenCV filters (the same maths).
Add `--no-refine` to see the raw model mask with the plain blur of the other folders.

**Note on gregblur's `selfie-segmenter` option.** gregblur always reads MediaPipe's
first mask as "background" and turns it around. That is right for the multiclass
model, but the selfie segmenter has only one mask, and it is the **person** (tested:
1.0 on the person, 0.0 on the background). So in gregblur that option would blur
the person instead of the background. This folder reads each model the right way.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows was tested)
- No GPU needed. On an Intel i7-13620H at 640x480 with `selfie-segmenter`:
  model about 6 ms, bilateral filter about 20 ms, blur a few ms. 18-24 FPS.
- About 500 MB of disk space for the virtual environment (MediaPipe is large)
- The first run compiles the numba steps (a few seconds). Later runs use the cache.

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd gregblur
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py --model selfie-segmenter
```

Linux / macOS:

```bash
cd gregblur
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py --model selfie-segmenter
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Without `--model`, gregblur's default (multiclass)
is used, which is slow on a CPU. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`gregblur/`).

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

### 2. Download the models

```
python download_model.py
```

This downloads both MediaPipe models (about 16.6 MB together) from Google's model
storage into `models/`. The links point to Google's "latest" version; if Google
updates a file, the script keeps it and prints a warning with the new checksum.

### 3. Run

```
python run.py --model selfie-segmenter
```
Uses the webcam (camera 0). Use `--camera 1` for a second camera.

```
python run.py                    # gregblur's default model (selfie-multiclass), slow
python run.py --model selfie-segmenter --image ../assets/samples/person_portrait.jpg
python run.py --model selfie-segmenter --video path/to/video.mp4
python run.py --model selfie-segmenter --no-refine      # raw mask and plain blur, to compare
```

#### Live video call view

```
python live.py --model selfie-segmenter
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Blur light uses blur radius 12, Blur strong gregblur's default 25.
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. Local only (it needs a webcam and a window).

Useful options for `run.py` (`live.py` has `--model` and `--no-refine` too):

| Option | What it does |
|--------|--------------|
| `--model NAME` | `selfie-multiclass` (default) or `selfie-segmenter` (faster) |
| `--no-refine` | Raw model mask and the plain blur of the other folders |
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

In `run_all_live.py` the model is chosen in `EXTRA_ARGS` (set to `selfie-segmenter`).

## Option B: Run with Docker

Run all commands from this folder (`gregblur/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-gregblur .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/gregblur/outputs" vbg-gregblur
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/gregblur/outputs" vbg-gregblur --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/gregblur/outputs" vbg-gregblur
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/gregblur/outputs" vbg-gregblur --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/gregblur/outputs" vbg-gregblur --camera 0
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
The mask panel shows the final alpha. The Blur panel uses gregblur's no-halo blur.
The top bar shows FPS (the whole pipeline) and the model's inference time in ms.

Measured on an Intel i7-13620H (CPU only), 640x480, filter on 4 threads:

| Model | Video file | Webcam | Raw (`--no-refine`) |
|-------|-----------|--------|---------------------|
| `selfie-segmenter` | 6 ms, 23 FPS | 6 ms, 18-20 FPS | 82 FPS video, 56 FPS webcam |
| `selfie-multiclass` | 114 ms, 4 FPS | 117 ms, 5 FPS | 8 FPS video |

Most of the time after the model is the 11x11 bilateral filter (about 20 ms). It
only runs where the mask changes (about 11% of the pixels on the test clip); the
rest of the mask is flat and would not change.

Quality on the test clip: the filter removes the small specks of flag that the raw
selfie segmenter mask lets through, and the blur shows no halo around the person.
On the webcam, a raised arm was followed closely.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| Very low FPS (about 4-5) | That is gregblur's default model on a CPU. Use `--model selfie-segmenter`. |
| `WARNING: ... is not the file this folder was tested with` | Google updated the model behind the "latest" link. It should still work; tell us if the mask looks wrong. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `W0000 ... inference_feedback_manager.cc ... Disabling support for feedback tensors` | Harmless MediaPipe log message. |
| The first start takes a few seconds | numba compiles the filter once, then caches it in `__pycache__/`. |
| Two OpenCV packages installed (`opencv-python` and `opencv-contrib-python`), strange `cv2` errors | Use a fresh venv with only `requirements.txt`. MediaPipe brings `opencv-contrib-python`; do not install `opencv-python` as well. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `gregblur/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

- **gregblur** (the pipeline) is **Apache-2.0**. The code here is our own Python
  version of its shaders.
- **MediaPipe** (code) is Apache-2.0. The two segmentation models come with model
  cards that describe their intended use and limits; read them before using a
  model in a product (links on the model page below).

- Repo: https://github.com/gregce/gregblur
- Live demo: https://gregce.github.io/gregblur/
- Pipeline explained: https://github.com/gregce/gregblur/blob/main/docs/pipeline.md
- Shaders: https://github.com/gregce/gregblur/blob/main/src/core/shaders.ts
- Model page: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
