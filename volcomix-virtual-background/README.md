# Volcomix virtual-background (Google Meet / ML Kit models)

## What this model is

**Volcomix/virtual-background** (Apache-2.0) is a browser demo of video-call virtual
backgrounds. It runs three small Google segmentation models and a WebGL2 pipeline.
This folder is a Python version of it (main branch, last changed 2023-10-07):

| `--model` | Model | Input | File size |
|-----------|-------|-------|-----------|
| `meet-lite` (default, as in the demo) | Google Meet segmentation, lite | 160x96 | 0.4 MB |
| `meet-full` | Google Meet segmentation, full | 256x144 | 0.4 MB |
| `mlkit` | ML Kit selfie segmentation | 256x256 | 0.25 MB |

The demo's fourth model, BodyPix, is already in `bodypix/`.

The pipeline, as in the demo's WebGL2 path, with its default settings:

1. **Model** (TFLite, run with LiteRT, Google's standalone TFLite runtime).
2. **Softmax**: the Meet models give two scores per pixel, [background, person].
3. **Joint bilateral filter** at the frame size: the mask is smoothed, but its edge
   stays where the picture has an edge (sigma space 1, sigma colour 0.1).
4. **Coverage**: `smoothstep(0.5, 0.75)` turns the soft mask into the final alpha.
   Weak mask areas (below 0.5) become background.
5. **Light wrapping** (only when replacing the background): a little of the new
   background's light is added to the person's edge, so the person blends in.

Not included: the demo's own blur. It is a mask-aware Gaussian blur on the GPU, so
the person does not smear into the background. Here the blur is the same cheap
blur as in the other folders.

The demo runs steps 2-5 as GPU shaders. Here they are plain Python loops compiled
with **numba**. Add `--no-refine` to see the raw model mask.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows was tested)
- No GPU needed. On an Intel i7-13620H at 640x480: the models take 1-5 ms, the
  filter 10-25 ms (4 threads). 21-47 FPS for the whole pipeline, see below.
- About 400 MB of disk space for the virtual environment
- The first run compiles the numba steps (a few seconds). Later runs use the cache.

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd volcomix-virtual-background
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd volcomix-virtual-background
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`volcomix-virtual-background/`).

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

This downloads the three `.tflite` files (about 1.1 MB together) from the repo's
main branch into `models/` and checks their checksums.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0) with the Meet lite model. Use `--camera 1` for a second camera.

```
python run.py --model meet-full
python run.py --model mlkit
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
python run.py --no-refine        # raw model mask, to compare
```

#### Live video call view

```
python live.py
python live.py --model mlkit
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. Local only (it needs a webcam and a window).

Useful options for `run.py` (`live.py` has `--model` and `--no-refine` too):

| Option | What it does |
|--------|--------------|
| `--model NAME` | `meet-lite` (default), `meet-full` or `mlkit` |
| `--no-refine` | Raw model mask: no joint bilateral filter, coverage or light wrap |
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

In `run_all_live.py` the model is chosen in `EXTRA_ARGS` (default `meet-lite`).

## Option B: Run with Docker

Run all commands from this folder (`volcomix-virtual-background/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-volcomix-virtual-background .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/volcomix-virtual-background/outputs" vbg-volcomix-virtual-background
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/volcomix-virtual-background/outputs" vbg-volcomix-virtual-background --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/volcomix-virtual-background/outputs" vbg-volcomix-virtual-background
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/volcomix-virtual-background/outputs" vbg-volcomix-virtual-background --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/volcomix-virtual-background/outputs" vbg-volcomix-virtual-background --camera 0
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
The mask panel shows the final alpha (after coverage). The top bar shows FPS (the
whole pipeline, including the filter) and the model's inference time in ms.

Measured on an Intel i7-13620H (CPU only), 640x480, 1 TFLite thread, filter on 4 threads:

| Model | Video file | Webcam | Raw (`--no-refine`), video |
|-------|-----------|--------|----------------------------|
| `meet-lite` | 1.3 ms, 24-25 FPS | 2.0-2.6 ms, 21-26 FPS | 1.2 ms, 150-160 FPS |
| `meet-full` | 2.8 ms, 30 FPS | 3.0 ms, 44 FPS | 2.8 ms, 113 FPS |
| `mlkit` | 4.8 ms, 45-47 FPS | 4.9 ms, 42 FPS | 4.7 ms, 87 FPS |

The models are very fast; most of the time is the joint bilateral filter. Meet lite
is the slowest overall: its mask is the smallest (160x96), so the filter's kernel
at 640x480 is the largest (7x7 samples instead of 6x6 or 5x5).

Quality on the test clip: coverage removes the flag patch that the raw Meet lite
mask lets through. Both Meet models leave a small hole at the shoulder patch.
ML Kit gave the cleanest mask on that clip. On the webcam, the people sitting
behind were left out. Light wrapping is subtle (strength 0.3, the demo's default).

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` | Harmless. LiteRT says it uses its fast CPU backend. |
| The first start takes a few seconds | numba compiles the filter once, then caches it in `__pycache__/`. |
| Low FPS on a big video | The filter grows with the frame size. Use a 640x480 or 1280x720 input, or `--no-refine`. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `volcomix-virtual-background/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

- **The demo's code** (Volcomix/virtual-background) is **Apache-2.0**. The code here
  is our own Python version of its pipeline.
- **Google Meet models: the license is unclear.** The repo's README says the
  Meet model card was first released under Apache-2.0, but seems to have switched
  to the Google Terms of Service on 2021-01-21. The repo keeps a copy of the model
  card that matches these files. Use the Meet models for learning and testing
  only, and do not ship them in a product without checking with Google.
- **ML Kit selfie model**: the repo says it is Apache-2.0 (copy of the model card
  in the repo). It was extracted from the ML Kit Android package.

- Repo: https://github.com/Volcomix/virtual-background
- Live demo: https://volcomix.github.io/virtual-background
- Meet model card copy: https://github.com/Volcomix/virtual-background/blob/main/docs/meet-segmentation-model-card-2020-10-12.pdf
- ML Kit model card copy: https://github.com/Volcomix/virtual-background/blob/main/docs/mlkit-selfie-model-card-2021-02-16.pdf
- Google's post on Meet backgrounds: https://research.google/blog/background-features-in-google-meet-powered-by-web-ml/
- ML Kit selfie segmentation: https://developers.google.com/ml-kit/vision/selfie-segmentation
- LiteRT: https://pypi.org/project/ai-edge-litert/
