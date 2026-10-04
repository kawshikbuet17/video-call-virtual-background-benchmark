# MediaPipe Selfie Multiclass

## What this model is

Google's multiclass selfie model (256x256 input, float32, about 16 MB). Instead of
person / background it labels each pixel as one of six classes: background, hair,
body skin, face skin, clothes, others (accessories). For a virtual background we
use "everything that is not background" as the person mask. It is more accurate
than the landscape selfie model, but much slower on CPU.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed, but slow on CPU: inference is about 115-135 ms per frame on an Intel i7-13620H (about 7-8 FPS).
- About 400 MB of disk space for the virtual environment

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd mediapipe-multiclass
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd mediapipe-multiclass
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`mediapipe-multiclass/`).

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

This saves the model into `models/` (about 16 MB).

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

Run all commands from this folder (`mediapipe-multiclass/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-mediapipe-multiclass .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/mediapipe-multiclass/outputs" vbg-mediapipe-multiclass
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/mediapipe-multiclass/outputs" vbg-mediapipe-multiclass --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/mediapipe-multiclass/outputs" vbg-mediapipe-multiclass
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/mediapipe-multiclass/outputs" vbg-mediapipe-multiclass --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/mediapipe-multiclass/outputs" vbg-mediapipe-multiclass --camera 0
xhost -local:docker
```

- `--device /dev/video0` gives the container your webcam.
- `-e DISPLAY` and `-v /tmp/.X11-unix` let the container open a window on your screen.
- `xhost +local:docker` allows that; `xhost -local:docker` removes the permission again.

TODO: not tested yet (Docker was not available, and no Linux machine with a
webcam).

### GPU

Not provided. This model would benefit from a GPU, but MediaPipe's Python package
does not offer GPU inference on Windows. TODO: try MediaPipe's GPU delegate on
Linux if needed.

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

Measured on an Intel i7-13620H (CPU only), 640x480: about 115-135 ms inference,
about 7-8 FPS. The video looks choppy, but the mask is the cleanest of the models
tested so far: busy backgrounds (for example a flag) mostly disappear.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `INFO: Created TensorFlow Lite XNNPACK delegate` and `W0000 ... inference_feedback_manager` lines | Harmless log messages from MediaPipe. |
| Two OpenCV packages installed (`opencv-python` and `opencv-contrib-python`), strange `cv2` errors | Use a fresh venv with only `requirements.txt`. MediaPipe brings `opencv-contrib-python`; do not install `opencv-python` as well. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `mediapipe-multiclass/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

MediaPipe (code) is licensed under Apache-2.0. The multiclass segmentation model
comes with a model card that describes its intended use and limits; read it
before using the model in a product.

- Model page: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
- Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf
- Repo: https://github.com/google-ai-edge/mediapipe
