# DeepLabV3 MobileNetV3-Large (torchvision)

## What this model is

DeepLabV3 (2017) is a general semantic segmentation model. This version uses a
MobileNetV3-Large backbone and comes with torchvision, trained on COCO with the
21 Pascal VOC classes (background, aeroplane, ..., person, ..., tvmonitor). We
use only the **person** class (class 15) as the mask. It is not made for video
calls, so it is a useful baseline: how well does a general model do?

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed, but slow on CPU at the official size: about 160-175 ms per frame
  on an Intel i7-13620H (about 4 FPS). With `--input-size 256`: about 54 ms
  (about 13-14 FPS).
- About 650 MB of disk space for the virtual environment (PyTorch + torchvision)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd deeplabv3-mobilenet
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd deeplabv3-mobilenet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`deeplabv3-mobilenet/`).

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
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Download the model

```
python download_model.py
```

This downloads torchvision's official weights (`deeplabv3_mobilenet_v3_large-fc3c493d.pth`,
about 44 MB) from download.pytorch.org into `models/` and checks the checksum.

Why the extra option on Linux: the normal Linux PyTorch from PyPI includes CUDA (GPU)
and is several GB. The extra index gives the CPU-only build. Windows and macOS
do not need it.

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
| `--input-size N` | Short side of the model input. Default 520 (official). 256 is about 3x faster but less detailed |

## Option B: Run with Docker

Run all commands from this folder (`deeplabv3-mobilenet/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-deeplabv3-mobilenet .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/deeplabv3-mobilenet/outputs" vbg-deeplabv3-mobilenet
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/deeplabv3-mobilenet/outputs" vbg-deeplabv3-mobilenet --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/deeplabv3-mobilenet/outputs" vbg-deeplabv3-mobilenet
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/deeplabv3-mobilenet/outputs" vbg-deeplabv3-mobilenet --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/deeplabv3-mobilenet/outputs" vbg-deeplabv3-mobilenet --camera 0
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

Measured on an Intel i7-13620H (CPU only), 640x480:

| Setting | Inference | FPS |
|---------|-----------|-----|
| `--input-size 520` (default) | about 160-175 ms | about 4 |
| `--input-size 256` | about 54 ms | about 13-14 |

The mask finds the person reliably (no background leaks), but its edges are very
soft and blobby: a wide halo around the person in blur/replace mode. Portrait
models (MODNet, RVM, MediaPipe) give much tighter edges.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| Low FPS | Use `--input-size 256`. |
| Linux: pip downloads several GB (`nvidia-...` packages) | You forgot `--extra-index-url https://download.pytorch.org/whl/cpu`. Delete `.venv` and install again with it. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `deeplabv3-mobilenet/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

torchvision (code and these weights) is BSD-3-Clause. Commercial use is allowed.
Note: the weights were trained on the COCO dataset; check COCO's terms if that
matters for your use.

- Docs: https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html
- Paper (DeepLabV3): https://arxiv.org/abs/1706.05587
- Paper (MobileNetV3): https://arxiv.org/abs/1905.02244
