# MODNet (portrait matting)

## What this model is

MODNet is a real-time portrait **matting** model (AAAI 2022). Matting means it
predicts a soft alpha value (0..1) for every pixel, so hair and edges blend
smoothly instead of being cut out hard. We use the official ONNX export of the
"photographic portrait matting" model and run it with ONNX Runtime (no PyTorch).

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed, but slow on CPU at the official input size: about 170 ms per
  frame on an Intel i7-13620H (about 5 FPS). Use `--ref-size 256` for about
  18 FPS with less detail.
- About 300 MB of disk space for the virtual environment (CPU)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd modnet
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd modnet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`modnet/`).

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

This downloads the official ONNX model from Google Drive into `models/` (about 25 MB)
and checks its checksum. If it times out, run it again.

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
| `--ref-size N` | Short side of the model input. Default 512 (official). 256 is about 4x faster but less detailed |

## Option B: Run with Docker

Run all commands from this folder (`modnet/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-modnet .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/modnet/outputs" vbg-modnet
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/modnet/outputs" vbg-modnet --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/modnet/outputs" vbg-modnet
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/modnet/outputs" vbg-modnet --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/modnet/outputs" vbg-modnet --camera 0
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
| `--ref-size 512` (default) | about 170 ms | about 5 |
| `--ref-size 256` | about 46 ms | about 18 |

The mask is the cleanest of the models tested so far: soft hair edges, and busy
backgrounds (for example a flag) do not leak in.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `download_model.py`: timeout, or "does not match the expected checksum" | Google Drive is slow or rate-limited. Run it again later. Or download the file by hand from the link in https://github.com/ZHKKKe/MODNet/tree/master/onnx and save it as `models/modnet_photographic_portrait_matting.onnx`. |
| Very low FPS | Expected at the default size on CPU. Try `--ref-size 256`. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `modnet/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

The MODNet repo states: "The code, models, and demos in this repository
(excluding GIF files under the folder `doc/gif`) are released under the Apache
License 2.0". Commercial use is allowed. Keep the license and notices if you
redistribute it.

- Repo: https://github.com/ZHKKKe/MODNet
- ONNX model: https://github.com/ZHKKKe/MODNet/tree/master/onnx
- Paper: https://arxiv.org/abs/2011.11961
