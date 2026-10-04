# Fast_Portrait_Segmentation (ERD seg-matting, PyTorch)

## What this model is

`lizhengwei1992/Fast_Portrait_Segmentation` (2018-2019) explored fast portrait
segmentation networks for phones. The repo has network code but no weights; its
README says the final model was released in the author's follow-up repo
`mobile_phone_human_matting`: **ERD seg-matting** (a light segmentation net plus a
small matting head that outputs a soft alpha matte), 256x256 input, 3.4 MB.

This is the **original PyTorch version** of the model that `ncnn-portrait-segmentation`
runs with ncnn. The masks are almost identical (average difference 0.58 out of 255),
so these two folders compare the same model in two runtimes.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows and Linux/Docker were tested)
- No GPU needed. Inference is about 17-20 ms per frame on an Intel i7-13620H.
- About 600 MB of disk space for the virtual environment (PyTorch)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd fast-portrait-segmentation
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd fast-portrait-segmentation
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`fast-portrait-segmentation/`).

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

This downloads the weights (`ckpt_lastest.pth`, about 3.4 MB) **and the network code**
(`segnet.py`) from `mobile_phone_human_matting` into `models/` and checks the checksums.
The code is downloaded instead of copied into this project because the author's repos
have no license (see "License note").

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

## Option B: Run with Docker

Run all commands from this folder (`fast-portrait-segmentation/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-fast-portrait-segmentation .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/fast-portrait-segmentation/outputs" vbg-fast-portrait-segmentation
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/fast-portrait-segmentation/outputs" vbg-fast-portrait-segmentation --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/fast-portrait-segmentation/outputs" vbg-fast-portrait-segmentation
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/fast-portrait-segmentation/outputs" vbg-fast-portrait-segmentation --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/fast-portrait-segmentation/outputs" vbg-fast-portrait-segmentation --camera 0
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

Measured on an Intel i7-13620H (CPU only), 640x480: about 17 ms inference and
41 FPS on a video file; about 19.5 ms and 36 FPS with the webcam. Same quality as
`ncnn-portrait-segmentation`: very clean, smooth mask with no background leaks.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| `No module named 'segnet'` | `models/segnet.py` is missing. Run `python download_model.py`. |
| Linux: pip downloads several GB (`nvidia-...` packages) | You forgot `--extra-index-url https://download.pytorch.org/whl/cpu`. Delete `.venv` and install again with it. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `fast-portrait-segmentation/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

**No license.** Neither `lizhengwei1992/Fast_Portrait_Segmentation` nor
`lizhengwei1992/mobile_phone_human_matting` has a license file, so by default the
code and weights are "all rights reserved". Use this for learning and benchmarking
only, and ask the author before using it in a product. This is also why this project
does not include the author's code: `download_model.py` downloads it to `models/`.

- Repo: https://github.com/lizhengwei1992/Fast_Portrait_Segmentation
- Weights and code: https://github.com/lizhengwei1992/mobile_phone_human_matting
