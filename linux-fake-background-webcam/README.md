# Linux-Fake-Background-Webcam (lfbw) recipe

## What this model is

**fangfufu/Linux-Fake-Background-Webcam** ("lfbw", **GPL-3.0**, master branch, last
changed 2026-02-07) is a Linux tool. It reads your webcam, replaces or blurs the
background, and sends the result to a virtual webcam (v4l2loopback), so any video
call app can use it. It needs Linux, so this folder only takes its **picture
processing** and shows the result in a window, like the other folders. The code
here is our own (see the license note).

The model is **MediaPipe selfie segmenter (landscape)**, the same file as in
`mediapipe-selfie/`. What is different is lfbw's simple recipe around it, all
plain OpenCV:

1. The frame is shrunk to 256 px wide, and MediaPipe gives a **hard mask**
   (each pixel is person or background, no in-between).
2. The mask is scaled up to the frame and blurred (Gaussian 7x7).
3. **Threshold 0.75**: pixels above it are person (1), the rest background (0).
4. **Postprocess**: dilate 5x5 (the person grows by 2 px), then a 10x10 box blur
   (a soft edge).
5. **Background**: a Gaussian blur with kernel 21, or an image.

Options with lfbw's names: `--threshold` (default 75), `--no-postprocess`, and
`--mask-update-speed` (see the next note).

**Note: lfbw's "mask update speed" does nothing.** lfbw is meant to mix each new
mask 50/50 with the previous one (less flicker). But it keeps the previous mask as
the same buffer that the next mask is written into, so it mixes the new mask with
itself. We tested this: a frame of 1.0 followed by a frame of 0.0 gives 0.0, not
0.5. So this folder does no smoothing by default, like lfbw really does. Use
`--mask-update-speed 50` to get the smoothing lfbw meant to have.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11). lfbw itself asks for Python 3.11+,
  but that is for its own code, which is not used here.
- Windows, Linux or macOS (only Windows was tested). No v4l2loopback needed.
- No GPU needed. On an Intel i7-13620H at 640x480: model about 6 ms, 53-68 FPS
  for the whole pipeline.
- About 350 MB of disk space for the virtual environment (MediaPipe is large)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd linux-fake-background-webcam
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd linux-fake-background-webcam
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`linux-fake-background-webcam/`).

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

This downloads `selfie_segmenter_landscape.tflite` (about 250 KB) from Google's
MediaPipe model storage into `models/` (the same URL lfbw uses). The link points to
Google's "latest" version; if Google updates the file, the script keeps it and
prints a warning with the new checksum.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0). Use `--camera 1` for a second camera.

```
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
python run.py --threshold 50                 # accept less certain pixels as the person
python run.py --no-postprocess               # hard edge, no dilate and no soft edge
python run.py --mask-update-speed 50         # smoothing between frames (see the note above)
```

#### Live video call view

```
python live.py
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Blur light is lfbw's default blur (kernel 21). Blur strong adds lfbw's `blur=50`
effect on top (a second Gaussian, kernel 51).
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. `live.py` also takes `--threshold`, `--no-postprocess` and
`--mask-update-speed`. Local only (it needs a webcam and a window).

Useful options for `run.py`:

| Option | What it does |
|--------|--------------|
| `--threshold N` | Percent a pixel must reach to count as the person (default 75) |
| `--no-postprocess` | No dilate and no box blur of the mask |
| `--mask-update-speed N` | Percent of the new mask mixed in per frame (default 100 = no smoothing) |
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

## Option B: Run with Docker

Run all commands from this folder (`linux-fake-background-webcam/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-linux-fake-background-webcam .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/linux-fake-background-webcam/outputs" vbg-linux-fake-background-webcam
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/linux-fake-background-webcam/outputs" vbg-linux-fake-background-webcam --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/linux-fake-background-webcam/outputs" vbg-linux-fake-background-webcam
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/linux-fake-background-webcam/outputs" vbg-linux-fake-background-webcam --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/linux-fake-background-webcam/outputs" vbg-linux-fake-background-webcam --camera 0
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

Measured on an Intel i7-13620H (CPU only), 640x480: about 6 ms inference and
63-68 FPS on a video file; about 7-8 ms and 53 FPS with the webcam. All the
post-processing is cheap OpenCV, so it is one of the fastest folders.

Quality: the outline is clean and stable on the webcam. On the test clip the
recipe shows its weak side: the threshold turns the mask hard, so a wrong patch
(a piece of the flag) shows up solid instead of faint, and there are a few small
holes in the body. The blur (kernel 21 on the full frame) is lighter than in the
other folders.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `WARNING: this is not the model file this folder was tested with` | Google updated the model behind the "latest" link. It should still work. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| Bits of the background show up solid | Raise `--threshold` (for example 85), or use `mediapipe-selfie/`, which keeps the soft mask. |
| Parts of the person are cut off | Lower `--threshold` (for example 50). |
| `W0000 ... inference_feedback_manager.cc ...` | Harmless MediaPipe log message. |
| Two OpenCV packages installed (`opencv-python` and `opencv-contrib-python`), strange `cv2` errors | Use a fresh venv with only `requirements.txt`. MediaPipe brings `opencv-contrib-python`; do not install `opencv-python` as well. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `linux-fake-background-webcam/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

- **Warning: lfbw is GPL-3.0.** If you copy or change its code and give it to
  others, your program must also be GPL-3.0 and you must share its source. This
  folder does not copy lfbw's code: it is our own short version of the same steps
  (a resize, a threshold, a dilate and blurs), written for this project.
- **MediaPipe** (code) is Apache-2.0. The model comes with a model card that
  describes its intended use and limits (see the model page).

- Repo: https://github.com/fangfufu/Linux-Fake-Background-Webcam
- The processing code: https://github.com/fangfufu/Linux-Fake-Background-Webcam/blob/master/lfbw/lfbw.py
- License: https://github.com/fangfufu/Linux-Fake-Background-Webcam/blob/master/LICENSE
- Model page: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
