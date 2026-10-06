# YOLO segmentation (Ultralytics YOLO26 nano)

> **License warning: AGPL-3.0.** Ultralytics' code and models are AGPL-3.0. If you
> ship YOLO inside a product, or run it on a server that other people use over a
> network (for example a video call service), AGPL-3.0 requires you to release
> that product's source code under the AGPL too, unless you buy Ultralytics'
> commercial license. Fine for learning and benchmarking; check with your legal
> team before any commercial use. See "License note" at the end.

## What this model is

YOLO is a general object detector from Ultralytics. The `-seg` models also draw
the outline of every object they find (instance segmentation). They know 80
everyday classes (the COCO dataset); this folder keeps only **person** and joins
the outlines of all people into one mask. It is not made for video calls like the
portrait models, so it is a useful comparison.

| `--weights` | Model | File size |
|-------------|-------|-----------|
| `yolo26n-seg` (default) | YOLO26 nano, the newest (2026) | 6.7 MB |
| `yolo11n-seg` | YOLO11 nano | 6.2 MB |
| `yolov8n-seg` | YOLOv8 nano | 7.1 MB |

All run with the official `ultralytics` Python package and PyTorch, on the CPU.
Ultralytics cuts each person's mask at 0.5, so the mask is hard (0 or 1, no soft
edge). `--imgsz` sets the model input size: 640 is Ultralytics' default, 320 is
about twice as fast.

### `--soft-edge` (optional)

YOLO's mask is hard and can have specks and holes. `--soft-edge` adds the clean-up
from [Number-L/realtime-person-cutout](https://github.com/Number-L/realtime-person-cutout)
(MIT; written here as our own code). For each person:

1. Keep only the **largest connected piece** (drops loose specks).
2. **Fill holes** inside the person. A hole that touches the picture border counts
   as outside and stays.
3. **Soft edge** with a distance transform: alpha is 0 at the edge and 1 from 2 px
   inside. This also moves the edge in a little, so no light halo of the old
   background shows.

Then the mask is smoothed over time (70% this frame, 30% the previous one). Number-L
does this per tracked person with ByteTrack (an extra package, `lap`); here it is done
on the joined mask, which is the same when one person is in the picture.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows was tested)
- No GPU needed. On an Intel i7-13620H at 640x480: about 45-65 ms per frame at
  `--imgsz 640` (15-19 FPS), about 25-35 ms at `--imgsz 320` (23-32 FPS).
- About 1 GB of disk space for the virtual environment (PyTorch and Ultralytics)

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd yolo-seg
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py --imgsz 320
```

Linux / macOS:

```bash
cd yolo-seg
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python download_model.py
.venv/bin/python live.py --imgsz 320
python live.py --imgsz 320 --soft-edge
python live.py --imgsz 320 --main-person --soft-edge   # only you, soft edge
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`yolo-seg/`).

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

On Linux the extra index gives the CPU-only PyTorch build. Without it, pip
downloads the CUDA build, which is several GB.

### 2. Download the models

```
python download_model.py
```

This downloads the three `.pt` files (about 20 MB together) from Ultralytics'
official assets release (v8.4.0) on GitHub into `models/` and checks their
checksums. `run.py` loads them from `models/`, so Ultralytics does not download
anything by itself.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0) with `yolo26n-seg` at 640. Use `--camera 1` for a second camera.

```
python run.py --imgsz 320                    # about twice as fast
python run.py --weights yolo11n-seg          # or yolov8n-seg
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
```

#### Live video call view

```
python live.py --imgsz 320
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. `live.py` also takes `--weights`. Local only (it needs a webcam
and a window).

Useful options for `run.py`:

| Option | What it does |
|--------|--------------|
| `--weights NAME` | `yolo26n-seg` (default), `yolo11n-seg` or `yolov8n-seg` |
| `--imgsz N` | Model input size, long side (default 640; 320 is faster) |
| `--soft-edge` | Largest piece only, holes filled, 2 px soft edge, smoothed over time |
| `--main-person` | Keep only the largest person (usually you, closest to the camera); people behind you are dropped |
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

In `run_all_live.py` YOLO runs with `--imgsz 320` (set in `EXTRA_ARGS`).

## Option B: Run with Docker

Run all commands from this folder (`yolo-seg/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-yolo-seg .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/yolo-seg/outputs" vbg-yolo-seg
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/yolo-seg/outputs" vbg-yolo-seg --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/yolo-seg/outputs" vbg-yolo-seg
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/yolo-seg/outputs" vbg-yolo-seg --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/yolo-seg/outputs" vbg-yolo-seg --camera 0
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
The mask is white where YOLO found a person. The top bar shows FPS (the whole
pipeline) and the inference time in ms. Here "inference" is the whole Ultralytics
call: its preprocessing (1-3 ms), the network, and the mask decoding (2-3 ms).

Measured on an Intel i7-13620H (CPU only), 640x480, PyTorch with 4 threads. This
laptop varied a lot during these runs (heat), so the numbers are ranges:

| Model | Video file | Webcam |
|-------|-----------|--------|
| `yolo26n-seg`, `--imgsz 640` | 47-66 ms, 14-18 FPS | 53-55 ms, 16-17 FPS |
| `yolo26n-seg`, `--imgsz 320` | 24-35 ms, 23-32 FPS | 25 ms, 32 FPS |
| `yolo11n-seg`, 640 | 45-48 ms, 18 FPS | - |
| `yolov8n-seg`, 640 | 50 ms, 18 FPS | - |

8 PyTorch threads were not reliably faster than 4.

Quality on the test clip: YOLO26 gave a clean mask with no flag leak and no holes,
even at 320 (better than YOLO11, which had holes at the shoulder patch and the
badge). The outline is hard and simplified, so hair edges look cut out rather
than soft.

With `--soft-edge` (same laptop, test video): about 5 ms more per person.

| Setting | Without | With `--soft-edge` |
|---------|---------|--------------------|
| `--imgsz 640` | 18.8-18.9 FPS | 16.0-17.1 FPS |
| `--imgsz 320` | 34.7 FPS | 26.3-27.0 FPS |
| `--imgsz 320`, webcam | 31.7 FPS | 19.4 FPS |

It filled YOLO11's hole at the shoulder patch, but not the holes at the badge,
because they touch the bottom of the picture. The edge is a little softer. People in
the background are still kept: YOLO finds every person, and the clean-up works per
person. Use `--main-person` for that: it keeps only the largest person. On the webcam
it removed a colleague sitting behind (tested). If someone comes closer to the camera
than you, they become the largest and are kept instead.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` from this folder. |
| `Could not open webcam 0` | Close other apps using the camera (Zoom, Teams, browser). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| Low FPS | Use `--imgsz 320`. |
| A `settings.json` appears in `%APPDATA%\Ultralytics` (Windows) | Normal: Ultralytics keeps its settings there. Delete the folder to remove it. |
| Linux: pip downloads several GB (CUDA) | Install with `--extra-index-url https://download.pytorch.org/whl/cpu` as shown above. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `yolo-seg/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

**AGPL-3.0** for Ultralytics' code and its models (stated in the repo's LICENSE
and on Ultralytics' license page). You may use, study and change them freely. But
if you distribute software that includes them, or let people use it over a network,
that software must also be released under AGPL-3.0 with its source code. For a
closed-source product, Ultralytics sells an Enterprise license. Check with your
legal team first.

- Repo: https://github.com/ultralytics/ultralytics
- License file: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Ultralytics licensing: https://www.ultralytics.com/license
- Segmentation docs: https://docs.ultralytics.com/tasks/segment/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- Model release: https://github.com/ultralytics/assets/releases/tag/v8.4.0
- Soft-edge recipe: https://github.com/Number-L/realtime-person-cutout (MIT)
