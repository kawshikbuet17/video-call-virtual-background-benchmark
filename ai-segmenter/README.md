# AI-Segmenter pipeline (matting model + YOLO object selection)

> **License warning: GPL-3.0 and AGPL-3.0 parts.** The default pipeline uses RVM
> (**GPL-3.0**) and Ultralytics YOLO (**AGPL-3.0**). Both require you to release your
> product's source code under the same license if you ship them (AGPL also when people
> use it over a network). Fine for learning and benchmarking; check with your legal team
> before any commercial use. See "License note" at the end.

## What this model is

**Models used by default (`run.py` and `live.py`): RVM makes the mask, and the
YOLO11 nano detector finds people and objects so only those areas are kept.**
BiRefNet is used only with `--matte birefnet`.

**Yerdnainspace/AI-Segmenter** (MIT, 2026, a student project at HAW Hamburg) is a
Windows desktop app for live video and broadcast work (camera or Blackmagic
DeckLink in, fill + matte out). It is built for an NVIDIA GPU with TensorRT. This
folder redoes its **live pipeline** as a short script, on the CPU (our own code):

1. **Matte**: a matting model makes a soft alpha (0..1) for the whole frame.
   - `rvm` (default): Robust Video Matting, MobileNetV3, as ONNX, downsample ratio 0.25
     (the app's value). The same model as `rvm/`.
   - `birefnet`: BiRefNet (`ZhengPeng7/BiRefNet`, MIT), a large, high-quality model, at
     512x512 like the app (its official size is 1024). Very slow on a CPU.
2. **YOLO**: every 3rd frame, the YOLO11 nano **detector** (boxes only, input 320,
   confidence 0.25, at most 8 objects) finds objects. The boxes are reused in between.
   The app keeps **all** detected objects by default ("select all"); `--persons-only`
   keeps only people.
3. **ROI**: the alpha is kept only inside the chosen boxes. Each box is made 6% bigger
   (at least 8 px) and the box mask is softened with a 15x15 blur. Outside the boxes
   everything becomes background. With no detection at all, the alpha is kept as it is.

So YOLO does **not** make the edges better. It only decides *which* parts of the matte
to keep. The edges come from the matting model.

Not included from the app:

- **ViTMatte**: it needs a trimap (sure person / sure background / unsure). The app
  makes it from brightness (bright = person, dark = background), which does not fit a
  normal webcam scene.
- **CorridorKey**: a green-screen keyer for 2048 px footage on a GPU.
- **MediaPipe selfie**: already in `mediapipe-selfie/`. MediaPipe needs
  `opencv-contrib-python` and Ultralytics needs `opencv-python`, and two OpenCV
  packages in one venv conflict. Note: the app treats MediaPipe's category mask
  `> 0` as the person, but in our test (`linux-fake-background-webcam/`) category 0
  was the person and 255 the background, so that option probably keeps the
  background. We did not run the app itself to confirm this.
- The app's GPU paths (CUDA, TensorRT), DeckLink output and user interface.

## Requirements

- Python 3.10 (tested with 3.10.0 on Windows 11)
- Windows, Linux or macOS (only Windows was tested)
- No GPU needed for `rvm`. On an Intel i7-13620H at 640x480: RVM alone about 29 ms,
  with YOLO about 60-66 ms (12-15 FPS). BiRefNet: 9-23 seconds per frame.
- About 1.2 GB of disk space for the virtual environment (PyTorch, Ultralytics,
  transformers), plus 445 MB if you download BiRefNet

## Quick start

From the project root, the first time:

Windows (PowerShell):

```powershell
cd ai-segmenter
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe live.py
```

Linux / macOS:

```bash
cd ai-segmenter
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python download_model.py
.venv/bin/python live.py
```

After that, only the last line is needed. Use `run.py` instead of `live.py` for
the 4-panel benchmark view. Details for each step are below.

## Option A: Run locally

Run all commands from this folder (`ai-segmenter/`).

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
python download_model.py              # YOLO11n + RVM, about 21 MB
python download_model.py --birefnet   # also BiRefNet, about 445 MB (optional)
```

YOLO11n comes from Ultralytics' official assets release (v8.4.0), RVM from the RVM
GitHub release (v1.0.0), and BiRefNet from Hugging Face, pinned to one revision. All
checksums are checked. BiRefNet's network code (`birefnet.py`) is saved in
`models/birefnet/` with the weights, so nothing is downloaded when you run it.

### 3. Run

```
python run.py
```
Uses the webcam (camera 0) with RVM + YOLO. Use `--camera 1` for a second camera.

```
python run.py --persons-only                 # keep only people, not other objects
python run.py --no-yolo                      # RVM alone, to compare
python run.py --matte birefnet --image ../assets/samples/person_portrait.jpg
python run.py --image ../assets/samples/person_portrait.jpg
python run.py --video path/to/video.mp4
```

Use BiRefNet on images only: one frame takes many seconds on a CPU.

#### Live video call view

```
python live.py                        # RVM + YOLO11n (default)
python live.py --matte birefnet       # BiRefNet: the picture updates only every 10-20 s on a CPU
```

One large, resizable window with your camera and the effect applied, like a video
call. A picker bar under the video lets you choose:
**None | Blur light | Blur strong | one tile per image in `assets/backgrounds/`**.
Click a tile or press its number key (`1`-`9`). To add your own background, copy
a `.jpg` or `.png` into `assets/backgrounds/` and restart. Use `--camera 1` for
another camera. `live.py` also takes `--persons-only` and `--no-yolo`. Local only
(it needs a webcam and a window).

Useful options for `run.py`:

| Option | What it does |
|--------|--------------|
| `--matte NAME` | `rvm` (default) or `birefnet` (needs `download_model.py --birefnet`; very slow) |
| `--persons-only` | YOLO keeps only people (the app keeps every detected object) |
| `--no-yolo` | Matting model alone, no object selection |
| `--save` | Save the result into `outputs/` (image: 4 PNG files, video/webcam: an MP4 of the side-by-side view) |
| `--no-display` | No window. Saves the output and prints the average FPS and inference time |
| `--background path` | Background image for replacement (default: `../assets/backgrounds/simple_room.png`) |
| `--max-frames N` | Stop after N frames (handy for webcam benchmarks) |

## Option B: Run with Docker

Run all commands from this folder (`ai-segmenter/`). The model is downloaded
during the build.

TODO: the Docker image has not been built or tested yet. Docker Desktop stopped
responding during this work.

### Build

```
docker build -t vbg-ai-segmenter .
```

### Run on an image or video (all operating systems)

Results are written to `outputs/` on your computer. `assets/` is mounted so you
can use any file in `assets/samples/`.

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/ai-segmenter/outputs" vbg-ai-segmenter
docker run --rm -v "${PWD}\..\assets:/app/assets" -v "${PWD}\outputs:/app/ai-segmenter/outputs" vbg-ai-segmenter --no-display --video ../assets/samples/my_clip.mp4
```

Linux / macOS:

```bash
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/ai-segmenter/outputs" vbg-ai-segmenter
docker run --rm -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/ai-segmenter/outputs" vbg-ai-segmenter --no-display --video ../assets/samples/my_clip.mp4
```

The first command (no arguments) processes `assets/samples/person_portrait.jpg`.
To use your own file, copy it into `assets/samples/` first.

### Webcam + display in Docker (Linux only)

Docker on Windows and macOS cannot easily access the webcam or open windows.
Use the local run (Option A) there.

On Linux with an X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/../assets:/app/assets" -v "$(pwd)/outputs:/app/ai-segmenter/outputs" vbg-ai-segmenter --camera 0
xhost -local:docker
```

- `--device /dev/video0` gives the container your webcam.
- `-e DISPLAY` and `-v /tmp/.X11-unix` let the container open a window on your screen.
- `xhost +local:docker` allows that; `xhost -local:docker` removes the permission again.

TODO: not tested yet (Docker was not available, and no Linux machine with a
webcam).


The Docker image downloads YOLO11n and RVM only, not BiRefNet.

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
The mask is white where the person is kept. The top bar shows FPS (the whole
pipeline) and the inference time in ms (YOLO, when it runs on that frame, plus the
matting model).

Measured on an Intel i7-13620H (CPU only), 640x480 test video. During these runs
other programs used about half of the CPU and most of the RAM, so the numbers
varied a lot:

| Setup | Time per frame | FPS |
|-------|----------------|-----|
| RVM alone (`--no-yolo`) | about 29 ms | 25-26 |
| RVM + YOLO (default) | 60-66 ms | 12-15 |
| BiRefNet (512) | 9-23 seconds | far from real time |

YOLO11n itself took 65-110 ms per run in these tests, which is slow for a nano
detector; on a quiet machine expect less.

Quality on the test clip:

- **YOLO selection does not remove the flag leak** on this clip. The person's box
  (plus 6%) already covers the flag area, so the matte's mistake stays. Box selection
  only removes mistakes outside the person's box (for example other people or a
  poster), which is what the app uses it for.
- **BiRefNet gave the cleanest mask we have seen on the test image**: no flag, sharp
  hair. But it is not usable live on a CPU.

## Common problems and fixes

| Problem | Fix |
|---------|-----|
| `Model not found ... Run first: python download_model.py` | Run `python download_model.py` (add `--birefnet` for BiRefNet). |
| `This modeling file requires the following packages ... einops, kornia` | Install everything from `requirements.txt`; BiRefNet's code needs them. |
| `Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed` | Harmless. The BiRefNet download uses normal HTTP instead (a few minutes). |
| BiRefNet takes many seconds per frame | Expected on a CPU. Use it on single images, or use `rvm`. |
| `Could not open webcam 0`, or no frames | Close other apps using the camera (Zoom, Teams, a browser tab). Try `--camera 1`. On Windows, check Settings > Privacy > Camera. |
| A `settings.json` appears in `%APPDATA%\Ultralytics` (Windows) | Normal: Ultralytics keeps its settings there. |
| Linux: pip downloads several GB (CUDA) | Install with `--extra-index-url https://download.pytorch.org/whl/cpu` as shown above. |
| PowerShell says running scripts is disabled when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run.py` directly without activating. |
| Docker: `Could not read background image` | The `assets` folder is not mounted. Run the command from inside `ai-segmenter/` exactly as shown. |
| Docker: `libGL.so.1` not found | You changed the Dockerfile base image. Keep the `apt-get install` line. |
| Docker on Linux: files in `outputs/` are owned by root | Add `--user "$(id -u):$(id -g)"` to the `docker run` command. |
| Docker on Linux: `could not connect to display` | Run `xhost +local:docker` first, and check that `echo $DISPLAY` is not empty. |

## License note

- **AI-Segmenter** (the app): **MIT**. Its LICENSE also says that model weights and
  other external parts have their own licenses. The code here is our own short version
  of its pipeline.
- **RVM**: **GPL-3.0** (code and model). See `rvm/` for details.
- **Ultralytics YOLO11**: **AGPL-3.0** (code and model), with a paid Enterprise license
  for closed-source use. See `yolo-seg/` for details.
- **BiRefNet**: MIT (code and weights on Hugging Face).

- Repo: https://github.com/Yerdnainspace/AI-Segmenter
- License: https://github.com/Yerdnainspace/AI-Segmenter/blob/main/LICENSE
- The YOLO selection code: https://github.com/Yerdnainspace/AI-Segmenter/blob/main/ai_segmenter/yolo_controls.py
- BiRefNet: https://github.com/ZhengPeng7/BiRefNet and https://huggingface.co/ZhengPeng7/BiRefNet
- RVM: https://github.com/PeterL1n/RobustVideoMatting
