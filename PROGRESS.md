# Progress

Status values: Not started / In progress / Done / Skipped (+ reason).

## Models

| # | Model | Folder | Status | Local run tested | Docker run tested | Notes |
|---|-------|--------|--------|------------------|-------------------|-------|
| 1 | PP-HumanSeg v1 (portrait Lite) | `pp-humanseg-v1/` | Done | Yes (webcam, image, video) | Partly: image + video OK. Display window: `libsm6` fix added but not yet re-tested (Docker Desktop hung) | Paddle 3.3.1 CPU. oneDNN disabled (crashes on this model). Has `live.py` (video-call view with a picker bar for blur and backgrounds) |
| 2 | PP-HumanSegV2 Lite (portrait) | `pp-humanseg-v2/` | In progress | Yes (image, video, webcam headless) | Not yet (Docker Desktop hung) | Copy of v1 with the V2 model (256x144). oneDNN also crashes. Has `live.py` |
| 3 | MediaPipe Selfie Segmentation (landscape) | `mediapipe-selfie/` | Not started | - | - | |
| 4 | MediaPipe Selfie Multiclass | `mediapipe-multiclass/` | Not started | - | - | |
| 5 | MODNet | `modnet/` | Not started | - | - | |
| 6 | Robust Video Matting (RVM) | `rvm/` | Not started | - | - | GPL-3.0 |
| 7 | SINet | `sinet/` | Not started | - | - | |
| 8 | U²-Net (portrait) | `u2net/` | Not started | - | - | Too slow for real time on CPU |
| 9 | DeepLabV3 MobileNet (torchvision) | `deeplabv3-mobilenet/` | Not started | - | - | |
| 10 | TF.js BodyPix | `bodypix/` | Not started | - | - | May need Node.js/browser; discuss first |
| 11 | YOLO segmentation (Ultralytics) | `yolo-seg/` | Not started | - | - | AGPL-3.0 |
| 12 | Apple Vision person segmentation | `apple-vision/` | Not started | - | - | macOS only, no Docker. Dev machine is Windows, so likely Skipped |
| 13 | NVIDIA Maxine | `nvidia-maxine/` | Not started | - | - | Needs RTX GPU (dev machine has RTX 4050); discuss first |

Folder names are planned and may change when each model is started.

## Project-level tasks

| Task | Status | Notes |
|------|--------|-------|
| Step 0: project skeleton | Done | Committed `076b8cf` |
| `system_info.py` | Done | Tested on Windows host and in a `python:3.10-slim` container |
| Hardware compatibility section | In progress | Results columns, Min hardware column and the "Which model for which machine?" placeholder are added. The full requirement text is still needed from the user |
| Sample video in `assets/samples/` | Not started | No freely licensed clip found yet |

## Log

| Date | What was done |
|------|---------------|
| 2026-10-04 | Step 0: checked the system (Windows 11, Python 3.10.0, Docker 26.1.1 with working `--gpus all`, RTX 4050, webcam). Created `CLAUDE.md`, `PROGRESS.md`, `README.md`, `.gitignore`, `.dockerignore`, `assets/` (2 generated backgrounds, 1 public-domain sample image). |
| 2026-10-04 | Added the hardware compatibility work: `system_info.py`, the Results table columns (device, RAM, resolution, local/Docker), the Min hardware column, and the "Which model for which machine?" placeholder. The full section text is still to come. |
| 2026-10-04 | PP-HumanSeg v1: added `pp-humanseg-v1/` (run.py, download_model.py, requirements.txt, Dockerfile, README). Paddle 3.3.1 on CPU, about 40 ms inference, about 20 FPS on webcam. oneDNN crashes on this old model in Paddle 3.x and is not faster in 2.6.2, so it is disabled. Switched compositing to `cv2.blendLinear` (113 ms down to 9 ms on a 960x1200 image). Docker needs `libgl1 libglib2.0-0 libgomp1`, plus `libsm6` for windows. Tested locally (webcam, image, video) and in Docker (image, video). The X11 display test via WSLg crashed because `libsm6` was missing. The fix is in the Dockerfile, but the rebuild could not finish because Docker Desktop stopped responding. Linux Docker webcam not tested. |
| 2026-10-04 | PP-HumanSeg v1: added `live.py`, a single large live window that works like a video call (b/r/o modes, n = next background). Reuses the model functions from `run.py`. Opened locally with the webcam. |
| 2026-10-04 | `live.py`: replaced the b/r/o/n keys with a picker bar (None, Blur light, Blur strong, one tile per background), chosen by mouse click or keys 1-9. |
| 2026-10-04 | Committed PP-HumanSeg v1 (`2e10454`). PP-HumanSeg V2: added `pp-humanseg-v2/` (same code as v1, V2 portrait model 256x144). Local image/video/webcam OK. V2 has a cleaner mask than v1 and is a bit faster (about 33 vs 38 ms on quiet runs). Benchmarks on this laptop vary up to 2x between runs; added a note to README Results. Docker not tested: Docker Desktop is not responding and needs a restart. |
