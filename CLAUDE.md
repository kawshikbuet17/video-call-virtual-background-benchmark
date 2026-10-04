# CLAUDE.md — notes for future Claude sessions

## Project purpose

Run and compare real-time person segmentation models for video call virtual
backgrounds (background blur and background replacement) on a local PC.
This is a learning and benchmarking project: keep everything simple, readable
and beginner-friendly. No frameworks, no unnecessary abstractions.

The full original brief is in `prompts.txt`. Progress lives in `PROGRESS.md`.

## Working rules (very important)

1. Work on ONE model at a time. Never set up several models in one go.
2. Before starting a model, briefly say what you will do (files, dependencies,
   model download, Docker setup) and wait for the user's "go".
3. After finishing a model, run it yourself (use `assets/samples/` if no webcam),
   both locally and in Docker, then tell the user how to run it.
4. NEVER commit automatically. When a step is done, ask "Shall I commit?" and
   suggest a short commit message. Commit only after the user says yes.
5. Keep `CLAUDE.md` and `PROGRESS.md` up to date after every step.
6. Never invent links. Only add paper/repo/doc links you have verified exist
   (HTTP 200 and correct content). If unsure, leave a `TODO` and say so.
7. If something fails (dependency conflicts, broken downloads, unsupported OS,
   Docker issues), explain the problem simply and propose options. Do not
   silently work around it.

## CPU only (user decision, 2026-10-04)

Run and benchmark every model on the CPU. Do not add `--gpu` options,
`requirements-gpu.txt`, GPU Dockerfiles or `--gpus all` commands, and do not
download GPU packages (onnxruntime-gpu, CUDA/cuDNN wheels, paddlepaddle-gpu,
CUDA PyTorch). In each model README the "GPU" section says: "Not provided. This
project runs every model on the CPU." The "GPU needed" column in README.md is
information only. This overrides the GPU parts of the Docker rules below.

## Developer machine (checked 2026-10-04)

- Windows 11 Home (10.0.26200), Intel i7-13620H, 16 GB RAM
- NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), driver 536.52
- Python 3.10.0 (default `python`), Python 3.7 also installed (`py -3.7`)
- Docker Desktop 26.1.1 (WSL2 backend). `docker run --gpus all` works
  (tested with `nvidia-smi -L` inside an `ubuntu:22.04` container).
- Webcam: Chicony USB2.0 Camera

Note: `python:3.10-slim` currently resolves to Python 3.10.22, while local is
3.10.0. Pin the exact versions that were tested in each `requirements.txt`.

## Hardware compatibility (low-end to high-end)

TODO: the user's full "Hardware compatibility" requirement has not been pasted
yet. Ask for it and add it here. Agreed so far:

- `system_info.py` (root, standard library only) prints OS, Python, CPU, RAM,
  GPU and Docker version. Run it on the host. Inside Docker on Windows/macOS it
  reports the VM's RAM and no GPU.
- Every Results row in `README.md` records: Model | Device (CPU/GPU name) | RAM |
  Resolution | Avg FPS | Avg inference (ms) | Local/Docker | Notes.
- The models table in `README.md` has a "Min hardware" column. Fill it from real
  test results only; leave `TBD` until then.
- `README.md` has a "Which model for which machine?" guide (low-end / mid-range
  / high-end). It is a placeholder until there are results.

## Folder conventions

```
assets/backgrounds/   shared background images for replacement
assets/samples/       shared test images/videos
<model-name>/         one folder per model, kebab-case (e.g. pp-humanseg-v1/)
  README.md           short, complete run guide (see structure below)
  requirements.txt    CPU packages, pinned
  requirements-gpu.txt  only if GPU needs different packages
  download_model.py   downloads weights into <model-name>/models/
  run.py              single entry point (benchmark view: 4 panels)
  live.py             optional live video-call view (one big window + picker bar:
                      None / Blur light / Blur strong / backgrounds; click or 1-9);
                      imports the model functions from run.py, local only
  Dockerfile          built from inside the model folder
  .dockerignore       copy of the root template
  models/             downloaded weights (git-ignored)
  outputs/            results and screenshots (git-ignored)
  .venv/              the model's own virtual environment (git-ignored)
```

Each model folder must work on its own with its own `.venv`, because models
have conflicting dependencies (Paddle, PyTorch, MediaPipe, TensorFlow, ...).

Adding a model: create the folder with the files above, then update the
models table (including Min hardware), resources and results in `README.md`,
and the row in `PROGRESS.md`.

### Model README sections

What this model is · Requirements · Option A: Run locally (venv for Windows and
Linux/macOS, download model, run webcam/image/video) · Option B: Run with Docker
(build, image/video run for PowerShell and Linux/macOS, Linux-only webcam +
display, GPU run if useful) · Controls · Expected result · Common problems and
fixes · License note (warn clearly for GPL/AGPL or other restrictive licenses).

## requirements.txt rules

- One `requirements.txt` per model folder. Never a shared one at the root.
- Pin exact versions (`package==x.y.z`) that were actually installed and tested.
- Only what is needed to run that model. No unused packages.
- First line: `# Tested with Python 3.10` (the real tested version).
- Short comments for anything non-obvious (old version needed, CPU vs GPU).
- GPU variants go in `requirements-gpu.txt`; CPU stays in `requirements.txt`.
- `opencv-python` for local runs. In Docker use `opencv-python-headless` only if
  the container is headless, and explain the choice in a comment.

## Docker rules

- Docker is optional; local run is the primary way.
- One `Dockerfile` per model folder: `docker build -t vbg-<model-name> .`
- Slim official Python base image matching the tested version
  (e.g. `python:3.10-slim`). Install only needed system libraries and comment why.
- Install from the same `requirements.txt` as the local run.
- Download weights during the build (`RUN python download_model.py`) so the image
  works out of the box. Manual downloads (login, license) go in the README.
- `run.py --no-display` (headless) is the default way to run in Docker.
- Mount `assets/` and the model's `outputs/` as volumes. Give exact `docker run`
  commands for Windows PowerShell and Linux/macOS.
- Linux only: optional webcam + display command (`--device /dev/video0`, X11).
- GPU models: optional `--gpus all` command; a GPU Dockerfile variant only if
  really needed.
- Keep images small. No docker-compose or orchestration unless asked.

## run.py behaviour (same for every model)

- Input: webcam by default; `--image path` and `--video path`.
- Window shows side by side: original | mask | blurred background | replaced
  background (image from `assets/backgrounds/`).
- FPS and inference time (ms) drawn on screen.
- Keys: `q` quit, `s` save screenshot to `outputs/`.
- `--save` writes the processed image/video to `outputs/`.
- `--no-display`: no window, saves output, prints average FPS and inference
  time at the end.
- OpenCV for camera, display and compositing. One file if possible, clearly
  split into: load model → preprocess → inference → postprocess → composite →
  display/save.
- Preprocess correctly per model (input size, normalization, RGB/BGR,
  NCHW/NHWC) and comment why.

## Lessons learned (apply to every model)

- Paddle's default CPU config uses 1 thread; more threads were not faster for
  PP-HumanSeg v1. oneDNN crashes on old `.pdmodel` files in Paddle 3.x.
- Blend with `cv2.blendLinear(frame, bg, alpha, 1 - alpha)` (float32 alpha).
  The numpy formula `a * frame + (1 - a) * bg` is about 10x slower.
- Blur by shrinking 8x, blurring, then scaling back up. A big Gaussian kernel
  on the full frame is much slower.
- `python:3.10-slim` needs `libgl1 libglib2.0-0` for `opencv-python`, plus
  `libsm6` to open a window (X11). Paddle also needs `libgomp1`.
- Docker X11 display can be tested on this Windows machine from WSL
  (`wsl -d Ubuntu-24.04`, WSLg gives `DISPLAY=:0`). The webcam cannot be tested in Docker here.
- Git Bash rewrites `/paths` in docker arguments; prefix commands with
  `MSYS_NO_PATHCONV=1` or use PowerShell.
- Mark `run.py` and `download_model.py` executable in git
  (`git add --chmod=+x`) and give them a `#!/usr/bin/env python3` line.

- Benchmarks on this laptop vary up to 2x between runs (P-/E-cores, turbo,
  heat). Close other heavy apps (and any open live.py window), run 2-3 times,
  report typical values. Pinning to P-cores (affinity 0xFFF) did not fix it.

- MediaPipe installs `opencv-contrib-python`; pin that instead of `opencv-python`
  (never both). Use the Tasks API (`mediapipe.tasks.python.vision`), not the
  deprecated `mp.solutions`.
- The MediaPipe docs moved from ai.google.dev to developers.google.com/edge/mediapipe/.

## Coding style

- Simple, readable Python. Minimal dependencies. Plain functions over classes.
- Comments where they help a beginner understand *why*.
- READMEs: short, simple, complete, plain English, copy-paste commands,
  no marketing language.

## Git

- This folder is its own git repo (branch `master`).
- Commit only when the user says yes (rule 4).
- Never commit model weights, virtual environments or outputs.
