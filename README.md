# Video Call Virtual Background Benchmark

Run and compare real-time person segmentation models for video call virtual
backgrounds (background blur and background replacement) on your own PC. Each
model lives in its own folder with its own virtual environment, a single
`run.py`, and an optional Dockerfile. Every `run.py` shows the same view
(original | mask | blurred background | replaced background) with FPS and
inference time, so results are easy to compare across models and machines.

## Folder structure

```
video-call-virtual-background-benchmark/
├── README.md            this file
├── PROGRESS.md          status of each model and a work log
├── CLAUDE.md            project rules and conventions
├── system_info.py       prints your hardware details (run this first)
├── .gitignore
├── .dockerignore        template, copied into each model folder
├── assets/
│   ├── backgrounds/     images for background replacement
│   └── samples/         test images/videos (add your own here)
└── <model-name>/        one folder per model, for example pp-humanseg-v1/
    ├── README.md        exact commands for this model
    ├── requirements.txt
    ├── download_model.py
    ├── run.py
    ├── Dockerfile
    ├── models/          downloaded weights (not in git)
    └── outputs/         results and screenshots (not in git)
```

## Quick start

First, check your hardware (no packages needed):

```
python system_info.py
```

Then pick a model folder and use one of two ways to run it.

**A. Locally (recommended, supports webcam on every OS)**

1. `cd <model-name>`
2. Create a virtual environment in `.venv` and install `requirements.txt`.
3. Run `python download_model.py`.
4. Run `python run.py` (webcam), or `--image` / `--video` with a file.
5. For a live video-call view where you pick blur or a background, run
   `python live.py` (in model folders that have it).

**B. With Docker (image/video files on every OS; webcam on Linux only)**

1. `cd <model-name>`
2. `docker build -t vbg-<model-name> .`
3. `docker run` with `assets/` and `outputs/` mounted, using `--no-display`.

See the model's own README for the exact commands.

## Prerequisites

- Python 3.10 (each model's `requirements.txt` states the version it was tested with): https://www.python.org/downloads/
- Docker (optional): https://docs.docker.com/get-started/get-docker/
- GPU in Docker (optional, NVIDIA only):
  - Linux: NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - Windows: Docker Desktop with the WSL2 backend and a recent NVIDIA driver: https://docs.docker.com/desktop/features/gpu/

## Hardware compatibility

TODO: the full "Hardware compatibility" section has not been added yet.

What is in place so far:
- `system_info.py` prints your OS, CPU, RAM and GPU. Run it on the host, not
  inside Docker (Docker on Windows/macOS runs in a VM and reports the VM's RAM
  and no GPU).
- The models table has a **Min hardware** column. It is filled in as each model
  is tested.
- The Results table records the device and RAM for every run.

## Models

Status and min hardware are filled in as each model is done. See `PROGRESS.md` for details.

| # | Model | Folder | Type | GPU needed | Min hardware | Docker support | License | Status |
|---|-------|--------|------|------------|--------------|----------------|---------|--------|
| 1 | PP-HumanSeg v1 (portrait Lite) | `pp-humanseg-v1/` | Segmentation | No | TBD (about 20 FPS on a laptop i7 CPU) | Yes (CPU) | Apache-2.0 | Done |
| 2 | PP-HumanSegV2 Lite (portrait) | `pp-humanseg-v2/` | Segmentation | No | TBD (about 25 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 | In progress |
| 3 | MediaPipe Selfie Segmentation (landscape) | `mediapipe-selfie/` | Segmentation | No | TBD (40+ FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 (see model card) | In progress |
| 4 | MediaPipe Selfie Multiclass | `mediapipe-multiclass/` | Segmentation | No (slow on CPU) | TBD (about 7 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 (see model card) | In progress |
| 5 | MODNet | `modnet/` | Matting | No | TBD | Planned | Apache-2.0 | Not started |
| 6 | Robust Video Matting (RVM) | `rvm/` | Matting | No (faster with GPU) | TBD | Planned | **GPL-3.0** | Not started |
| 7 | SINet | `sinet/` | Segmentation | No | TBD | Planned | MIT | Not started |
| 8 | U²-Net (portrait) | `u2net/` | Segmentation | Recommended (too slow on CPU) | TBD | Planned | Apache-2.0 | Not started |
| 9 | DeepLabV3 MobileNet (torchvision) | `deeplabv3-mobilenet/` | Segmentation | No | TBD | Planned | BSD-3-Clause | Not started |
| 10 | TF.js BodyPix | `bodypix/` | Segmentation | No | TBD | TBD (may need Node.js/browser) | Apache-2.0 | Not started |
| 11 | YOLO segmentation (Ultralytics) | `yolo-seg/` | Segmentation | No (faster with GPU) | TBD | Planned | **AGPL-3.0** | Not started |
| 12 | Apple Vision person segmentation | `apple-vision/` | Segmentation | No | Mac only | No (macOS only) | Apple OS API | Not started |
| 13 | NVIDIA Maxine | `nvidia-maxine/` | Segmentation | Yes (NVIDIA RTX) | NVIDIA RTX GPU | TBD | NVIDIA SDK license | Not started |

GPL-3.0 and AGPL-3.0 are restrictive for closed-source commercial use. Read the
license before using those models in a product.

## Resources

| Model | Paper | Code / docs |
|-------|-------|-------------|
| PP-HumanSeg v1 / V2 | [PP-HumanSeg (arXiv 2112.07146)](https://arxiv.org/abs/2112.07146) | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PP-HumanSeg folder](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg) |
| MediaPipe Selfie / Multiclass | TODO | [Image segmenter guide](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter), [Selfie model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf), [Multiclass model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf), [MediaPipe repo](https://github.com/google-ai-edge/mediapipe) |
| MODNet | [MODNet (arXiv 2011.11961)](https://arxiv.org/abs/2011.11961) | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) |
| Robust Video Matting | [RVM (arXiv 2108.11515)](https://arxiv.org/abs/2108.11515) | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) |
| SINet | [SINet (arXiv 1911.09099)](https://arxiv.org/abs/1911.09099) | [clovaai/c3_sinet](https://github.com/clovaai/c3_sinet) |
| U²-Net | [U²-Net (arXiv 2005.09007)](https://arxiv.org/abs/2005.09007) | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) |
| DeepLabV3 MobileNet | [DeepLabV3 (arXiv 1706.05587)](https://arxiv.org/abs/1706.05587), [MobileNetV3 (arXiv 1905.02244)](https://arxiv.org/abs/1905.02244) | [torchvision docs](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html) |
| TF.js BodyPix | TODO | [tfjs-models/body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix) |
| YOLO segmentation | TODO | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), [Segmentation docs](https://docs.ultralytics.com/tasks/segment/) |
| Apple Vision | - | [VNGeneratePersonSegmentationRequest](https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest) |
| NVIDIA Maxine | - | [NVIDIA Maxine](https://developer.nvidia.com/maxine) |

## Results

Before adding a row, run `python system_info.py` on the host and copy your
device and RAM details into the row. Results are only comparable when the
hardware is known.

Laptop numbers can vary by 2x between runs of the same model (turbo boost,
temperature, Windows moving work between fast and slow CPU cores). For fair
numbers: plug in the charger, close other heavy apps (including Docker Desktop if
not needed), run each model 2-3 times, and report the typical value.

| Model | Device (CPU/GPU name) | RAM | Resolution | Avg FPS | Avg inference (ms) | Local/Docker | Notes |
|-------|-----------------------|-----|------------|---------|--------------------|--------------|-------|
| PP-HumanSeg v1 | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 20.3 | 41.5 | Local | Windows 11, Paddle 3.3.1, 150 frames |
| PP-HumanSeg v1 | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 22.5 | 38.2 | Local | Windows 11, test clip, 75 frames |
| PP-HumanSeg v1 | CPU: Intel i7-13620H (no GPU used) | 16 GB host (8 GB WSL2 VM) | 640x480 (video) | 20.6 | 44.4 | Docker | Docker Desktop on WSL2, same test clip |
| PP-HumanSeg V2 | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 26-27 | 32-34 | Local | Windows 11, Paddle 3.3.1, test clip, typical of quiet runs |
| PP-HumanSeg v1 | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 22-23 | 37-39 | Local | Re-measured the same session for comparison with V2 |
| MediaPipe Selfie (landscape) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 110 | 3.6 | Local | Windows 11, mediapipe 1.0.1, test clip, 2 runs |
| MediaPipe Selfie (landscape) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 41-63 | 6-11 | Local | 150 frames, 2 runs |
| MediaPipe Selfie Multiclass | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 7.2 | 133 | Local | Windows 11, mediapipe 1.0.1, test clip |
| MediaPipe Selfie Multiclass | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 8.3 | 116 | Local | 60 frames |

### Which model for which machine?

To be filled in once we have results.

| Machine | Example hardware | Suggested model(s) |
|---------|------------------|--------------------|
| Low-end | TBD | TBD |
| Mid-range | TBD | TBD |
| High-end | TBD | TBD |

## How to add a new model

1. Create a new folder `<model-name>/` (kebab-case).
2. Add `README.md`, `requirements.txt`, `download_model.py`, `run.py`,
   `Dockerfile`, and a copy of the root `.dockerignore`. Copy the layout of an
   existing model folder.
3. `requirements.txt`: pin exact tested versions, CPU packages only, with
   `# Tested with Python X.Y` on the first line. GPU packages go in
   `requirements-gpu.txt`.
4. `Dockerfile`: slim Python base image of the same version, install from
   `requirements.txt`, run `download_model.py` during the build.
5. Test it locally and in Docker.
6. Add the model to the Models, Resources and Results tables above, and to `PROGRESS.md`.

---

**Prepared By**  
Kawshik Kumar Paul  
Senior Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
kawshikbuet17@gmail.com
