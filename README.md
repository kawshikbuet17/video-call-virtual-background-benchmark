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
├── run_all_live.py      runs several models' live.py side by side on one webcam
├── camera_share.py      shares one webcam between those windows (used by run_all_live.py)
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

**Run several models at once (live, side by side)**

Open `run_all_live.py` and comment out (`#`) the models you do not want in the
`MODELS` list. Then, from the project root:

```
python run_all_live.py
```

Each selected model opens its own `live.py` window, all showing your webcam.
Each window has its own picker bar. Close a window with `q`; Ctrl+C in the
terminal closes all of them.

The windows share the screen, as large as possible:

| Models selected | Layout |
|-----------------|--------|
| 1 | one big window |
| 2 | side by side |
| 3 or 4 | 2 x 2 grid |
| 5 or 6 | 3 x 2 grid |
| 7 to 9 | 3 x 3 grid (and so on) |

Run up to 4 at once. More models make every window slow, because they all share
one CPU. The screen size is detected; if the windows do not fit, set `SCREEN` in
`run_all_live.py`.

Every window runs in its model's own venv, so set up each model once first.
This sets up all of them (takes a while; run from the project root):

Windows (PowerShell):

```powershell
foreach ($m in "pp-humanseg-v1", "pp-humanseg-v2", "mediapipe-selfie", "mediapipe-multiclass", "modnet", "rvm", "sinet", "u2net", "deeplabv3-mobilenet", "bodypix", "fast-person-segmentation", "ncnn-portrait-segmentation", "fast-portrait-segmentation", "slimnet", "webinar-humanseg") {
    python -m venv "$m\.venv"
    & "$m\.venv\Scripts\python.exe" -m pip install -r "$m\requirements.txt"
    & "$m\.venv\Scripts\python.exe" "$m\download_model.py"
}
```

Linux / macOS:

```bash
for m in pp-humanseg-v1 pp-humanseg-v2 mediapipe-selfie mediapipe-multiclass modnet rvm sinet u2net deeplabv3-mobilenet bodypix fast-person-segmentation ncnn-portrait-segmentation fast-portrait-segmentation slimnet webinar-humanseg; do
    python3 -m venv "$m/.venv"
    # The extra index gives CPU-only PyTorch on Linux (sinet, u2net, deeplabv3-mobilenet, fast-portrait-segmentation); other packages still come from PyPI.
    "$m/.venv/bin/python" -m pip install -r "$m/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cpu
    "$m/.venv/bin/python" "$m/download_model.py"
done
```

The webcam can only be opened by one program at a time, so `camera_share.py`
opens it once and shares the frames with all windows. All models share one CPU,
so each runs slower than alone. Use a single `live.py` or `run.py` for FPS numbers.

## Prerequisites

- Python 3.10 (each model's `requirements.txt` states the version it was tested with): https://www.python.org/downloads/
- Docker (optional): https://docs.docker.com/get-started/get-docker/
- No GPU needed: every model in this project runs and is benchmarked on the CPU.

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

Status, model size and min hardware are filled in as each model is done. Model size = the model file(s) in
`models/` after download. See `PROGRESS.md` for details.

| # | Model | Folder | Type | Model size | GPU needed | Min hardware | Docker support | License | Status |
|---|-------|--------|------|------------|------------|--------------|----------------|---------|--------|
| 1 | PP-HumanSeg v1 (portrait Lite) | `pp-humanseg-v1/` | Segmentation | 2.3 MB | No | TBD (about 20 FPS on a laptop i7 CPU) | Yes (CPU) | Apache-2.0 | Done |
| 2 | PP-HumanSegV2 Lite (portrait) | `pp-humanseg-v2/` | Segmentation | 5.6 MB | No | TBD (about 25 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 | In progress |
| 3 | MediaPipe Selfie Segmentation (landscape) | `mediapipe-selfie/` | Segmentation | 0.25 MB | No | TBD (40+ FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 (see model card) | In progress |
| 4 | MediaPipe Selfie Multiclass | `mediapipe-multiclass/` | Segmentation | 16.4 MB | No (slow on CPU) | TBD (about 7 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 (see model card) | In progress |
| 5 | MODNet | `modnet/` | Matting | 25.9 MB | No (slow on CPU at full size) | TBD (about 5 FPS at 512, about 20 FPS at 256 on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 | In progress |
| 6 | Robust Video Matting (RVM) | `rvm/` | Matting | 15.0 MB | No | TBD (about 10 FPS default, 25-29 FPS at ratio 0.4 on a laptop i7 CPU) | Yes (CPU), not tested yet | **GPL-3.0** | In progress |
| 7 | SINet | `sinet/` | Segmentation | 0.44 MB | No | TBD (about 23 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | MIT | In progress |
| 8 | U²-Net (human segmentation) | `u2net/` | Segmentation | 176.3 MB | No (very slow on CPU) | TBD (about 2 FPS on a laptop i7 CPU: not real time) | Yes (CPU), not tested yet | Apache-2.0 | In progress |
| 9 | DeepLabV3 MobileNet (torchvision) | `deeplabv3-mobilenet/` | Segmentation | 44.4 MB | No (slow on CPU at full size) | TBD (about 4 FPS at 520, about 13 FPS at 256 on a laptop i7 CPU) | Yes (CPU), not tested yet | BSD-3-Clause | In progress |
| 10 | TF.js BodyPix (run in Python via tf-bodypix) | `bodypix/` | Segmentation | 5.2 MB | No | TBD (about 11 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | Apache-2.0 (model), MIT (tf-bodypix) | In progress |
| 11 | Deepixel Upperbody-Segmentation | - | Segmentation | - | No (CPU-only SDK) | TBD | TBD | Proprietary (license from Deepixel) | Blocked: SDK not publicly available |
| 12 | fast_person_segmentation (Mobile-UNet) | `fast-person-segmentation/` | Segmentation | 14.7 MB | No | TBD (about 23-44 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | No license (repo); original model MIT | In progress |
| 13 | ncnn-portrait-segmentation (ERDNet) | `ncnn-portrait-segmentation/` | Segmentation | 3.3 MB | No | TBD (about 40-47 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | MIT (code); model origin has no license | In progress |
| 14 | Fast_Portrait_Segmentation (ERD seg-matting, PyTorch) | `fast-portrait-segmentation/` | Matting | 3.4 MB | No | TBD (about 36-41 FPS on a laptop i7 CPU) | Yes (CPU), not tested yet | No license | In progress |
| 15 | SlimNet (anilsathyan7/Portrait-Segmentation) | `slimnet/` | Segmentation | 1.5 MB | No | TBD (about 41-45 FPS on video, 15-17 FPS with webcam on a laptop i7 CPU) | Yes (CPU), not tested yet | MIT | In progress |
| 16 | Webinar HumanSeg (netesh3/webinar: PP-HumanSegV2 ONNX + temporal and edge steps) | `webinar-humanseg/` | Segmentation | 3.8 MB | No | TBD (about 34 FPS on video and webcam on a laptop i7 CPU; 88 FPS without the refine steps) | Yes (CPU), not tested yet | Model Apache-2.0; webinar repo has **no license** | In progress |
| 17 | YOLO segmentation (Ultralytics) | `yolo-seg/` | Segmentation | TBD | No (faster with GPU) | TBD | Planned | **AGPL-3.0** | Not started |
| 18 | Apple Vision person segmentation | `apple-vision/` | Segmentation | TBD | No | Mac only | No (macOS only) | Apple OS API | Not started |
| 19 | NVIDIA Maxine | `nvidia-maxine/` | Segmentation | TBD | Yes (NVIDIA RTX) | NVIDIA RTX GPU | TBD | NVIDIA SDK license | Not started |

GPL-3.0 and AGPL-3.0 are restrictive for closed-source commercial use. Read the
license before using those models in a product.

## Resources

| Model | Paper | Code / docs |
|-------|-------|-------------|
| PP-HumanSeg v1 / V2 | [PP-HumanSeg (arXiv 2112.07146)](https://arxiv.org/abs/2112.07146) | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PP-HumanSeg folder](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg) |
| MediaPipe Selfie / Multiclass | TODO | [Image segmenter guide](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter), [Selfie model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf), [Multiclass model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf), [MediaPipe repo](https://github.com/google-ai-edge/mediapipe) |
| MODNet | [MODNet (arXiv 2011.11961)](https://arxiv.org/abs/2011.11961) | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) |
| Robust Video Matting | [RVM (arXiv 2108.11515)](https://arxiv.org/abs/2108.11515) | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) |
| SINet | [SINet (arXiv 1911.09099)](https://arxiv.org/abs/1911.09099) | [clovaai/ext_portrait_segmentation](https://github.com/clovaai/ext_portrait_segmentation) (portrait model, used here), [clovaai/c3_sinet](https://github.com/clovaai/c3_sinet) (street-scene version) |
| U²-Net (human segmentation) | [U²-Net (arXiv 2005.09007)](https://arxiv.org/abs/2005.09007) | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) (we use `u2net_human_seg.pth`; `u2net_portrait.pth` is a drawing model) |
| DeepLabV3 MobileNet | [DeepLabV3 (arXiv 1706.05587)](https://arxiv.org/abs/1706.05587), [MobileNetV3 (arXiv 1905.02244)](https://arxiv.org/abs/1905.02244) | [torchvision docs](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html) |
| TF.js BodyPix | TODO | [tfjs-models/body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix) (deprecated), [de-code/python-tf-bodypix](https://github.com/de-code/python-tf-bodypix) (Python wrapper used here) |
| Deepixel Upperbody-Segmentation | - | [deepixel-inc/Upperbody-Segmentation](https://github.com/deepixel-inc/Upperbody-Segmentation) (README and API docs only), [evaluation script](https://github.com/deepixel-inc/Upperbody-Evaluation) |
| fast_person_segmentation | - | [SamSamhuns/fast_person_segmentation](https://github.com/SamSamhuns/fast_person_segmentation) (model used here), [anilsathyan7/Portrait-Segmentation](https://github.com/anilsathyan7/Portrait-Segmentation) (original, MIT) |
| ncnn-portrait-segmentation (ERDNet) | - | [leeys888/ncnn-portrait-segmentation](https://github.com/leeys888/ncnn-portrait-segmentation) (model used here), [lizhengwei1992/Fast_Portrait_Segmentation](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation) (model origin), [Tencent/ncnn](https://github.com/Tencent/ncnn) |
| Fast_Portrait_Segmentation (ERD seg-matting) | - | [lizhengwei1992/Fast_Portrait_Segmentation](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation), [lizhengwei1992/mobile_phone_human_matting](https://github.com/lizhengwei1992/mobile_phone_human_matting) (weights and code used here) |
| SlimNet | - | [anilsathyan7/Portrait-Segmentation](https://github.com/anilsathyan7/Portrait-Segmentation) (MIT) |
| Webinar HumanSeg | - | [netesh3/webinar](https://github.com/netesh3/webinar) (no license), [PR #242](https://github.com/netesh3/webinar/pull/242), [humanseg.ts](https://github.com/netesh3/webinar/blob/main/web/lib/humanseg.ts), model: [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg) (Apache-2.0) |
| YOLO segmentation | TODO | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), [Segmentation docs](https://docs.ultralytics.com/tasks/segment/) |
| Apple Vision | - | [VNGeneratePersonSegmentationRequest](https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest) |
| NVIDIA Maxine | - | [NVIDIA Maxine](https://developer.nvidia.com/maxine) |

## Results

Before adding a row, run `python system_info.py` on the host and copy your
device and RAM details into the row. "Model size" is the size of the model
file(s) in `models/` after download (for Paddle models: graph + weights, unpacked from a smaller zip). Results are only comparable when the
hardware is known.

Laptop numbers can vary by 2x between runs of the same model (turbo boost,
temperature, Windows moving work between fast and slow CPU cores). For fair
numbers: plug in the charger, close other heavy apps (including Docker Desktop if
not needed), run each model 2-3 times, and report the typical value.

| Model | Model size | Device (CPU/GPU name) | RAM | Resolution | Avg FPS | Avg inference (ms) | Local/Docker | Notes |
|-------|------------|-----------------------|-----|------------|---------|--------------------|--------------|-------|
| PP-HumanSeg v1 | 2.3 MB (weights 0.56 MB) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 20.3 | 41.5 | Local | Windows 11, Paddle 3.3.1, 150 frames |
| PP-HumanSeg v1 | 2.3 MB (weights 0.56 MB) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 22.5 | 38.2 | Local | Windows 11, test clip, 75 frames |
| PP-HumanSeg v1 | 2.3 MB (weights 0.56 MB) | CPU: Intel i7-13620H (no GPU used) | 16 GB host (8 GB WSL2 VM) | 640x480 (video) | 20.6 | 44.4 | Docker | Docker Desktop on WSL2, same test clip |
| PP-HumanSeg V2 | 5.6 MB (weights 3.8 MB) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 26-27 | 32-34 | Local | Windows 11, Paddle 3.3.1, test clip, typical of quiet runs |
| PP-HumanSeg v1 | 2.3 MB (weights 0.56 MB) | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 22-23 | 37-39 | Local | Re-measured the same session for comparison with V2 |
| MediaPipe Selfie (landscape) | 0.25 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 110 | 3.6 | Local | Windows 11, mediapipe 1.0.1, test clip, 2 runs |
| MediaPipe Selfie (landscape) | 0.25 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 41-63 | 6-11 | Local | 150 frames, 2 runs |
| MediaPipe Selfie Multiclass | 16.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 7.2 | 133 | Local | Windows 11, mediapipe 1.0.1, test clip |
| MediaPipe Selfie Multiclass | 16.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 8.3 | 116 | Local | 60 frames |
| MODNet (ref-size 512) | 25.9 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 5.6-5.8 | 161-166 | Local | onnxruntime 1.23.2, official ONNX model |
| MODNet (ref-size 256) | 25.9 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 19.6-20.5 | 40-43 | Local | Same model, smaller input (less detail) |
| RVM MobileNetV3 (auto ratio 0.8) | 15.0 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 9.9-10.1 | 88 | Local | onnxruntime 1.23.2, official fp32 ONNX |
| RVM MobileNetV3 (ratio 0.4) | 15.0 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 25.3-29.3 | 25-29 | Local | Same model, smaller first stage |
| SINet | 0.44 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 22.7-24.3 | 35-37 | Local | PyTorch 2.14.1 (CPU), official weights |
| U²-Net (human seg) | 176.3 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 1.9 | 518-523 | Local | PyTorch 2.14.1 (CPU), official weights; 30 frames |
| DeepLabV3 MobileNetV3 (input 520) | 44.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 4.3 | 162-176 | Local | torch 2.14.1 + torchvision 0.29.1 (CPU), official weights |
| DeepLabV3 MobileNetV3 (input 256) | 44.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 13.4-13.9 | 54-55 | Local | Same model, smaller input |
| BodyPix MobileNet 0.75 (stride 16) | 5.2 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 10.5-11.1 | 63-66 | Local | TensorFlow 2.11 + tf-bodypix 0.4.2, official TF.js model |
| fast_person_segmentation (Mobile-UNet e260) | 14.7 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 43.7 | 14.2 | Local | OpenCV 4.11 DNN (no TensorFlow) |
| fast_person_segmentation (Mobile-UNet e260) | 14.7 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 21.7-23.4 | 26.6-27.9 | Local | Webcam reading uses CPU too; 150 frames, 3 runs |
| ncnn ERDNet | 3.3 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 47.1 | 15.2 | Local | ncnn 1.0.20260526 Python package, 4 threads |
| ncnn ERDNet | 3.3 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 38.1-42.9 | 17.1-18.9 | Local | 150 frames, 2 runs |
| Fast_Portrait_Segmentation ERD (PyTorch) | 3.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 41.5 | 17.1 | Local | torch 2.14.1 (CPU); same model as ncnn ERDNet |
| Fast_Portrait_Segmentation ERD (PyTorch) | 3.4 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 36.5 | 19.5 | Local | 150 frames |
| SlimNet (512) | 1.5 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 41.0-45.5 | 12.7-13.3 | Local | ai-edge-litert 2.2.0, 2 threads |
| SlimNet (512) | 1.5 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 15.2-16.6 | 39-44 | Local | Webcam reading uses a lot of CPU; 150 frames |
| Webinar HumanSeg (temporal + edge) | 3.8 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video) | 33.0-34.8 | 4.9-5.2 | Local | onnxruntime 1.23.2 + numba 0.68 (edge step 4 threads). Inference = model only; temporal ~4 ms, edge ~14 ms |
| Webinar HumanSeg (temporal + edge) | 3.8 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (webcam) | 34.2-34.6 | 4.9-5.3 | Local | 150 frames, 3 of 4 runs (one outlier run: 8 FPS) |
| Webinar HumanSeg (`--no-refine`) | 3.8 MB | CPU: Intel i7-13620H (no GPU used) | 16 GB | 640x480 (video / webcam) | 86-89 | 4.3-4.4 | Local | Raw PP-HumanSegV2 mask on ONNX Runtime (Paddle took 33 ms for the same network) |

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
5. Optional `live.py`: copy one from another model. Keep its `--shared-camera`,
   `--window-pos` and `--window-width` options, and add the folder to `MODELS`
   in `run_all_live.py`.
6. Test it locally and in Docker.
7. Add the model to the Models, Resources and Results tables above, and to `PROGRESS.md`.

---

**Prepared By**  
Kawshik Kumar Paul  
Senior Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
kawshikbuet17@gmail.com
