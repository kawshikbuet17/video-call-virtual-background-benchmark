#!/usr/bin/env python3
"""AI-Segmenter pipeline: a matting model for the soft alpha + YOLO to pick objects.

The Windows app github.com/Yerdnainspace/AI-Segmenter, its live pipeline redone
here as a short script (our own code):
  1. matte    a matting model makes a soft alpha (0..1) for the whole frame:
              RVM (default, MobileNetV3, ONNX) or BiRefNet (large, slow on CPU)
  2. YOLO     every 3rd frame, the YOLO11 nano detector finds objects (boxes only).
              The app keeps all of them by default ("select all"); --persons-only
              keeps only people.
  3. ROI      the alpha is kept only inside the chosen boxes (each box made 6%
              bigger, the box mask softened with a 15x15 blur). Anything outside
              becomes background, for example a person-like poster on the wall.

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam, RVM + YOLO
    python run.py --matte birefnet --image ../assets/samples/person_portrait.jpg
    python run.py --persons-only                             # keep only people
    python run.py --no-yolo                                  # matte only, to compare
    python run.py --video my_clip.mp4 --save

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

MATTE = "rvm"                  # rvm (default) or birefnet; --matte
USE_YOLO = True                # --no-yolo turns the object selection off
PERSONS_ONLY = False           # --persons-only; the app keeps every detected object
NUM_THREADS = 4                # PyTorch CPU threads (YOLO and BiRefNet)
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view

# AI-Segmenter's defaults (ai_segmenter/app.py, yolo_controls.py, models/*.py).
YOLO_IMGSZ = 320               # YOLO input size
YOLO_CONF = 0.25               # YOLO confidence
YOLO_MAX_DET = 8               # at most 8 objects
YOLO_EVERY = 3                 # YOLO runs on every 3rd frame; boxes are reused in between
RVM_DOWNSAMPLE = 0.25          # RVM's internal downsample ratio
BIREFNET_SIZE = 512            # BiRefNet input (its official size is 1024)
PERSON_CLASS = 0               # "person" in the COCO class list


def model_title():
    selection = "" if not USE_YOLO else (" + YOLO persons" if PERSONS_ONLY else " + YOLO")
    return f"AI-Segmenter {MATTE.upper() if MATTE == 'rvm' else 'BiRefNet'}{selection}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--matte", choices=["rvm", "birefnet"], default=MATTE,
                   help="rvm (default) or birefnet (needs: python download_model.py --birefnet; slow on CPU)")
    p.add_argument("--no-yolo", action="store_true", help="matting model only, no YOLO object selection")
    p.add_argument("--persons-only", action="store_true", help="keep only YOLO's people (the app keeps all objects)")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def need(path, hint="python download_model.py"):
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nRun first:  {hint}")
    return path


def load_model():
    torch.set_num_threads(NUM_THREADS)
    model = {"frame_count": 0, "boxes": [], "rvm_states": None, "rvm_size": None}
    if MATTE == "rvm":
        path = need(MODELS_DIR / "rvm_mobilenetv3_fp32.onnx")
        # CPU only: this project benchmarks models on the CPU.
        model["rvm"] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    else:
        from transformers import AutoModelForImageSegmentation
        path = need(MODELS_DIR / "birefnet" / "model.safetensors", "python download_model.py --birefnet")
        # trust_remote_code: BiRefNet's network code (birefnet.py) comes with the model.
        # It is loaded from models/birefnet/, which download_model.py pinned to one revision.
        net = AutoModelForImageSegmentation.from_pretrained(str(path.parent), trust_remote_code=True)
        model["birefnet"] = net.eval()
    if USE_YOLO:
        # A full path, so Ultralytics does not download anything by itself.
        model["yolo"] = YOLO(str(need(MODELS_DIR / "yolo11n.pt")), task="detect")

    # Warm-up run so the first real frame is not slower than the rest.
    run_inference(model, preprocess(np.zeros((480, 640, 3), np.uint8)))
    model.update(frame_count=0, boxes=[], rvm_states=None)  # forget the black warm-up frame
    return model


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # Each model prepares its own input below (RVM: full frame RGB 0..1;
    # BiRefNet: 512x512 RGB, ImageNet normalisation; YOLO: Ultralytics does it).
    return frame_bgr


# ---------------------------------------------------------------- inference
def rvm_alpha(model, frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])  # NCHW
    h, w = frame_bgr.shape[:2]
    # RVM remembers earlier frames in four recurrent states; reset them when the size changes.
    if model["rvm_states"] is None or model["rvm_size"] != (h, w):
        model["rvm_states"] = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        model["rvm_size"] = (h, w)
    r1, r2, r3, r4 = model["rvm_states"]
    _, pha, *states = model["rvm"].run(None, {
        "src": src, "r1i": r1, "r2i": r2, "r3i": r3, "r4i": r4,
        "downsample_ratio": np.array([RVM_DOWNSAMPLE], np.float32),
    })
    model["rvm_states"] = states
    return pha[0, 0]


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def birefnet_alpha(model, frame_bgr):
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (BIREFNET_SIZE, BIREFNET_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    with torch.inference_mode():
        out = model["birefnet"](x)
        pred = (out[-1] if isinstance(out, (list, tuple)) else out).sigmoid()[0, 0].numpy()
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


def yolo_boxes(model, frame_bgr):
    result = model["yolo"].predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, max_det=YOLO_MAX_DET,
                                   classes=[PERSON_CLASS] if PERSONS_ONLY else None,
                                   device="cpu", verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy().round().astype(int).tolist()


def run_inference(model, frame_bgr):
    model["frame_count"] += 1
    if USE_YOLO and model["frame_count"] % YOLO_EVERY == 1:  # frames 1, 4, 7, ...
        model["boxes"] = yolo_boxes(model, frame_bgr)
    alpha = rvm_alpha(model, frame_bgr) if MATTE == "rvm" else birefnet_alpha(model, frame_bgr)
    return alpha, list(model["boxes"])


# ---------------------------------------------------------------- postprocess
def box_mask(boxes, h, w):
    """1 inside the chosen boxes (each grown by 6%, at least 8 px), softened."""
    mask = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        pad_x, pad_y = max(8, int((x2 - x1) * 0.06)), max(8, int((y2 - y1) * 0.06))
        mask[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)] = 1.0
    return cv2.GaussianBlur(mask, (15, 15), 0)


def postprocess(output, frame_shape):
    alpha, boxes = output
    h, w = frame_shape[:2]
    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    if USE_YOLO and boxes:  # no detections: the app leaves the alpha as it is
        alpha = alpha * box_mask(boxes, h, w)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)  # float mask in [0, 1]


# ---------------------------------------------------------------- composite
def composite(frame, alpha, background):
    h, w = frame.shape[:2]
    # Strong blur, done cheaply: shrink the frame 8x, blur it, scale it back up.
    # A big Gaussian kernel on the full-size frame looks the same but is much slower.
    small = cv2.resize(frame, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    blurred_bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = cv2.resize(background, (w, h))

    # result = alpha * person + (1 - alpha) * background, for every pixel.
    # cv2.blendLinear does exactly this, about 10x faster than the same formula in numpy.
    inv_alpha = 1.0 - alpha
    blurred = cv2.blendLinear(frame, blurred_bg, alpha, inv_alpha)
    replaced = cv2.blendLinear(frame, bg, alpha, inv_alpha)

    mask = cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return mask, blurred, replaced


def build_view(frame, mask, blurred, replaced, fps, infer_ms):
    """Put the four images side by side with labels and the speed numbers on top."""
    h, w = frame.shape[:2]
    panel_w = int(w * PANEL_HEIGHT / h)
    panels = []
    for img, label in [(frame, "Original"), (mask, "Mask"), (blurred, "Blur"), (replaced, "Replace")]:
        panel = cv2.resize(img, (panel_w, PANEL_HEIGHT))
        cv2.putText(panel, label, (8, PANEL_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(panel)
    row = np.hstack(panels)

    header = np.zeros((36, row.shape[1], 3), np.uint8)
    text = f"{model_title()} | FPS: {fps:.1f} | Inference: {infer_ms:.1f} ms"
    cv2.putText(header, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([header, row])


# ---------------------------------------------------------------- main loop
def process_frame(predictor, frame, background):
    """Run the full pipeline on one frame. Returns the outputs and inference time in ms."""
    input_tensor = preprocess(frame)
    t0 = time.perf_counter()
    output = run_inference(predictor, input_tensor)
    infer_ms = (time.perf_counter() - t0) * 1000
    alpha = postprocess(output, frame.shape)
    mask, blurred, replaced = composite(frame, alpha, background)
    return mask, blurred, replaced, infer_ms


def run_image(args, predictor, background):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    t0 = time.perf_counter()
    mask, blurred, replaced, infer_ms = process_frame(predictor, frame, background)
    fps = 1.0 / (time.perf_counter() - t0)
    view = build_view(frame, mask, blurred, replaced, fps, infer_ms)

    if args.save or args.no_display:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stem = Path(args.image).stem
        for name, img in [("view", view), ("mask", mask), ("blur", blurred), ("replace", replaced)]:
            path = OUTPUT_DIR / f"{stem}_{name}.png"
            cv2.imwrite(str(path), img)
            print(f"Saved {path}")

    print(f"Inference: {infer_ms:.1f} ms | Full pipeline: {fps:.1f} FPS")
    if not args.no_display:
        print("Press q to close, s to save a screenshot.")
        cv2.imshow(model_title(), view)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                save_screenshot(view)
            elif key in (ord("q"), 27) or cv2.getWindowProperty(model_title(), cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def run_stream(args, predictor, background):
    """Webcam or video file."""
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + args.video if args.video else f'webcam {args.camera}'}")

    save = args.save or args.no_display
    writer = None
    out_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if out_fps <= 0 or out_fps > 120:
        out_fps = 25  # webcams often do not report a usable FPS

    frames, total_infer_ms, total_s = 0, 0.0, 0.0
    fps = 0.0
    print("Press q to quit, s to save a screenshot." if not args.no_display else "Running headless, press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video, or camera error
            t0 = time.perf_counter()
            mask, blurred, replaced, infer_ms = process_frame(predictor, frame, background)
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed  # smoothed for display
            frames += 1
            total_infer_ms += infer_ms
            total_s += elapsed

            view = build_view(frame, mask, blurred, replaced, fps, infer_ms)
            if save:
                if writer is None:
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    stem = Path(args.video).stem if args.video else "webcam"
                    out_path = OUTPUT_DIR / f"{stem}_view.mp4"
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             out_fps, (view.shape[1], view.shape[0]))
                writer.write(view)

            if not args.no_display:
                cv2.imshow(model_title(), view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(model_title(), cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    save_screenshot(view)
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved {out_path}")
        cv2.destroyAllWindows()

    if frames:
        print(f"Frames: {frames} | Average FPS (full pipeline): {frames / total_s:.1f} "
              f"| Average inference: {total_infer_ms / frames:.1f} ms")


def save_screenshot(view):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), view)
    print(f"Saved {path}")


def main():
    args = parse_args()
    background = cv2.imread(args.background)
    if background is None:
        raise SystemExit(f"Could not read background image: {args.background}")

    global MATTE, USE_YOLO, PERSONS_ONLY
    MATTE = args.matte
    USE_YOLO = not args.no_yolo
    PERSONS_ONLY = args.persons_only
    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
