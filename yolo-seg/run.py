#!/usr/bin/env python3
"""YOLO segmentation (Ultralytics YOLO26 / YOLO11 / YOLOv8, nano): background blur and replacement demo.

YOLO is a general object detector that also outlines each object (instance
segmentation). It knows 80 everyday classes (COCO); we keep only "person" and join
the outlines of all people into one mask.

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam, yolo26n-seg
    python run.py --weights yolo11n-seg                      # older YOLO11
    python run.py --imgsz 320                                # faster, coarser mask
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

WEIGHTS = ["yolo26n-seg", "yolo11n-seg", "yolov8n-seg"]   # nano models; yolo26 is the newest
WEIGHTS_NAME = "yolo26n-seg"   # changed with --weights
IMG_SIZE = 640                 # long side of the model input (Ultralytics' default); --imgsz
CONF = 0.25                    # minimum confidence for a person (Ultralytics' default)
PERSON_CLASS = 0               # "person" in the COCO class list
NUM_THREADS = 4                # PyTorch CPU threads
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view


def model_title():
    return f"YOLO {WEIGHTS_NAME} ({IMG_SIZE})"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--weights", choices=WEIGHTS, default=WEIGHTS_NAME,
                   help="yolo26n-seg (default, newest), yolo11n-seg or yolov8n-seg")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE,
                   help="model input size, long side (default 640; 320 is faster but coarser)")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    path = HERE / "models" / f"{WEIGHTS_NAME}.pt"
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nRun first:  python download_model.py")
    torch.set_num_threads(NUM_THREADS)
    model = YOLO(str(path), task="segment")  # a full path, so Ultralytics does not download anything

    # Warm-up run so the first real frame is not slower than the rest.
    run_inference(model, preprocess(np.zeros((480, 640, 3), np.uint8)))
    return model


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # Ultralytics does its own preprocessing: it takes an OpenCV BGR image, scales it
    # so the long side is --imgsz, pads it to a multiple of 32 ("letterbox"), and
    # converts it to RGB 0..1 NCHW. So there is nothing to do here.
    return frame_bgr


# ---------------------------------------------------------------- inference
def run_inference(model, frame_bgr):
    # One call runs preprocessing, the network and the mask decoding.
    # retina_masks=True: masks at the full frame size (not the 4x smaller default).
    return model.predict(frame_bgr, imgsz=IMG_SIZE, conf=CONF, classes=[PERSON_CLASS],
                         retina_masks=True, device="cpu", verbose=False)[0]


# ---------------------------------------------------------------- postprocess
def postprocess(result, frame_shape):
    h, w = frame_shape[:2]
    if result.masks is None:  # nobody found
        return np.zeros((h, w), np.float32)
    # One 0/1 mask per person. Ultralytics already cut each one at 0.5, so the
    # edges are hard. Join all people: a pixel is "person" if any mask says so.
    masks = result.masks.data.cpu().numpy()          # shape (people, H, W)
    alpha = masks.max(axis=0).astype(np.float32)
    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)  # float mask in [0, 1], same size as the frame


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

    global WEIGHTS_NAME, IMG_SIZE
    WEIGHTS_NAME = args.weights
    IMG_SIZE = args.imgsz
    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
