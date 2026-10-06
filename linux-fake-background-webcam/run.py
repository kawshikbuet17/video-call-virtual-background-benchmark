#!/usr/bin/env python3
"""Linux-Fake-Background-Webcam (lfbw): its MediaPipe mask recipe, in a window.

The Linux tool github.com/fangfufu/Linux-Fake-Background-Webcam sends a virtual
webcam with a fake background. This folder redoes its picture processing (our own
code, see the README's license note) and shows it in a window instead:
  1. model       MediaPipe selfie segmenter (landscape) on the frame shrunk to 256 px
                 wide; its hard "category" mask (person / not person) is used
  2. smooth      mask scaled up to the frame, 7x7 Gaussian blur
  3. threshold   pixels above 0.75 become person (1), the rest background (0)
  4. postprocess dilate 5x5 (grow the person a little), then a 10x10 box blur
                 (soft edge)
  5. background  Gaussian blur, kernel 21 (lfbw's default), or an image

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save
    python run.py --threshold 50                             # lfbw's --threshold
    python run.py --no-postprocess                           # lfbw's --no-postprocess

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "selfie_segmenter_landscape.tflite"
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

MODEL_TITLE = "lfbw (MediaPipe selfie)"
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view

# lfbw's defaults (lfbw/lfbw.py). Changed with the options of the same name.
THRESHOLD = 0.75          # --threshold 75
POSTPROCESS = True        # dilate + box blur (off with --no-postprocess)
MASK_UPDATE_SPEED = 1.0   # 1.0 = no smoothing between frames (see README: what lfbw really does)
BLUR_KERNEL = 21          # background blur kernel; sigma = kernel / 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--threshold", type=int, default=75,
                   help="minimum percentage to accept a pixel as the person (default 75, as lfbw)")
    p.add_argument("--no-postprocess", action="store_true", help="no dilate + box blur of the mask (as lfbw)")
    p.add_argument("--mask-update-speed", type=int, default=100,
                   help="percent of the new mask mixed in each frame; 100 = no smoothing (default, "
                        "what lfbw really does); 50 = what lfbw's setting was meant to do")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun first:  python download_model.py")
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,  # every frame on its own, as in lfbw
        output_category_mask=True,              # lfbw uses the hard person / background mask
        output_confidence_masks=False,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)

    # Warm-up run so the first real frame is not slower than the rest.
    run_inference(segmenter, preprocess(np.zeros((480, 640, 3), np.uint8)))
    return segmenter


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # lfbw shrinks the frame to 256 px wide (same aspect ratio) before MediaPipe.
    # MediaPipe then resizes to the model input (256x144) and normalizes internally.
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (256, int(256 * h / w)), interpolation=cv2.INTER_AREA)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- inference
def run_inference(segmenter, mp_image):
    category = segmenter.segment(mp_image).category_mask.numpy_view()
    if category.ndim == 3:
        category = category[..., 0]
    # Category 0 is the person, 255 is the background (tested). 1.0 = person.
    return (category == 0).astype(np.float32)


# ---------------------------------------------------------------- postprocess
# For --mask-update-speed below 100: the previous mask, kept between frames.
_prev = {"mask": None}


def reset_temporal():
    _prev["mask"] = None


def postprocess(person, frame_shape):
    h, w = frame_shape[:2]
    mask = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # soften the hard mask's steps
    if THRESHOLD < 1:
        # Back to hard 0 / 1: only pixels clearly inside the (blurred) person stay.
        _, mask = cv2.threshold(mask, THRESHOLD, 1.0, cv2.THRESH_BINARY)
    if POSTPROCESS:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # grow the person by 2 px
        mask = cv2.blur(mask, (10, 10))                      # soft 10 px edge
    if MASK_UPDATE_SPEED < 1:
        # Mix with the previous mask (less flicker, but a moving person lags behind).
        if _prev["mask"] is not None and _prev["mask"].shape == mask.shape:
            mask = cv2.addWeighted(mask, MASK_UPDATE_SPEED, _prev["mask"], 1 - MASK_UPDATE_SPEED, 0)
        _prev["mask"] = mask
    return mask  # float mask in [0, 1], same size as the frame


# ---------------------------------------------------------------- composite
def blur_background(frame, kernel=BLUR_KERNEL):
    # lfbw's default blurred background: one Gaussian blur, sigma = kernel / 3.
    kernel = kernel if kernel % 2 else kernel + 1
    return cv2.GaussianBlur(frame, (kernel, kernel), kernel / 3)


def composite(frame, alpha, background):
    h, w = frame.shape[:2]
    blurred_bg = blur_background(frame)
    bg = cv2.resize(background, (w, h))

    # result = alpha * person + (1 - alpha) * background, for every pixel.
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
    text = f"{MODEL_TITLE} | FPS: {fps:.1f} | Inference: {infer_ms:.1f} ms"
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
    reset_temporal()  # a single image has no previous frame
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
        cv2.imshow(MODEL_TITLE, view)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                save_screenshot(view)
            elif key in (ord("q"), 27) or cv2.getWindowProperty(MODEL_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def run_stream(args, predictor, background):
    """Webcam or video file."""
    reset_temporal()
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
                cv2.imshow(MODEL_TITLE, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(MODEL_TITLE, cv2.WND_PROP_VISIBLE) < 1:
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

    global THRESHOLD, POSTPROCESS, MASK_UPDATE_SPEED
    THRESHOLD = min(max(args.threshold, 0), 100) / 100
    POSTPROCESS = not args.no_postprocess
    MASK_UPDATE_SPEED = min(max(args.mask_update_speed, 0), 100) / 100
    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
