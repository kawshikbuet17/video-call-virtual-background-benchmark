#!/usr/bin/env python3
"""gregblur: MediaPipe selfie segmentation + gregblur's WebGL2 blur pipeline.

The browser library github.com/gregce/gregblur, redone in Python:
  1. model       MediaPipe selfie multiclass 256x256 (gregblur's default) or
                 selfie segmenter 256x256 (faster), with MediaPipe Tasks
  2. bilateral   joint bilateral filter (11x11) at the frame size: smooths the
                 mask, keeps it sharp where the picture has an edge
  3. temporal    mask = 76% this frame + 24% the previous mask (less flicker)
  4. blur        the frame is shrunk to half size without the person's pixels, then
                 blurred with a mask-weighted Gaussian, so the person leaves no halo
  5. composite   alpha = smoothstep(0.26, 0.72, mask + 0.035)

gregblur only blurs. The Replace panel uses the same alpha with a background image.

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam, selfie multiclass
    python run.py --model selfie-segmenter                   # faster model
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save
    python run.py --no-refine                                # raw mask, plain blur

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numba
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from numba import njit, prange

HERE = Path(__file__).resolve().parent
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

# gregblur's two MediaPipe models (src/segmentation/mediapipe.ts).
MODELS = {
    "selfie-multiclass": "selfie_multiclass_256x256.tflite",   # gregblur's default
    "selfie-segmenter": "selfie_segmenter.tflite",
}
MODEL_NAME = "selfie-multiclass"   # changed with --model
REFINE = True                      # bilateral + temporal + gregblur blur (off with --no-refine)
FILTER_THREADS = 4                 # numba threads for the bilateral filter
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view

# gregblur's defaults (src/core/pipeline.ts and shaders.ts).
SIGMA_SPACE = 4.0                  # bilateral filter, in frame pixels
SIGMA_COLOR = 0.1                  # bilateral filter, colour difference 0..1
BILATERAL_RADIUS = 5               # 11x11 samples
TEMPORAL_BLEND = 0.24              # share of the previous mask
BLUR_RADIUS = 25                   # Gaussian blur radius (sigma), in frame pixels
DOWNSAMPLE = 2                     # the blur runs at half size


def model_title():
    return f"gregblur {MODEL_NAME}" + ("" if REFINE else " (raw)")


def model_path():
    return HERE / "models" / MODELS[MODEL_NAME]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--model", choices=list(MODELS), default=MODEL_NAME,
                   help="selfie-multiclass (gregblur's default) or selfie-segmenter (faster)")
    p.add_argument("--no-refine", action="store_true",
                   help="raw model mask and the plain blur of the other folders, to compare")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    path = model_path()
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nRun first:  python download_model.py")
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,  # every frame on its own (gregblur: VIDEO mode)
        output_confidence_masks=True,           # soft 0..1 masks
        output_category_mask=False,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)
    numba.set_num_threads(FILTER_THREADS)

    # Warm-up run so the first real frame is not slower than the rest. This also
    # compiles the numba filter: a few seconds the very first time, then cached.
    frame = np.zeros((480, 640, 3), np.uint8)
    mask = postprocess(run_inference(segmenter, preprocess(frame)), frame)
    composite(frame, mask, frame)
    reset_temporal()
    return segmenter


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # MediaPipe resizes to the model input (256x256) and normalizes internally,
    # so we only convert OpenCV's BGR to the RGB that MediaPipe expects.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ---------------------------------------------------------------- inference
def run_inference(segmenter, mp_image):
    """Returns the person probability at the frame size, float32 (H, W)."""
    result = segmenter.segment(mp_image)
    first = result.confidence_masks[0].numpy_view()
    if first.ndim == 3:
        first = first[..., 0]
    if MODEL_NAME == "selfie-multiclass":
        # Six masks (background, hair, body skin, face skin, clothes, others):
        # mask 0 is the background, so person = 1 - background.
        return (1.0 - first).astype(np.float32)
    # Selfie segmenter: one mask, and it is the person.
    return np.ascontiguousarray(first, dtype=np.float32)


# ---------------------------------------------------------------- postprocess
@njit(cache=True, parallel=True, fastmath=True)
def _bilateral_pixels(frame, mask, todo, radius, sigma_space, sigma_color):
    """gregblur's BILATERAL_FILTER_SHADER as a loop (numba compiles it).
    Only pixels with todo != 0 are computed; the others keep their mask value."""
    h, w = mask.shape
    k_space = -1.0 / (2.0 * sigma_space * sigma_space)
    k_color = -1.0 / (2.0 * sigma_color * sigma_color)
    out = mask.copy()
    for y in prange(h):
        for x in range(w):
            if todo[y, x] == 0:
                continue
            c0 = np.float32(frame[y, x, 0])
            c1 = np.float32(frame[y, x, 1])
            c2 = np.float32(frame[y, x, 2])
            total = np.float32(0.0)
            acc = np.float32(0.0)
            for dy in range(-radius, radius + 1):
                sy = min(max(y + dy, 0), h - 1)          # edge pixels repeat (CLAMP_TO_EDGE)
                for dx in range(-radius, radius + 1):
                    sx = min(max(x + dx, 0), w - 1)
                    d0 = (np.float32(frame[sy, sx, 0]) - c0) / 255.0
                    d1 = (np.float32(frame[sy, sx, 1]) - c1) / 255.0
                    d2 = (np.float32(frame[sy, sx, 2]) - c2) / 255.0
                    # close in space AND close in colour = large weight
                    weight = math.exp((dx * dx + dy * dy) * k_space + (d0 * d0 + d1 * d1 + d2 * d2) * k_color)
                    acc += weight * mask[sy, sx]
                    total += weight
            out[y, x] = acc / max(total, 0.001)
    return out


def bilateral_filter(frame, mask):
    """Joint bilateral filter, guided by the frame's colours.

    The result is a weighted average of the mask around each pixel, so where the
    mask is (almost) flat it stays the same. We skip pixels whose 11x11
    neighbourhood varies by less than 0.001 (found quickly with OpenCV's
    dilate/erode); they would change by less than 0.001 anyway.
    """
    size = 2 * BILATERAL_RADIUS + 1
    kernel = np.ones((size, size), np.uint8)
    spread = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
    todo = (spread >= 0.001).astype(np.uint8)
    return _bilateral_pixels(frame, mask, todo, BILATERAL_RADIUS, SIGMA_SPACE, SIGMA_COLOR)


# The temporal blend needs the previous mask, so it keeps state between frames.
_prev = {"mask": None}


def reset_temporal():
    _prev["mask"] = None


def postprocess(person, frame):
    """Person probability -> bilateral filter -> temporal blend. Returns the
    person mask at frame size (0..1). Needs the frame itself as the colour guide."""
    h, w = frame.shape[:2]
    if person.shape != (h, w):  # MediaPipe normally returns the frame size already
        person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
    if not REFINE:
        return np.clip(person, 0.0, 1.0)
    mask = bilateral_filter(frame, person)
    previous = _prev["mask"]
    if previous is not None and previous.shape == mask.shape:
        mask = cv2.addWeighted(mask, 1.0 - TEMPORAL_BLEND, previous, TEMPORAL_BLEND, 0.0)
    _prev["mask"] = mask
    return mask


@njit(cache=True, parallel=True)
def _alpha(mask, bias, lo, hi):
    h, w = mask.shape
    out = np.empty((h, w), np.float32)
    for y in prange(h):
        for x in range(w):
            m = min(max(mask[y, x] + bias, 0.0), 1.0)
            t = min(max((m - lo) / (hi - lo), 0.0), 1.0)
            out[y, x] = t * t * (3.0 - 2.0 * t)          # smoothstep
    return out


def to_alpha(mask):
    """gregblur's composite: slightly biased towards the person (+0.035), so ears and
    temples are kept, then a soft edge between 0.26 and 0.72 (smoothstep)."""
    if not REFINE:
        return mask
    return _alpha(mask, 0.035, 0.26, 0.72)


@njit(cache=True, parallel=True)
def _masked_downsample(frame, mask):
    """gregblur's MASKED_DOWNSAMPLE_SHADER: frame to half size, leaving out the person.
    Each half-size pixel takes 3x3 samples; a sample falls between pixels, so it is
    the average of a 2x2 block. Its weight is 1 - smoothstep(0.12, 0.55, person)."""
    h, w = mask.shape
    hh, hw = h // 2, w // 2
    out = np.empty((hh, hw, 3), np.float32)
    for Y in prange(hh):
        for X in range(hw):
            acc0 = acc1 = acc2 = 0.0
            total = 0.0
            centre0 = centre1 = centre2 = 0.0
            for dy in range(-1, 2):
                y0 = min(max(2 * Y + dy, 0), h - 1)
                y1 = min(max(2 * Y + dy + 1, 0), h - 1)
                for dx in range(-1, 2):
                    x0 = min(max(2 * X + dx, 0), w - 1)
                    x1 = min(max(2 * X + dx + 1, 0), w - 1)
                    fg = (mask[y0, x0] + mask[y0, x1] + mask[y1, x0] + mask[y1, x1]) * 0.25
                    t = min(max((fg - 0.12) / (0.55 - 0.12), 0.0), 1.0)
                    weight = 1.0 - t * t * (3.0 - 2.0 * t)
                    c0 = (float(frame[y0, x0, 0]) + frame[y0, x1, 0] + frame[y1, x0, 0] + frame[y1, x1, 0]) * 0.25
                    c1 = (float(frame[y0, x0, 1]) + frame[y0, x1, 1] + frame[y1, x0, 1] + frame[y1, x1, 1]) * 0.25
                    c2 = (float(frame[y0, x0, 2]) + frame[y0, x1, 2] + frame[y1, x0, 2] + frame[y1, x1, 2]) * 0.25
                    if dy == 0 and dx == 0:
                        centre0, centre1, centre2 = c0, c1, c2
                    acc0 += c0 * weight
                    acc1 += c1 * weight
                    acc2 += c2 * weight
                    total += weight
            if total < 0.001:   # all samples are the person: keep the centre sample
                out[Y, X, 0], out[Y, X, 1], out[Y, X, 2] = centre0, centre1, centre2
            else:
                out[Y, X, 0], out[Y, X, 1], out[Y, X, 2] = acc0 / total, acc1 / total, acc2 / total
    return out


def gregblur_background(frame, mask, radius=BLUR_RADIUS):
    """The blurred background, without the person bleeding into it (no halo).

    1. Masked downsample to half size (person pixels get almost no weight).
    2. Gaussian blur in two passes (across, then down). Each sample's weight is
       gauss * (1 - person mask), so person pixels are left out. Because the weight
       depends only on the sample, each pass is exactly
       blur(weight * image) / blur(weight), which OpenCV computes fast.
    """
    h, w = frame.shape[:2]
    small = _masked_downsample(frame, mask)
    sigma = max(1.0, radius / DOWNSAMPLE)
    r = int(min(16, math.ceil(sigma)))            # the shader takes at most 16 samples per side
    offsets = np.arange(-r, r + 1, dtype=np.float32)
    gauss = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    one = np.ones(1, np.float32)
    # The shader reads the full-size mask between pixels: a 2x2 average = INTER_AREA.
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    weight = np.maximum(1.0 - small_mask, 0.001)
    for kx, ky in ((gauss, one), (one, gauss)):  # across, then down
        num = cv2.sepFilter2D(small * weight[:, :, None], -1, kx, ky, borderType=cv2.BORDER_REPLICATE)
        den = cv2.sepFilter2D(weight, -1, kx, ky, borderType=cv2.BORDER_REPLICATE)
        small = num / np.maximum(den, 0.001)[:, :, None]
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def plain_background_blur(frame):
    # The cheap blur of the other folders (used with --no-refine).
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- composite
def composite(frame, mask, background):
    h, w = frame.shape[:2]
    alpha = to_alpha(mask)
    blurred_bg = gregblur_background(frame, mask) if REFINE else plain_background_blur(frame)
    bg = cv2.resize(background, (w, h))

    # result = alpha * person + (1 - alpha) * background, for every pixel.
    # cv2.blendLinear does exactly this, about 10x faster than the same formula in numpy.
    inv_alpha = 1.0 - alpha
    blurred = cv2.blendLinear(frame, blurred_bg, alpha, inv_alpha)
    replaced = cv2.blendLinear(frame, bg, alpha, inv_alpha)

    mask_view = cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return mask_view, blurred, replaced


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
    person_mask = postprocess(output, frame)
    mask, blurred, replaced = composite(frame, person_mask, background)
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

    global REFINE, MODEL_NAME
    REFINE = not args.no_refine
    MODEL_NAME = args.model
    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
