#!/usr/bin/env python3
"""Volcomix virtual-background: Google Meet / ML Kit selfie models + its WebGL2 pipeline.

The browser demo github.com/Volcomix/virtual-background, redone in Python:
  1. model        Meet segmentation (lite 160x96 or full 256x144) or ML Kit selfie
                  (256x256), TFLite files run with LiteRT (no full TensorFlow)
  2. softmax      the Meet models give two scores [background, person] per pixel
  3. joint bilateral filter  at the frame size: smooths the mask, but keeps it
                  sharp where the picture has an edge
  4. coverage     smoothstep(0.5, 0.75) turns the soft mask into the final alpha
  5. light wrap   (replace only) a little of the new background's light is
                  added to the person's edge, so they blend into it

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam, Meet lite
    python run.py --model meet-full                          # Meet 256x144
    python run.py --model mlkit                              # ML Kit selfie 256x256
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save
    python run.py --no-refine                                # model only (no steps 3-5)

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numba
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from numba import njit, prange

HERE = Path(__file__).resolve().parent
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

# The three TFLite models of the demo: file name, input width and height.
MODELS = {
    "meet-lite": ("segm_lite_v681.tflite", 160, 96),    # the demo's default
    "meet-full": ("segm_full_v679.tflite", 256, 144),
    "mlkit": ("selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite", 256, 256),
}
MODEL_NAME = "meet-lite"      # changed with --model
REFINE = True                 # joint bilateral filter + coverage + light wrap (off with --no-refine)
NUM_THREADS = 1               # TFLite threads; the demo's WebAssembly build uses 1 as well
FILTER_THREADS = 4            # numba threads for the joint bilateral filter
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view

# Post-processing settings: the demo's defaults (src/App.tsx).
SIGMA_SPACE = 1.0             # in model pixels (scaled to frame pixels below)
SIGMA_COLOR = 0.1             # colour difference, 0..1 RGB
COVERAGE = (0.5, 0.75)        # mask below 0.5 = background, above 0.75 = person
LIGHT_WRAPPING = 0.3          # strength of the light wrap


def model_title():
    return f"Volcomix {MODEL_NAME}" + ("" if REFINE else " (raw)")


def model_path():
    return HERE / "models" / MODELS[MODEL_NAME][0]


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
                   help="meet-lite (160x96, default), meet-full (256x144) or mlkit (256x256)")
    p.add_argument("--no-refine", action="store_true",
                   help="raw model mask: no joint bilateral filter, coverage or light wrap")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    path = model_path()
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nRun first:  python download_model.py")
    interpreter = Interpreter(model_path=str(path), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()  # reserve memory for all tensors once
    numba.set_num_threads(FILTER_THREADS)

    # Warm-up run so the first real frame is not slower than the rest. This also
    # compiles the numba steps: a few seconds the very first time, then cached.
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage_and_wrap(frame, postprocess(run_inference(interpreter, preprocess(frame)), frame), frame)
    return interpreter


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # As in the demo: RGB, resized to the model input, values 0..1, NHWC.
    _, w, h = MODELS[MODEL_NAME]
    img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img[np.newaxis]  # shape (1, h, w, 3)


# ---------------------------------------------------------------- inference
def run_inference(interpreter, input_tensor):
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_tensor)
    interpreter.invoke()
    # Meet: (1, h, w, 2) scores [background, person]. ML Kit: (1, h, w, 1) person 0..1.
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


# ---------------------------------------------------------------- postprocess
@njit(cache=True, parallel=True, fastmath=True)
def _filter_pixels(frame, mask, taps, space_w, k_color):
    """The per-pixel loop of the joint bilateral filter. taps[i] = (dy, dx) offsets in
    frame pixels, space_w[i] = their distance weight (the same for every pixel).
    frame is the uint8 BGR frame; colour differences are scaled to 0..1."""
    out_h, out_w = frame.shape[0], frame.shape[1]
    mask_h, mask_w = mask.shape
    n = taps.shape[0]
    out = np.empty((out_h, out_w), np.float32)
    for y in prange(out_h):
        rows = np.empty(n, np.int32)      # nearest frame row of each tap
        mrows = np.empty(n, np.int32)     # nearest mask row of each tap
        for i in range(n):
            py = y + 0.5 + taps[i, 0]
            rows[i] = min(max(int(math.floor(py)), 0), out_h - 1)
            mrows[i] = min(max(int(math.floor(py * mask_h / out_h)), 0), mask_h - 1)
        for x in range(out_w):
            # If every mask sample is the same (all background or all person), the
            # weighted average is that value. Most pixels end here.
            first = mask[mrows[0], min(max(int(math.floor((x + 0.5 + taps[0, 1]) * mask_w / out_w)), 0), mask_w - 1)]
            same = True
            for i in range(1, n):
                mx = min(max(int(math.floor((x + 0.5 + taps[i, 1]) * mask_w / out_w)), 0), mask_w - 1)
                if mask[mrows[i], mx] != first:
                    same = False
                    break
            if same:
                out[y, x] = first
                continue
            c0, c1, c2 = np.float32(frame[y, x, 0]), np.float32(frame[y, x, 1]), np.float32(frame[y, x, 2])
            total = np.float32(0.0)
            acc = np.float32(0.0)
            for i in range(n):
                px = x + 0.5 + taps[i, 1]
                sx = min(max(int(math.floor(px)), 0), out_w - 1)
                mx = min(max(int(math.floor(px * mask_w / out_w)), 0), mask_w - 1)
                d0 = (np.float32(frame[rows[i], sx, 0]) - c0) / 255.0
                d1 = (np.float32(frame[rows[i], sx, 1]) - c1) / 255.0
                d2 = (np.float32(frame[rows[i], sx, 2]) - c2) / 255.0
                weight = space_w[i] * math.exp((d0 * d0 + d1 * d1 + d2 * d2) * k_color)
                acc += weight * mask[mrows[i], mx]
                total += weight
            out[y, x] = acc / total
    return out


def joint_bilateral_filter(frame, mask, sigma_space, sigma_color):
    """The demo's joint bilateral filter (jointBilateralFilterStage.ts).

    For every frame pixel, mask values around it are averaged. A neighbour counts
    more when it is close (space weight) and when its colour in the frame is close
    to this pixel's colour (colour weight), so the mask edge follows picture edges.
    frame: uint8 BGR frame (H, W, 3). mask: model mask (h, w), 0..1.
    """
    out_h, out_w = frame.shape[:2]
    mask_h, mask_w = mask.shape
    # Kernel size in frame pixels; it is sampled sparsely (every `step` pixels).
    sigma = sigma_space * max(out_w / mask_w, out_h / mask_h)
    step = max(1.0, math.sqrt(sigma) * 0.66)
    offset = step * 0.5 if step > 1.0 else 0.0
    n = int((2.0 * sigma - offset) / step) + 1
    d = -sigma + offset + np.arange(n) * step
    dy, dx = np.meshgrid(d, d, indexing="ij")
    taps = np.stack([dy.ravel(), dx.ravel()], axis=1).astype(np.float32)
    # The shader measures distance in texture coordinates (0..1) and uses
    # exp(-0.5 * d^2 / (4 * sigma^2)).
    sigma_texel = max(1.0 / out_w, 1.0 / out_h) * sigma
    k_space = -0.5 / (sigma_texel ** 2 * 4.0 + 1e-6)
    space_w = np.exp(((taps[:, 1] / out_w) ** 2 + (taps[:, 0] / out_h) ** 2) * k_space).astype(np.float32)
    k_color = np.float32(-0.5 / (sigma_color ** 2 * 4.0 + 1e-6))
    return _filter_pixels(frame, mask, taps, space_w, k_color)


@njit(cache=True, parallel=True)
def _wrap_and_alpha(frame, mask, background, lo, hi, strength):
    """Coverage and light wrap, pixel by pixel (in numpy this took about 25 ms).
    alpha = smoothstep(lo, hi, mask). The light wrap adds a little of the new
    background at the person's edge with a "screen" blend: 1 - (1 - a) * (1 - b)."""
    h, w = mask.shape
    alpha = np.empty((h, w), np.float32)
    wrapped = np.empty_like(frame)
    for y in prange(h):
        for x in range(w):
            m = mask[y, x]
            t = min(max((m - lo) / (hi - lo), 0.0), 1.0)
            alpha[y, x] = t * t * (3.0 - 2.0 * t)
            wrap = strength * (1.0 - max(0.0, m - hi) / (1.0 - hi))   # 1 at the edge, 0 deep inside
            for c in range(3):
                a = frame[y, x, c] / 255.0
                b = wrap * background[y, x, c] / 255.0
                wrapped[y, x, c] = np.uint8((1.0 - (1.0 - a) * (1.0 - b)) * 255.0 + 0.5)
    return alpha, wrapped


def person_probability(output):
    if output.shape[-1] == 2:
        # Meet: softmax over [background, person]. For two classes that is
        # sigmoid(person - background).
        return (1.0 / (1.0 + np.exp(output[0, :, :, 0] - output[0, :, :, 1]))).astype(np.float32)
    return output[0, :, :, 0].astype(np.float32)  # ML Kit: already 0..1


def postprocess(output, frame):
    """Returns the person mask at frame size (before coverage, see coverage_and_wrap).
    Needs the frame itself (not only its shape) for the joint bilateral filter."""
    person = person_probability(output)
    h, w = frame.shape[:2]
    if not REFINE:
        return np.clip(cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    # The demo keeps the mask in an 8-bit texture: 0..255 steps.
    person = np.round(person * 255.0).astype(np.float32) / 255.0
    return joint_bilateral_filter(frame, person, SIGMA_SPACE, SIGMA_COLOR)


def coverage_and_wrap(frame, mask, background=None):
    """Final alpha (coverage: smoothstep(0.5, 0.75) of the mask) and, when replacing
    the background, the frame with the light wrap. The demo does no light wrap for blur."""
    if not REFINE:
        return mask, frame
    strength = LIGHT_WRAPPING if background is not None else 0.0
    return _wrap_and_alpha(frame, mask, frame if background is None else background,
                           COVERAGE[0], COVERAGE[1], strength)


# ---------------------------------------------------------------- composite
def composite(frame, mask, background):
    h, w = frame.shape[:2]
    # Strong blur, done cheaply: shrink the frame 8x, blur it, scale it back up.
    # A big Gaussian kernel on the full-size frame looks the same but is much slower.
    small = cv2.resize(frame, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    blurred_bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = cv2.resize(background, (w, h))
    alpha, wrapped = coverage_and_wrap(frame, mask, bg)

    # result = alpha * person + (1 - alpha) * background, for every pixel.
    # cv2.blendLinear does exactly this, about 10x faster than the same formula in numpy.
    inv_alpha = 1.0 - alpha
    blurred = cv2.blendLinear(frame, blurred_bg, alpha, inv_alpha)
    replaced = cv2.blendLinear(wrapped, bg, alpha, inv_alpha)

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
