#!/usr/bin/env python3
"""Webinar HumanSeg: PP-HumanSegV2-Lite (ONNX) + temporal step + joint edge upsample.

The background pipeline of the netesh3/webinar browser app (its main branch as of
2026-10-03, commit d9051ea; first added in pull request #242), redone in Python:
  1. model     PP-HumanSegV2-Lite at 256x144 on ONNX Runtime (CPU)
  2. temporal  block-matching motion between the previous and the current frame;
               the previous mask is moved along it and partly kept where the picture
               did not change (less flicker, fewer holes)
  3. edge      joint bilateral upsample to the frame size: at the person's edge, the
               mask follows colour edges in the picture instead of the coarse model grid

Shows side by side: original | mask | blurred background | replaced background.

Examples:
    python run.py                                            # webcam
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save
    python run.py --image ../assets/samples/person_portrait.jpg --no-display
    python run.py --no-refine                                # model only, no steps 2-3

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numba
import numpy as np
import onnxruntime as ort
from numba import njit, prange

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "humanseg.onnx"
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

MODEL_TITLE = "Webinar HumanSeg"
MODEL_W, MODEL_H = 256, 144   # fixed in the ONNX export
REFINE = True                 # temporal step + edge step (turn off with --no-refine)
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view

# Temporal step settings (the values used in the pull request).
LUMA_W, LUMA_H = 160, 90      # motion is found on a small grey image
FLOW_BLOCK = 8                # block size in pixels
FLOW_RADIUS = 6               # search up to 6 pixels in each direction
STILL_KEEP = 0.5              # where nothing changed: blend 50% of the old mask in
HOLD = 0.85                   # where the mask dropped but the picture did not change: keep 85%
HOLD_CORE = 0.5               # ...only for pixels that were at least 50% person
ERR_LO, ERR_HI = 0.04, 0.12   # picture difference where we start / stop trusting the old mask

# Edge step settings (joint bilateral upsample, as in the webinar app).
MATTE_LONG_SIDE = 512         # the mask is brought to 512 px (long side) before the edge step
SIGMA_SPACE = 1.22            # neighbourhood, in matte texels
SIGMA_COLOR = 0.12            # colour match, in 0..1 RGB distance (smaller = snappier edges)
EDGE_THREADS = 4              # 4 threads: 20 ms per 640x480 frame here; 16 threads were only 16 ms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--no-refine", action="store_true",
                   help="use the raw model mask (no temporal step, no edge step), to compare")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun first:  python download_model.py")
    # CPU only: this project benchmarks models on the CPU.
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    numba.set_num_threads(EDGE_THREADS)

    # Warm-up run so the first real frame is not slower than the rest. This also
    # compiles the numba steps: a few seconds the very first time, then cached.
    frame = np.zeros((480, 640, 3), np.uint8)
    output = run_inference(session, preprocess(frame))
    for _ in range(2):  # the second frame runs the motion search
        postprocess(output, frame)
    reset_temporal()
    return session


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # model was trained on RGB, OpenCV gives BGR
    img = img.astype(np.float32) / 127.5 - 1.0                  # PaddleSeg: (x / 255 - 0.5) / 0.5, range -1..1
    img = img.transpose(2, 0, 1)[np.newaxis]                    # HWC -> NCHW (batch of 1)
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------- inference
def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})[0]  # (1, 2, 144, 256): softmax [background, person]


# ---------------------------------------------------------------- postprocess
# The temporal step needs the previous frame and the previous mask, so it keeps state
# between calls. reset_temporal() forgets it (new video, or a single image).
_prev = {"luma": None, "mask": None}


def reset_temporal():
    _prev["luma"] = None
    _prev["mask"] = None


def grey(frame_bgr, w, h):
    """Small grey (luma) image in 0..1, used to find motion."""
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


@njit(cache=True, parallel=True)
def block_motion(cur, prev):
    """For each 8x8 block of `cur`, find where it came from in `prev` (shift up to
    FLOW_RADIUS pixels), by the smallest sum of absolute differences (SAD).
    Returns two small arrays (blocks high x blocks wide): dx and dy.
    A plain loop compiled by numba; in numpy this step took about 15 ms."""
    h, w = cur.shape
    r, b = FLOW_RADIUS, FLOW_BLOCK
    bh, bw = (h + b - 1) // b, (w + b - 1) // b   # number of blocks, rounded up
    flow_x = np.zeros((bh, bw), np.float32)
    flow_y = np.zeros((bh, bw), np.float32)
    for i in prange(bh * bw):
        by, bx = i // bw, i % bw
        y0, x0 = by * b, bx * b
        y1, x1 = min(h, y0 + b), min(w, x0 + b)
        best = np.inf
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # A small preference for small motion, so a flat wall does not "move".
                sad = 0.002 * (dx * dx + dy * dy)
                for y in range(y0, y1):
                    if sad >= best:
                        break  # already worse than the best shift: stop early
                    py = min(h - 1, max(0, y + dy))
                    for x in range(x0, x1):
                        sad += abs(cur[y, x] - prev[py, min(w - 1, max(0, x + dx))])
                if sad < best:
                    best = sad
                    flow_x[by, bx] = dx
                    flow_y[by, bx] = dy
    return flow_x, flow_y


def temporal_step(mask, luma):
    """Mix the new model mask with the previous mask, moved along the motion."""
    if _prev["luma"] is None:
        _prev["luma"], _prev["mask"] = luma, mask
        return mask
    prev_luma, prev_mask = _prev["luma"], _prev["mask"]
    fx, fy = block_motion(luma, prev_luma)

    # Pixel centres of the 256x144 mask, in luma (160x90) coordinates.
    sx, sy = LUMA_W / MODEL_W, LUMA_H / MODEL_H
    xs = (np.arange(MODEL_W, dtype=np.float32) + 0.5) * sx - 0.5
    ys = (np.arange(MODEL_H, dtype=np.float32) + 0.5) * sy - 0.5
    lx, ly = np.meshgrid(xs, ys)
    # Smooth motion per pixel: bilinear between block centres.
    bh, bw = fx.shape
    gx = np.clip(lx / FLOW_BLOCK - 0.5, 0, bw - 1)
    gy = np.clip(ly / FLOW_BLOCK - 0.5, 0, bh - 1)
    dx = cv2.remap(fx, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(fy, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # The previous mask, moved to where things are now.
    gridx, gridy = np.meshgrid(np.arange(MODEL_W, dtype=np.float32), np.arange(MODEL_H, dtype=np.float32))
    moved = cv2.remap(prev_mask, gridx + dx / sx, gridy + dy / sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # How well the moved previous picture matches this one: small error = trust the old mask.
    now = cv2.remap(luma, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    before = cv2.remap(prev_luma, lx + dx, ly + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.abs(now - before)
    trust = 1.0 - np.clip((err - ERR_LO) / (ERR_HI - ERR_LO), 0.0, 1.0)

    # Light blend everywhere the picture did not change (against shimmer) ...
    out = mask + (moved - mask) * STILL_KEEP * trust
    # ... and where the model dropped a confident person pixel without the picture
    # changing, keep most of the old value (fewer holes in the person).
    held = mask + (moved - mask) * HOLD * trust
    drop = (moved > mask) & (moved >= HOLD_CORE)
    out = np.where(drop, np.maximum(out, held), out).astype(np.float32)

    _prev["luma"], _prev["mask"] = luma, out
    return out


@njit(cache=True, parallel=True, fastmath=True)
def edge_pixels(coarse, matte_a, matte_rgb, ref):
    """The per-pixel loop of the edge step (in the browser this is a GPU shader).
    numba compiles it to machine code and runs the rows on EDGE_THREADS threads;
    the same loop in plain numpy took about 150 ms per 640x480 frame."""
    h, w = coarse.shape
    mh, mw = matte_a.shape
    k_space = 1.0 / (2.0 * SIGMA_SPACE * SIGMA_SPACE)
    k_color = 1.0 / (2.0 * SIGMA_COLOR * SIGMA_COLOR)
    out = coarse.copy()
    for y in prange(h):
        # Position of this pixel in matte texels: a whole texel (by) and a fraction (fy).
        py = (y + 0.5) * mh / h - 0.5
        by = math.floor(py)
        fy = py - by
        for x in range(w):
            c = coarse[y, x]
            if c <= 0.02 or c >= 0.98:
                continue  # clearly person or clearly background: nothing to refine
            px = (x + 0.5) * mw / w - 0.5
            bx = math.floor(px)
            fx = px - bx
            # The 4x4 matte texels around this pixel vote. A texel's vote counts more
            # when it is close, and when its colour is close to this pixel's colour.
            total = 0.0
            acc = 0.0
            for oy in range(-1, 3):
                qy = min(max(by + oy, 0), mh - 1)
                for ox in range(-1, 3):
                    qx = min(max(bx + ox, 0), mw - 1)
                    d0 = matte_rgb[qy, qx, 0] - ref[y, x, 0]
                    d1 = matte_rgb[qy, qx, 1] - ref[y, x, 1]
                    d2 = matte_rgb[qy, qx, 2] - ref[y, x, 2]
                    dist2 = (ox - fx) ** 2 + (oy - fy) ** 2
                    weight = math.exp(-dist2 * k_space - (d0 * d0 + d1 * d1 + d2 * d2) * k_color)
                    acc += matte_a[qy, qx] * weight
                    total += weight
            voted = acc / max(total, 1e-6)
            # Where no neighbour matches the colour (small total), keep the plain value.
            t = min(max((total - 0.02) / (0.2 - 0.02), 0.0), 1.0)
            out[y, x] = c + (voted - c) * (t * t * (3.0 - 2.0 * t))
    return out


def joint_upsample(mask, frame):
    """Bring the 256x144 mask to the frame size so its edge follows the picture.

    1. The "matte": the mask at 512 px on the long side (nearest neighbour), next to
       the frame's average colour at the same size.
    2. Every frame pixel gets the plain (bilinear) matte value.
    3. Only uncertain pixels (the person's edge) are redone by a joint bilateral
       filter (edge_pixels), so a shoulder edge lands on the shirt/wall boundary
       instead of on the coarse model grid.
    """
    h, w = frame.shape[:2]
    long = min(MATTE_LONG_SIDE, max(w, h))
    mw, mh = (long, max(1, round(long * h / w))) if w >= h else (max(1, round(long * w / h)), long)
    matte_a = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
    coarse = cv2.resize(matte_a, (w, h), interpolation=cv2.INTER_LINEAR)
    matte_rgb = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # This pixel's colour, slightly smoothed (the browser reads mip level 1 = half size).
    half = cv2.resize(frame, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
    ref = cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return edge_pixels(coarse, matte_a, matte_rgb, ref)


def postprocess(output, frame):
    """Person probability -> temporal step -> joint upsample -> mask at frame size.
    Needs the frame itself (not only its shape) for the two refine steps."""
    mask = np.ascontiguousarray(output[0, 1], dtype=np.float32)  # channel 1 = person
    if REFINE:
        mask = temporal_step(mask, grey(frame, LUMA_W, LUMA_H))
        alpha = joint_upsample(mask, frame)
    else:
        h, w = frame.shape[:2]
        alpha = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)  # float mask in [0, 1], same size as the frame


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
    text = f"{MODEL_TITLE}{'' if REFINE else ' (raw)'} | FPS: {fps:.1f} | Inference: {infer_ms:.1f} ms"
    cv2.putText(header, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([header, row])


# ---------------------------------------------------------------- main loop
def process_frame(predictor, frame, background):
    """Run the full pipeline on one frame. Returns the outputs and inference time in ms."""
    input_tensor = preprocess(frame)
    t0 = time.perf_counter()
    output = run_inference(predictor, input_tensor)
    infer_ms = (time.perf_counter() - t0) * 1000
    alpha = postprocess(output, frame)
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

    global REFINE
    REFINE = not args.no_refine
    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
