#!/usr/bin/env python3
"""SlimNet (anilsathyan7/Portrait-Segmentation, TFLite): background blur and replacement demo.

Shows side by side: original | mask | blurred background | replaced background.

The model is a TFLite file. We run it with LiteRT ("ai-edge-litert"), Google's
standalone TFLite runtime, so the full TensorFlow package is not needed.

Examples:
    python run.py                                            # webcam
    python run.py --image ../assets/samples/person_portrait.jpg
    python run.py --video my_clip.mp4 --save
    python run.py --image ../assets/samples/person_portrait.jpg --no-display

Keys: q = quit, s = save a screenshot to outputs/
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "portrait_segmentation.tflite"
DEFAULT_BACKGROUND = HERE.parent / "assets" / "backgrounds" / "simple_room.png"
OUTPUT_DIR = HERE / "outputs"

MODEL_TITLE = "SlimNet (512)"
INPUT_SIZE = 512  # SlimNet works on 512x512 images (it shrinks them to 128 inside)
NUM_THREADS = 2   # CPU threads for TFLite. 2 was as fast as 4 here (12.7 vs 13.6 ms) and uses less CPU; 1 was 22.5 ms
PANEL_HEIGHT = 240  # height of each panel in the side-by-side view


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an input image")
    p.add_argument("--video", help="path to an input video")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--background", default=str(DEFAULT_BACKGROUND), help="background image for replacement")
    p.add_argument("--save", action="store_true", help="save the result to outputs/")
    p.add_argument("--no-display", action="store_true", help="no window; save output and print stats (for Docker/servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    return p.parse_args()


# ---------------------------------------------------------------- load model
def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun first:  python download_model.py")
    interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()  # reserve memory for all tensors once

    # Warm-up run so the first real frame is not slower than the rest.
    run_inference(interpreter, preprocess(np.zeros((480, 640, 3), np.uint8)))
    return interpreter


# ---------------------------------------------------------------- preprocess
def preprocess(frame_bgr):
    # Same as the repo's notebook (slim512.ipynb) and its MediaPipe graph:
    # RGB, 512x512, values scaled to 0..1. TFLite uses NHWC (batch, height, width, channels).
    img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img[np.newaxis]  # shape (1, 512, 512, 3)


# ---------------------------------------------------------------- inference
def run_inference(interpreter, input_tensor):
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_tensor)
    interpreter.invoke()
    # Shape (1, 512, 512, 2): a score per pixel for [background, person].
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


# ---------------------------------------------------------------- postprocess
def postprocess(output, frame_shape):
    # Channel 0 = background score, channel 1 = person score. Softmax over two
    # classes is the same as sigmoid(person - background), which is cheaper.
    # (The repo's notebook uses argmax over the same 2 channels for a hard mask.)
    background, person = output[0, :, :, 0], output[0, :, :, 1]
    person_prob = (1.0 / (1.0 + np.exp(background - person))).astype(np.float32)
    h, w = frame_shape[:2]
    alpha = cv2.resize(person_prob, (w, h), interpolation=cv2.INTER_LINEAR)
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

    print("Loading model...")
    predictor = load_model()
    if args.image:
        run_image(args, predictor, background)
    else:
        run_stream(args, predictor, background)


if __name__ == "__main__":
    main()
