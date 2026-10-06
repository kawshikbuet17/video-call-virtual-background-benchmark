#!/usr/bin/env python3
"""Live webcam with virtual background, like a video call.

One large window with the processed camera image, and a picker bar below it:
    None | Blur light | Blur strong | one tile per image in assets/backgrounds/

    python live.py
    python live.py --camera 1
    python live.py --imgsz 320          # faster, coarser mask
    python live.py --weights yolo11n-seg
    python live.py --imgsz 320 --soft-edge   # soft edge, no specks or holes
    python live.py --imgsz 320 --main-person # only the largest person (you), not people behind

Choose an effect by clicking a tile, or press its number key (1-9).
Other keys: s = save screenshot, q = quit.
To add your own background, copy a .jpg/.png into assets/backgrounds/ and restart.

The model code (load, preprocess, inference, postprocess) is reused from run.py.
"""
import argparse
import sys
import time

import cv2
import numpy as np

import run as model  # load_model, preprocess, run_inference, postprocess

WINDOW = "YOLO - Live"
BACKGROUNDS_DIR = model.HERE.parent / "assets" / "backgrounds"
BAR_HEIGHT = 90     # height of the picker bar under the video
MAX_TILES = 9       # number keys 1-9

# Blur strength = how much the frame is shrunk before blurring (bigger = blurrier).
BLUR_LEVELS = {"Blur light": 4, "Blur strong": 12}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--shared-camera", action="store_true",
                   help="read frames from camera_share.py instead of opening the webcam (used by run_all_live.py)")
    p.add_argument("--window-pos", type=int, nargs=2, metavar=("X", "Y"), help="place the window at X, Y on screen")
    p.add_argument("--window-width", type=int, help="initial window width in pixels (height follows)")
    p.add_argument("--weights", choices=model.WEIGHTS, default=model.WEIGHTS_NAME,
                   help="yolo26n-seg (default, newest), yolo11n-seg or yolov8n-seg")
    p.add_argument("--imgsz", type=int, default=model.IMG_SIZE, help="model input size (default 640; 320 is faster)")
    p.add_argument("--soft-edge", action="store_true", help="largest piece, filled holes, soft edge, smoothed over time")
    p.add_argument("--main-person", action="store_true", help="keep only the largest person (usually you)")
    return p.parse_args()


def load_effects():
    """List of effects shown as tiles: (name, kind, value)."""
    effects = [("None", "none", None)]
    effects += [(name, "blur", factor) for name, factor in BLUR_LEVELS.items()]
    paths = sorted(p for p in BACKGROUNDS_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            effects.append((p.stem, "image", img))
    if len(effects) > MAX_TILES:
        print(f"Only the first {MAX_TILES} effects are shown ({len(effects)} found).")
    return effects[:MAX_TILES]


def apply_effect(frame, alpha, effect, bg_cache):
    """Return the frame with the chosen effect applied."""
    name, kind, value = effect
    if kind == "none":
        return frame
    h, w = frame.shape[:2]
    if kind == "blur":
        # Shrink, blur, scale up: a strong blur that is cheap to compute.
        small = cv2.resize(frame, (max(w // value, 1), max(h // value, 1)), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.GaussianBlur(small, (5, 5), 0), (w, h))
    else:
        key = (name, w, h)
        if key not in bg_cache:  # resize each background once, not every frame
            bg_cache[key] = cv2.resize(value, (w, h))
        bg = bg_cache[key]
    # result = alpha * person + (1 - alpha) * background
    return cv2.blendLinear(frame, bg, alpha, 1.0 - alpha)


def make_bar(effects, width):
    """Draw the picker bar once. Each tile shows a preview and its number + name."""
    tile_w = width // len(effects)
    bar = np.full((BAR_HEIGHT, width, 3), 40, np.uint8)
    for i, (name, kind, value) in enumerate(effects):
        x0 = i * tile_w
        tw, th = tile_w - 8, BAR_HEIGHT - 8
        if kind == "image":
            preview = cv2.resize(value, (tw, th))
        elif kind == "blur":
            preview = cv2.GaussianBlur(np.full((th, tw, 3), (150, 120, 90), np.uint8), (0, 0), 3)
            cv2.circle(preview, (tw // 2, th // 2), th // 4, (200, 200, 200), -1)
            preview = cv2.blur(preview, (value + 1, value + 1))  # stronger blur level looks blurrier
        else:
            preview = np.full((th, tw, 3), 70, np.uint8)
        bar[4:4 + th, x0 + 4:x0 + 4 + tw] = preview
        label = f"{i + 1} {name}"
        cv2.putText(bar, label, (x0 + 8, BAR_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(bar, label, (x0 + 8, BAR_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return bar, tile_w


def draw_status(img, effect_name, fps, infer_ms):
    text = f"{effect_name} | {fps:.0f} FPS | {infer_ms:.0f} ms | click a tile or press 1-9, q quit"
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)        # dark outline
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)  # white text


def main():
    args = parse_args()
    effects = load_effects()
    selected = 1  # start with "Blur light"

    model.WEIGHTS_NAME = args.weights
    model.IMG_SIZE = args.imgsz
    model.SOFT_EDGE = args.soft_edge
    model.MAIN_PERSON = args.main_person
    print("Loading model...")
    predictor = model.load_model()

    if args.shared_camera:
        # The webcam is opened once by camera_share.py (in the project root),
        # so several live.py windows can show it at the same time.
        sys.path.insert(0, str(model.HERE.parent))
        from camera_share import SharedCameraReader
        cap = SharedCameraReader()
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}. Close other apps using the camera, or try --camera 1.")

    # Mouse clicks: remember where the user clicked; the main loop decides which tile it was.
    clicks = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable: drag the corner or maximize it
    if args.window_pos:
        cv2.moveWindow(WINDOW, *args.window_pos)
    window_sized = False
    cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, param:
                         clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)

    bar, tile_w, bg_cache = None, 0, {}
    fps = 0.0
    print("Live. Click a tile or press 1-9 to choose an effect. s = screenshot, q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from the webcam.")
            break
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Same pipeline as run.py: preprocess -> inference -> postprocess -> composite.
        t_inf = time.perf_counter()
        output = model.run_inference(predictor, model.preprocess(frame))
        infer_ms = (time.perf_counter() - t_inf) * 1000
        alpha = model.postprocess(output, frame.shape)
        result = apply_effect(frame, alpha, effects[selected], bg_cache)

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed  # smoothed

        # Build the window image: video on top, picker bar below, selected tile highlighted.
        if bar is None or bar.shape[1] != w:
            bar, tile_w = make_bar(effects, w)
        shown_bar = bar.copy()
        cv2.rectangle(shown_bar, (selected * tile_w + 1, 1), ((selected + 1) * tile_w - 2, BAR_HEIGHT - 2), (0, 220, 255), 3)
        top = result.copy()
        draw_status(top, effects[selected][0], fps, infer_ms)
        view = np.vstack([top, shown_bar])
        if args.window_width and not window_sized:  # keep the aspect ratio of the view
            cv2.resizeWindow(WINDOW, args.window_width, int(args.window_width * view.shape[0] / view.shape[1]))
            window_sized = True
        cv2.imshow(WINDOW, view)

        # Handle mouse clicks on the bar.
        for x, y in clicks:
            if y >= h and tile_w and x // tile_w < len(effects):
                selected = x // tile_w
        clicks.clear()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if ord("1") <= key <= ord("9") and key - ord("1") < len(effects):
            selected = key - ord("1")
        elif key == ord("s"):
            model.OUTPUT_DIR.mkdir(exist_ok=True)
            path = model.OUTPUT_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), result)  # saved without the status text and bar
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
