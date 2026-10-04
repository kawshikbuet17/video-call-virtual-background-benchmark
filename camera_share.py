#!/usr/bin/env python3
"""Share one webcam with several model processes at the same time.

A webcam can only be opened by one process at a time (on Windows, a second
process gets no frames). So this script opens the webcam once and copies every
frame into shared memory. Each model's live.py (started with --shared-camera)
reads the newest frame from there instead of opening the camera itself.

You normally do not run this yourself: run_all_live.py starts it for you.
It needs OpenCV, so it is run with one of the model folders' .venv Python.

    python camera_share.py --camera 0      # serve the webcam until Ctrl+C

Shared memory layout (name "vbg_camera"):
    bytes 0-7    frame counter (uint64). Odd while a frame is being written.
    bytes 8-11   height (uint32)
    bytes 12-15  width  (uint32)
    bytes 16-    the frame, height x width x 3 bytes (BGR, like OpenCV)
"""
import argparse
import signal
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np

SHM_NAME = "vbg_camera"
HEADER = struct.Struct("<QII")  # counter, height, width
MAX_BYTES = HEADER.size + 1920 * 1080 * 3  # room for up to a 1080p frame


# ---------------------------------------------------------------- reader (used by live.py)
class SharedCameraReader:
    """Drop-in replacement for cv2.VideoCapture: has read(), isOpened(), release()."""

    def __init__(self, timeout_s=10.0):
        start = time.time()
        while True:
            try:
                self.shm = shared_memory.SharedMemory(name=SHM_NAME)
                break
            except FileNotFoundError:
                if time.time() - start > timeout_s:
                    raise SystemExit("Shared camera not found. Start it with run_all_live.py.")
                time.sleep(0.2)
        if sys.platform != "win32":
            # On Linux/macOS Python would delete the shared memory when this
            # reader exits. Only the camera server should do that.
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.shm._name, "shared_memory")
        self.last_counter = 0

    def isOpened(self):
        return True

    def read(self):
        """Wait for a new frame and return (True, frame), like cv2.VideoCapture.read()."""
        buf = self.shm.buf
        deadline = time.time() + 5.0
        while time.time() < deadline:
            counter, h, w = HEADER.unpack_from(buf, 0)
            if counter % 2 == 0 and counter != self.last_counter and h > 0:
                frame = np.frombuffer(buf, np.uint8, h * w * 3, HEADER.size).reshape(h, w, 3).copy()
                # If the counter changed while copying, the frame may be half old/half new: try again.
                if HEADER.unpack_from(buf, 0)[0] == counter:
                    self.last_counter = counter
                    return True, frame
            time.sleep(0.002)
        return False, None  # no new frame for 5 s: camera server stopped

    def get(self, prop):
        return 0

    def release(self):
        self.shm.close()


# ---------------------------------------------------------------- server
def raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def serve(camera_index):
    import cv2  # only the server needs OpenCV

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {camera_index}. Close other apps using the camera.")
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=MAX_BYTES)
    buf = shm.buf
    counter = 0
    # run_all_live.py stops us with terminate() (SIGTERM on Linux/macOS). Treat it
    # like Ctrl+C so the cleanup below still runs and the shared memory is freed.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    print("Shared camera ready.", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from the webcam.", flush=True)
                break
            h, w = frame.shape[:2]
            if HEADER.size + h * w * 3 > MAX_BYTES:
                raise SystemExit(f"Camera resolution {w}x{h} is larger than 1920x1080.")
            counter += 1  # odd = writing
            HEADER.pack_into(buf, 0, counter, h, w)
            buf[HEADER.size:HEADER.size + h * w * 3] = frame.tobytes()
            counter += 1  # even = frame complete
            HEADER.pack_into(buf, 0, counter, h, w)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        del buf
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    serve(p.parse_args().camera)
