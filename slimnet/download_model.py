#!/usr/bin/env python3
"""Download the SlimNet portrait segmentation model (TFLite) into models/.

Usage:  python download_model.py

From https://github.com/anilsathyan7/Portrait-Segmentation (MIT):
mediapipe_slimnet/portrait_segmentation.tflite (about 1.5 MB). This is the final
SlimNet checkpoint (slim-net-157) converted to TFLite; the author's MediaPipe
Android demo uses the same file. Input 512x512.
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://github.com/anilsathyan7/Portrait-Segmentation/raw/master/mediapipe_slimnet/portrait_segmentation.tflite"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "cc33b5c06b8cfa4c0a69cc0c7b86d67e1f2b85c79bad85048de92f57194356b5"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "portrait_segmentation.tflite"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if MODEL_FILE.exists():
        print(f"Model already downloaded: {MODEL_FILE}")
        return
    MODEL_FILE.parent.mkdir(exist_ok=True)
    tmp = MODEL_FILE.with_suffix(".part")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    if sha256_of(tmp) != SHA256:
        tmp.unlink()
        raise SystemExit("Downloaded file does not match the expected checksum. Try again later.")
    tmp.rename(MODEL_FILE)
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
