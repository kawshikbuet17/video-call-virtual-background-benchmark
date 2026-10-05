#!/usr/bin/env python3
"""Download gregblur's two MediaPipe segmentation models into models/.

Usage:  python download_model.py

- selfie_multiclass_256x256.tflite  gregblur's default (float32, about 16 MB)
- selfie_segmenter.tflite           faster, one person mask (float16, about 0.25 MB)

These are the URLs in gregblur's src/segmentation/mediapipe.ts (Google's MediaPipe
model storage, "latest" version). Model page:
https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
"""
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
# name: (URL, sha256 of the file we tested with on 2026-10-05)
FILES = {
    "selfie_multiclass_256x256.tflite": (
        BASE + "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
    "selfie_segmenter.tflite": (
        BASE + "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b"),
}
MODELS_DIR = Path(__file__).resolve().parent / "models"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"Already downloaded: {name}")
            continue
        tmp = target.with_suffix(".part")
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, tmp)
        checksum = sha256_of(tmp)
        tmp.rename(target)
        if checksum != expected:
            # The URL points to Google's "latest" version, which can change.
            print(f"WARNING: {name} is not the file this folder was tested with "
                  f"(Google may have updated it). New sha256: {checksum}")
    print(f"Done. Models are in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
