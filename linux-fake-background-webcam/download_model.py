#!/usr/bin/env python3
"""Download the MediaPipe Selfie Segmenter (landscape) model into models/.

Usage:  python download_model.py

This is the model Linux-Fake-Background-Webcam downloads (same URL as in its
lfbw/lfbw.py): Google's MediaPipe model storage, "latest" version, float16,
about 250 KB. Model page:
https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
"""
import hashlib
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
       "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite")
# Checksum of the file we tested with (2026-10-06). The "latest" file can change.
SHA256 = "490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "selfie_segmenter_landscape.tflite"


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
    checksum = sha256_of(tmp)
    tmp.rename(MODEL_FILE)
    if checksum != SHA256:
        # The URL points to Google's "latest" version, which can change.
        print("WARNING: this is not the model file this folder was tested with "
              f"(Google may have updated it). New sha256: {checksum}")
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
