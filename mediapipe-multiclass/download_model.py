#!/usr/bin/env python3
"""Download the MediaPipe Selfie Multiclass model into models/.

Usage:  python download_model.py

Model page: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
(SelfieMulticlass (256 x 256), input 256x256, float32, about 16 MB).
"""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
       "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODEL_FILE = Path(__file__).resolve().parent / "models" / "selfie_multiclass_256x256.tflite"


def main():
    if MODEL_FILE.exists():
        print(f"Model already downloaded: {MODEL_FILE}")
        return
    MODEL_FILE.parent.mkdir(exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, MODEL_FILE)
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
