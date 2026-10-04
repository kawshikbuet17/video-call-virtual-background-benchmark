#!/usr/bin/env python3
"""Download the MediaPipe Selfie Segmenter (landscape) model into models/.

Usage:  python download_model.py

Model page: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
(SelfieSegmenter (Landscape), input 144x256, float16, about 250 KB).
"""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
       "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite")
MODEL_FILE = Path(__file__).resolve().parent / "models" / "selfie_segmenter_landscape.tflite"


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
