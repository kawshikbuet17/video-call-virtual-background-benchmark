#!/usr/bin/env python3
"""Download the official Robust Video Matting ONNX model into models/.

Usage:  python download_model.py

Model: rvm_mobilenetv3_fp32.onnx (MobileNetV3 backbone, 32-bit float, about 15 MB),
from the official GitHub release v1.0.0:
https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "rvm_mobilenetv3_fp32.onnx"


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
