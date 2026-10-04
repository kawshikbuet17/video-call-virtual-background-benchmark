#!/usr/bin/env python3
"""Download the ERDNet portrait segmentation model (ncnn format) into models/.

Usage:  python download_model.py

From https://github.com/leeys888/ncnn-portrait-segmentation (MIT): models/erdnet.param
(the network structure) and models/erdnet.bin (the weights), about 3.3 MB in total.
That repo says the model was taken from lizhengwei1992's Fast_Portrait_Segmentation /
mobile_phone_human_matting projects and converted to ncnn.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/leeys888/ncnn-portrait-segmentation/master/models/"
MODELS_DIR = Path(__file__).resolve().parent / "models"
# File name -> SHA256 of the file we tested with, to catch broken or changed downloads.
FILES = {
    "erdnet.param": "d8eb4901260c24e642bbd4f693206ed509327c90deffc62548a5b7f8fbc2821b",
    "erdnet.bin": "9949b2285dfe1fb09a86393dd3ab2448ceee80a8a4fa21f26e0d4e9b51838b01",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if all((MODELS_DIR / name).exists() for name in FILES):
        print(f"Model already downloaded: {MODELS_DIR}")
        return
    MODELS_DIR.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        path = MODELS_DIR / name
        print(f"Downloading {BASE_URL + name}")
        urllib.request.urlretrieve(BASE_URL + name, path)
        if sha256_of(path) != expected:
            path.unlink()
            raise SystemExit(f"{name} does not match the expected checksum. Try again later.")
    print(f"Done. Model saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
