#!/usr/bin/env python3
"""Download the official BodyPix model (TF.js format) into models/.

Usage:  python download_model.py

Model: MobileNetV1, multiplier 0.75, output stride 16, float. This is BodyPix's
default configuration. Google hosts it in TF.js graph format: one model.json
plus binary weight files (about 5.2 MB in total).
"""
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/075/"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "bodypix_mobilenet_float_075_stride16"

# Remote file name -> (local file name, SHA256 of the file we tested with).
# model.json is saved under that name because the loader looks for it.
FILES = {
    "model-stride16.json": ("model.json", "a2e08279b82dad9d8a780c32e7428b6f1f71db5139b423898e4454dae35616c4"),
    "group1-shard1of2.bin": ("group1-shard1of2.bin", "a134113cb4b60ed0acc79205cbe50cf035f031866dfa22482c2ce60e067182a2"),
    "group1-shard2of2.bin": ("group1-shard2of2.bin", "13f0db2400200d3a920a9badcccfb371a7440965f940f3a5fcb5806efd49d255"),
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if all((MODEL_DIR / local).exists() for local, _ in FILES.values()):
        print(f"Model already downloaded: {MODEL_DIR}")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected) in FILES.items():
        path = MODEL_DIR / local
        print(f"Downloading {BASE_URL + remote}")
        urllib.request.urlretrieve(BASE_URL + remote, path)
        if sha256_of(path) != expected:
            path.unlink()
            raise SystemExit(f"{local} does not match the expected checksum. Try again later.")
    print(f"Done. Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
