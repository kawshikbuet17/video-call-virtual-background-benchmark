#!/usr/bin/env python3
"""Download the official torchvision DeepLabV3 MobileNetV3-Large weights into models/.

Usage:  python download_model.py

Weights: DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 (about 44 MB),
the same file torchvision would download itself. We save it in models/ (instead of
torchvision's cache folder) so every model in this project keeps its weights in one place.
Docs: https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "deeplabv3_mobilenet_v3_large-fc3c493d.pth"


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
