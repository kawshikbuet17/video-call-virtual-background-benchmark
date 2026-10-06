#!/usr/bin/env python3
"""Download the Ultralytics YOLO nano segmentation models into models/.

Usage:  python download_model.py

- yolo26n-seg.pt   YOLO26, the newest (default in run.py), about 6.7 MB
- yolo11n-seg.pt   YOLO11, about 6.2 MB
- yolov8n-seg.pt   YOLOv8, about 7.1 MB

From Ultralytics' official assets release v8.4.0 on GitHub. The checksums below
are the ones GitHub lists for these files. License: AGPL-3.0 (see the README).
"""
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
FILES = {
    "yolo26n-seg.pt": "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f",
    "yolo11n-seg.pt": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "yolov8n-seg.pt": "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c",
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
    for name, expected in FILES.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"Already downloaded: {name}")
            continue
        tmp = target.with_suffix(".part")
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        if sha256_of(tmp) != expected:
            tmp.unlink()
            raise SystemExit(f"{name} does not match the expected checksum. Try again, or download it "
                             f"manually from:\n{BASE_URL + name}\nand save it in: {MODELS_DIR}")
        tmp.rename(target)
    print(f"Done. Models are in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
