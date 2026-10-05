#!/usr/bin/env python3
"""Download the three TFLite models of Volcomix/virtual-background into models/.

Usage:  python download_model.py

- segm_lite_v681.tflite   Google Meet segmentation, 160x96 (about 0.4 MB)
- segm_full_v679.tflite   Google Meet segmentation, 256x144 (about 0.4 MB)
- selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite
                          ML Kit selfie segmentation, 256x256 (about 0.25 MB)

They are taken from the repo's main branch (public/models/). See the README's
license note: the Meet models' license is unclear.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/Volcomix/virtual-background/main/public/models/"
# Checksums of the files we tested with (main branch, 2026-10-05).
FILES = {
    "segm_lite_v681.tflite": "b13571477760f99012fbd6365b38406e94020b8eaaf513da8e8eff6332f38b44",
    "segm_full_v679.tflite": "4fa38a8d2e48e6e3a6fc59a8c88f8766147544947689dec299f654bfdb2b171e",
    "selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite":
        "8d13b7fae74af625c641226813616a2117bd6bca19eb3b75574621fc08557f27",
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
            raise SystemExit(
                f"{name} does not match the expected checksum.\n"
                f"Try again, or download it manually from:\n{BASE_URL + name}\n"
                f"and save it in: {MODELS_DIR}")
        tmp.rename(target)
    print(f"Done. Models are in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
