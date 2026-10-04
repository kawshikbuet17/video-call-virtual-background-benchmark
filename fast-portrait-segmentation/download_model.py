#!/usr/bin/env python3
"""Download the Fast_Portrait_Segmentation final model (ERD seg-matting) into models/.

Usage:  python download_model.py

Fast_Portrait_Segmentation (https://github.com/lizhengwei1992/Fast_Portrait_Segmentation)
has network code but no weights. Its README says the final model was released in the
author's follow-up repo https://github.com/lizhengwei1992/mobile_phone_human_matting.
From there we download:
  - pre_trained/erd_seg_matting/model/ckpt_lastest.pth  (the weights, about 3.4 MB)
  - model/segnet.py                                     (the network code)

Both repos have no license file, so we do not copy the network code into this
project: it is downloaded into models/ (which git ignores), like the weights.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/lizhengwei1992/mobile_phone_human_matting/master/"
MODELS_DIR = Path(__file__).resolve().parent / "models"
# Remote path -> (local file name, SHA256 of the file we tested with).
FILES = {
    "pre_trained/erd_seg_matting/model/ckpt_lastest.pth":
        ("ckpt_lastest.pth", "d4a41d006e3abad49b2337f54b46251f0aed5900f1958b53a323b12237c3a62a"),
    "model/segnet.py":
        ("segnet.py", "344ae5ef12f651b71a0411fa29d1bd3d6853aed6118e57b129bd184a45d9df28"),
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if all((MODELS_DIR / local).exists() for local, _ in FILES.values()):
        print(f"Model already downloaded: {MODELS_DIR}")
        return
    MODELS_DIR.mkdir(exist_ok=True)
    for remote, (local, expected) in FILES.items():
        path = MODELS_DIR / local
        print(f"Downloading {BASE + remote}")
        urllib.request.urlretrieve(BASE + remote, path)
        if sha256_of(path) != expected:
            path.unlink()
            raise SystemExit(f"{local} does not match the expected checksum. Try again later.")
    print(f"Done. Model saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
