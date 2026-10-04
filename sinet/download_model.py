#!/usr/bin/env python3
"""Download the official SINet portrait segmentation weights into models/.

Usage:  python download_model.py

Weights: result/SINet/SINet.pth (about 445 KB) from the official repo
https://github.com/clovaai/ext_portrait_segmentation (MIT license),
trained on the EG1800 portrait dataset.
(Not the clovaai/c3_sinet repo: its SINet.pth is a 19-class street-scene model.)
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://github.com/clovaai/ext_portrait_segmentation/raw/master/result/SINet/SINet.pth"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "d164a9643c6e6765fc6dd4ec73f6018ad2526c21336911b4484715f33e6981a4"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "SINet.pth"


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
