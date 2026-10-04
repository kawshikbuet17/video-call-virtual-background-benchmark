#!/usr/bin/env python3
"""Download the official MODNet ONNX model into models/.

Usage:  python download_model.py

This is the "Image Matting Model" (photographic portrait matting) in ONNX format,
linked from the official repo: https://github.com/ZHKKKe/MODNet/tree/master/onnx
The file is hosted on Google Drive (about 25 MB).
"""
import hashlib
import urllib.request
from pathlib import Path

DRIVE_FILE_ID = "1cgycTQlYXpTh26gB9FTnthE7AvruV8hd"
# Direct download URL for a Google Drive file. "confirm=t" skips Drive's
# "file is too large to scan for viruses" page.
URL = f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}&export=download&confirm=t"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "modnet_photographic_portrait_matting.onnx"


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
    print("Downloading MODNet ONNX model from Google Drive (about 25 MB)...")
    urllib.request.urlretrieve(URL, tmp)

    if sha256_of(tmp) != SHA256:
        tmp.unlink()
        raise SystemExit(
            "Downloaded file does not match the expected checksum.\n"
            "Google Drive may have returned an error page (for example a download quota).\n"
            "Try again later, or download it manually from the link in the official repo:\n"
            "https://github.com/ZHKKKe/MODNet/tree/master/onnx\n"
            f"and save it as: {MODEL_FILE}")
    tmp.rename(MODEL_FILE)
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
