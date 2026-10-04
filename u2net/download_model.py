#!/usr/bin/env python3
"""Download the official U^2-Net human segmentation weights into models/.

Usage:  python download_model.py

This is u2net_human_seg.pth (about 176 MB), the human segmentation model linked
from the official repo README: https://github.com/xuebinqin/U-2-Net
(trained on the Supervisely Person dataset). The file is hosted on Google Drive.

Not u2net_portrait.pth: that model turns photos into pencil drawings, it does
not separate a person from the background.
"""
import hashlib
import urllib.request
from pathlib import Path

DRIVE_FILE_ID = "1m_Kgs91b21gayc2XLW0ou8yugAIadWVP"
# Direct download URL for a Google Drive file. "confirm=t" skips Drive's
# "file is too large to scan for viruses" page.
URL = f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}&export=download&confirm=t"
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "13f0396e3f6fbc1d7ab55a8558640bef306a59b9110a5c595eb949ff69cf2414"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "u2net_human_seg.pth"


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
    print("Downloading U^2-Net human segmentation weights from Google Drive (about 176 MB)...")
    urllib.request.urlretrieve(URL, tmp)

    if sha256_of(tmp) != SHA256:
        tmp.unlink()
        raise SystemExit(
            "Downloaded file does not match the expected checksum.\n"
            "Google Drive may have returned an error page (for example a download quota).\n"
            "Try again later, or download it manually from the link in the official repo:\n"
            "https://github.com/xuebinqin/U-2-Net\n"
            f"and save it as: {MODEL_FILE}")
    tmp.rename(MODEL_FILE)
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
