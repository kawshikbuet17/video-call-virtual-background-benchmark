#!/usr/bin/env python3
"""Download the PP-HumanSegV2-Lite ONNX model used by netesh3/webinar into models/.

Usage:  python download_model.py

The file is the 256x144 PP-HumanSegV2-Lite portrait model from PaddleSeg
(Apache-2.0), converted to ONNX by the webinar project (see its
web/public/models/README.md). About 3.8 MB. It is always taken from the repo's
main branch (the latest version).
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/netesh3/webinar/main/web/public/models/humanseg.onnx"
# Checksum of the file we tested with (main on 2026-10-03, also written in the webinar
# repo's README). If main gets a new model later, the download still works but warns.
SHA256 = "e2e9445d874b5fb119a01be5b2c5f33baa2a5de7bf7959f5b62fceb26c66a0e2"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "humanseg.onnx"


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
    print("Downloading PP-HumanSegV2-Lite ONNX model from GitHub (about 3.8 MB)...")
    urllib.request.urlretrieve(URL, tmp)

    checksum = sha256_of(tmp)
    tmp.rename(MODEL_FILE)
    if checksum != SHA256:
        # Not an error page: GitHub answers a missing file with HTTP 404 and urlretrieve
        # stops with an error. So the webinar repo has a newer model than the tested one.
        print("WARNING: this is not the model file this folder was tested with.\n"
              "The webinar repo's main branch has a newer humanseg.onnx. run.py should still\n"
              "work if its input/output did not change (1x3x144x256 in, 1x2x144x256 out).\n"
              f"New sha256: {checksum}")
    print(f"Done. Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
