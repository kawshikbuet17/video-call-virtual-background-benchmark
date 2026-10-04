#!/usr/bin/env python3
"""Download the fast_person_segmentation model into models/.

Usage:  python download_model.py

Model: mnetv2_munet_transpose_e260_s128.pb (about 14.7 MB), a frozen TensorFlow
graph of a Mobile-UNet (MobileNetV2 encoder, transposed-convolution decoder,
128x128 input). The repo's model_info.json calls it the "best & fastest model".
It is stored in the GitHub repo https://github.com/SamSamhuns/fast_person_segmentation
(cpp/cppflow_pseg/examples/pseg_model/model_zoo/). It comes from the
"deconv_model-260" checkpoint of https://github.com/anilsathyan7/Portrait-Segmentation (MIT).
"""
import hashlib
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/SamSamhuns/fast_person_segmentation/master/"
       "cpp/cppflow_pseg/examples/pseg_model/model_zoo/mnetv2_munet_transpose_e260_s128.pb")
# Checksum of the file we tested with, to catch broken or changed downloads.
SHA256 = "20359e8aeb07882d4fc395ebbccb092b8df57e78ffffb8f8e052542b3486ad2b"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "mnetv2_munet_transpose_e260_s128.pb"


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
