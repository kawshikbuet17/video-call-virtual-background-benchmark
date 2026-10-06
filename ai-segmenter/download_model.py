#!/usr/bin/env python3
"""Download the models for the AI-Segmenter pipeline into models/.

Usage:
    python download_model.py              # YOLO11n + RVM (about 21 MB)
    python download_model.py --birefnet   # also BiRefNet (about 445 MB, optional)

- yolo11n.pt                  Ultralytics YOLO11 nano detector (boxes only), the one
                              AI-Segmenter uses to pick objects. AGPL-3.0.
- rvm_mobilenetv3_fp32.onnx   Robust Video Matting, MobileNetV3, the soft alpha. GPL-3.0.
- birefnet/                   BiRefNet (ZhengPeng7/BiRefNet on Hugging Face), a large
                              high-quality model. MIT. Pinned to one revision, and its
                              code is saved here too, so nothing is fetched at run time.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
FILES = {
    # name: (URL, sha256 of the file we tested with)
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "rvm_mobilenetv3_fp32.onnx": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"),
}
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"   # tested revision (2026-10-06)
BIREFNET_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"  # model.safetensors


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(name, url, expected):
    target = MODELS_DIR / name
    if target.exists():
        print(f"Already downloaded: {name}")
        return
    tmp = target.with_suffix(".part")
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, tmp)
    if sha256_of(tmp) != expected:
        tmp.unlink()
        raise SystemExit(f"{name} does not match the expected checksum. Try again, or download it "
                         f"manually from:\n{url}\nand save it in: {MODELS_DIR}")
    tmp.rename(target)


def download_birefnet():
    from huggingface_hub import snapshot_download  # installed together with transformers

    target = MODELS_DIR / "birefnet"
    print("Downloading BiRefNet from Hugging Face (about 445 MB) ...")
    snapshot_download(BIREFNET_REPO, revision=BIREFNET_REVISION, local_dir=target,
                      allow_patterns=["*.py", "*.json", "model.safetensors"])
    if sha256_of(target / "model.safetensors") != BIREFNET_SHA256:
        raise SystemExit("BiRefNet's model.safetensors does not match the expected checksum. "
                         f"Delete {target} and try again.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--birefnet", action="store_true", help="also download BiRefNet (about 445 MB)")
    args = p.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        download_file(name, url, expected)
    if args.birefnet:
        download_birefnet()
    print(f"Done. Models are in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
