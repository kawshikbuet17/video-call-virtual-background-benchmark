#!/usr/bin/env python3
"""Download the PP-HumanSeg V2 portrait Lite inference model into models/.

Usage:  python download_model.py

The model is the official PaddleSeg "Inference Model (Softmax)" for
PP-HumanSegV2-Lite (portrait, input 256x144). Softmax means the model outputs a
person probability per pixel (0..1), which gives smoother edges than a hard
0/1 mask.
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/"
       "portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip")
MODEL_NAME = "portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax"

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    target = MODELS_DIR / MODEL_NAME
    if (target / "model.pdiparams").exists():
        print(f"Model already downloaded: {target}")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    zip_path = MODELS_DIR / f"{MODEL_NAME}.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODELS_DIR)
    zip_path.unlink()
    # The zip was made on a Mac and contains an unneeded __MACOSX folder.
    shutil.rmtree(MODELS_DIR / "__MACOSX", ignore_errors=True)

    print(f"Done. Model saved to: {target}")


if __name__ == "__main__":
    main()
