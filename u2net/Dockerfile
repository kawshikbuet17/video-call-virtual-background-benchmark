# U^2-Net human segmentation (PyTorch), CPU. The image is large (176 MB weights + PyTorch).
# Build from this folder:  docker build -t vbg-u2net .
FROM python:3.10-slim

# libgl1 + libglib2.0-0: opencv-python fails to import without them.
# libsm6: needed only to open a window (Linux webcam + display command).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Same layout as the repo, so run.py finds ../assets (mounted at run time).
WORKDIR /app/u2net

COPY requirements.txt .
# The extra index makes pip pick the CPU-only PyTorch build (the default Linux
# build from PyPI includes CUDA and is several GB).
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY download_model.py run.py u2net_model.py LICENSE-U2Net ./
# Download the weights now so the image works out of the box.
RUN python download_model.py

ENTRYPOINT ["python", "run.py"]
# Default: process the sample image headless and save results to outputs/.
CMD ["--no-display", "--image", "../assets/samples/person_portrait.jpg"]
