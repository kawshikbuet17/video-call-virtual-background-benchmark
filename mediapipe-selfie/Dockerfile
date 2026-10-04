# MediaPipe Selfie Segmentation (landscape), CPU.
# Build from this folder:  docker build -t vbg-mediapipe-selfie .
FROM python:3.10-slim

# libgl1 + libglib2.0-0: OpenCV (and MediaPipe) fail to import without them.
# libsm6: needed only to open a window (Linux webcam + display command).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Same layout as the repo, so run.py finds ../assets (mounted at run time).
WORKDIR /app/mediapipe-selfie

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py run.py ./
# Download the weights now so the image works out of the box.
RUN python download_model.py

ENTRYPOINT ["python", "run.py"]
# Default: process the sample image headless and save results to outputs/.
CMD ["--no-display", "--image", "../assets/samples/person_portrait.jpg"]
