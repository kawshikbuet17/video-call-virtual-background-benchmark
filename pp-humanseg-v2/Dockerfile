# PP-HumanSeg V2 (portrait Lite), CPU.
# Build from this folder:  docker build -t vbg-pp-humanseg-v2 .
FROM python:3.10-slim

# libgl1 + libglib2.0-0: opencv-python fails to import without them.
# libsm6: needed only to open a window (Linux webcam + display command).
# libgomp1: OpenMP runtime that paddlepaddle needs to import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Same layout as the repo, so run.py finds ../assets (mounted at run time).
WORKDIR /app/pp-humanseg-v2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py run.py ./
# Download the weights now so the image works out of the box.
RUN python download_model.py

ENTRYPOINT ["python", "run.py"]
# Default: process the sample image headless and save results to outputs/.
CMD ["--no-display", "--image", "../assets/samples/person_portrait.jpg"]
