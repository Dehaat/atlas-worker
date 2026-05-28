# infra/docker/Dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---------------- system packages ----------------

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    colmap \
 && rm -rf /var/lib/apt/lists/*

# ---------------- working dir ----------------

WORKDIR /app

# ---------------- copy repo ----------------

COPY . /app

# ---------------- python ----------------

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install \
    nerfstudio \
    opencv-python \
    numpy \
    pillow \
    runpod \
    requests

# ---------------- env ----------------

ENV PYTHONPATH=/app:/app/sdk:/app/intelligence/src

# ---------------- startup ----------------

CMD ["python3", "infra/runpod/handler.py"]