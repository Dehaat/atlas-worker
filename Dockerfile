FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    zip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY sdk /app/sdk
COPY schemas /app/schemas
COPY workers /app/workers
COPY infra /app/infra

RUN pip install --upgrade pip

RUN pip install -e /app/sdk
RUN pip install -e /app/schemas

RUN pip install \
    pycolmap \
    opencv-python \
    pillow \
    numpy \
    runpod \
    requests

RUN pip install nerfstudio

RUN python -c "import torch; print(torch.cuda.is_available())"

CMD ["python3", "/app/infra/runpod/handler.py"]