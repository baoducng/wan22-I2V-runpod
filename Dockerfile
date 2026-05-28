FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update
Run apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python tooling
RUN pip install --upgrade pip setuptools wheel


# Install remaining deps (PyPI + git)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY . .

# Model is downloaded from HuggingFace Hub at container startup (not baked in)
# To cache on a persistent volume, set HF_HOME=/runpod-volume/hf_cache in endpoint env vars
#RUN python3 preload_model.py

CMD ["python3", "handler.py"]
