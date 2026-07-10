# Reproducible alternative to startup.sh — build once, run anywhere with an NVIDIA GPU.
# Host needs a recent NVIDIA driver + nvidia-container-toolkit (the GCP DLVM images have both).
#
#   docker build -t seethrough-svc service/seethrough
#   docker run --gpus all -p 8000:8000 -v hf-cache:/root/.cache/huggingface seethrough-svc
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.12 python3.12-venv python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/shitagaki-lab/see-through.git /opt/see-through
WORKDIR /opt/see-through
RUN ln -sf common/assets assets

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt && \
    pip install fastapi "uvicorn[standard]"

COPY app.py /opt/app.py
WORKDIR /opt
ENV SEE_THROUGH_DIR=/opt/see-through
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
