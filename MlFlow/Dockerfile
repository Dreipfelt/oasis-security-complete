# ─────────────────────────────────────────────────────────
# OASIS MLflow — Hugging Face Space
# MLflow Tracking Server
# Backend  : NeonDB (PostgreSQL)
# Artifacts: HuggingFace Dataset (hf://datasets/Dreipfelt/oasis-mlflow-artifacts)
# ─────────────────────────────────────────────────────────
FROM continuumio/miniconda3

LABEL maintainer="Dreipfelt" \
      project="OASIS Security" \
      component="MLflow Tracking Server" \
      version="2.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces: non-root user required
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy files with correct ownership
COPY --chown=user . $HOME/app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Authenticate to HuggingFace Hub using the HF_TOKEN secret
# This allows MLflow to push artefacts to the HF Dataset
RUN --mount=type=secret,id=HF_TOKEN,mode=0444 \
    pip install --no-cache-dir huggingface_hub && \
    echo "HF auth configured"

# Launch MLflow tracking server
# All env vars are injected as HF Space Secrets — never hardcoded
CMD huggingface-cli login --token $HF_TOKEN --add-to-git-credential && \
    mlflow server \
    --host 0.0.0.0 \
    --port $PORT \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_STORE_URI \
    --serve-artifacts \
    --workers 2
