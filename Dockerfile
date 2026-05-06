# oasis/models/crime_predictor/Dockerfile

# ── Stage 1: Training ──────────────────────────
FROM python:3.11-slim AS trainer

LABEL stage="training"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ARG DATA_URL
RUN python train.py --data-url ${DATA_URL}

# ── Stage 2: Production (minimal) ─────────────
FROM python:3.11-slim AS production

LABEL maintainer="Dreipfelt" \
      project="OASIS Security" \
      version="1.0" \
      description="Crime predictor inference API — CDSD Project"

# Utilisateur non-root (sécurité)
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Dépendances de serving uniquement (pas sklearn, statsmodels, etc.)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serving.txt

# Artefact ML depuis le stage trainer
COPY --from=trainer /app/models/crime_predictor.pkl ./models/

# Code applicatif
COPY predict.py model.py config.yaml ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]