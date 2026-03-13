# ML Worker Dockerfile — §9.1
# Image: nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
# Role: Celery consumer executing GPU-bound ML inference tasks
# GPU: All (§9.3)
# SPEC-AMEND-001: Downgraded from cuDNN 9 → cuDNN 8 for Pascal (SM 6.1) compatibility

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps including FFmpeg for audio resampling (§4.C.2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    adb \
    && rm -rf /var/lib/apt/lists/*

# §3.2 — Build context is monorepo root so /packages is accessible
WORKDIR /app

COPY requirements/worker.txt /app/requirements/worker.txt
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/worker.txt

COPY packages/ /app/packages/
COPY services/worker/ /app/services/worker/

# TODO: Download spaCy model
# RUN python3.11 -m spacy download en_core_web_sm

CMD ["celery", "-A", "services.worker.celery_app", "worker", "--loglevel=info"]
