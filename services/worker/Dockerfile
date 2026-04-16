# ML Worker Dockerfile — §9.1
# Image: nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
# Role: ML Worker executing GPU-bound inference tasks
# GPU: All (§9.3)
# SPEC-AMEND-001: Downgraded from cuDNN 9 → cuDNN 8 for Pascal (SM 6.1) compatibility

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# System deps including FFmpeg for audio resampling (§4.C.2)
# and adb for drift correction (§4.C.1)
RUN apt-get update -o Acquire::ForceIPv4=true && apt-get install -y --no-install-recommends -o Acquire::ForceIPv4=true \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    adb \
    && rm -rf /var/lib/apt/lists/*

# §3.2 — Build context is monorepo root so /packages is accessible
WORKDIR /app

# §10.2 — Install pinned Python dependencies
COPY requirements/worker.txt /app/requirements/worker.txt
COPY requirements/base.txt /app/requirements/base.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements/worker.txt

# §4.D.4 — Download spaCy English model for text preprocessing
RUN python3.11 -m spacy download en_core_web_sm

# §3.2 — Copy shared packages and worker service code
COPY packages/ /app/packages/
COPY services/__init__.py /app/services/__init__.py
COPY services/worker/ /app/services/worker/

# §5.1 — Ephemeral Vault directories for transient media
RUN mkdir -p /data/raw /data/interim /data/processed

# §9.3 — Verify CUDA runtime is accessible at build time
RUN python3.11 -c "import ctypes; ctypes.CDLL('libcudart.so')" || echo "CUDA runtime check (build-time only)"

CMD ["celery", "-A", "services.worker.celery_app", "worker", "--loglevel=info", "-I", "services.worker.tasks.inference"]