FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

LABEL org.lsie_mlf.cloud_control_plane.app="services.cloud_api.main:app" \
      org.lsie_mlf.cloud_control_plane.host="0.0.0.0" \
      org.lsie_mlf.cloud_control_plane.port="8000" \
      org.lsie_mlf.cloud_control_plane.bootstrap_command="python -m services.cloud_api.db.bootstrap"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY packages/__init__.py ./packages/__init__.py
COPY packages/schemas ./packages/schemas
COPY services/__init__.py ./services/__init__.py
COPY services/cloud_api ./services/cloud_api

RUN uv sync --frozen --no-dev --no-install-project

EXPOSE 8000

CMD ["uvicorn", "services.cloud_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
