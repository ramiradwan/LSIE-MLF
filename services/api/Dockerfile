# API Server Dockerfile — §9.1
# Image: python:3.11-slim
# Role: API Server REST endpoints on port 8000
# §3.2 — MUST exclude heavy ML dependencies

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# §10.2 — Install pinned Python dependencies
COPY requirements/api.txt /app/requirements/api.txt
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/api.txt

# §3.2 — Only API logic and shared schemas; no ML packages
COPY packages/__init__.py /app/packages/__init__.py
COPY packages/schemas/ /app/packages/schemas/
COPY services/__init__.py /app/services/__init__.py
COPY services/api/ /app/services/api/

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
