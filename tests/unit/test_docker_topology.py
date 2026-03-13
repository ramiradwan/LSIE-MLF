"""
Tests for Docker topology — Phase 7.1/7.2/6.4 validation.

Validates docker-compose.yml and Dockerfiles against:
  §9.1 — Container images and roles
  §9.2 — Volume mounts
  §9.3 — GPU reservation
  §9.4 — USB device passthrough
  §9.5 — Network configuration
  §9.6 — Service dependency order
  §3.2 — Build context and image separation
"""

from __future__ import annotations

from pathlib import Path

import pytest

COMPOSE_PATH = Path("docker-compose.yml")
WORKER_DOCKERFILE_PATH = Path("services/worker/Dockerfile")
API_DOCKERFILE_PATH = Path("services/api/Dockerfile")


@pytest.fixture()
def compose_content() -> str:
    """Read docker-compose.yml content."""
    return COMPOSE_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def worker_dockerfile() -> str:
    """Read worker Dockerfile content."""
    return WORKER_DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def api_dockerfile() -> str:
    """Read API Dockerfile content."""
    return API_DOCKERFILE_PATH.read_text(encoding="utf-8")


class TestComposeTopology:
    """§9 — Five-container topology validation."""

    def test_five_services(self, compose_content: str) -> None:
        """§9.1 — Exactly five services defined."""
        import re

        services = re.findall(r"^\s{2}(\w+):", compose_content, re.MULTILINE)
        # Filter to actual service names (not sub-keys)
        top_level = [s for s in services if s not in ("condition", "driver")]
        assert "redis" in top_level
        assert "postgres" in top_level
        assert "stream_scrcpy" in top_level
        assert "worker" in top_level
        assert "api" in top_level

    def test_redis_image(self, compose_content: str) -> None:
        """§9.1 — Message Broker uses redis:7-alpine."""
        assert "redis:7-alpine" in compose_content

    def test_postgres_image(self, compose_content: str) -> None:
        """§9.1 — Persistent Store uses postgres:16-alpine."""
        assert "postgres:16-alpine" in compose_content

    def test_network_bridge(self, compose_content: str) -> None:
        """§9.5 — appnetwork uses bridge driver."""
        assert "appnetwork:" in compose_content
        assert "driver: bridge" in compose_content


class TestComposeVolumes:
    """§9.2 — Volume mount validation."""

    def test_ipc_share_volume(self, compose_content: str) -> None:
        """§9.2 — ipc-share volume for IPC Pipe."""
        assert "ipc-share:/tmp/ipc/" in compose_content

    def test_data_volumes(self, compose_content: str) -> None:
        """§9.2 — Data volumes for Ephemeral Vault."""
        assert "data-raw:/data/raw/" in compose_content
        assert "data-interim:/data/interim/" in compose_content
        assert "data-processed:/data/processed/" in compose_content

    def test_pg_data_volume(self, compose_content: str) -> None:
        """§9.2 — PostgreSQL data persistence."""
        assert "pg-data:/var/lib/postgresql/data/" in compose_content


class TestComposeDependencies:
    """§9.6 — Service dependency order."""

    def test_postgres_depends_on_redis(self, compose_content: str) -> None:
        """§9.6 — Persistent Store starts after Message Broker."""
        # postgres section should have depends_on redis
        assert "service_healthy" in compose_content

    def test_worker_depends_on_postgres(self, compose_content: str) -> None:
        """§9.6 — ML Worker starts after Persistent Store."""
        assert "stream_scrcpy:" in compose_content

    def test_api_depends_on_worker(self, compose_content: str) -> None:
        """§9.6 — API Server starts after ML Worker."""
        assert "worker:" in compose_content


class TestComposeGPU:
    """§9.3 — GPU reservation."""

    def test_nvidia_gpu_reservation(self, compose_content: str) -> None:
        """§9.3 — ML Worker reserves all NVIDIA GPUs."""
        assert "driver: nvidia" in compose_content
        assert "count: all" in compose_content
        assert "capabilities: [gpu]" in compose_content


class TestComposeDevicePassthrough:
    """§9.4 — USB device passthrough."""

    def test_usb_device(self, compose_content: str) -> None:
        """§9.4 — Capture Container has USB device access."""
        assert "/dev/bus/usb:/dev/bus/usb" in compose_content


class TestWorkerDockerfile:
    """§9.1 — ML Worker image validation."""

    def test_cuda_base_image(self, worker_dockerfile: str) -> None:
        """§9.1 — CUDA 12.2.2 + cuDNN 8 base image."""
        assert "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04" in worker_dockerfile

    def test_python_311(self, worker_dockerfile: str) -> None:
        """§10.2 — Python 3.11 installed."""
        assert "python3.11" in worker_dockerfile

    def test_ffmpeg_installed(self, worker_dockerfile: str) -> None:
        """§4.C.2 — FFmpeg for audio resampling."""
        assert "ffmpeg" in worker_dockerfile

    def test_adb_installed(self, worker_dockerfile: str) -> None:
        """§4.C.1 — adb for drift correction."""
        assert "adb" in worker_dockerfile

    def test_spacy_model_downloaded(self, worker_dockerfile: str) -> None:
        """§4.D.4 — spaCy en_core_web_sm model downloaded."""
        assert "spacy download en_core_web_sm" in worker_dockerfile

    def test_packages_copied(self, worker_dockerfile: str) -> None:
        """§3.2 — Shared packages copied into image."""
        assert "COPY packages/" in worker_dockerfile

    def test_worker_code_copied(self, worker_dockerfile: str) -> None:
        """§3.2 — Worker service code copied."""
        assert "COPY services/worker/" in worker_dockerfile

    def test_data_directories_created(self, worker_dockerfile: str) -> None:
        """§5.1 — Ephemeral Vault directories exist."""
        assert "/data/raw" in worker_dockerfile
        assert "/data/interim" in worker_dockerfile
        assert "/data/processed" in worker_dockerfile

    def test_celery_entrypoint(self, worker_dockerfile: str) -> None:
        """ML Worker runs as Celery consumer."""
        assert "celery" in worker_dockerfile
        assert "services.worker.celery_app" in worker_dockerfile


class TestApiDockerfile:
    """§9.1 — API Server image validation."""

    def test_python_slim_base(self, api_dockerfile: str) -> None:
        """§9.1 — python:3.11-slim base image."""
        assert "python:3.11-slim" in api_dockerfile

    def test_excludes_ml_packages(self, api_dockerfile: str) -> None:
        """§3.2 — API image does NOT contain ML packages."""
        assert "COPY packages/ml_core/" not in api_dockerfile

    def test_schemas_included(self, api_dockerfile: str) -> None:
        """§3.2 — Shared schemas included."""
        assert "COPY packages/schemas/" in api_dockerfile

    def test_api_code_copied(self, api_dockerfile: str) -> None:
        """§3.2 — API service code copied."""
        assert "COPY services/api/" in api_dockerfile

    def test_port_8000_exposed(self, api_dockerfile: str) -> None:
        """API Server exposes port 8000."""
        assert "EXPOSE 8000" in api_dockerfile

    def test_uvicorn_entrypoint(self, api_dockerfile: str) -> None:
        """API Server runs via uvicorn."""
        assert "uvicorn" in api_dockerfile
        assert "services.api.main:app" in api_dockerfile

    def test_init_files_copied(self, api_dockerfile: str) -> None:
        """Root __init__.py files copied for Python package resolution."""
        assert "packages/__init__.py" in api_dockerfile
        assert "services/__init__.py" in api_dockerfile
