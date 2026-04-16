---
name: docker-topology
description: Docker Compose container topology, GPU allocation, volume mounts, and service dependency order. Use when editing docker-compose.yml, Dockerfiles, or debugging container startup, networking, or GPU access issues.
---

# Docker Compose Runtime Topology (§9)

## Service dependency order (§9.6)

1. redis (health: `redis-cli ping`)
2. postgres (health: `pg_isready`, depends: redis)
3. stream_scrcpy (depends: redis)
4. worker (depends: redis + postgres + stream_scrcpy)
5. orchestrator (depends: redis + postgres + stream_scrcpy)
6. api (depends: redis + worker + postgres; redis is required for Oura webhook physiology ingress enqueue)

## Container specs (§9.1)

api: python:3.11-slim, port 8000:8000, restart unless-stopped.
worker: nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 (SPEC-AMEND-001: cuDNN 8 for Pascal SM 6.1 dp4a compatibility), no port, restart on-failure:5, ALL GPUs reserved.
orchestrator: same image as worker (nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04), command `python3.11 -m services.worker.run_orchestrator`, no port, restart on-failure:5, ALL GPUs reserved (SPEC-AMEND-003: separate container for Module C).
redis: redis:7-alpine, port 6379 internal only, restart unless-stopped.
postgres: postgres:16-alpine, port 5432 internal only, restart unless-stopped.
stream_scrcpy: ubuntu:24.04 + scrcpy v3.3.4 (SPEC-AMEND-002: Ubuntu 24.04 for GLIBC 2.38+, SPEC-AMEND-004: scrcpy v3.3.4 dual-instance architecture), no port, restart on-failure:3, device /dev/bus/usb.

## Environment variables

stream_scrcpy:
- SDL_VIDEODRIVER=dummy (headless, no display)
- XDG_RUNTIME_DIR=/tmp (prevent Wayland/Display abort)
- ADB_SERVER_SOCKET=tcp:host.docker.internal:5037 (remote ADB server)

worker:
- CELERYD_CONCURRENCY=1 (single worker process, prevent GPU contention)
- OMP_NUM_THREADS=1 (prevent CPU thread collisions in PyTorch)
- HF_HOME=/data/interim/hf_cache (persist 3GB Whisper model across restarts)
- ADB_SERVER_SOCKET=tcp:host.docker.internal:5037

orchestrator:
- ADB_SERVER_SOCKET=tcp:host.docker.internal:5037

## Volumes (§9.2)

ipc-share → /tmp/ipc/ (stream_scrcpy + worker + orchestrator).
data-raw → /data/raw/ (worker + orchestrator). data-interim → /data/interim/ (worker + orchestrator). data-processed → /data/processed/ (worker + orchestrator + api).
pg-data → /var/lib/postgresql/data/ (postgres).

## Build context (§3.2)

ALL Dockerfiles MUST set build context to monorepo root. `.dockerignore` prevents data/ and docs/ from entering images. API image excludes ML deps via `requirements/api.txt`. Worker image excludes web assets.

## GPU (§9.3)

Worker and orchestrator containers only. `deploy.resources.reservations.devices: driver nvidia, count all, capabilities [gpu]`. Requires Docker Compose v2.x.