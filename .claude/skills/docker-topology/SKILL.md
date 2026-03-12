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
5. api (depends: worker + postgres)

## Container specs (§9.1)

api: python:3.11-slim, port 8000:8000, restart unless-stopped.
worker: nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04, no port, restart on-failure:5, ALL GPUs reserved.
redis: redis:7-alpine, port 6379 internal only, restart unless-stopped.
postgres: postgres:16-alpine, port 5432 internal only, restart unless-stopped.
stream_scrcpy: ubuntu:22.04 + scrcpy v2.x, no port, restart on-failure:3, device /dev/bus/usb.

## Volumes (§9.2)

ipc-share → /tmp/ipc/ (stream_scrcpy + worker).
data-raw → /data/raw/ (worker). data-interim → /data/interim/ (worker). data-processed → /data/processed/ (worker + api).
pg-data → /var/lib/postgresql/data/ (postgres).

## Build context (§3.2)

ALL Dockerfiles MUST set build context to monorepo root. `.dockerignore` prevents data/ and docs/ from entering images. API image excludes ML deps via `requirements/api.txt`. Worker image excludes web assets.

## GPU (§9.3)

Worker container only. `deploy.resources.reservations.devices: driver nvidia, count all, capabilities [gpu]`. Requires Docker Compose v2.x.
