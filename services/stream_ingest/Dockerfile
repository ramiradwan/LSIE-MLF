# Capture Container Dockerfile — §9.1
# Image: ubuntu:22.04 + scrcpy v2.x
# Role: Raw USB audio/video stream extraction
# Device passthrough: /dev/bus/usb (§9.4)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    adb \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# TODO: Install scrcpy v2.x from official release

RUN mkdir -p /tmp/ipc

COPY services/stream_ingest/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
