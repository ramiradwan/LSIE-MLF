# Capture Container Dockerfile — §9.1
# Image: ubuntu:22.04 + scrcpy v2.x
# Role: Raw USB audio/video stream extraction
# Device passthrough: /dev/bus/usb (§9.4)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# §9.1 — System dependencies: adb for USB, ffmpeg for fallback, wget for scrcpy download
RUN apt-get update && apt-get install -y --no-install-recommends \
    adb \
    ffmpeg \
    wget \
    libsdl2-2.0-0 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# §9.1 — Install scrcpy v2.7 from official GitHub release
# scrcpy provides raw PCM audio capture from Android via USB (§4.A.1)
ARG SCRCPY_VERSION=2.7
RUN mkdir -p /usr/local/share/scrcpy \
    && wget -q "https://github.com/Genymobile/scrcpy/releases/download/v${SCRCPY_VERSION}/scrcpy-server-v${SCRCPY_VERSION}" \
        -O /usr/local/share/scrcpy/scrcpy-server \
    && wget -q "https://github.com/Genymobile/scrcpy/releases/download/v${SCRCPY_VERSION}/scrcpy-linux-x86_64-v${SCRCPY_VERSION}.tar.gz" \
        -O /tmp/scrcpy.tar.gz \
    && tar xzf /tmp/scrcpy.tar.gz -C /tmp/ \
    && cp /tmp/scrcpy-linux-x86_64-v${SCRCPY_VERSION}/scrcpy /usr/local/bin/scrcpy \
    && chmod +x /usr/local/bin/scrcpy \
    && rm -rf /tmp/scrcpy* \
    && scrcpy --version

# §4.A.1 — IPC Pipe mount point (shared volume with ML Worker)
RUN mkdir -p /tmp/ipc

# §3.2 — Build context is monorepo root so services/ is accessible
COPY services/stream_ingest/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
