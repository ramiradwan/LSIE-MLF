"""SharedMemory transport for 30 s PCM windows.

Producer side (``module_c_orchestrator``): :func:`write_pcm_block`
allocates a SharedMemory block named ``lsie_ipc_pcm_{uuid4}``, writes
the audio bytes, and returns a :class:`PcmBlock` whose handle is kept
alive by the producer. On Windows this matters: the kernel reclaims
an anonymous-named file mapping the moment the last handle closes, so
if the producer drops its handle before the consumer attaches, the
attach fails with ``FileNotFoundError``. POSIX is more permissive, but
the same lifecycle keeps both platforms consistent.

The producer dispatches the block's :class:`PcmBlockMetadata` over the
IPC queue, and calls :meth:`PcmBlock.close_and_unlink` once the consumer
ack lands (or on shutdown / orchestrator session end).

Consumer side (``gpu_ml_worker``): :func:`read_pcm_block` attaches by
name with ``create=False``, copies the bytes out, verifies SHA-256,
then detaches. The consumer never unlinks — that is the producer's
lifecycle. Python 3.11's ``shared_memory`` module does not register
attach-only ``SharedMemory`` instances with the parent's
``resource_tracker`` (``register`` is called only when ``create=True``),
so the consumer naturally has attach-only tracking semantics. Python
3.13+'s explicit ``track=False`` flag would express the same intent
directly.

Crash safety: a producer that crashes mid-flow leaves the block
allocated. On POSIX the ``ipc.cleanup`` Dirty State Recovery sweep
reclaims it via ``/dev/shm`` enumeration. On Windows the kernel
reclaims the mapping when the producer process exits and its handle
disappears.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

SHM_PREFIX = "lsie_ipc_pcm_"


@dataclass(frozen=True)
class PcmBlockMetadata:
    """Producer→consumer descriptor for one SharedMemory PCM block."""

    name: str
    byte_length: int
    sha256: str


@dataclass
class PcmBlock:
    """Producer-side handle to a live SharedMemory PCM block.

    Hold this until the consumer has acked, then call
    :meth:`close_and_unlink`. Dropping the reference without unlinking
    leaks the block on POSIX (Dirty State Recovery cleans it later); on
    Windows the kernel reclaims it on producer-process exit.
    """

    metadata: PcmBlockMetadata
    _shm: shared_memory.SharedMemory = field(repr=False)
    _closed: bool = field(default=False, repr=False)

    def close_and_unlink(self) -> None:
        """Zeroize the buffer, release the producer handle, remove the OS name.

        Idempotent. The zeroize-before-close ordering is deliberate per
        §5.2: ``close()`` releases the local handle and on POSIX may
        immediately make the page reusable by another process, so we
        wipe the bytes while we still own the live mapping.
        """
        if self._closed:
            return
        self._closed = True
        # Late import keeps shared_buffers free of a privacy → ipc
        # cycle if the privacy package ever needs PcmBlock.
        from services.desktop_app.privacy.zeroize import zeroize_pcm_block

        try:
            zeroize_pcm_block(self._shm)
        except Exception:  # noqa: BLE001
            logger.debug("PcmBlock zeroize failed for %s", self.metadata.name, exc_info=True)
        try:
            self._shm.close()
        except Exception:  # noqa: BLE001
            logger.debug("PcmBlock close failed for %s", self.metadata.name, exc_info=True)
        try:
            self._shm.unlink()  # no-op on Windows; shm_unlink on POSIX
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("PcmBlock unlink failed for %s", self.metadata.name, exc_info=True)


def write_pcm_block(audio: bytes) -> PcmBlock:
    """Allocate a SharedMemory block, write ``audio``, return a live handle.

    The producer's local SharedMemory handle remains open inside the
    returned :class:`PcmBlock` — the caller is responsible for calling
    :meth:`PcmBlock.close_and_unlink` once the consumer is done.

    Each block is given a fresh ``uuid4``-suffixed name so two
    concurrent segments cannot collide.
    """
    if not audio:
        raise ValueError("write_pcm_block: audio is empty")

    name = f"{SHM_PREFIX}{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=len(audio))
    try:
        buf = shm.buf
        if buf is None:
            raise RuntimeError(f"PCM block {name} created with null buffer")
        buf[: len(audio)] = audio
    except BaseException:
        # Roll back the allocation on any write failure.
        shm.close()
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()
        raise

    metadata = PcmBlockMetadata(
        name=name,
        byte_length=len(audio),
        sha256=hashlib.sha256(audio).hexdigest(),
    )
    return PcmBlock(metadata=metadata, _shm=shm)


def read_pcm_block(metadata: PcmBlockMetadata) -> bytes:
    """Attach to ``metadata.name``, copy bytes out, verify SHA-256.

    Raises ``FileNotFoundError`` if the block has been unlinked (or, on
    Windows, the producer dropped its last handle). Raises
    ``ValueError`` if the byte length or checksum disagree with the
    metadata — the caller MUST treat that as a corrupted segment and
    discard rather than retry.
    """
    shm = shared_memory.SharedMemory(name=metadata.name, create=False)
    try:
        buf = shm.buf
        if buf is None:
            raise ValueError(f"PCM block {metadata.name} attached with null buffer")
        if len(buf) < metadata.byte_length:
            raise ValueError(
                f"PCM block {metadata.name} short read: "
                f"buf={len(buf)} < expected={metadata.byte_length}"
            )
        audio = bytes(buf[: metadata.byte_length])
    finally:
        shm.close()

    actual = hashlib.sha256(audio).hexdigest()
    if actual != metadata.sha256:
        raise ValueError(
            f"PCM block {metadata.name} checksum mismatch: "
            f"expected {metadata.sha256[:16]}…, got {actual[:16]}…"
        )
    return audio
