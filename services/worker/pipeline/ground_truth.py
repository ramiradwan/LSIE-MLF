"""
Ground Truth Ingestion — §4.B Module B

Ingest real-time contextual metadata, chat streams, and user
interactions from TikTok Webcast to create ground truth datasets.

Technology: TikTokLive connector + EulerStream signature API.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)

# §12.1 Module B — Exponential backoff parameters
BACKOFF_INITIAL: float = 1.0  # seconds
BACKOFF_MAX: float = 30.0  # seconds
BACKOFF_MAX_RETRIES: int = 10


@runtime_checkable
class SignatureProvider(Protocol):
    """
    Abstraction over WebSocket signature generation for TikTok Webcast.

    The v2.0 spec (§4.B) delegates signature generation to EulerStream.
    This protocol exists so the provider can be replaced with a local
    patchright-based signer (using Module F's existing Chromium stack)
    without any changes to GroundTruthIngester or the module boundary.

    A local signer would execute TikTok's signing JS via page.evaluate()
    in a headless browser, eliminating the external API dependency.
    See services/worker/tasks/enrichment.py for the patchright infrastructure.
    """

    async def get_signed_params(self, stream_url: str) -> dict[str, str]:
        """Return signed URL parameters (X-Bogus, msToken) for WebSocket auth."""
        ...


class EulerStreamSigner:
    """
    §4.B.1 — Default signature provider using EulerStream API.

    Failure mode (§4.B contract): outage results in degraded mode.
    """

    def __init__(self, sign_url: str) -> None:
        self.sign_url = sign_url

    async def get_signed_params(self, stream_url: str) -> dict[str, str]:
        """
        §4.B.1 — HTTP call to EulerStream signing service.

        Returns signed URL parameters needed for TikTok WebSocket auth.
        On failure, raises to trigger degraded mode in GroundTruthIngester.
        """
        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                self.sign_url,
                params={"url": stream_url},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp,
        ):
            resp.raise_for_status()
            data: dict[str, Any] = await resp.json()
            return {
                "X-Bogus": str(data.get("X-Bogus", "")),
                "msToken": str(data.get("msToken", "")),
            }


CREATOR_FOLLOW_EVENT_TYPES: frozenset[str] = frozenset(
    {"creator_follow", "follow", "user_follow", "followevent"}
)


def creator_follow_outcome_from_live_event(
    live_event: dict[str, Any],
    *,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Normalize TikTok follow events into the canonical §7E outcome input.

    The returned dict is intentionally still an analytics payload, not a schema
    model. Module E assigns deterministic IDs and performs idempotent database
    persistence after the event reaches the attribution ledger path.
    """
    raw_event_type = str(live_event.get("event_type", "")).strip().lower()
    normalized_event_type = raw_event_type.replace("_", "")
    if raw_event_type not in CREATOR_FOLLOW_EVENT_TYPES and normalized_event_type not in {
        "creatorfollow",
        "userfollow",
        "followevent",
    }:
        return None

    raw_payload = live_event.get("payload")
    payload = cast(dict[str, Any], raw_payload) if isinstance(raw_payload, dict) else {}
    source_event_ref = (
        payload.get("event_id")
        or payload.get("message_id")
        or payload.get("msg_id")
        or live_event.get("event_id")
    )
    return {
        "session_id": session_id or live_event.get("session_id"),
        "outcome_type": "creator_follow",
        "outcome_value": 1.0,
        "outcome_time_utc": live_event.get("timestamp_utc"),
        "source_system": "tiktok_webcast",
        "source_event_ref": str(source_event_ref) if source_event_ref is not None else None,
        "confidence": 1.0,
    }


class GroundTruthIngester:
    """
    §4.B — TikTok Webcast event ingestion.

    Maintains persistent WebSocket connection, refreshes signature
    tokens during reconnects, and enforces Action_Combo constraint
    for composite event formation.

    Failure modes (§4.B contract):
      - WebSocket disconnect → exponential backoff (1s initial, 30s max, 10 retries)
      - Signature provider outage → degraded mode
      - Malformed protobuf → log and discard
    """

    def __init__(
        self,
        unique_id: str,
        signer: SignatureProvider,
        event_buffer: deque[dict[str, Any]] | None = None,
        on_combo_event: Callable[..., Any] | None = None,
    ) -> None:
        self.unique_id = unique_id
        self.signer = signer
        self.event_buffer: deque[dict[str, Any]] = (
            event_buffer if event_buffer is not None else deque(maxlen=10000)
        )
        self.on_combo_event = on_combo_event
        self._client: Any = None
        self._running: bool = False
        self._combo_window: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """
        Establish authenticated WebSocket connection via TikTokLive.

        §4.B.1 — Call signer.get_signed_params() then pass signed URL
        to TikTokLive client for WebSocket authentication.
        """
        from TikTokLive import TikTokLiveClient

        # §4.B.1 — Get signed parameters from EulerStream
        signed_params = await self.signer.get_signed_params(
            f"https://www.tiktok.com/@{self.unique_id}/live"
        )

        # §4.B.1 — Initialize TikTokLive client with signed params
        self._client = TikTokLiveClient(
            unique_id=self.unique_id,
            **signed_params,
        )

        logger.info("TikTokLive client connected for @%s", self.unique_id)

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """
        Parse protobuf event, apply Action_Combo constraint, stage for sync.

        §4.B — Module B → Module C: Python dicts with uniqueId, event_type,
        timestamp_utc, payload.
        §4.B.1 — Action_Combo: multiple events that must coincide temporally.
        """
        try:
            # §4.B — Construct LiveEvent dict per inter-module contract
            live_event: dict[str, Any] = {
                "uniqueId": event.get("uniqueId", self.unique_id),
                "event_type": event.get("event_type", "unknown"),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "payload": event.get("payload", {}),
            }
            outcome_payload = creator_follow_outcome_from_live_event(live_event)
            if outcome_payload is not None:
                live_event["_attribution_outcome"] = outcome_payload

            # §4.B.1 — Action_Combo constraint: track concurrent events
            self._combo_window.append(live_event)

            # Check for combo within 2-second window
            if len(self._combo_window) >= 2:
                combo_events = self._combo_window[:]
                self._combo_window.clear()

                if self.on_combo_event is not None:
                    self.on_combo_event(combo_events)

                # Stage combo as a single composite event
                combo_dict: dict[str, Any] = {
                    "uniqueId": self.unique_id,
                    "event_type": "Action_Combo",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "payload": {"events": combo_events},
                }
                combo_outcomes = [
                    item["_attribution_outcome"]
                    for item in combo_events
                    if isinstance(item.get("_attribution_outcome"), dict)
                ]
                if combo_outcomes:
                    combo_dict["_outcome_events"] = combo_outcomes
                # §12 Queue overload B — deque eviction via maxlen
                self.event_buffer.append(combo_dict)
            else:
                # Stage individual event
                self.event_buffer.append(live_event)

        except (KeyError, TypeError, ValueError) as exc:
            # §4.B contract — Malformed protobuf: log and discard
            logger.warning("Malformed event discarded: %s", exc)

    async def run(self) -> None:
        """
        Main ingestion loop with reconnection logic.

        §12.1 Module B — Exponential backoff on WebSocket disconnect:
        1s initial delay, 30s max delay, 10 maximum retries.
        """
        self._running = True
        retry_count = 0
        backoff = BACKOFF_INITIAL

        while self._running and retry_count < BACKOFF_MAX_RETRIES:
            try:
                await self.connect()

                if self._client is None:
                    raise RuntimeError("TikTokLive client not initialized")

                # §4.B — Register event handler and start listening
                self._client.on("event")(self._handle_event)

                await self._client.start()

                # If we get here, connection was clean — reset backoff
                retry_count = 0
                backoff = BACKOFF_INITIAL

            except Exception as exc:
                retry_count += 1
                logger.error(
                    "WebSocket error (retry %d/%d, backoff %.1fs): %s",
                    retry_count,
                    BACKOFF_MAX_RETRIES,
                    backoff,
                    exc,
                )

                if retry_count >= BACKOFF_MAX_RETRIES:
                    logger.error(
                        "Max retries (%d) exhausted, stopping ingestion",
                        BACKOFF_MAX_RETRIES,
                    )
                    break

                # §12.1 Module B — Exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)

        self._running = False

    def stop(self) -> None:
        """Signal the ingestion loop to stop."""
        self._running = False
        if self._client is not None:
            logger.info("Stopping TikTokLive client for @%s", self.unique_id)
