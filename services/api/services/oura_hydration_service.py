from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from packages.schemas.data_tiers import DataTier, mark_data_tier
from packages.schemas.physiology import PhysiologicalChunkEvent
from services.api.clients.oura_client import OuraAPIClient, create_oura_client_from_env

logger = logging.getLogger(__name__)

_PHYSIO_HYDRATE_QUEUE = "physio:hydrate"
_PHYSIO_EVENTS_QUEUE = "physio:events"
_ALLOWED_SOURCE_KINDS = {"ibi", "session"}
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "lsie-mlf:oura-hydration")
_DEFAULT_IDLE_SLEEP_S = 1.0


class OuraHydrationService:
    """Hydrates queued Oura notifications into validated chunk events."""

    def __init__(
        self,
        *,
        redis_client: Any,
        oura_client: OuraAPIClient | None = None,
        oura_client_factory: Callable[[], OuraAPIClient | None] = create_oura_client_from_env,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] | None = None,
        hydrate_queue: str = _PHYSIO_HYDRATE_QUEUE,
        events_queue: str = _PHYSIO_EVENTS_QUEUE,
    ) -> None:
        self._redis = redis_client
        self._oura_client = oura_client
        self._oura_client_factory = oura_client_factory
        self._clock = clock
        self._sleep = sleep
        self._hydrate_queue = hydrate_queue
        self._events_queue = events_queue

    def drain_once(self) -> int:
        client = self._get_oura_client()
        if client is None:
            logger.info("Oura hydration disabled; OURA_CLIENT_ID not configured")
            return 0

        drained = 0
        while True:
            raw = self._pop_notification()
            if raw is None:
                break
            drained += 1
            try:
                self._process_notification(raw, client)
            except Exception:
                logger.warning(
                    "Unexpected Oura hydration failure; notification skipped",
                    exc_info=True,
                )
        return drained

    def run_forever(self, *, idle_sleep_s: float = _DEFAULT_IDLE_SLEEP_S) -> None:
        while True:
            try:
                drained = self.drain_once()
            except Exception:
                logger.warning("Oura hydration loop iteration failed", exc_info=True)
                drained = 0
            if drained == 0 and self._sleep is not None:
                self._sleep(idle_sleep_s)

    def _get_oura_client(self) -> OuraAPIClient | None:
        if self._oura_client is not None:
            return self._oura_client
        self._oura_client = self._oura_client_factory()
        return self._oura_client

    def _pop_notification(self) -> str | None:
        raw = mark_data_tier(
            self._redis.lpop(self._hydrate_queue),
            DataTier.TRANSIENT,
            spec_ref="§5.2.1",
            purpose="Oura hydration notification while in Redis transit",
        )  # §5.2.1 Transient Data
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def _process_notification(self, raw_notification: str, oura_client: OuraAPIClient) -> None:
        notification = self._parse_notification(raw_notification)
        if notification is None:
            return

        source_kind = self._source_kind_for_notification(notification)
        if source_kind is None:
            logger.warning(
                "Unsupported Oura notification skipped: data_type=%s",
                notification.get("data_type"),
            )
            return

        resource = self._fetch_resource(notification, oura_client, source_kind)
        if resource is None:
            return

        for event in self._normalize_events(notification, resource, source_kind=source_kind):
            try:
                validated = PhysiologicalChunkEvent.model_validate(event)
            except ValidationError:
                logger.warning("Malformed Oura provider record skipped", exc_info=True)
                continue
            event_json = mark_data_tier(
                validated.model_dump_json(),
                DataTier.TRANSIENT,
                spec_ref="§5.2.1",
                purpose="Normalized PhysiologicalChunkEvent while in Redis transit",
            )  # §5.2.1 Transient Data
            self._redis.rpush(
                self._events_queue,
                event_json,
            )

    def _parse_notification(self, raw_notification: str) -> dict[str, Any] | None:
        try:
            payload = mark_data_tier(
                json.loads(raw_notification),
                DataTier.TRANSIENT,
                spec_ref="§5.2.1",
                purpose="Raw Oura hydration notification decoded from Redis transit",
            )  # §5.2.1 Transient Data
        except json.JSONDecodeError:
            logger.warning("Malformed Oura hydration notification skipped")
            return None
        if not isinstance(payload, dict):
            logger.warning("Malformed Oura hydration notification skipped")
            return None

        subject_role = payload.get("subject_role")
        start_datetime = payload.get("start_datetime")
        end_datetime = payload.get("end_datetime")
        data_type = payload.get("data_type")
        unique_id = payload.get("unique_id")
        if subject_role not in ("streamer", "operator"):
            logger.warning("Malformed Oura hydration notification skipped: invalid subject_role")
            return None
        if not isinstance(data_type, str) or not data_type.strip():
            logger.warning("Malformed Oura hydration notification skipped: invalid data_type")
            return None
        if not isinstance(start_datetime, str) or not isinstance(end_datetime, str):
            logger.warning("Malformed Oura hydration notification skipped: invalid window")
            return None
        if not isinstance(unique_id, str) or not unique_id.strip():
            logger.warning("Malformed Oura hydration notification skipped: missing unique_id")
            return None
        return payload

    def _source_kind_for_notification(self, notification: dict[str, Any]) -> str | None:
        data_type = str(notification["data_type"]).strip().lower()
        if data_type in {"heartrate", "heart_rate", "ibi"}:
            return "ibi"
        if data_type in {"session", "workout", "sleep"}:
            return "session"
        return None

    def _fetch_resource(
        self,
        notification: dict[str, Any],
        oura_client: OuraAPIClient,
        source_kind: str,
    ) -> dict[str, Any] | None:
        try:
            resource = mark_data_tier(
                oura_client.get_json(
                    self._resource_path_for_source_kind(source_kind),
                    query={
                        "start_datetime": notification["start_datetime"],
                        "end_datetime": notification["end_datetime"],
                    },
                ),
                DataTier.TRANSIENT,
                spec_ref="§5.2.1",
                purpose="Hydrated raw Oura provider response before chunk normalization",
            )  # §5.2.1 Transient Data
            return resource
        except Exception:
            logger.warning("Oura resource fetch failed; notification skipped", exc_info=True)
            return None

    def _normalize_events(
        self,
        notification: dict[str, Any],
        resource: dict[str, Any],
        *,
        source_kind: str,
    ) -> list[dict[str, Any]]:
        data = resource.get("data")
        if not isinstance(data, list):
            logger.warning("Malformed Oura provider record skipped")
            return []

        events: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning("Malformed Oura provider record skipped")
                continue
            normalized = self._normalize_item(notification, item, source_kind=source_kind)
            if normalized is not None:
                events.append(normalized)
        return events

    def _normalize_item(
        self,
        notification: dict[str, Any],
        item: dict[str, Any],
        *,
        source_kind: str,
    ) -> dict[str, Any] | None:
        if source_kind not in _ALLOWED_SOURCE_KINDS:
            logger.warning("Malformed Oura provider record skipped: invalid source_kind")
            return None

        start = self._parse_datetime(item.get("timestamp") or item.get("start_datetime"))
        end = self._parse_datetime(item.get("end_datetime"))
        if start is None:
            logger.warning("Malformed Oura provider record skipped")
            return None
        if end is None:
            end = start

        payload = self._build_payload(item, source_kind=source_kind, start=start, end=end)
        if payload is None:
            logger.warning("Malformed Oura provider record skipped")
            return None

        resource_id = item.get("id") or item.get("session_id") or item.get("day") or "window"
        unique_id = uuid.uuid5(
            _UUID_NAMESPACE,
            "|".join(
                [
                    str(notification["subject_role"]),
                    source_kind,
                    str(resource_id),
                    start.isoformat(),
                    end.isoformat(),
                ]
            ),
        )

        return {
            "unique_id": unique_id,
            "event_type": "physiological_chunk",
            "provider": "oura",
            "subject_role": notification["subject_role"],
            "source_kind": source_kind,
            "window_start_utc": start,
            "window_end_utc": end,
            "ingest_timestamp_utc": self._clock(),
            "payload": payload,
        }

    def _build_payload(
        self,
        item: dict[str, Any],
        *,
        source_kind: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any] | None:
        if source_kind == "ibi":
            ibi_items = item.get("ibi_ms") or item.get("ibi_ms_items")
            if not isinstance(ibi_items, list) or not ibi_items:
                return None
            heart_rate = item.get("heart_rate_bpm") or item.get("heart_rate_items_bpm")
            heart_rate_items = heart_rate if isinstance(heart_rate, list) and heart_rate else None
            sample_interval_s = self._coerce_positive_int(
                item.get("sample_interval_s"),
                default=1,
            )
            valid_sample_count = self._coerce_non_negative_int(
                item.get("valid_sample_count"), default=len(ibi_items)
            )
            return {
                "sample_interval_s": sample_interval_s,
                "valid_sample_count": valid_sample_count,
                "expected_sample_count": self._expected_sample_count(
                    item,
                    start=start,
                    end=end,
                    sample_interval_s=sample_interval_s,
                    default=len(ibi_items),
                ),
                "derivation_method": "provider",
                "ibi_ms_items": ibi_items,
                "heart_rate_items_bpm": heart_rate_items,
                "rmssd_items_ms": None,
                "motion_items": None,
            }
        if source_kind == "session":
            rmssd_items = item.get("rmssd_ms") or item.get("rmssd_items_ms")
            heart_rate = item.get("heart_rate_bpm") or item.get("heart_rate_items_bpm")
            motion_items = item.get("motion_items")
            if not isinstance(rmssd_items, list) or not rmssd_items:
                return None
            sample_interval_s = self._coerce_positive_int(
                item.get("sample_interval_s"),
                default=300,
            )
            valid_sample_count = self._coerce_non_negative_int(
                item.get("valid_sample_count"), default=len(rmssd_items)
            )
            return {
                "sample_interval_s": sample_interval_s,
                "valid_sample_count": valid_sample_count,
                "expected_sample_count": self._expected_sample_count(
                    item,
                    start=start,
                    end=end,
                    sample_interval_s=sample_interval_s,
                    default=len(rmssd_items),
                ),
                "derivation_method": "provider",
                "ibi_ms_items": None,
                "heart_rate_items_bpm": heart_rate if isinstance(heart_rate, list) else None,
                "rmssd_items_ms": rmssd_items,
                "motion_items": motion_items if isinstance(motion_items, list) else None,
            }
        return None

    def _resource_path_for_source_kind(self, source_kind: str) -> str:
        if source_kind == "ibi":
            return "/v2/usercollection/heartrate"
        return "/v2/usercollection/session"

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid integers")
        coerced = int(value)
        if coerced <= 0:
            raise ValueError("Expected positive integer")
        return coerced

    @staticmethod
    def _coerce_non_negative_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid integers")
        coerced = int(value)
        if coerced < 0:
            raise ValueError("Expected non-negative integer")
        return coerced

    @staticmethod
    def _expected_sample_count(
        item: dict[str, Any],
        *,
        start: datetime,
        end: datetime,
        sample_interval_s: int,
        default: int,
    ) -> int:
        explicit = item.get("expected_sample_count")
        if explicit is not None:
            return OuraHydrationService._coerce_non_negative_int(explicit, default=default)
        window_seconds = max(0.0, (end - start).total_seconds())
        if window_seconds <= 0:
            return default
        derived = int(window_seconds // sample_interval_s)
        return max(default, derived)
