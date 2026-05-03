"""Service boundary for cloud telemetry ingestion."""

from __future__ import annotations

from typing import Any

from packages.schemas.cloud import (
    CloudIngestResponse,
    TelemetryPosteriorDeltaBatch,
    TelemetrySegmentBatch,
)
from services.cloud_api.repos.telemetry import (
    PosteriorDeltaApplyError,
    insert_posterior_delta_batch,
    insert_segment_batch,
)
from services.cloud_api.services.transactions import run_in_transaction


class PosteriorDeltaAuthorizationError(RuntimeError):
    pass


class TelemetryIngestService:
    def ingest_segments(
        self,
        batch: TelemetrySegmentBatch,
        *,
        client_id: str,
    ) -> CloudIngestResponse:
        def _write(conn: Any) -> int:
            with conn.cursor() as cur:
                return insert_segment_batch(cur, batch, client_id=client_id)

        inserted = run_in_transaction(_write)
        accepted = len(batch.segments) + len(batch.attribution_events)
        return CloudIngestResponse(accepted_count=accepted, inserted_count=inserted)

    def ingest_posterior_deltas(
        self,
        batch: TelemetryPosteriorDeltaBatch,
        *,
        client_id: str,
    ) -> CloudIngestResponse:
        for delta in batch.deltas:
            if delta.client_id != client_id:
                raise PosteriorDeltaAuthorizationError(
                    "posterior delta client_id does not match authenticated client"
                )

        def _write(conn: Any) -> int:
            with conn.cursor() as cur:
                return insert_posterior_delta_batch(
                    cur,
                    batch.deltas,
                    authenticated_client_id=client_id,
                )

        try:
            inserted = run_in_transaction(_write)
        except PosteriorDeltaApplyError as exc:
            raise PosteriorDeltaAuthorizationError(str(exc)) from exc
        return CloudIngestResponse(accepted_count=len(batch.deltas), inserted_count=inserted)
