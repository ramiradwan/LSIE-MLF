"""Service boundary for cloud telemetry ingestion."""

from __future__ import annotations

from typing import Any

from packages.schemas.cloud import (
    CloudIngestResponse,
    TelemetryPosteriorDeltaBatch,
    TelemetrySegmentBatch,
)
from services.cloud_api.repos.telemetry import insert_posterior_delta_batch, insert_segment_batch
from services.cloud_api.services.transactions import run_in_transaction


class TelemetryIngestService:
    def ingest_segments(self, batch: TelemetrySegmentBatch) -> CloudIngestResponse:
        def _write(conn: Any) -> int:
            with conn.cursor() as cur:
                return insert_segment_batch(cur, batch)

        inserted = run_in_transaction(_write)
        return CloudIngestResponse(accepted_count=len(batch.segments), inserted_count=inserted)

    def ingest_posterior_deltas(self, batch: TelemetryPosteriorDeltaBatch) -> CloudIngestResponse:
        def _write(conn: Any) -> int:
            with conn.cursor() as cur:
                return insert_posterior_delta_batch(cur, batch.deltas)

        inserted = run_in_transaction(_write)
        return CloudIngestResponse(accepted_count=len(batch.deltas), inserted_count=inserted)
