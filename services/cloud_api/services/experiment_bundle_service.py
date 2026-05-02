from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
)
from services.cloud_api.repos.experiments import fetch_active_experiment_rows
from services.cloud_api.services.transactions import run_in_transaction


class ExperimentBundleUnavailableError(RuntimeError):
    pass


class ExperimentBundleService:
    def build_bundle(self) -> ExperimentBundle:
        def _read(conn: Any) -> list[dict[str, object]]:
            with conn.cursor() as cur:
                return fetch_active_experiment_rows(cur)

        rows = run_in_transaction(_read)
        if not rows:
            raise ExperimentBundleUnavailableError("no active experiment arms")

        issued_at = datetime.now(UTC)
        payload = ExperimentBundlePayload(
            bundle_id=f"bundle-{issued_at.strftime('%Y%m%d%H%M%S')}",
            issued_at_utc=issued_at,
            expires_at_utc=issued_at + timedelta(hours=24),
            policy_version=os.environ.get("LSIE_CLOUD_POLICY_VERSION", "v4.0"),
            experiments=_rows_to_experiments(rows),
        )
        signature = _sign_payload(payload)
        return ExperimentBundle(**payload.model_dump(), signature=signature)


def _rows_to_experiments(rows: list[dict[str, object]]) -> list[ExperimentBundleExperiment]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        experiment_id = str(row["experiment_id"])
        grouped[experiment_id].append(row)
        label = row.get("label")
        labels[experiment_id] = str(label) if label else experiment_id

    experiments: list[ExperimentBundleExperiment] = []
    for experiment_id, arm_rows in grouped.items():
        arms = [_row_to_arm(row) for row in arm_rows]
        experiments.append(
            ExperimentBundleExperiment(
                experiment_id=experiment_id,
                label=labels[experiment_id],
                arms=arms,
            )
        )
    return experiments


def _row_to_arm(row: dict[str, object]) -> ExperimentBundleArm:
    arm_id = str(row["arm_id"])
    return ExperimentBundleArm(
        arm_id=arm_id,
        greeting_text=str(row.get("greeting_text") or arm_id),
        posterior_alpha=float(cast(float | int | str, row["posterior_alpha"])),
        posterior_beta=float(cast(float | int | str, row["posterior_beta"])),
        selection_count=int(cast(int | str, row.get("selection_count") or 0)),
        enabled=bool(row["enabled"]),
    )


def _sign_payload(payload: ExperimentBundlePayload) -> str:
    secret = os.environ.get("LSIE_CLOUD_BUNDLE_SIGNING_SECRET", "dev-cloud-bundle-secret")
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
