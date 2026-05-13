from __future__ import annotations

import base64
import binascii
import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
)
from packages.schemas.evaluation import StimulusDefinition
from services.cloud_api.repos.experiments import fetch_active_experiment_rows
from services.cloud_api.services.transactions import run_in_transaction

LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY = "LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY"
Ed25519PrivateKey = ECC.EccKey


class ExperimentBundleUnavailableError(RuntimeError):
    pass


class ExperimentBundleSigningError(RuntimeError):
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
        stimulus_definition=StimulusDefinition.model_validate(row["stimulus_definition"]),
        posterior_alpha=float(cast(float | int | str, row["posterior_alpha"])),
        posterior_beta=float(cast(float | int | str, row["posterior_beta"])),
        selection_count=int(cast(int | str, row.get("selection_count") or 0)),
        enabled=bool(row["enabled"]),
    )


def _sign_payload(payload: ExperimentBundlePayload) -> str:
    private_key = _load_private_key_from_env()
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _urlsafe_b64encode(eddsa.new(private_key, "rfc8032").sign(canonical))


def _load_private_key_from_env() -> Ed25519PrivateKey:
    raw_key = os.environ.get(LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY, "").strip()
    if not raw_key:
        raise ExperimentBundleSigningError(
            f"{LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY} is not configured"
        )
    try:
        if raw_key.startswith("-----BEGIN"):
            key = ECC.import_key(raw_key)
        else:
            key = ECC.import_key(_urlsafe_b64decode(raw_key))
    except (binascii.Error, ValueError) as exc:
        raise ExperimentBundleSigningError("Ed25519 bundle private key is invalid") from exc
    if key.has_private() and "Ed25519" in str(key.curve):
        return key
    raise ExperimentBundleSigningError("Ed25519 bundle private key is invalid")


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")
