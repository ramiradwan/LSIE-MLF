from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from packages.schemas.cloud import ExperimentBundle, ExperimentBundlePayload
from services.desktop_app.state.sqlite_schema import bootstrap_schema

BundleSignatureMode = Literal["hmac-sha256", "ed25519"]


@dataclass(frozen=True)
class BundleVerificationConfig:
    signature_mode: BundleSignatureMode = "hmac-sha256"
    hmac_secret: str | None = None
    ed25519_public_key: bytes | None = None


class ExperimentBundleVerificationError(RuntimeError):
    pass


class ExperimentBundleFetchError(RuntimeError):
    pass


class ExperimentBundleClient:
    def __init__(self, base_url: str, *, timeout_s: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def fetch_bundle(self, *, access_token: str) -> ExperimentBundle:
        try:
            response = httpx.get(
                f"{self._base_url}/v4/experiments/bundle",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            return ExperimentBundle.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ExperimentBundleFetchError("experiment bundle fetch failed") from exc


class ExperimentBundleStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        bootstrap_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def cache_verified_bundle(
        self,
        bundle: ExperimentBundle,
        *,
        config: BundleVerificationConfig | None = None,
        applied_at_utc: datetime | None = None,
    ) -> None:
        verify_bundle(bundle, config=config)
        self.cache_bundle(bundle, applied_at_utc=applied_at_utc)

    def cache_bundle(
        self,
        bundle: ExperimentBundle,
        *,
        applied_at_utc: datetime | None = None,
    ) -> None:
        updated_at = _format_timestamp(applied_at_utc or datetime.now(UTC))
        active_keys: set[tuple[str, str]] = set()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for experiment in bundle.experiments:
                for arm in experiment.arms:
                    active_keys.add((experiment.experiment_id, arm.arm_id))
                    end_dated_at = None if arm.enabled else updated_at
                    self._conn.execute(
                        """
                        INSERT INTO experiments (
                            experiment_id, label, arm, greeting_text, alpha_param,
                            beta_param, enabled, end_dated_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(experiment_id, arm) DO UPDATE SET
                            label = excluded.label,
                            greeting_text = excluded.greeting_text,
                            alpha_param = excluded.alpha_param,
                            beta_param = excluded.beta_param,
                            enabled = excluded.enabled,
                            end_dated_at = excluded.end_dated_at,
                            updated_at = excluded.updated_at
                        """,
                        (
                            experiment.experiment_id,
                            experiment.label,
                            arm.arm_id,
                            arm.greeting_text,
                            arm.posterior_alpha,
                            arm.posterior_beta,
                            1 if arm.enabled else 0,
                            end_dated_at,
                            updated_at,
                        ),
                    )
            rows = self._conn.execute("SELECT experiment_id, arm FROM experiments").fetchall()
            missing = [
                (updated_at, str(row["experiment_id"]), str(row["arm"]))
                for row in rows
                if (str(row["experiment_id"]), str(row["arm"])) not in active_keys
            ]
            disable_rows = [
                (timestamp, timestamp, experiment_id, arm)
                for timestamp, experiment_id, arm in missing
            ]
            self._conn.executemany(
                """
                UPDATE experiments
                SET enabled = 0, end_dated_at = ?, updated_at = ?
                WHERE experiment_id = ? AND arm = ? AND enabled = 1
                """,
                disable_rows,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


def verify_bundle(
    bundle: ExperimentBundle,
    *,
    config: BundleVerificationConfig | None = None,
    now_utc: datetime | None = None,
) -> None:
    effective_config = config or BundleVerificationConfig()
    now = now_utc or datetime.now(UTC)
    if bundle.issued_at_utc > now or bundle.expires_at_utc <= now:
        raise ExperimentBundleVerificationError("experiment bundle is outside its validity window")
    if not bundle.signature:
        raise ExperimentBundleVerificationError("experiment bundle is unsigned")
    if effective_config.signature_mode == "hmac-sha256":
        _verify_hmac_signature(bundle, effective_config)
        return
    if effective_config.ed25519_public_key is None:
        raise ExperimentBundleVerificationError("Ed25519 bundle verification is not configured")
    raise ExperimentBundleVerificationError("Ed25519 bundle verification is not available")


def canonical_bundle_payload(bundle: ExperimentBundle) -> str:
    payload = ExperimentBundlePayload.model_validate(bundle.model_dump(exclude={"signature"}))
    return canonical_payload(payload)


def canonical_payload(payload: ExperimentBundlePayload) -> str:
    return json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def sign_bundle_payload(payload: ExperimentBundlePayload, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        canonical_payload(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_hmac_signature(
    bundle: ExperimentBundle,
    config: BundleVerificationConfig,
) -> None:
    if config.hmac_secret is None:
        raise ExperimentBundleVerificationError("HMAC bundle verification secret is not configured")
    expected = sign_bundle_payload(
        ExperimentBundlePayload.model_validate(bundle.model_dump(exclude={"signature"})),
        config.hmac_secret,
    )
    if not hmac.compare_digest(expected, bundle.signature):
        raise ExperimentBundleVerificationError("experiment bundle signature verification failed")


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "BundleSignatureMode",
    "BundleVerificationConfig",
    "ExperimentBundleClient",
    "ExperimentBundleFetchError",
    "ExperimentBundleStore",
    "ExperimentBundleVerificationError",
    "canonical_bundle_payload",
    "canonical_payload",
    "sign_bundle_payload",
    "verify_bundle",
]
