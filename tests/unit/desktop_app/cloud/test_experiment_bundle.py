from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
)
from services.desktop_app.cloud.experiment_bundle import (
    BundleVerificationConfig,
    ExperimentBundleClient,
    ExperimentBundleStore,
    ExperimentBundleVerificationError,
    sign_bundle_payload,
    verify_bundle,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

ISSUED_AT = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
APPLIED_AT = datetime(2026, 5, 2, 13, 0, tzinfo=UTC)
SECRET = "bundle-secret"


def _payload(*, arms: list[ExperimentBundleArm] | None = None) -> ExperimentBundlePayload:
    return ExperimentBundlePayload(
        bundle_id="bundle-a",
        issued_at_utc=ISSUED_AT,
        expires_at_utc=ISSUED_AT + timedelta(hours=24),
        policy_version="v4.0",
        experiments=[
            ExperimentBundleExperiment(
                experiment_id="experiment-a",
                label="Experiment A",
                arms=arms
                or [
                    ExperimentBundleArm(
                        arm_id="arm-a",
                        greeting_text="Hello A",
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                        selection_count=5,
                    )
                ],
            )
        ],
    )


def _signed_bundle(payload: ExperimentBundlePayload | None = None) -> ExperimentBundle:
    effective_payload = payload or _payload()
    return ExperimentBundle(
        **effective_payload.model_dump(),
        signature=sign_bundle_payload(effective_payload, SECRET),
    )


def test_verify_bundle_accepts_hmac_signed_canonical_json() -> None:
    verify_bundle(_signed_bundle(), config=BundleVerificationConfig(hmac_secret=SECRET))


def test_verify_bundle_requires_explicit_hmac_secret() -> None:
    with pytest.raises(ExperimentBundleVerificationError, match="secret"):
        verify_bundle(_signed_bundle(), config=BundleVerificationConfig())


def test_verify_bundle_rejects_expired_bundle() -> None:
    expired = _signed_bundle(
        _payload().model_copy(update={"expires_at_utc": ISSUED_AT + timedelta(minutes=5)})
    )

    with pytest.raises(ExperimentBundleVerificationError, match="validity"):
        verify_bundle(
            expired,
            config=BundleVerificationConfig(hmac_secret=SECRET),
            now_utc=ISSUED_AT + timedelta(minutes=10),
        )


def test_verify_bundle_rejects_bad_or_missing_signatures() -> None:
    bundle = _signed_bundle()
    tampered = bundle.model_copy(update={"signature": "0" * 64})
    unsigned = bundle.model_copy(update={"signature": " "})

    with pytest.raises(ExperimentBundleVerificationError):
        verify_bundle(tampered, config=BundleVerificationConfig(hmac_secret=SECRET))
    with pytest.raises(ExperimentBundleVerificationError):
        verify_bundle(unsigned, config=BundleVerificationConfig(hmac_secret=SECRET))


def test_verify_bundle_fails_closed_for_ed25519_mode() -> None:
    with pytest.raises(ExperimentBundleVerificationError, match="Ed25519"):
        verify_bundle(
            _signed_bundle(),
            config=BundleVerificationConfig(signature_mode="ed25519"),
        )


def test_bundle_client_fetches_and_validates_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _signed_bundle()
    seen_headers: list[str] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        assert url == "https://cloud.example.test/v4/experiments/bundle"
        assert timeout == 15.0
        seen_headers.append(headers["Authorization"])
        request = httpx.Request("GET", url)
        return httpx.Response(200, json=bundle.model_dump(mode="json"), request=request)

    monkeypatch.setattr("services.desktop_app.cloud.experiment_bundle.httpx.get", fake_get)
    client = ExperimentBundleClient("https://cloud.example.test/")

    fetched = client.fetch_bundle(access_token="access-a")

    assert fetched.bundle_id == bundle.bundle_id
    assert seen_headers == ["Bearer access-a"]


def test_cache_verified_bundle_upserts_arms_and_disables_missing_arms(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, greeting_text, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("experiment-a", "Experiment A", "missing-arm", "Old", 1.0, 1.0, 1),
    )
    conn.close()
    bundle = _signed_bundle(
        _payload(
            arms=[
                ExperimentBundleArm(
                    arm_id="arm-a",
                    greeting_text="Updated A",
                    posterior_alpha=4.0,
                    posterior_beta=5.0,
                    selection_count=8,
                ),
                ExperimentBundleArm(
                    arm_id="arm-b",
                    greeting_text="Hello B",
                    posterior_alpha=6.0,
                    posterior_beta=7.0,
                    selection_count=9,
                ),
            ]
        )
    )
    store = ExperimentBundleStore(db_path)
    try:
        store.cache_verified_bundle(
            bundle,
            config=BundleVerificationConfig(hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    rows = {
        row[0]: row[1:]
        for row in conn.execute(
            """
            SELECT arm, greeting_text, alpha_param, beta_param, enabled, end_dated_at, updated_at
            FROM experiments
            WHERE experiment_id = 'experiment-a'
            ORDER BY arm
            """
        ).fetchall()
    }
    conn.close()

    assert rows["arm-a"] == ("Updated A", 4.0, 5.0, 1, None, "2026-05-02T13:00:00Z")
    assert rows["arm-b"] == ("Hello B", 6.0, 7.0, 1, None, "2026-05-02T13:00:00Z")
    assert rows["missing-arm"] == (
        "Old",
        1.0,
        1.0,
        0,
        "2026-05-02T13:00:00Z",
        "2026-05-02T13:00:00Z",
    )


def test_cache_verified_bundle_rejects_unsigned_bundle_without_writing(tmp_path: Path) -> None:
    bundle = _signed_bundle().model_copy(update={"signature": "0" * 64})
    store = ExperimentBundleStore(tmp_path / "desktop.sqlite")
    try:
        with pytest.raises(ExperimentBundleVerificationError):
            store.cache_verified_bundle(bundle, config=BundleVerificationConfig(hmac_secret=SECRET))
        row = store._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
            ("experiment-a",),
        ).fetchone()
    finally:
        store.close()

    assert row is not None
    assert row[0] == 0
