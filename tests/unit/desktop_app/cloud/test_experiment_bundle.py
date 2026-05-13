from __future__ import annotations

import base64
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
)
from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from services.desktop_app.cloud.experiment_bundle import (
    BundleVerificationConfig,
    ExperimentBundleClient,
    ExperimentBundleFetchError,
    ExperimentBundleStore,
    ExperimentBundleVerificationError,
    canonical_payload,
    encode_ed25519_public_key,
    sign_bundle_payload,
    verify_bundle,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

ISSUED_AT = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
EXPIRES_AT = datetime(2036, 5, 3, 12, 0, tzinfo=UTC)
APPLIED_AT = datetime(2036, 5, 2, 13, 0, tzinfo=UTC)
SECRET = "bundle-secret"


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text=text,
        ),
        expected_stimulus_rule="Deliver the spoken greeting to the creator",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


def _payload(*, arms: list[ExperimentBundleArm] | None = None) -> ExperimentBundlePayload:
    return ExperimentBundlePayload(
        bundle_id="bundle-a",
        issued_at_utc=ISSUED_AT,
        expires_at_utc=EXPIRES_AT,
        policy_version="v4.0",
        experiments=[
            ExperimentBundleExperiment(
                experiment_id="experiment-a",
                label="Experiment A",
                arms=arms
                or [
                    ExperimentBundleArm(
                        arm_id="arm-a",
                        stimulus_definition=_stimulus_definition("Hello A"),
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


def _ed25519_signed_bundle(
    payload: ExperimentBundlePayload | None = None,
    private_key: ECC.EccKey | None = None,
) -> tuple[ExperimentBundle, ECC.EccKey]:
    effective_payload = payload or _payload()
    effective_private_key = private_key or ECC.generate(curve="Ed25519")
    signature = eddsa.new(effective_private_key, "rfc8032").sign(
        canonical_payload(effective_payload).encode("utf-8")
    )
    return (
        ExperimentBundle(
            **effective_payload.model_dump(),
            signature=_urlsafe_b64encode(signature),
        ),
        effective_private_key,
    )


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def test_verify_bundle_accepts_hmac_signed_canonical_json() -> None:
    verify_bundle(
        _signed_bundle(),
        config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
        now_utc=ISSUED_AT + timedelta(hours=1),
    )


def test_verify_bundle_accepts_ed25519_signed_canonical_json() -> None:
    bundle, private_key = _ed25519_signed_bundle()

    verify_bundle(
        bundle,
        config=BundleVerificationConfig(
            ed25519_public_key=encode_ed25519_public_key(private_key.public_key()).encode("utf-8")
        ),
        now_utc=ISSUED_AT + timedelta(hours=1),
    )


def test_verify_bundle_requires_explicit_ed25519_public_key() -> None:
    bundle, _private_key = _ed25519_signed_bundle()

    with pytest.raises(ExperimentBundleVerificationError, match="Ed25519"):
        verify_bundle(
            bundle,
            config=BundleVerificationConfig(),
            now_utc=ISSUED_AT + timedelta(hours=1),
        )


def test_verify_bundle_allows_small_not_before_clock_skew() -> None:
    future_issued = _signed_bundle(
        _payload().model_copy(
            update={
                "issued_at_utc": ISSUED_AT + timedelta(seconds=2),
                "expires_at_utc": ISSUED_AT + timedelta(hours=24),
            }
        )
    )

    verify_bundle(
        future_issued,
        config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
        now_utc=ISSUED_AT,
    )


def test_verify_bundle_rejects_expired_bundle() -> None:
    expired = _signed_bundle(
        _payload().model_copy(update={"expires_at_utc": ISSUED_AT + timedelta(minutes=5)})
    )

    with pytest.raises(ExperimentBundleVerificationError, match="validity"):
        verify_bundle(
            expired,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            now_utc=ISSUED_AT + timedelta(minutes=10),
        )


def test_verify_bundle_rejects_large_not_before_clock_skew() -> None:
    future_issued = _signed_bundle(
        _payload().model_copy(
            update={
                "issued_at_utc": ISSUED_AT + timedelta(seconds=6),
                "expires_at_utc": ISSUED_AT + timedelta(hours=24),
            }
        )
    )

    with pytest.raises(ExperimentBundleVerificationError, match="validity"):
        verify_bundle(
            future_issued,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            now_utc=ISSUED_AT,
        )


def test_verify_bundle_rejects_bad_or_missing_signatures() -> None:
    bundle = _signed_bundle()
    tampered = bundle.model_copy(update={"signature": "0" * 64})
    unsigned = bundle.model_copy(update={"signature": " "})

    with pytest.raises(ExperimentBundleVerificationError):
        verify_bundle(
            tampered,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            now_utc=ISSUED_AT + timedelta(hours=1),
        )
    with pytest.raises(ExperimentBundleVerificationError):
        verify_bundle(
            unsigned,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            now_utc=ISSUED_AT + timedelta(hours=1),
        )


def test_verify_bundle_rejects_bad_ed25519_signature() -> None:
    bundle, private_key = _ed25519_signed_bundle()

    with pytest.raises(ExperimentBundleVerificationError, match="signature"):
        verify_bundle(
            bundle.model_copy(update={"signature": "0" * 64}),
            config=BundleVerificationConfig(
                ed25519_public_key=encode_ed25519_public_key(private_key.public_key()).encode(
                    "utf-8"
                )
            ),
            now_utc=ISSUED_AT + timedelta(hours=1),
        )


def test_bundle_client_preserves_http_status_for_bounded_operator_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del headers, timeout
        request = httpx.Request("GET", url)
        return httpx.Response(429, json={"detail": "rate limited"}, request=request)

    monkeypatch.setattr("services.desktop_app.cloud.experiment_bundle.httpx.get", fake_get)
    client = ExperimentBundleClient("https://cloud.example.test/")

    with pytest.raises(ExperimentBundleFetchError) as exc_info:
        client.fetch_bundle(access_token="access-a")

    assert exc_info.value.status_code == 429


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


def test_cache_verified_bundle_preserves_local_posterior_for_existing_arm(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "experiment-a",
            "Experiment A",
            "arm-a",
            _stimulus_definition("Old A").model_dump_json(),
            11.0,
            12.0,
            1,
        ),
    )
    conn.close()
    bundle = _signed_bundle(
        _payload(
            arms=[
                ExperimentBundleArm(
                    arm_id="arm-a",
                    stimulus_definition=_stimulus_definition("Updated A"),
                    posterior_alpha=4.0,
                    posterior_beta=5.0,
                    selection_count=8,
                ),
            ]
        )
    )
    store = ExperimentBundleStore(db_path)
    try:
        store.cache_verified_bundle(
            bundle,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    row = conn.execute(
        """
        SELECT stimulus_definition, alpha_param, beta_param, enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = 'experiment-a' AND arm = 'arm-a'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0]) == _stimulus_definition("Updated A").model_dump(mode="json")
    assert row[1:] == (11.0, 12.0, 1, None, "2036-05-02T13:00:00Z")


def test_cache_verified_bundle_seeds_new_arms_from_bundle(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    bundle = _signed_bundle(
        _payload(
            arms=[
                ExperimentBundleArm(
                    arm_id="arm-b",
                    stimulus_definition=_stimulus_definition("Hello B"),
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
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    row = conn.execute(
        """
        SELECT stimulus_definition, alpha_param, beta_param, enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = 'experiment-a' AND arm = 'arm-b'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0]) == _stimulus_definition("Hello B").model_dump(mode="json")
    assert row[1:] == (6.0, 7.0, 1, None, "2036-05-02T13:00:00Z")


def test_cache_verified_bundle_disables_missing_arms_without_deleting_them(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "experiment-a",
            "Experiment A",
            "missing-arm",
            _stimulus_definition("Old").model_dump_json(),
            1.0,
            1.0,
            1,
        ),
    )
    conn.close()
    store = ExperimentBundleStore(db_path)
    try:
        store.cache_verified_bundle(
            _signed_bundle(),
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    row = conn.execute(
        """
        SELECT stimulus_definition, alpha_param, beta_param, enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = 'experiment-a' AND arm = 'missing-arm'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0]) == _stimulus_definition("Old").model_dump(mode="json")
    assert row[1:] == (
        1.0,
        1.0,
        0,
        "2036-05-02T13:00:00Z",
        "2036-05-02T13:00:00Z",
    )


def test_preview_verified_bundle_reports_changes_without_mutating_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn, seed_experiments=False)
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "experiment-a",
            "Experiment A",
            "arm-a",
            _stimulus_definition("Old A").model_dump_json(),
            11.0,
            12.0,
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "experiment-a",
            "Experiment A",
            "missing-arm",
            _stimulus_definition("Old").model_dump_json(),
            1.0,
            1.0,
            1,
        ),
    )
    before_rows = conn.execute(
        """
        SELECT arm, stimulus_definition, alpha_param, beta_param, enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = 'experiment-a'
        ORDER BY arm
        """
    ).fetchall()
    conn.close()
    bundle = _signed_bundle(
        _payload(
            arms=[
                ExperimentBundleArm(
                    arm_id="arm-a",
                    stimulus_definition=_stimulus_definition("Updated A"),
                    posterior_alpha=4.0,
                    posterior_beta=5.0,
                    selection_count=8,
                ),
                ExperimentBundleArm(
                    arm_id="arm-b",
                    stimulus_definition=_stimulus_definition("Hello B"),
                    posterior_alpha=6.0,
                    posterior_beta=7.0,
                    selection_count=9,
                ),
            ]
        )
    )
    store = ExperimentBundleStore(db_path)
    try:
        preview = store.preview_verified_bundle(
            bundle,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            checked_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    assert preview.added_count == 1
    assert preview.updated_count == 1
    assert preview.disabled_count >= 1
    assert preview.existing_preserved_count == 1
    assert [change.action for change in preview.changes[:2]] == ["update", "add"]
    assert any(
        change.action == "disable" and change.arm_id == "missing-arm" for change in preview.changes
    )

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    rows = conn.execute(
        """
        SELECT arm, stimulus_definition, alpha_param, beta_param, enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = 'experiment-a'
        ORDER BY arm
        """
    ).fetchall()
    conn.close()

    assert rows == before_rows


def test_preview_token_allows_freshly_signed_equivalent_bundle(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    first = _signed_bundle()
    second = _signed_bundle(
        _payload().model_copy(
            update={
                "bundle_id": "bundle-b",
                "issued_at_utc": ISSUED_AT + timedelta(minutes=1),
                "expires_at_utc": EXPIRES_AT + timedelta(minutes=1),
            }
        )
    )
    store = ExperimentBundleStore(db_path)
    try:
        preview_token = store.preview_token(first)
        store.cache_verified_bundle(
            second,
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
            expected_preview_token=preview_token,
        )
        row = store._conn.execute(
            """
            SELECT stimulus_definition, alpha_param, beta_param
            FROM experiments
            WHERE experiment_id = ? AND arm = ?
            """,
            ("experiment-a", "arm-a"),
        ).fetchone()
    finally:
        store.close()

    assert first.signature != second.signature
    assert row is not None
    assert json.loads(str(row[0])) == _stimulus_definition("Hello A").model_dump(mode="json")
    assert tuple(row[1:]) == (2.0, 3.0)


def test_cache_verified_bundle_leaves_encounter_log_untouched(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        """
        INSERT INTO sessions (session_id, stream_url, started_at)
        VALUES (?, ?, ?)
        """,
        ("session-a", "https://example.test/stream", "2036-05-02T12:00:00Z"),
    )
    conn.execute(
        """
        INSERT INTO encounter_log (
            session_id, segment_id, experiment_id, arm, timestamp_utc, gated_reward,
            p90_intensity, semantic_gate, n_frames_in_window
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "session-a",
            "segment-a",
            "experiment-a",
            "arm-a",
            "2036-05-02T12:00:00Z",
            0.75,
            0.75,
            1,
            60,
        ),
    )
    conn.close()
    store = ExperimentBundleStore(db_path)
    try:
        store.cache_verified_bundle(
            _signed_bundle(),
            config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            applied_at_utc=APPLIED_AT,
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    row = conn.execute(
        """
        SELECT session_id, segment_id, experiment_id, arm, gated_reward
        FROM encounter_log
        """
    ).fetchone()
    conn.close()

    assert row == ("session-a", "segment-a", "experiment-a", "arm-a", 0.75)


def test_cache_verified_bundle_rejects_unsigned_bundle_without_writing(tmp_path: Path) -> None:
    bundle = _signed_bundle().model_copy(update={"signature": "0" * 64})
    store = ExperimentBundleStore(tmp_path / "desktop.sqlite")
    try:
        with pytest.raises(ExperimentBundleVerificationError):
            store.cache_verified_bundle(
                bundle,
                config=BundleVerificationConfig(signature_mode="hmac-sha256", hmac_secret=SECRET),
            )
        row = store._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
            ("experiment-a",),
        ).fetchone()
    finally:
        store.close()

    assert row is not None
    assert row[0] == 0
