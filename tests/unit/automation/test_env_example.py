from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"

_REQUIRED_VARIABLES = {
    "TRUSTED_SPEC_SIGNERS",
    "LSIE_STATE_DIR",
    "LSIE_CAPTURE_DIR",
    "LSIE_ADB_PATH",
    "LSIE_SCRCPY_PATH",
    "LSIE_FFMPEG_PATH",
    "LSIE_API_PORT",
    "LSIE_API_URL",
    "LSIE_DEV_FORCE_CPU_SPEECH",
    "LSIE_CLOUD_BASE_URL",
    "LSIE_CLOUD_CLIENT_ID",
    "LSIE_CLOUD_SYNC_INTERVAL_S",
    "LSIE_CLOUD_SYNC_BATCH_SIZE",
    "LSIE_CLOUD_SYNC_TIMEOUT_S",
    "LSIE_CLOUD_BUNDLE_ED25519_PUBLIC_KEY",
    "CLOUD_POSTGRES_HOST",
    "CLOUD_POSTGRES_PORT",
    "CLOUD_POSTGRES_USER",
    "CLOUD_POSTGRES_PASSWORD",
    "CLOUD_POSTGRES_DB",
    "LSIE_CLOUD_TOKEN_SIGNING_SECRET",
    "LSIE_CLOUD_ALLOWED_CLIENT_IDS",
    "LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY",
    "LSIE_CLOUD_POLICY_VERSION",
    "SEMANTIC_DEVICE_MODE",
    "SEMANTIC_GRAY_BAND_FALLBACK_ENABLED",
    "SEMANTIC_SHADOW_MODE_ENABLED",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "OURA_CLIENT_ID",
    "OURA_CLIENT_SECRET",
    "OURA_WEBHOOK_SECRET",
    "OURA_TOKEN_FILE",
    "OURA_STATE_DIR",
}

_LEGACY_OR_CONFUSING_VARIABLES = {
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "REDIS_URL",
    "PHYSIO_BUFFER_RETENTION_S",
    "PHYSIO_DERIVE_WINDOW_S",
    "PHYSIO_VALIDITY_MIN",
    "PHYSIO_STALENESS_THRESHOLD_S",
    "EULERSTREAM_SIGN_URL",
    "VAULT_SHRED_INTERVAL_HOURS",
    "DRIFT_POLL_INTERVAL_SECONDS",
    "MAX_TOLERATED_DRIFT_MS",
    "AUTO_STIMULUS_DELAY_S",
}


def _env_assignments() -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        assert separator == "=", stripped
        assignments[key] = value
    return assignments


def test_env_example_documents_current_desktop_and_cloud_variables() -> None:
    assignments = _env_assignments()

    assert _REQUIRED_VARIABLES.issubset(assignments)


def test_env_example_drops_legacy_runtime_variables() -> None:
    assignments = _env_assignments()

    assert _LEGACY_OR_CONFUSING_VARIABLES.isdisjoint(assignments)


def test_cloud_base_url_example_is_unversioned_origin() -> None:
    assignments = _env_assignments()

    assert assignments["LSIE_CLOUD_BASE_URL"] == "https://cloud.example.test"
    assert not assignments["LSIE_CLOUD_BASE_URL"].rstrip("/").endswith("/v4")


def test_env_example_contains_no_private_material() -> None:
    content = _ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "-----BEGIN" not in content
    assert "PRIVATE KEY" not in content
    assert "postgres://" not in content
