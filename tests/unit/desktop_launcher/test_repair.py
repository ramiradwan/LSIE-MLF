"""WS1 P4 — runtime repair tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from services.desktop_app.state.sqlite_schema import bootstrap_schema
from services.desktop_launcher import health_check, install_manager, repair


def test_repair_runtime_reinstalls_and_runs_smoke_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    python_exe = runtime_dir / "python" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("python", encoding="utf-8")
    calls: list[tuple[Path, Path, Path, bool]] = []
    statuses: list[str] = []
    logs: list[str] = []

    def fake_find_runtime_python(root: Path) -> Path:
        assert root == runtime_dir / "python"
        return python_exe

    def fake_run_uv_sync(
        *,
        repo_root: Path,
        staging_dir: Path,
        python_exe: Path,
        log: install_manager.LogCallback,
        reinstall: bool = False,
    ) -> None:
        calls.append((repo_root, staging_dir, python_exe, reinstall))
        log("uv sync complete")

    monkeypatch.setattr(install_manager, "find_runtime_python", fake_find_runtime_python)
    monkeypatch.setattr(install_manager, "run_uv_sync", fake_run_uv_sync)
    monkeypatch.setattr(health_check, "run_runtime_smoke_test", lambda root: f"smoke {root.name}")

    result = repair.repair_runtime(
        runtime_dir=runtime_dir,
        repo_root=tmp_path / "repo",
        sqlite_path=tmp_path / "state" / "desktop.sqlite",
        status=statuses.append,
        log=logs.append,
    )

    assert calls == [(tmp_path / "repo", runtime_dir, python_exe, True)]
    assert logs == ["uv sync complete", "smoke runtime"]
    assert statuses == [
        "Repairing desktop runtime",
        "Rebuilding ML backend",
        "Running runtime health check",
        "Repair complete",
    ]
    assert result.runtime_dir == runtime_dir
    assert result.preserved_tables == ("attribution_event", "metrics", "physiology_log")


def test_repair_runtime_removes_leftover_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    staging_dir = tmp_path / "runtime.staging"
    python_exe = runtime_dir / "python" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("python", encoding="utf-8")
    staging_dir.mkdir()
    (staging_dir / "leftover.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(install_manager, "find_runtime_python", lambda _root: python_exe)
    monkeypatch.setattr(install_manager, "run_uv_sync", lambda **_kwargs: None)
    monkeypatch.setattr(health_check, "run_runtime_smoke_test", lambda _root: "")

    repair.repair_runtime(runtime_dir=runtime_dir, repo_root=tmp_path)

    assert not staging_dir.exists()


def test_repair_runtime_preserves_local_sqlite_analytics_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    python_exe = runtime_dir / "python" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("python", encoding="utf-8")
    sqlite_path = tmp_path / "state" / "desktop.sqlite"
    sqlite_path.parent.mkdir()
    _seed_sqlite(sqlite_path)

    monkeypatch.setattr(install_manager, "find_runtime_python", lambda _root: python_exe)
    monkeypatch.setattr(install_manager, "run_uv_sync", lambda **_kwargs: None)
    monkeypatch.setattr(health_check, "run_runtime_smoke_test", lambda _root: "")

    repair.repair_runtime(runtime_dir=runtime_dir, repo_root=tmp_path, sqlite_path=sqlite_path)

    with sqlite3.connect(str(sqlite_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM physiology_log").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM attribution_event").fetchone() == (1,)


def _seed_sqlite(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        bootstrap_schema(conn)
        conn.execute(
            "INSERT INTO sessions (session_id, stream_url, started_at) VALUES (?, ?, ?)",
            ("session-1", "https://example.test/live", "2026-05-02T10:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO metrics (
                session_id,
                segment_id,
                timestamp_utc,
                au12_intensity
            ) VALUES (?, ?, ?, ?)
            """,
            ("session-1", "segment-1", "2026-05-02T10:00:01Z", 0.42),
        )
        conn.execute(
            """
            INSERT INTO physiology_log (
                session_id,
                segment_id,
                subject_role,
                rmssd_ms,
                heart_rate_bpm,
                freshness_s,
                is_stale,
                provider,
                source_kind,
                derivation_method,
                window_s,
                validity_ratio,
                is_valid,
                source_timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                "segment-1",
                "streamer",
                41.2,
                72,
                12.0,
                0,
                "oura",
                "ibi",
                "rmssd_trailing_window",
                300,
                1.0,
                1,
                "2026-05-02T10:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO attribution_event (
                event_id,
                session_id,
                segment_id,
                event_type,
                event_time_utc,
                selected_arm_id,
                expected_rule_text_hash,
                semantic_method,
                semantic_method_version,
                reward_path_version,
                bandit_decision_snapshot,
                finality,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "event-1",
                "session-1",
                "segment-1",
                "greeting_interaction",
                "2026-05-02T10:00:01Z",
                "arm-a",
                "0" * 64,
                "cross_encoder",
                "v1",
                "reward-v1",
                "{}",
                "online_provisional",
                "v1",
            ),
        )
