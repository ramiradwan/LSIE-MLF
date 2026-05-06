"""Integration coverage for the v4 desktop fixture benchmark harness."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from scripts import run_fixture_benchmark
from scripts.generate_capture_fixture import main as generate_capture_fixture

BASELINE_TEMPLATE = (
    "# LSIE-MLF Performance Baseline Log\n\n"
    "Fresh v4 baseline.\n\n"
    + run_fixture_benchmark._format_markdown_row(run_fixture_benchmark.BASELINE_COLUMNS)
    + "\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _generate_tiny_fixture(path: Path) -> None:
    exit_code = generate_capture_fixture(
        [
            str(path),
            "--segments",
            "4",
            "--segment-duration-s",
            "1.0",
            "--width",
            "96",
            "--height",
            "72",
            "--seed",
            "2026",
            "--overwrite",
        ]
    )
    assert exit_code == 0


def _row_mapping(row: str) -> dict[str, str]:
    cells = run_fixture_benchmark._parse_markdown_row(row)
    assert len(cells) == len(run_fixture_benchmark.BASELINE_COLUMNS)
    return dict(zip(run_fixture_benchmark.BASELINE_COLUMNS, cells, strict=True))


def _timing(mapping: dict[str, str], column: str) -> float:
    value = run_fixture_benchmark._parse_ms_cell(mapping[column])
    assert value is not None, f"{column} is not numeric: {mapping[column]!r}"
    return value


def _fixture_row_count(rows: list[tuple[str, ...]]) -> int:
    scenario_index = run_fixture_benchmark.BASELINE_COLUMNS.index("Scenario")
    return sum(1 for row in rows if row[scenario_index] == run_fixture_benchmark.FIXTURE_LABEL)


@pytest.mark.integration
def test_fixture_benchmark_appends_parseable_v4_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "tiny-fixture"
    _generate_tiny_fixture(fixture)
    baseline_copy = tmp_path / "performance_baseline.md"
    baseline_copy.write_text(BASELINE_TEMPLATE, encoding="utf-8")

    for env_name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "OURA_CLIENT_ID",
        "OURA_CLIENT_SECRET",
    ):
        monkeypatch.delenv(env_name, raising=False)

    def forbidden_live_access(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise KeyboardInterrupt("live hardware/network/model access attempted")

    monkeypatch.setattr(
        "packages.ml_core.transcription.TranscriptionEngine",
        forbidden_live_access,
    )

    _columns, original_rows = run_fixture_benchmark._read_baseline_table(baseline_copy)
    assert _fixture_row_count(original_rows) == 0

    exit_code = run_fixture_benchmark.main(
        [
            str(fixture),
            "--segments",
            "3",
            "--baseline-path",
            str(baseline_copy),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    output_rows = [line for line in captured.out.splitlines() if line.startswith("|")]
    assert len(output_rows) == 1
    mapping = _row_mapping(output_rows[0])

    assert mapping["Scenario"] == run_fixture_benchmark.FIXTURE_LABEL
    assert mapping["Segments"] == "3"
    assert all(cell.strip() and cell.strip().upper() != "TBD" for cell in mapping.values())
    assert "v4 desktop IPC/SQLite fixture benchmark" in mapping["Notes"]
    assert "persisted=3" in mapping["Notes"]
    assert "live ADB/scrcpy path not measured" in mapping["Notes"]
    assert "Celery" not in captured.out
    assert "Redis" not in captured.out
    assert "Cycle / PR" not in captured.out

    dispatch_p50 = _timing(mapping, "Dispatch p50 (ms)")
    dispatch_p95 = _timing(mapping, "Dispatch p95 (ms)")
    publish_p50 = _timing(mapping, "ML publish p50 (ms)")
    publish_p95 = _timing(mapping, "ML publish p95 (ms)")
    state_p50 = _timing(mapping, "Analytics state p50 (ms)")
    state_p95 = _timing(mapping, "Analytics state p95 (ms)")
    visual_p50 = _timing(mapping, "Visual AU12 tick p50 (ms)")
    e2e_p95 = _timing(mapping, "End-to-end p95 (ms)")

    assert 0.0 < dispatch_p50 <= dispatch_p95 < 1_000.0
    assert 0.0 < publish_p50 <= publish_p95 < 5_000.0
    assert 0.0 < state_p50 <= state_p95 < 1_000.0
    assert 0.0 <= visual_p50 < 100.0
    assert max(dispatch_p95, publish_p95, state_p95) <= e2e_p95 < 10_000.0

    _columns, updated_rows = run_fixture_benchmark._read_baseline_table(baseline_copy)
    assert _fixture_row_count(updated_rows) == 1


@pytest.mark.integration
def test_fixture_benchmark_rejects_old_baseline_table(tmp_path: Path) -> None:
    old_baseline = tmp_path / "old_performance_baseline.md"
    old_baseline.write_text(
        """# old baseline

| Date | Commit SHA | Cycle / PR | Segment-assembly p50 (ms) | Notes |
|---|---|---|---|---|
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="performance baseline table columns changed"):
        run_fixture_benchmark._read_baseline_table(old_baseline)


@pytest.mark.integration
def test_fixture_benchmark_regression_warning_is_observational_only() -> None:
    columns = run_fixture_benchmark.BASELINE_COLUMNS
    previous = (
        "2026-05-06",
        "`abc1234`",
        run_fixture_benchmark.FIXTURE_LABEL,
        "3",
        "1.000",
        "1.000",
        "1.000",
        "1.000",
        "1.000",
        "1.000",
        "0.100",
        "3.000",
        "previous fixture row",
    )
    current = (
        "2026-05-06",
        "`abc1234`",
        run_fixture_benchmark.FIXTURE_LABEL,
        "3",
        "1.000",
        "1.250",
        "1.000",
        "1.300",
        "1.000",
        "1.240",
        "0.100",
        "3.700",
        "current fixture row",
    )

    assert len(previous) == len(columns)
    assert len(current) == len(columns)
    warnings = run_fixture_benchmark._observational_warnings(current, previous)

    assert len(warnings) == 4
    assert "Dispatch p95" in warnings[0]
    assert "ML publish p95" in warnings[1]
    assert "Analytics state p95" in warnings[2]
    assert "End-to-end p95" in warnings[3]


@pytest.mark.integration
def test_fixture_benchmark_does_not_modify_repository_baseline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "tiny-fixture"
    _generate_tiny_fixture(fixture)
    repo_baseline = Path("docs/artifacts/performance_baseline.md")
    repo_baseline_before = repo_baseline.read_text(encoding="utf-8")
    baseline_copy = tmp_path / "performance_baseline.md"
    shutil.copyfile(repo_baseline, baseline_copy)

    exit_code = run_fixture_benchmark.main(
        [
            str(fixture),
            "--segments",
            "1",
            "--baseline-path",
            str(baseline_copy),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert repo_baseline.read_text(encoding="utf-8") == repo_baseline_before
