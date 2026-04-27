"""Integration coverage for the offline fixture benchmark harness."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from scripts import run_fixture_benchmark
from scripts.generate_capture_fixture import main as generate_capture_fixture


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
    cycle_index = run_fixture_benchmark.BASELINE_COLUMNS.index("Cycle / PR")
    return sum(1 for row in rows if row[cycle_index] == run_fixture_benchmark.FIXTURE_LABEL)


@pytest.mark.integration
def test_fixture_benchmark_appends_parseable_offline_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "tiny-fixture"
    _generate_tiny_fixture(fixture)

    repo_baseline = Path("docs/artifacts/performance_baseline.md")
    repo_baseline_before = repo_baseline.read_text(encoding="utf-8")
    baseline_copy = tmp_path / "performance_baseline.md"
    shutil.copyfile(repo_baseline, baseline_copy)

    for env_name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "OURA_CLIENT_ID",
        "OURA_CLIENT_SECRET",
    ):
        monkeypatch.delenv(env_name, raising=False)

    from packages.ml_core.semantic import SemanticEvaluator
    from packages.ml_core.transcription import TranscriptionEngine
    from services.worker.pipeline.orchestrator import DriftCorrector

    def forbidden_live_access(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise KeyboardInterrupt("live hardware/network/model access attempted")

    monkeypatch.setattr(DriftCorrector, "poll", forbidden_live_access)
    monkeypatch.setattr(TranscriptionEngine, "load_model", forbidden_live_access)
    monkeypatch.setattr(SemanticEvaluator, "__init__", forbidden_live_access)

    _columns, original_rows = run_fixture_benchmark._read_baseline_table(baseline_copy)
    original_fixture_rows = _fixture_row_count(original_rows)
    assert not any(
        row[2] == run_fixture_benchmark.FIXTURE_LABEL and "N=3" in row[-1] for row in original_rows
    ), "checked-in baseline must not keep the non-compliant N=3 fixture row"
    original_live_rows = [row for row in original_rows if row[2].startswith("PR 91")]
    assert original_live_rows, "operator-owned live-stack row must be present in baseline"

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

    assert mapping["Cycle / PR"] == run_fixture_benchmark.FIXTURE_LABEL
    assert "PR" not in mapping["Cycle / PR"]
    assert all(cell.strip() and cell.strip().upper() != "TBD" for cell in mapping.values())
    assert "TBD" not in mapping["Notes"].upper()
    assert "offline" in mapping["Notes"].lower()
    assert "N=3" in mapping["Notes"]

    segment_p50 = _timing(mapping, "Segment-assembly p50 (ms)")
    segment_p95 = _timing(mapping, "Segment-assembly p95 (ms)")
    inference_p50 = _timing(mapping, "ML inference p50 (ms)")
    inference_p95 = _timing(mapping, "ML inference p95 (ms)")
    au12_p50 = _timing(mapping, "AU12 per-frame p50 (ms)")
    comod_ms = _timing(mapping, "Co-Modulation window compute (ms)")

    assert 0.0 < segment_p50 <= segment_p95 < 1_000.0
    assert 0.0 < inference_p50 <= inference_p95 < 5_000.0
    assert 0.0 < au12_p50 < 50.0
    assert comod_ms == pytest.approx(0.0, abs=1e-9)

    _columns, updated_rows = run_fixture_benchmark._read_baseline_table(baseline_copy)
    assert _fixture_row_count(updated_rows) == original_fixture_rows + 1
    updated_live_rows = [row for row in updated_rows if row[2].startswith("PR 91")]
    assert updated_live_rows == original_live_rows
    assert repo_baseline.read_text(encoding="utf-8") == repo_baseline_before


@pytest.mark.integration
def test_fixture_benchmark_regression_warning_is_observational_only() -> None:
    columns = run_fixture_benchmark.BASELINE_COLUMNS
    previous = (
        "2026-04-16",
        "`abc1234`",
        run_fixture_benchmark.FIXTURE_LABEL,
        "1.000",
        "1.000",
        "1.000",
        "1.000",
        "0.100",
        "0.000",
        "previous fixture row",
    )
    current = (
        "2026-04-16",
        "`abc1234`",
        run_fixture_benchmark.FIXTURE_LABEL,
        "1.000",
        "1.250",
        "1.000",
        "1.300",
        "0.100",
        "0.000",
        "current fixture row",
    )

    assert len(previous) == len(columns)
    assert len(current) == len(columns)
    warnings = run_fixture_benchmark._observational_warnings(current, previous)

    assert len(warnings) == 2
    assert "Segment-assembly p95" in warnings[0]
    assert "ML inference p95" in warnings[1]
