from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from automation.schemas.spec_work_item import SpecWorkItem, load_work_item, main


def _valid_work_item_payload() -> dict[str, object]:
    return {
        "type": "spec_work_item",
        "title": "Persist desktop attribution ledger",
        "spec_refs": ["§6.1", "§7E"],
        "source_artifacts": ["docs/tech-spec-v4.0.pdf"],
        "target_files": [
            "services/desktop_app/processes/analytics_state_worker.py",
            "tests/unit/desktop_app/processes/test_analytics_state_worker.py",
        ],
        "target_symbols": ["analytics_state_worker", "AttributionLedgerRecords"],
        "dependencies": ["packages/ml_core/attribution.py"],
        "guarded_activation_risks": ["offline_final replay remains guarded by tests"],
        "acceptance_criteria": {
            "invariants": ["only online_provisional records are emitted"],
            "tests": ["tests/unit/desktop_app/processes/test_analytics_state_worker.py"],
            "required_gates": [
                "uv run pytest tests/unit/desktop_app/processes/test_analytics_state_worker.py -q"
            ],
            "forbidden_changes": [
                "do not add an offline_final replay producer without updating its guard"
            ],
        },
        "local_artifacts": ["automation/work-items/active/attribution-ledger.json"],
    }


def test_spec_work_item_accepts_valid_payload() -> None:
    work_item = SpecWorkItem.model_validate(_valid_work_item_payload())

    assert work_item.title == "Persist desktop attribution ledger"
    assert work_item.acceptance_criteria.forbidden_changes == [
        "do not add an offline_final replay producer without updating its guard"
    ]


def test_spec_work_item_rejects_duplicate_target_files() -> None:
    payload = _valid_work_item_payload()
    target_files = payload["target_files"]
    assert isinstance(target_files, list)
    target_files.append("services/desktop_app/processes/analytics_state_worker.py")

    with pytest.raises(ValidationError, match="duplicate target_files"):
        SpecWorkItem.model_validate(payload)


def test_spec_work_item_rejects_duplicate_target_symbols() -> None:
    payload = _valid_work_item_payload()
    target_symbols = payload["target_symbols"]
    assert isinstance(target_symbols, list)
    target_symbols.append("analytics_state_worker")

    with pytest.raises(ValidationError, match="duplicate target_symbols"):
        SpecWorkItem.model_validate(payload)


def test_spec_work_item_requires_spec_refs() -> None:
    payload = _valid_work_item_payload()
    payload["spec_refs"] = []

    with pytest.raises(ValidationError):
        SpecWorkItem.model_validate(payload)


def test_spec_work_item_rejects_unknown_fields() -> None:
    payload = _valid_work_item_payload()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SpecWorkItem.model_validate(payload)


def test_spec_work_item_rejects_local_artifacts_outside_active_dir() -> None:
    payload = _valid_work_item_payload()
    payload["local_artifacts"] = ["automation/work-items/templates/spec_work_item.json"]

    with pytest.raises(ValidationError, match="local_artifacts entries must live under"):
        SpecWorkItem.model_validate(payload)


def test_load_work_item_reads_json_file(tmp_path: Path) -> None:
    work_item_path = tmp_path / "work-item.json"
    work_item_path.write_text(json.dumps(_valid_work_item_payload()), encoding="utf-8")

    work_item = load_work_item(str(work_item_path))

    assert work_item.spec_refs == ["§6.1", "§7E"]


def test_main_returns_zero_for_valid_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work_item_path = tmp_path / "work-item.json"
    work_item_path.write_text(json.dumps(_valid_work_item_payload()), encoding="utf-8")

    exit_code = main([str(work_item_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Persist desktop attribution ledger" in captured.out


def test_main_reads_stdin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", _Stdin(json.dumps(_valid_work_item_payload())))

    exit_code = main([])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "automation/work-items/active/attribution-ledger.json" in captured.out


class _Stdin:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value
