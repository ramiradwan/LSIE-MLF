"""v3.4-specific verifier PASS/FAIL and behavioral binding coverage."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.audit.registry import AuditContext
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item
from scripts.audit.verifiers import behavioral
from scripts.audit.verifiers.mechanical import (
    MECHANICAL_VERIFIERS,
    verify_derived_only_attribution_persistence,
    verify_semantic_reason_codes,
)

_REASON_CODES = (
    "cross_encoder_high_match",
    "cross_encoder_high_nonmatch",
    "gray_band_llm_match",
    "gray_band_llm_nonmatch",
    "semantic_local_failure_fallback",
    "semantic_timeout",
    "semantic_error",
)

_SEMANTIC_METHODS = (
    "cross_encoder",
    "llm_gray_band",
    "azure_llm_legacy",
)

_DERIVED_ONLY_BODY = (
    "No raw audio, raw video, complete transient PhysiologicalChunkEvent payload "
    "bodies, or free-form semantic rationales are persisted by the attribution extension."
)

_REQUIRED_DERIVED_FIELDS = (
    "expected_rule_text_hash",
    "semantic_method",
    "semantic_method_version",
    "semantic_p_match",
    "semantic_reason_code",
    "bandit_decision_snapshot",
    "finality",
    "schema_version",
)

_V34_BEHAVIORAL_ITEMS = (
    "13.24",
    "13.25",
    "13.26",
    "13.27",
    "13.28",
    "13.29",
    "13.31",
)
_V34_BEHAVIORAL_MARKER_ONLY_ITEMS = (
    "13.24",
    "13.25",
    "13.26",
    "13.28",
    "13.29",
    "13.31",
)
_V34_BEHAVIORAL_TARGET_FILES = {
    "tests/unit/worker/pipeline/test_orchestrator.py",
    "tests/unit/ml_core/test_semantic.py",
    "tests/unit/worker/pipeline/test_analytics.py",
    "tests/unit/worker/tasks/test_inference_v3.py",
}
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _item(item_id: str, *, body: str | None = None) -> Section13Item:
    return Section13Item(
        item_id=item_id,
        title=f"v3.4 verifier {item_id}",
        body=body or f"Synthetic v3.4 criterion for §{item_id}.",
    )


def _context(repo_root: Path, spec_content: dict[str, Any]) -> AuditContext:
    return AuditContext(repo_root=repo_root, spec_content=spec_content)


CompletedPytestRun = subprocess.CompletedProcess[str]


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> CompletedPytestRun:
    return subprocess.CompletedProcess(
        args=[sys.executable, "-m", "pytest"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _node_suffix(item_id: str) -> str:
    return item_id.replace(".", "_")


def _stub_behavioral_pytest(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    item_id: str,
    *,
    run_passes: bool,
) -> tuple[list[tuple[str, bool, tuple[str, ...]]], str]:
    suffix = _node_suffix(item_id)
    nodeid = f"tests/test_v34_behavior.py::test_audit_{suffix}"
    location = f"tests/test_v34_behavior.py:1:test_audit_{suffix}"
    calls: list[tuple[str, bool, tuple[str, ...]]] = []

    def fake_run(
        observed_repo_root: Path,
        observed_item_id: str,
        *,
        collect_only: bool,
        test_paths: Sequence[str] | None = None,
    ) -> CompletedPytestRun:
        assert observed_repo_root == repo_root
        assert observed_item_id == item_id
        selected_paths = tuple(test_paths or ())
        assert selected_paths == ("tests/test_v34_behavior.py",)
        calls.append((observed_item_id, collect_only, selected_paths))
        if collect_only:
            return _completed(0, stdout=f"{nodeid}\n")
        if run_passes:
            return _completed(0, stdout=".\n1 passed")
        return _completed(1, stdout=f"F\nFAILED {nodeid}")

    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda _tests_root: {item_id: (location,)},
    )
    monkeypatch.setattr(behavioral, "_run_pytest_for_audit_item", fake_run)
    return calls, nodeid


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _assert_concrete_verifier_result(result: AuditResult) -> None:
    assert "Verifier for §" not in result.evidence
    assert "not yet implemented" not in result.evidence
    assert "placeholder" not in result.evidence.lower()


def _assert_pass(result: AuditResult, *snippets: str) -> None:
    assert result.passed is True
    assert result.follow_up is None
    _assert_concrete_verifier_result(result)
    for snippet in snippets:
        assert snippet in result.evidence


def _assert_fail(result: AuditResult, *snippets: str) -> None:
    assert result.passed is False
    assert result.follow_up is not None
    _assert_concrete_verifier_result(result)
    for snippet in snippets:
        assert snippet in result.evidence


def _semantic_spec() -> dict[str, Any]:
    schema = {"properties": {"reasoning": {"enum": list(_REASON_CODES)}}}
    return {
        "llm_prompt": {
            "output_schema": {"source": json.dumps(schema)},
            "inference_parameters": [
                {"parameter": "match_threshold", "value": "0.72"},
                {"parameter": "gray_band_interval", "value": "0.58 <= score < 0.72"},
            ],
        },
        "semantic_method_registry": [
            {"method_id": "cross_encoder", "role": "primary", "execution_mode": "local"},
            {
                "method_id": "llm_gray_band",
                "role": "fallback",
                "execution_mode": "azure_openai",
            },
            {
                "method_id": "azure_llm_legacy",
                "role": "legacy",
                "execution_mode": "azure_openai",
            },
        ],
    }


def _write_semantic_fixture(
    repo_root: Path,
    *,
    methods: tuple[str, ...] = _SEMANTIC_METHODS,
) -> None:
    _write(
        repo_root,
        "packages/schemas/evaluation.py",
        f"SEMANTIC_REASON_CODES = {_REASON_CODES!r}\nSEMANTIC_METHODS = {methods!r}\n",
    )
    _write(
        repo_root,
        "packages/ml_core/semantic.py",
        "from packages.schemas.evaluation import SEMANTIC_REASON_CODES\n\n"
        "GRAY_BAND_LOWER_THRESHOLD = 0.58\n"
        "MATCH_THRESHOLD = 0.72\n"
        'OUTPUT_SCHEMA = {"properties": {"reasoning": '
        '{"enum": list(SEMANTIC_REASON_CODES)}}}\n\n'
        "class Evaluator:\n"
        "    def _evaluate_gray_band_fallback(self, primary_score):\n"
        "        return False\n\n"
        "    def evaluate(self, primary_score):\n"
        "        if (primary_score >= MATCH_THRESHOLD "
        "or primary_score < GRAY_BAND_LOWER_THRESHOLD):\n"
        "            return True\n"
        "        if self.gray_band_fallback_enabled:\n"
        "            return self._evaluate_gray_band_fallback(primary_score)\n"
        "        return False\n",
    )


def _attribution_schema_source(*, forbidden_field: str | None = None) -> str:
    optional = f"    {forbidden_field}: bytes\n" if forbidden_field else ""
    field_lines = "".join(f"    {field}: str\n" for field in _REQUIRED_DERIVED_FIELDS)
    return (
        "class AttributionEvent:\n"
        "    event_id: str\n"
        f"{field_lines}"
        f"{optional}"
        "\n"
        "class OutcomeEvent:\n"
        "    outcome_id: str\n"
        "    finality: str\n"
        "    schema_version: str\n"
        "\n"
        "class EventOutcomeLink:\n"
        "    link_id: str\n"
        "    finality: str\n"
        "    schema_version: str\n"
        "\n"
        "class AttributionScore:\n"
        "    score_id: str\n"
        "    finality: str\n"
        "    schema_version: str\n"
    )


def _attribution_sql_source(*, forbidden_column: str | None = None) -> str:
    optional = f"    {forbidden_column} BYTEA,\n" if forbidden_column else ""
    required_columns = "\n".join(f"    {field} TEXT," for field in _REQUIRED_DERIVED_FIELDS)
    return f"""
CREATE TABLE IF NOT EXISTS attribution_event (
    event_id UUID PRIMARY KEY,
{required_columns}
{optional}    created_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS outcome_event (
    outcome_id UUID PRIMARY KEY,
    finality TEXT,
    schema_version TEXT
);
CREATE TABLE IF NOT EXISTS event_outcome_link (
    link_id UUID PRIMARY KEY,
    finality TEXT,
    schema_version TEXT
);
CREATE TABLE IF NOT EXISTS attribution_score (
    score_id UUID PRIMARY KEY,
    finality TEXT,
    schema_version TEXT
);
"""


def _analytics_source() -> str:
    column_list = ", ".join(_REQUIRED_DERIVED_FIELDS)
    return f'''
class MetricsStore:
    def _write_attribution_ledger(self, ledger):
        insert_event_sql = """
        INSERT INTO attribution_event ({column_list}) VALUES ({column_list})
        """
        event_params = ledger.event.model_dump(mode="json")
        return insert_event_sql, event_params
'''


def _write_derived_only_fixture(
    repo_root: Path,
    *,
    forbidden_field: str | None = None,
    forbidden_column: str | None = None,
) -> None:
    _write(
        repo_root,
        "packages/schemas/attribution.py",
        _attribution_schema_source(forbidden_field=forbidden_field),
    )
    _write(
        repo_root,
        "data/sql/05-attribution.sql",
        _attribution_sql_source(forbidden_column=forbidden_column),
    )
    _write(repo_root, "services/worker/pipeline/analytics.py", _analytics_source())


def test_v34_mechanical_verifiers_are_registered() -> None:
    assert "13.27" not in MECHANICAL_VERIFIERS
    assert "13.30" in MECHANICAL_VERIFIERS


def test_v34_behavioral_verifiers_are_registered() -> None:
    assert set(_V34_BEHAVIORAL_ITEMS).issubset(behavioral.BEHAVIORAL_VERIFIERS)
    assert "13.30" not in behavioral.BEHAVIORAL_VERIFIERS
    assert set(MECHANICAL_VERIFIERS).isdisjoint(behavioral.BEHAVIORAL_VERIFIERS)
    assert behavioral.BEHAVIORAL_VERIFIERS["13.27"] is behavioral.verify_behavioral_item


def test_v34_behavioral_items_have_concrete_marker_bindings() -> None:
    discovered = behavioral.discover_audit_item_markers(_REPO_ROOT / "tests")
    missing = [item_id for item_id in _V34_BEHAVIORAL_ITEMS if not discovered.get(item_id)]

    assert missing == []
    for item_id in _V34_BEHAVIORAL_ITEMS:
        locations = discovered[item_id]
        assert all("test_v34_verifiers.py" not in location for location in locations)
        assert any(
            location.split(":", maxsplit=1)[0] in _V34_BEHAVIORAL_TARGET_FILES
            for location in locations
        ), f"§{item_id} has no target-file audit_item binding: {locations!r}"


@pytest.mark.parametrize("item_id", _V34_BEHAVIORAL_MARKER_ONLY_ITEMS)
def test_v34_behavioral_verifier_path_passes_for_marker_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item_id: str,
) -> None:
    calls, nodeid = _stub_behavioral_pytest(
        monkeypatch,
        tmp_path,
        item_id,
        run_passes=True,
    )

    result = behavioral.verify_behavioral_item(_context(tmp_path, {}), _item(item_id))

    _assert_pass(
        result,
        f'@pytest.mark.audit_item("{item_id}")',
        "selected test(s) passed",
        nodeid,
    )
    assert calls == [
        (item_id, True, ("tests/test_v34_behavior.py",)),
        (item_id, False, ("tests/test_v34_behavior.py",)),
    ]


@pytest.mark.parametrize("item_id", _V34_BEHAVIORAL_MARKER_ONLY_ITEMS)
def test_v34_behavioral_verifier_path_fails_for_marker_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item_id: str,
) -> None:
    calls, nodeid = _stub_behavioral_pytest(
        monkeypatch,
        tmp_path,
        item_id,
        run_passes=False,
    )

    result = behavioral.verify_behavioral_item(_context(tmp_path, {}), _item(item_id))

    _assert_fail(
        result,
        f'@pytest.mark.audit_item("{item_id}")',
        "Selected tests",
        nodeid,
    )
    assert calls == [
        (item_id, True, ("tests/test_v34_behavior.py",)),
        (item_id, False, ("tests/test_v34_behavior.py",)),
    ]


def test_v34_behavioral_missing_binding_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda _tests_root: {},
    )

    result = behavioral.verify_behavioral_item(_context(tmp_path, {}), _item("13.24"))

    _assert_fail(
        result,
        'No tests discovered for @pytest.mark.audit_item("13.24") under tests/.',
    )


def test_v34_future_behavioral_item_fails_instead_of_placeholder(tmp_path: Path) -> None:
    result = behavioral.verify_behavioral_item(_context(tmp_path, {}), _item("13.32"))

    _assert_fail(result, "§13.32 is not an in-scope behavioral audit item.")


def test_v34_1327_composite_verifier_passes_with_mechanical_and_behavioral_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_semantic_fixture(tmp_path)
    calls, nodeid = _stub_behavioral_pytest(
        monkeypatch,
        tmp_path,
        "13.27",
        run_passes=True,
    )

    result = behavioral.verify_behavioral_item(
        _context(tmp_path, _semantic_spec()),
        _item("13.27"),
    )

    _assert_pass(
        result,
        "PASS mechanical semantic registry/threshold surface",
        "§8.1/§13.27",
        "SEMANTIC_METHODS",
        "fallback branch is bounded",
        "PASS behavioral marker/evidence surface",
        '@pytest.mark.audit_item("13.27")',
        nodeid,
    )
    assert calls == [
        ("13.27", True, ("tests/test_v34_behavior.py",)),
        ("13.27", False, ("tests/test_v34_behavior.py",)),
    ]


def test_v34_1327_composite_verifier_fails_without_losing_behavioral_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_semantic_fixture(tmp_path, methods=("cross_encoder", "llm_gray_band"))
    _, nodeid = _stub_behavioral_pytest(
        monkeypatch,
        tmp_path,
        "13.27",
        run_passes=True,
    )

    result = behavioral.verify_behavioral_item(
        _context(tmp_path, _semantic_spec()),
        _item("13.27"),
    )

    _assert_fail(
        result,
        "FAIL mechanical semantic registry/threshold surface",
        "SEMANTIC_METHODS=('cross_encoder', 'llm_gray_band')",
        "PASS behavioral marker/evidence surface",
        nodeid,
    )


def test_v34_1327_composite_verifier_fails_without_losing_mechanical_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_semantic_fixture(tmp_path)
    _, nodeid = _stub_behavioral_pytest(
        monkeypatch,
        tmp_path,
        "13.27",
        run_passes=False,
    )

    result = behavioral.verify_behavioral_item(
        _context(tmp_path, _semantic_spec()),
        _item("13.27"),
    )

    _assert_fail(
        result,
        "PASS mechanical semantic registry/threshold surface",
        "SEMANTIC_METHODS",
        "FAIL behavioral marker/evidence surface",
        "Selected tests",
        nodeid,
    )


def test_semantic_method_registry_pass_and_fail_are_concrete(tmp_path: Path) -> None:
    pass_root = tmp_path / "semantic_pass"
    _write_semantic_fixture(pass_root)

    pass_result = verify_semantic_reason_codes(
        _context(pass_root, _semantic_spec()),
        _item("13.27"),
    )

    _assert_pass(
        pass_result,
        "§8.1/§13.27",
        "SEMANTIC_METHODS",
        "azure_llm_legacy",
        "fallback branch is bounded",
    )

    fail_root = tmp_path / "semantic_fail"
    _write_semantic_fixture(fail_root, methods=("cross_encoder", "llm_gray_band"))

    fail_result = verify_semantic_reason_codes(
        _context(fail_root, _semantic_spec()),
        _item("13.27"),
    )

    _assert_fail(
        fail_result,
        "§8.1/§13.27",
        "SEMANTIC_METHODS=('cross_encoder', 'llm_gray_band')",
        "expected ('cross_encoder', 'llm_gray_band', 'azure_llm_legacy')",
    )


def test_derived_only_attribution_persistence_pass_and_fail_are_concrete(
    tmp_path: Path,
) -> None:
    pass_root = tmp_path / "derived_only_pass"
    _write_derived_only_fixture(pass_root)

    pass_result = verify_derived_only_attribution_persistence(
        _context(pass_root, {}),
        _item("13.30", body=_DERIVED_ONLY_BODY),
    )

    _assert_pass(
        pass_result,
        "§13.30",
        "no forbidden raw media",
        "packages/schemas/attribution.py exposes derived/versioned fields",
        "data/sql/05-attribution.sql persists derived/versioned columns",
        "services/worker/pipeline/analytics.py writes the approved derived attribution fields",
    )

    fail_root = tmp_path / "derived_only_fail"
    _write_derived_only_fixture(fail_root, forbidden_field="raw_audio")

    fail_result = verify_derived_only_attribution_persistence(
        _context(fail_root, {}),
        _item("13.30", body=_DERIVED_ONLY_BODY),
    )

    _assert_fail(
        fail_result,
        "§13.30 forbidden attribution persistence fields matched",
        "forbidden raw audio field 'raw_audio'",
        "packages/schemas/attribution.py",
    )
