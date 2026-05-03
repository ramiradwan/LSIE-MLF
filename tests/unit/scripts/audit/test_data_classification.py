from __future__ import annotations

from pathlib import Path

from packages.schemas.data_tiers import DataTier
from scripts.audit.registry import AuditContext
from scripts.audit.spec_items import Section13Item
from scripts.audit.verifiers.data_classification import (
    scan_data_classification,
    verify_data_classification,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _item() -> Section13Item:
    return Section13Item(
        item_id="13.14",
        title="Data classification",
        body="Transient, debug, and permanent storage tiers enforced.",
    )


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_valid_inbound(repo_root: Path) -> None:
    _write(
        repo_root,
        "services/api/routes/example.py",
        """
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier


async def inbound(request):
    body = mark_data_tier(
        await request.body(),
        DataTier.TRANSIENT,
        spec_ref="§5.2.1",
    )
    return body
""".lstrip(),
    )


def test_data_tier_enum_is_spec_bounded() -> None:
    assert {tier.value for tier in DataTier} == {
        "Transient Data",
        "Debug Storage",
        "Permanent Analytical Storage",
    }
    assert [tier.name for tier in DataTier] == ["TRANSIENT", "DEBUG", "PERMANENT"]


def test_current_repo_data_classification_verifier_passes_with_file_line_evidence() -> None:
    result = verify_data_classification(
        AuditContext(repo_root=_REPO_ROOT, spec_content={}),
        _item(),
    )

    assert result.passed is True
    assert result.follow_up is None
    assert "services/worker/pipeline/analytics.py:" in result.evidence
    assert "services/api/routes/physiology.py:" in result.evidence
    assert "services/cloud_api/middleware/forbid_raw.py:" in result.evidence
    assert "services/cloud_api/repos/telemetry.py:" in result.evidence
    assert "INSERT tier=PERMANENT §5.2.3" in result.evidence
    assert "params tier=PERMANENT §5.2.3" in result.evidence
    assert "inbound tier=TRANSIENT §5.2.1" in result.evidence


def test_reward_py_is_scanned_and_unannotated_insert_fails_with_file_line(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "services/worker/pipeline/reward.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier

INSERT_SQL = """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
"""


def _example_insert_params() -> dict[str, str]:
    return mark_data_tier(
        {"session_id": "s"},
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
    )


def persist(cur):
    cur.execute(INSERT_SQL, _example_insert_params())
'''.lstrip(),
    )
    _write_valid_inbound(tmp_path)

    scan = scan_data_classification(tmp_path)

    rendered = "\n".join(finding.render() for finding in scan.findings)
    assert scan.passed is False
    assert "services/worker/pipeline/reward.py:19" in rendered
    assert "missing/invalid DataTier.PERMANENT annotation" in rendered


def test_scan_reports_missing_inbound_annotation_with_concrete_file_line(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "services/worker/pipeline/example.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier

INSERT_SQL = mark_data_tier(
    """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


def _example_insert_params() -> dict[str, str]:
    return mark_data_tier(
        {"session_id": "s"},
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
    )


def persist(cur):
    cur.execute(INSERT_SQL, _example_insert_params())
'''.lstrip(),
    )
    _write(
        tmp_path,
        "services/api/routes/example.py",
        """
from __future__ import annotations


async def inbound(request):
    body = await request.body()
    return body
""".lstrip(),
    )

    scan = scan_data_classification(tmp_path)

    rendered = "\n".join(finding.render() for finding in scan.findings)
    assert scan.passed is False
    assert "services/api/routes/example.py:5" in rendered
    assert "missing/invalid DataTier.TRANSIENT annotation" in rendered


def test_scan_accepts_derived_json_parse_from_marked_inbound_body(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "services/api/routes/example.py",
        """
from __future__ import annotations

import json

from packages.schemas.data_tiers import DataTier, mark_data_tier


async def inbound(request):
    body = mark_data_tier(
        await request.body(),
        DataTier.TRANSIENT,
        spec_ref="§5.2.1",
    )
    payload = json.loads(body)
    return payload
""".lstrip(),
    )
    _write(
        tmp_path,
        "services/worker/pipeline/example.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier

INSERT_SQL = mark_data_tier(
    """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


def persist(cur):
    cur.execute(
        INSERT_SQL,
        mark_data_tier({"session_id": "s"}, DataTier.PERMANENT, spec_ref="§5.2.3"),
    )
'''.lstrip(),
    )

    scan = scan_data_classification(tmp_path)

    assert scan.passed is True
    assert scan.findings == ()
    assert (
        "services/api/routes/example.py:9 inbound tier=TRANSIENT §5.2.1" in scan.inbound_annotations
    )


def test_permanent_insert_params_without_normalization_evidence_fail(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "services/worker/pipeline/example.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier

INSERT_SQL = mark_data_tier(
    """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


def persist(cur):
    cur.execute(INSERT_SQL, {"session_id": "s"})
'''.lstrip(),
    )
    _write_valid_inbound(tmp_path)

    scan = scan_data_classification(tmp_path)

    rendered = "\n".join(finding.render() for finding in scan.findings)
    assert scan.passed is False
    assert "services/worker/pipeline/example.py:15" in rendered
    assert "missing/invalid DataTier.PERMANENT normalization evidence" in rendered
    assert "params_line=15" in rendered


def test_raw_transient_params_fail_at_permanent_insert_boundary(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "services/worker/pipeline/example.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, mark_data_tier

INSERT_SQL = mark_data_tier(
    """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


def persist(cur):
    transient_params = mark_data_tier(
        {"session_id": "s"},
        DataTier.TRANSIENT,
        spec_ref="§5.2.1",
    )
    cur.execute(INSERT_SQL, transient_params)
'''.lstrip(),
    )
    _write_valid_inbound(tmp_path)

    scan = scan_data_classification(tmp_path)

    rendered = "\n".join(finding.render() for finding in scan.findings)
    assert scan.passed is False
    assert "services/worker/pipeline/example.py:20" in rendered
    assert "missing/invalid DataTier.PERMANENT normalization evidence" in rendered
    assert "params_line=15" in rendered


def test_scan_accepts_marked_insert_inbound_and_normalized_params(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "services/worker/pipeline/example.py",
        '''
from __future__ import annotations

from packages.schemas.data_tiers import DataTier, data_tier, mark_data_tier

INSERT_SQL = mark_data_tier(
    """
INSERT INTO metrics (session_id) VALUES (%(session_id)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


@data_tier(
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Dedicated normalized metrics row constructor",
)
def _example_insert_params(session_id: str) -> dict[str, str]:
    return {"session_id": session_id}


def persist(cur):
    cur.execute(INSERT_SQL, params=_example_insert_params("s"))
'''.lstrip(),
    )
    _write_valid_inbound(tmp_path)

    scan = scan_data_classification(tmp_path)

    assert scan.passed is True
    assert scan.findings == ()
    assert scan.insert_annotations == (
        "services/worker/pipeline/example.py:24 INSERT tier=PERMANENT §5.2.3 "
        "params tier=PERMANENT §5.2.3",
    )
    assert scan.inbound_annotations == (
        "services/api/routes/example.py:7 inbound tier=TRANSIENT §5.2.1",
    )
