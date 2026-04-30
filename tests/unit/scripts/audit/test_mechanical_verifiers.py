"""Evidence-focused tests for mechanical §13 audit verifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.audit.registry import AuditContext
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item
from scripts.audit.verifiers.mechanical import (
    MECHANICAL_VERIFIERS,
    verify_au12_geometry,
    verify_canonical_terminology,
    verify_dependency_pins,
    verify_derived_only_attribution_persistence,
    verify_directory_structure,
    verify_docker_topology,
    verify_drift_correction,
    verify_ephemeral_vault,
    verify_ffmpeg_resample,
    verify_ipc_lifecycle,
    verify_module_contracts,
    verify_schema_validation,
    verify_semantic_determinism,
    verify_semantic_reason_codes,
)

_FFMPEG_CMD = (
    "ffmpeg",
    "-f",
    "s16le",
    "-ar",
    "48000",
    "-ac",
    "1",
    "-i",
    "/tmp/ipc/audio_stream.raw",
    "-ar",
    "16000",
    "-f",
    "s16le",
    "-ac",
    "1",
    "pipe:1",
)

_AU12_LANDMARKS = (61, 291, 33, 133, 362, 263)

_LLM_PARAMS: dict[str, object] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 500,
    "seed": 42,
}
_CROSS_ENCODER_MODEL_ID = "local://models/semantic/lsie-greeting-cross-encoder-v1"
_CROSS_ENCODER_MODEL_VERSION = "lsie-greeting-cross-encoder-v1.0.0"
_SEMANTIC_CALIBRATION_VERSION = "semantic-greeting-calibration-v1.0.0"

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

_EXPECTED_MECHANICAL_ITEMS = {
    "13.1",
    "13.2",
    "13.3",
    "13.4",
    "13.5",
    "13.6",
    "13.7",
    "13.8",
    "13.9",
    "13.10",
    "13.12",
    "13.15",
    "13.30",
}


def _item(item_id: str) -> Section13Item:
    return Section13Item(
        item_id=item_id,
        title=f"Mechanical verifier {item_id}",
        body=f"Synthetic unit criterion for §{item_id}.",
    )


def _context(repo_root: Path, spec_content: Mapping[str, Any] | None = None) -> AuditContext:
    return AuditContext(
        repo_root=repo_root, spec_content={} if spec_content is None else spec_content
    )


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _assert_pass_evidence(result: AuditResult, *snippets: str) -> None:
    assert result.passed is True
    assert result.follow_up is None
    for snippet in snippets:
        assert snippet in result.evidence


def _assert_fail_evidence(result: AuditResult, *snippets: str) -> None:
    assert result.passed is False
    assert result.follow_up is not None
    for snippet in snippets:
        assert snippet in result.evidence


def _ffmpeg_spec() -> dict[str, Any]:
    return {
        "core_modules": {
            "modules": [
                {
                    "module_id": "C",
                    "resampling_command": {"source": " ".join(_FFMPEG_CMD)},
                }
            ]
        }
    }


def _au12_spec() -> dict[str, Any]:
    variable_dictionary = [
        {"code_name": f"landmarks[{landmark}]", "definition": "Required AU12 point."}
        for landmark in _AU12_LANDMARKS
    ]
    variable_dictionary.append({"code_name": "self.alpha", "definition": "Default 6.0"})
    return {
        "math_specifications": {
            "topics": [
                {
                    "topic_id": "au12",
                    "variable_dictionary": variable_dictionary,
                    "reference_implementation": {"source": "EPSILON: float = 1e-6"},
                }
            ]
        }
    }


def _llm_spec() -> dict[str, Any]:
    rows = [{"parameter": key, "value": str(value)} for key, value in _LLM_PARAMS.items()]
    rows.extend(
        [
            {"parameter": "response_format", "value": "JSON Schema Structured Outputs"},
            {"parameter": "cross_encoder_model_id", "value": _CROSS_ENCODER_MODEL_ID},
            {
                "parameter": "cross_encoder_model_version",
                "value": _CROSS_ENCODER_MODEL_VERSION,
            },
            {
                "parameter": "semantic_calibration_version",
                "value": _SEMANTIC_CALIBRATION_VERSION,
            },
        ]
    )
    return {"llm_prompt": {"inference_parameters": rows}}


def _vault_spec() -> dict[str, Any]:
    return {
        "data_governance": {
            "vault_parameters": [
                {"parameter": "Encryption algorithm", "specification": "AES-256-GCM"},
                {"parameter": "Key generation", "specification": "os.urandom(32)"},
                {"parameter": "IV/Nonce length", "specification": "12 bytes"},
                {
                    "parameter": "Secure deletion method",
                    "specification": (
                        "shred -vfz -n 3 executed by internal cron every 24 hours "
                        "on /data/raw/ and /data/interim/"
                    ),
                },
            ]
        }
    }


def _dependency_spec() -> dict[str, Any]:
    return {
        "dependency_matrix": {
            "pinned_packages": [
                {
                    "package": "numpy",
                    "version": "1.26.x",
                    "container_targets": ["worker", "orchestrator"],
                },
                {
                    "package": "fastapi",
                    "version": "0.110.x",
                    "container_targets": ["api"],
                },
                {
                    "package": "PySide6",
                    "version": ">=6.11",
                    "container_targets": ["operator_host"],
                },
            ]
        }
    }


def _canonical_spec() -> dict[str, Any]:
    return {
        "document_control": {
            "canonical_terms": [
                {"canonical_name": "ML Worker", "retired_synonym_list": ["GPU worker"]},
                {"canonical_name": "IPC Pipe", "retired_synonym_list": ["video pipe"]},
                {"canonical_name": "API Server", "retired_synonym_list": ["api"]},
            ],
        }
    }


def _reason_spec() -> dict[str, Any]:
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
            {"method_id": method, "role": "registry", "execution_mode": "test"}
            for method in _SEMANTIC_METHODS
        ],
    }


def _write_ffmpeg_fixture(repo_root: Path, command: Sequence[str]) -> None:
    _write(
        repo_root,
        "services/worker/pipeline/orchestrator.py",
        f"FFMPEG_RESAMPLE_CMD = {list(command)!r}\n\n"
        "def spawn(subprocess):\n"
        "    return subprocess.Popen(FFMPEG_RESAMPLE_CMD, stdout=-1)\n",
    )


def _write_au12_fixture(repo_root: Path, epsilon: float = 1e-6) -> None:
    _write(
        repo_root,
        "packages/ml_core/au12.py",
        f"EPSILON = {epsilon!r}\n"
        "DEFAULT_ALPHA_SCALE = 6.0\n\n"
        "def compute(landmarks):\n"
        "    left_outer = landmarks[61]\n"
        "    right_outer = landmarks[291]\n"
        "    left_eye_inner = landmarks[133]\n"
        "    right_eye_inner = landmarks[362]\n"
        "    iod = abs(landmarks[33][0] - landmarks[263][0])\n"
        "    if iod < EPSILON:\n"
        "        return 0.0\n"
        "    return DEFAULT_ALPHA_SCALE + left_outer[0] + right_outer[0] "
        "+ left_eye_inner[0] + right_eye_inner[0]\n",
    )


def _write_semantic_determinism_fixture(
    repo_root: Path,
    params: Mapping[str, object] | None = None,
) -> None:
    actual_params = dict(_LLM_PARAMS)
    if params is not None:
        actual_params.update(params)
    _write(
        repo_root,
        "packages/ml_core/semantic.py",
        f"LLM_PARAMS = {actual_params!r}\n"
        f"CROSS_ENCODER_MODEL_ID = {_CROSS_ENCODER_MODEL_ID!r}\n"
        f"CROSS_ENCODER_MODEL_VERSION = {_CROSS_ENCODER_MODEL_VERSION!r}\n"
        f"SEMANTIC_CALIBRATION_VERSION = {_SEMANTIC_CALIBRATION_VERSION!r}\n\n"
        "GRAY_BAND_RESPONSE_FORMAT = {\n"
        '    "type": "json_schema",\n'
        '    "json_schema": {"strict": True},\n'
        "}\n\n"
        "def call(client):\n"
        "    return client.responses.create(response_format=GRAY_BAND_RESPONSE_FORMAT)\n",
    )


def _write_vault_fixture(repo_root: Path, nonce_length: int = 12) -> None:
    _write(
        repo_root,
        "packages/ml_core/encryption.py",
        "import os\n\n"
        "from Crypto.Cipher import AES\n\n"
        f"nonce_length = {nonce_length}\n"
        'SHRED_COMMAND = ["shred", "-vfz", "-n", "3"]\n\n'
        "class EphemeralVault:\n"
        "    def encrypt(self):\n"
        "        key = os.urandom(32)\n"
        "        nonce = os.urandom(self.nonce_length)\n"
        "        return AES.new(key, AES.MODE_GCM, nonce=nonce)\n",
    )
    _write(
        repo_root,
        "services/worker/vault_cron.py",
        'SHRED_TARGETS = ("/data/raw/", "/data/interim/")\nINTERVAL_HOURS = 24\n',
    )


def _write_dependency_fixture(repo_root: Path, numpy_line: str = "numpy==1.26.4") -> None:
    _write(repo_root, "requirements/base.txt", f"{numpy_line}\n")
    _write(repo_root, "requirements/worker.txt", "-r base.txt\n")
    _write(repo_root, "requirements/api.txt", "fastapi==0.110.2\n")
    _write(repo_root, "requirements/cli.txt", "PySide6>=6.11\n")


def _write_canonical_scan_fixture(repo_root: Path, retired_line: str | None = None) -> None:
    _write(repo_root, "services/good.py", 'COMPONENT = "ML Worker"\n')
    _write(repo_root, "packages/good.py", 'COMPONENT = "IPC Pipe"\n')
    _write(repo_root, "scripts/good.py", 'COMPONENT = "Ephemeral Vault"\n')
    if retired_line is not None:
        _write(repo_root, "services/bad.py", f"COMMENT = {retired_line!r}\n")


def _write_reason_fixture(
    repo_root: Path,
    lower_threshold: float = 0.58,
    branch_mode: str = "bounded",
) -> None:
    branch_sources = {
        "bounded": (
            "        if (primary_score >= MATCH_THRESHOLD "
            "or primary_score < GRAY_BAND_LOWER_THRESHOLD):\n"
            "            return True\n"
            "        if self.gray_band_fallback_enabled:\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
        "flag_only": (
            "        if self.gray_band_fallback_enabled:\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
        "missing_lower": (
            "        if primary_score >= MATCH_THRESHOLD:\n"
            "            return True\n"
            "        if self.gray_band_fallback_enabled:\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
        "missing_upper": (
            "        if primary_score < GRAY_BAND_LOWER_THRESHOLD:\n"
            "            return True\n"
            "        if self.gray_band_fallback_enabled:\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
        "mixed": (
            "        if (self.gray_band_fallback_enabled "
            "and GRAY_BAND_LOWER_THRESHOLD <= primary_score < MATCH_THRESHOLD):\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
            "        return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
        "disjunctive_bypass": (
            "        if (GRAY_BAND_LOWER_THRESHOLD <= primary_score < MATCH_THRESHOLD "
            "and (self.gray_band_fallback_enabled or bypass_flag)):\n"
            "            return self._evaluate_gray_band_fallback(primary_score)\n"
        ),
    }
    branch_source = branch_sources[branch_mode]
    _write(
        repo_root,
        "packages/schemas/evaluation.py",
        f"SEMANTIC_REASON_CODES = {_REASON_CODES!r}\nSEMANTIC_METHODS = {_SEMANTIC_METHODS!r}\n",
    )
    _write(
        repo_root,
        "packages/ml_core/semantic.py",
        "from packages.schemas.evaluation import SEMANTIC_REASON_CODES\n\n"
        f"GRAY_BAND_LOWER_THRESHOLD = {lower_threshold!r}\n"
        "MATCH_THRESHOLD = 0.72\n"
        'OUTPUT_SCHEMA = {"properties": {"reasoning": '
        '{"enum": list(SEMANTIC_REASON_CODES)}}}\n\n'
        "class Evaluator:\n"
        "    def _evaluate_gray_band_fallback(self, primary_score):\n"
        "        return False\n\n"
        "    def evaluate(self, primary_score):\n"
        f"{branch_source}"
        "        return False\n",
    )


class TestMechanicalVerifiers:
    def test_mechanical_registry_keys_are_unit_covered(self) -> None:
        assert set(MECHANICAL_VERIFIERS) == _EXPECTED_MECHANICAL_ITEMS

    def test_spec_extraction_missing_fails_closed(self, tmp_path: Path) -> None:
        cases = []

        ffmpeg_root = tmp_path / "ffmpeg_missing_spec"
        _write_ffmpeg_fixture(ffmpeg_root, _FFMPEG_CMD)
        cases.append((verify_ffmpeg_resample, ffmpeg_root, "13.4", "§4.C.2/§13.4"))

        au12_root = tmp_path / "au12_missing_spec"
        _write_au12_fixture(au12_root)
        cases.append((verify_au12_geometry, au12_root, "13.6", "§7A/§13.6"))

        determinism_root = tmp_path / "determinism_missing_spec"
        _write_semantic_determinism_fixture(determinism_root)
        cases.append((verify_semantic_determinism, determinism_root, "13.7", "§8.2.1/§13.7"))

        vault_root = tmp_path / "vault_missing_spec"
        _write_vault_fixture(vault_root)
        cases.append((verify_ephemeral_vault, vault_root, "13.8", "§5.1/§13.8"))

        dependency_root = tmp_path / "dependency_missing_spec"
        _write_dependency_fixture(dependency_root)
        cases.append((verify_dependency_pins, dependency_root, "13.12", "§10.2/§13.12"))

        canonical_root = tmp_path / "canonical_missing_spec"
        _write_canonical_scan_fixture(canonical_root)
        cases.append((verify_canonical_terminology, canonical_root, "13.15", "§0.3/§13.15"))

        reason_root = tmp_path / "reason_missing_spec"
        _write_reason_fixture(reason_root)
        cases.append((verify_semantic_reason_codes, reason_root, "13.27", "§8.3/§13.27"))

        derived_only_root = tmp_path / "derived_only_missing_spec"
        cases.append(
            (
                verify_derived_only_attribution_persistence,
                derived_only_root,
                "13.30",
                "§13.30",
            )
        )

        for verifier, repo_root, item_id, section_ref in cases:
            result = verifier(_context(repo_root, {}), _item(item_id))

            _assert_fail_evidence(
                result,
                section_ref,
                "unable to extract",
                "Failing closed instead of substituting verifier-local fallback constants",
            )

    def test_verify_ffmpeg_resample_pass_and_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_ffmpeg_fixture(pass_root, _FFMPEG_CMD)

        pass_result = verify_ffmpeg_resample(_context(pass_root, _ffmpeg_spec()), _item("13.4"))

        _assert_pass_evidence(
            pass_result,
            "§4.C.2/§13.4",
            "services/worker/pipeline/orchestrator.py:1",
            "pipe:1",
            "invokes FFMPEG_RESAMPLE_CMD",
        )

        fail_root = tmp_path / "fail"
        mismatched_command = list(_FFMPEG_CMD)
        mismatched_command[4] = "44100"
        _write_ffmpeg_fixture(fail_root, mismatched_command)

        fail_result = verify_ffmpeg_resample(_context(fail_root, _ffmpeg_spec()), _item("13.4"))

        _assert_fail_evidence(
            fail_result,
            "§4.C.2/§13.4",
            "44100",
            "expected ('ffmpeg', '-f', 's16le', '-ar', '48000'",
        )

    def test_verify_au12_geometry_pass_and_epsilon_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_au12_fixture(pass_root)

        pass_result = verify_au12_geometry(_context(pass_root, _au12_spec()), _item("13.6"))

        _assert_pass_evidence(
            pass_result,
            "§7A.1/§7A.2/§13.6",
            "packages/ml_core/au12.py",
            "landmark indices",
            "§7A.5/§13.6",
            "EPSILON=1e-06",
            "DEFAULT_ALPHA_SCALE=6.0",
        )

        fail_root = tmp_path / "fail"
        _write_au12_fixture(fail_root, epsilon=1e-5)

        fail_result = verify_au12_geometry(_context(fail_root, _au12_spec()), _item("13.6"))

        _assert_fail_evidence(
            fail_result,
            "§7A.5/§13.6",
            "EPSILON=1e-05",
            "expected 1e-06",
        )

    def test_verify_semantic_determinism_pass_and_seed_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_semantic_determinism_fixture(pass_root)

        pass_result = verify_semantic_determinism(_context(pass_root, _llm_spec()), _item("13.7"))

        _assert_pass_evidence(
            pass_result,
            "§8.2.1/§13.7",
            "packages/ml_core/semantic.py:1",
            "LLM_PARAMS",
            "CROSS_ENCODER_MODEL_ID",
            "JSON Schema Structured Outputs",
        )

        fail_root = tmp_path / "fail"
        _write_semantic_determinism_fixture(fail_root, {"seed": 41})

        fail_result = verify_semantic_determinism(_context(fail_root, _llm_spec()), _item("13.7"))

        _assert_fail_evidence(
            fail_result,
            "§8.2.1/§13.7",
            "'seed': 41",
            "'seed': 42",
        )

    def test_verify_ephemeral_vault_pass_and_nonce_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_vault_fixture(pass_root)

        pass_result = verify_ephemeral_vault(_context(pass_root, _vault_spec()), _item("13.8"))

        _assert_pass_evidence(
            pass_result,
            "§5.1.5/§13.8",
            "packages/ml_core/encryption.py",
            "os.urandom(32)",
            "§5.1.8/§13.8",
            "services/worker/vault_cron.py:1",
            "('/data/raw/', '/data/interim/')",
        )

        fail_root = tmp_path / "fail"
        _write_vault_fixture(fail_root, nonce_length=16)

        fail_result = verify_ephemeral_vault(_context(fail_root, _vault_spec()), _item("13.8"))

        _assert_fail_evidence(
            fail_result,
            "§5.1.4/§13.8",
            "nonce_length=16",
            "expected 12 bytes",
        )

    def test_verify_dependency_pins_pass_and_version_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_dependency_fixture(pass_root)

        pass_result = verify_dependency_pins(
            _context(pass_root, _dependency_spec()), _item("13.12")
        )

        _assert_pass_evidence(
            pass_result,
            "§10.2/§13.12",
            "requirements/base.txt:1",
            "numpy==1.26.4",
            "requirements/api.txt:1",
            "fastapi==0.110.2",
            "requirements/cli.txt:1",
            "PySide6>=6.11",
        )

        fail_root = tmp_path / "fail"
        _write_dependency_fixture(fail_root, numpy_line="numpy==2.0.0")

        fail_result = verify_dependency_pins(
            _context(fail_root, _dependency_spec()), _item("13.12")
        )

        _assert_fail_evidence(
            fail_result,
            "§10.2/§13.12",
            "numpy==2.0.0",
            "expected numpy 1.26.x",
        )

    def test_verify_canonical_terminology_pass_and_retired_term_match(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_canonical_scan_fixture(pass_root)
        _write(pass_root, "docker-compose.yml", "services:\n  api:\n    image: example\n")

        pass_result = verify_canonical_terminology(
            _context(pass_root, _canonical_spec()), _item("13.15")
        )

        _assert_pass_evidence(
            pass_result,
            "§0.3/§13.15",
            "scanned 4 code/config files",
            "no retired terminology matched",
            "parsed §0.3/prose patterns",
            "excluded 1 broad §0.3 terms",
        )

        fail_root = tmp_path / "fail"
        _write_canonical_scan_fixture(fail_root, retired_line="GPU worker")

        fail_result = verify_canonical_terminology(
            _context(fail_root, _canonical_spec()), _item("13.15")
        )

        _assert_fail_evidence(
            fail_result,
            "§0.3/§13.15",
            "services/bad.py:1 matched 'GPU worker'",
            "COMMENT = 'GPU worker'",
        )

        parsed_spec_term_root = tmp_path / "parsed_spec_term"
        _write_canonical_scan_fixture(parsed_spec_term_root, retired_line="video pipe")

        parsed_spec_term_result = verify_canonical_terminology(
            _context(parsed_spec_term_root, _canonical_spec()), _item("13.15")
        )

        _assert_fail_evidence(
            parsed_spec_term_result,
            "§0.3/§13.15",
            "services/bad.py:1 matched 'video pipe'",
            "retired synonym for IPC Pipe: 'video pipe'",
        )

    def test_verify_canonical_terminology_scans_config_surfaces(self, tmp_path: Path) -> None:
        config_root = tmp_path / "config_surface"
        _write_canonical_scan_fixture(config_root)
        _write(config_root, ".github/workflows/ci.yml", "name: GPU worker gate\n")

        config_result = verify_canonical_terminology(
            _context(config_root, _canonical_spec()), _item("13.15")
        )

        _assert_fail_evidence(
            config_result,
            "§0.3/§13.15",
            ".github/workflows/ci.yml:1 matched 'GPU worker'",
            "code/config files",
        )

    def test_verify_semantic_reason_codes_pass_and_threshold_mismatch(self, tmp_path: Path) -> None:
        pass_root = tmp_path / "pass"
        _write_reason_fixture(pass_root)

        pass_result = verify_semantic_reason_codes(
            _context(pass_root, _reason_spec()), _item("13.27")
        )

        _assert_pass_evidence(
            pass_result,
            "§8.3/§13.27",
            "packages/schemas/evaluation.py:1",
            "SEMANTIC_REASON_CODES",
            "§8.2.1/§13.27",
            "GRAY_BAND_LOWER_THRESHOLD=0.58",
            "MATCH_THRESHOLD=0.72",
            "fallback branch is bounded",
            "lower_inclusive=True",
            "upper_exclusive=True",
        )

        fail_root = tmp_path / "fail"
        _write_reason_fixture(fail_root, lower_threshold=0.57)

        fail_result = verify_semantic_reason_codes(
            _context(fail_root, _reason_spec()), _item("13.27")
        )

        _assert_fail_evidence(
            fail_result,
            "§8.2.1/§13.27",
            "GRAY_BAND_LOWER_THRESHOLD=0.57",
            "expected lower bound 0.58",
        )

    def test_verify_semantic_reason_codes_fails_unbounded_fallback_branch(
        self, tmp_path: Path
    ) -> None:
        for branch_mode, missing_evidence in (
            ("flag_only", "lower_inclusive=False"),
            ("missing_lower", "lower_inclusive=False"),
            ("missing_upper", "upper_exclusive=False"),
        ):
            root = tmp_path / branch_mode
            _write_reason_fixture(root, branch_mode=branch_mode)

            result = verify_semantic_reason_codes(_context(root, _reason_spec()), _item("13.27"))

            _assert_fail_evidence(
                result,
                "§8.1/§8.2.1/§13.27",
                "fallback branch must require the feature flag",
                missing_evidence,
            )

    def test_check_sh_delegates_audit_gate_to_strict_harness(self) -> None:
        check_script = Path("scripts/check.sh").read_text(encoding="utf-8")
        audit_section = check_script.split("── §13 audit harness ──", maxsplit=1)[1]
        audit_section = audit_section.split("── Docker compose config ──", maxsplit=1)[0]

        assert "python scripts/run_audit.py --strict" in audit_section
        assert "--item" not in audit_section
        assert "grep -R" not in audit_section

    def test_verify_semantic_reason_codes_fails_mixed_bounded_and_unbounded_fallback_paths(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "mixed"
        _write_reason_fixture(root, branch_mode="mixed")

        result = verify_semantic_reason_codes(_context(root, _reason_spec()), _item("13.27"))

        _assert_fail_evidence(
            result,
            "§8.1/§8.2.1/§13.27",
            "non-compliant fallback branches",
            "feature=False",
            "lower_inclusive=False",
            "upper_exclusive=False",
            "observed all fallback branches",
            "feature=True lower_inclusive=True upper_exclusive=True",
        )

    def test_verify_semantic_reason_codes_fails_disjunctive_feature_gate_bypass(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "disjunctive_bypass"
        _write_reason_fixture(root, branch_mode="disjunctive_bypass")

        result = verify_semantic_reason_codes(_context(root, _reason_spec()), _item("13.27"))

        _assert_fail_evidence(
            result,
            "§8.1/§8.2.1/§13.27",
            "fallback branch must require the feature flag",
            "gray_band_fallback_enabled",
            "bypass_flag",
            "feature=False",
            "lower_inclusive=True",
            "upper_exclusive=True",
        )

    # ---- §13.1 — Directory structure ----------------------------------------

    def test_verify_directory_structure_pass_and_missing_path(self, tmp_path: Path) -> None:
        spec = {
            "codebase_architecture": {
                "directory_hierarchy": [
                    {"path": "/services/api/", "purpose": "API"},
                    {"path": "/packages/schemas/", "purpose": "Schemas"},
                    {"path": "/docker-compose.yml", "purpose": "Topology"},
                ]
            }
        }
        good = tmp_path / "good"
        (good / "services" / "api").mkdir(parents=True)
        (good / "packages" / "schemas").mkdir(parents=True)
        (good / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

        result = verify_directory_structure(_context(good, spec), _item("13.1"))
        _assert_pass_evidence(
            result,
            "§3/§13.1: extracted 3 directory_hierarchy entries",
            "services/api directory present",
            "docker-compose.yml file present",
        )

        bad = tmp_path / "bad"
        (bad / "services" / "api").mkdir(parents=True)
        # packages/schemas is missing on purpose.
        (bad / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

        result_fail = verify_directory_structure(_context(bad, spec), _item("13.1"))
        _assert_fail_evidence(
            result_fail,
            "packages/schemas directory MISSING",
        )

    def test_verify_directory_structure_fails_closed_without_spec(self, tmp_path: Path) -> None:
        result = verify_directory_structure(_context(tmp_path, {}), _item("13.1"))
        assert result.passed is False
        assert "missing codebase_architecture object" in result.evidence

    # ---- §13.2 — Docker topology --------------------------------------------

    def test_verify_docker_topology_pass_and_missing_service(self, tmp_path: Path) -> None:
        spec = {
            "docker_topology": {
                "containers": [
                    {
                        "container_name": "api",
                        "image_base": "python:3.11-slim",
                        "network": "appnetwork",
                        "restart_policy": "unless-stopped",
                        "depends_on": ["postgres"],
                        "gpu_required": False,
                    },
                    {
                        "container_name": "worker",
                        "image_base": "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
                        "network": "appnetwork",
                        "restart_policy": "on-failure:5",
                        "depends_on": ["postgres"],
                        "gpu_required": True,
                    },
                ],
                "volumes": [
                    {
                        "volume_name": "ipc-share",
                        "mount_path": "/tmp/ipc/",
                        "container_targets": ["worker"],
                    }
                ],
            }
        }
        good_compose = """\
services:
  api:
    image: python:3.11-slim
    networks:
      - appnetwork
    restart: unless-stopped
    depends_on:
      - postgres
  worker:
    build:
      context: .
    networks:
      - appnetwork
    restart: on-failure
    deploy:
      restart_policy:
        condition: on-failure
        max_attempts: 5
    depends_on:
      - postgres
    volumes:
      - ipc-share:/tmp/ipc/
volumes:
  ipc-share: {}
networks:
  appnetwork:
    driver: bridge
"""
        good = tmp_path / "good"
        good.mkdir()
        (good / "docker-compose.yml").write_text(good_compose, encoding="utf-8")

        result = verify_docker_topology(_context(good, spec), _item("13.2"))
        _assert_pass_evidence(
            result,
            "§9/§13.2: extracted 2 containers and 1 volumes",
            "service api image='python:3.11-slim' matches spec",
            "service worker mounts volume 'ipc-share' at '/tmp/ipc/'",
        )

        bad_compose = good_compose.replace("worker:", "worker_misnamed:")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "docker-compose.yml").write_text(bad_compose, encoding="utf-8")

        result_fail = verify_docker_topology(_context(bad, spec), _item("13.2"))
        _assert_fail_evidence(
            result_fail,
            "service 'worker' is missing from docker-compose.yml",
        )

    def test_verify_docker_topology_missing_compose_fails_closed(self, tmp_path: Path) -> None:
        spec = {
            "docker_topology": {
                "containers": [
                    {
                        "container_name": "api",
                        "image_base": "python:3.11-slim",
                        "network": "appnetwork",
                        "restart_policy": "unless-stopped",
                        "depends_on": [],
                    }
                ],
                "volumes": [],
            }
        }
        empty = tmp_path / "no_compose"
        empty.mkdir()
        result = verify_docker_topology(_context(empty, spec), _item("13.2"))
        _assert_fail_evidence(result, "docker-compose.yml is missing")

    # ---- §13.3 — IPC lifecycle ----------------------------------------------

    def test_verify_ipc_lifecycle_pass_and_missing_token(self, tmp_path: Path) -> None:
        spec = {
            "core_modules": {
                "modules": [
                    {
                        "module_id": "A",
                        "ipc_lifecycle_steps": [
                            {
                                "step_number": 1,
                                "title": "Init",
                                "description": (
                                    "<func>setup_pipes()</func> creates "
                                    "<path>/tmp/ipc/audio_stream.raw</path>."
                                ),
                            },
                            {
                                "step_number": 2,
                                "title": "Wait",
                                "description": (
                                    "<func>wait_for_device()</func> polls <cmd>adb devices</cmd>."
                                ),
                            },
                        ],
                    }
                ]
            }
        }
        good = tmp_path / "good"
        _write(
            good,
            "services/stream_ingest/entrypoint.sh",
            "#!/usr/bin/env bash\n"
            'AUDIO_PIPE="/tmp/ipc/audio_stream.raw"\n'
            'setup_pipes() { mkfifo "$AUDIO_PIPE"; }\n'
            "wait_for_device() { adb devices; }\n",
        )
        result = verify_ipc_lifecycle(_context(good, spec), _item("13.3"))
        _assert_pass_evidence(
            result,
            "§4.A/§13.3: extracted 2 IPC lifecycle steps",
            "step 1 (Init): all tokens present",
            "step 2 (Wait): all tokens present",
        )

        bad = tmp_path / "bad"
        _write(
            bad,
            "services/stream_ingest/entrypoint.sh",
            "#!/usr/bin/env bash\nsetup_pipes() { :; }\nwait_for_device() { :; }\n",
        )
        result_fail = verify_ipc_lifecycle(_context(bad, spec), _item("13.3"))
        _assert_fail_evidence(
            result_fail,
            "step 1 (Init): services/stream_ingest/entrypoint.sh missing tokens",
            "/tmp/ipc/audio_stream.raw",
        )

    def test_verify_ipc_lifecycle_missing_entrypoint_fails_closed(self, tmp_path: Path) -> None:
        spec = {
            "core_modules": {
                "modules": [
                    {
                        "module_id": "A",
                        "ipc_lifecycle_steps": [
                            {
                                "step_number": 1,
                                "title": "Init",
                                "description": "<path>/tmp/x</path>",
                            }
                        ],
                    }
                ]
            }
        }
        empty = tmp_path / "no_entrypoint"
        empty.mkdir()
        result = verify_ipc_lifecycle(_context(empty, spec), _item("13.3"))
        _assert_fail_evidence(result, "services/stream_ingest/entrypoint.sh is missing")

    # ---- §13.5 — Drift correction -------------------------------------------

    def test_verify_drift_correction_pass_and_wrong_constant(self, tmp_path: Path) -> None:
        spec = {
            "core_modules": {
                "modules": [
                    {
                        "module_id": "C",
                        "drift_polling_parameters": [
                            {"parameter": "Polling interval", "value": "30 seconds"},
                            {
                                "parameter": "Maximum tolerated drift",
                                "value": "150 milliseconds",
                            },
                            {"parameter": "Freeze threshold", "value": "3 consecutive failures"},
                            {"parameter": "Reset timeout", "value": "300 seconds"},
                            {
                                "parameter": "Drift formula",
                                "value": "drift_offset = host_utc - android_epoch",
                            },
                        ],
                    }
                ]
            }
        }
        good = tmp_path / "good"
        _write(
            good,
            "services/worker/pipeline/orchestrator.py",
            "from __future__ import annotations\n"
            "DRIFT_POLL_INTERVAL: int = 30\n"
            "MAX_TOLERATED_DRIFT_MS: int = 150\n"
            "DRIFT_FREEZE_AFTER_FAILURES: int = 3\n"
            "DRIFT_RESET_TIMEOUT: int = 300\n"
            "def correct(host_utc: float, android_epoch: float) -> float:\n"
            "    drift_offset = host_utc - android_epoch\n"
            "    return drift_offset\n",
        )
        result = verify_drift_correction(_context(good, spec), _item("13.5"))
        _assert_pass_evidence(
            result,
            "§4.C/§13.5: extracted drift_polling_parameters",
            "DRIFT_POLL_INTERVAL=30; expected 30",
            "DRIFT_RESET_TIMEOUT=300; expected 300",
        )

        bad = tmp_path / "bad"
        _write(
            bad,
            "services/worker/pipeline/orchestrator.py",
            "DRIFT_POLL_INTERVAL: int = 60\n"
            "MAX_TOLERATED_DRIFT_MS: int = 150\n"
            "DRIFT_FREEZE_AFTER_FAILURES: int = 3\n"
            "DRIFT_RESET_TIMEOUT: int = 300\n"
            "def correct(host_utc: float, android_epoch: float) -> float:\n"
            "    drift_offset = host_utc - android_epoch\n"
            "    return drift_offset\n",
        )
        result_fail = verify_drift_correction(_context(bad, spec), _item("13.5"))
        _assert_fail_evidence(
            result_fail,
            "DRIFT_POLL_INTERVAL=60",
            "expected 30",
        )

    # ---- §13.9 — Schema validation -------------------------------------------

    def test_verify_schema_validation_pass_and_missing_field(self, tmp_path: Path) -> None:
        schema_source = json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "InferenceHandoffPayload",
                "required": ["session_id", "_active_arm", "_x_max"],
                "properties": {
                    "session_id": {"type": "string"},
                    "_active_arm": {"type": "string"},
                    "_x_max": {
                        "type": ["number", "null"],
                        "description": (
                            "Calibration telemetry; reward computation does not divide P90 by it."
                        ),
                    },
                },
            }
        )
        spec = {"interface_contracts": {"schema_definition": {"source": schema_source}}}

        good = tmp_path / "good"
        _write(
            good,
            "packages/schemas/inference_handoff.py",
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, ConfigDict, Field\n"
            "class InferenceHandoffPayload(BaseModel):\n"
            "    session_id: str\n"
            "    active_arm: str = Field(..., alias='_active_arm')\n"
            "    model_config = ConfigDict(extra='forbid')\n",
        )
        _write(
            good,
            "services/worker/pipeline/orchestrator.py",
            "from packages.schemas.inference_handoff import InferenceHandoffPayload\n"
            "def dispatch(p):\n"
            "    return InferenceHandoffPayload(**p)\n",
        )
        result = verify_schema_validation(_context(good, spec), _item("13.9"))
        _assert_pass_evidence(
            result,
            "InferenceHandoffPayload subclasses pydantic BaseModel",
            "model_config extra='forbid'",
            "deprecated/diagnostic carve-outs accepted: ['_x_max']",
            "validated at dispatch boundary",
        )

        bad = tmp_path / "bad"
        _write(
            bad,
            "packages/schemas/inference_handoff.py",
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, ConfigDict\n"
            "class InferenceHandoffPayload(BaseModel):\n"
            "    session_id: str\n"
            "    model_config = ConfigDict(extra='forbid')\n",
        )
        _write(
            bad,
            "services/worker/pipeline/orchestrator.py",
            "from packages.schemas.inference_handoff import InferenceHandoffPayload\n"
            "def dispatch(p):\n"
            "    return InferenceHandoffPayload(**p)\n",
        )
        result_fail = verify_schema_validation(_context(bad, spec), _item("13.9"))
        _assert_fail_evidence(
            result_fail,
            "missing required fields ['_active_arm']",
        )

    def test_verify_schema_validation_missing_module_fails_closed(self, tmp_path: Path) -> None:
        spec = {
            "interface_contracts": {
                "schema_definition": {
                    "source": json.dumps(
                        {
                            "title": "InferenceHandoffPayload",
                            "required": ["session_id"],
                            "properties": {"session_id": {"type": "string"}},
                        }
                    )
                }
            }
        }
        empty = tmp_path / "no_schema"
        empty.mkdir()
        result = verify_schema_validation(_context(empty, spec), _item("13.9"))
        _assert_fail_evidence(result, "packages/schemas/inference_handoff.py is missing")

    # ---- §13.10 — Module contracts -------------------------------------------

    def test_verify_module_contracts_pass_and_missing_token(self, tmp_path: Path) -> None:
        spec = {
            "core_modules": {
                "modules": [
                    {
                        "module_id": "A",
                        "contract": {
                            "outputs": "<path>/tmp/ipc/audio_stream.raw</path>",
                            "failure_modes": "<func>wait_for_device()</func>",
                        },
                    },
                    {
                        "module_id": "C",
                        "contract": {
                            "outputs": "<class>InferenceHandoffPayload</class>",
                        },
                    },
                ]
            }
        }

        good = tmp_path / "good"
        _write(
            good,
            "services/stream_ingest/entrypoint.sh",
            'AUDIO_PIPE="/tmp/ipc/audio_stream.raw"\nwait_for_device() { :; }\n',
        )
        _write(
            good,
            "packages/schemas/inference_handoff.py",
            "class InferenceHandoffPayload:\n    pass\n",
        )
        # Module C scans pipeline/, schemas/, ml_core/ — the schemas hit covers it.
        (good / "services" / "worker" / "pipeline").mkdir(parents=True)
        (good / "packages" / "ml_core").mkdir(parents=True)

        result = verify_module_contracts(_context(good, spec), _item("13.10"))
        _assert_pass_evidence(
            result,
            "§4/§13.10: extracted",
            "module A contract.outputs <path>/tmp/ipc/audio_stream.raw</path>: present",
            "module A contract.failure_modes <func>wait_for_device()</func>: present",
            "module C contract.outputs <class>InferenceHandoffPayload</class>: present",
        )

        bad = tmp_path / "bad"
        # entrypoint.sh missing the audio path.
        _write(
            bad,
            "services/stream_ingest/entrypoint.sh",
            "wait_for_device() { :; }\n",
        )
        _write(
            bad,
            "packages/schemas/inference_handoff.py",
            "class InferenceHandoffPayload:\n    pass\n",
        )
        (bad / "services" / "worker" / "pipeline").mkdir(parents=True)
        (bad / "packages" / "ml_core").mkdir(parents=True)

        result_fail = verify_module_contracts(_context(bad, spec), _item("13.10"))
        _assert_fail_evidence(
            result_fail,
            "module A contract.outputs <path>/tmp/ipc/audio_stream.raw</path>: MISSING",
        )

    def test_verify_module_contracts_documents_x_max_carve_out(self, tmp_path: Path) -> None:
        spec = {
            "core_modules": {
                "modules": [
                    {
                        "module_id": "C",
                        "contract": {"outputs": "<field>_x_max</field>"},
                    }
                ]
            }
        }
        # No source declares _x_max, but the spec carve-out should still PASS.
        empty = tmp_path / "empty"
        (empty / "services" / "worker" / "pipeline").mkdir(parents=True)
        (empty / "packages" / "schemas").mkdir(parents=True)
        (empty / "packages" / "ml_core").mkdir(parents=True)
        result = verify_module_contracts(_context(empty, spec), _item("13.10"))
        _assert_pass_evidence(
            result,
            "<field>_x_max</field>: documented carve-out",
        )
