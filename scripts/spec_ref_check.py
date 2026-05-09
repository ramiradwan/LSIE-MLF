#!/usr/bin/env python3
"""
spec_ref_check.py — Resolve, index, and validate spec references
================================================================

Auto-generates a spec_ref index from the content.json payload embedded in the
currently committed tech-spec PDF, then scans the project for all spec_ref usage
and reports which refs resolve, which don't, which indexed content paths are
broken, and which content paths have no ref pointing at them.

The content.json can be loaded from three sources (tried in order):
  1. An explicit PDF passed with --from-pdf
  2. An explicit standalone content.json file passed with --content
  3. The single docs/tech-spec-v*.pdf file discovered in the repo root

The index is built in two layers:
  1. STRUCTURAL (auto-generated): Walks the payload and infers refs from
     section_number fields, module_id letters, subsection numbers, array
     indices mapped to conventional numbering, and math topic structure.
  2. EXPLICIT (from schema): If sections contain spec_refs arrays, those
     override structural heuristics.

This script also validates that indexed content_path targets actually resolve
against the loaded content payload, including selector-style paths such as:
  - core_modules.modules[module_id=B]
  - core_modules.modules[module_id=B].subsections[number=4.B.2]
  - error_handling.matrix[failure_category=Physiological Ingestion]

Usage:
    python scripts/spec_ref_check.py --index
    python scripts/spec_ref_check.py --validate
    python scripts/spec_ref_check.py --resolve "7A.4"
    python scripts/spec_ref_check.py --resolve "4.A.1"
    python scripts/spec_ref_check.py --extract > docs/content.json
    python scripts/spec_ref_check.py --from-pdf docs/tech-spec-v*.pdf --extract > docs/content.json

Dependencies: PyMuPDF (fitz) is used for PDF attachment extraction when
available. A built-in FlateDecode fallback handles the committed spec PDF in
environments where PyMuPDF is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast


# =====================================================================
# Content Source: PDF Extraction
# =====================================================================


def extract_content_from_pdf(pdf_path: Path) -> dict[str, Any]:
    """Extract content.json from a PDF/A-3 meta-attachment."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _extract_with_stream_scan() -> dict[str, Any]:
        """Fallback extractor for the committed spec PDF when PyMuPDF is absent."""
        import zlib

        raw_pdf = pdf_path.read_bytes()
        errors: list[str] = []

        for marker in re.finditer(rb"/Type\s*/EmbeddedFile", raw_pdf):
            stream_marker = raw_pdf.find(b"stream", marker.end())
            endobj_marker = raw_pdf.find(b"endobj", marker.end())
            if stream_marker == -1 or (endobj_marker != -1 and stream_marker > endobj_marker):
                continue

            header_start = raw_pdf.rfind(b"<<", 0, marker.start())
            if header_start == -1:
                header_start = marker.start()
            header = raw_pdf[header_start:stream_marker]
            if b"application#2Fjson" not in header and b"application/json" not in header:
                continue

            length_match = re.search(rb"/Length\s+(\d+)", header)
            if not length_match:
                errors.append("embedded JSON stream is missing /Length")
                continue

            stream_start = stream_marker + len(b"stream")
            if raw_pdf[stream_start:stream_start + 2] == b"\r\n":
                stream_start += 2
            elif raw_pdf[stream_start:stream_start + 1] == b"\n":
                stream_start += 1

            length = int(length_match.group(1))
            raw_bytes = raw_pdf[stream_start:stream_start + length]
            try:
                if b"/FlateDecode" in header:
                    raw_bytes = zlib.decompress(raw_bytes)
                loaded = json.loads(raw_bytes.decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("embedded JSON content is not an object")
                return cast(dict[str, Any], loaded)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, zlib.error) as exc:
                errors.append(str(exc))

        detail = f" Errors: {'; '.join(errors)}" if errors else ""
        raise ValueError(f"No embedded JSON content stream found in {pdf_path}.{detail}")

    try:
        import fitz
    except ImportError:
        return _extract_with_stream_scan()

    doc = fitz.open(str(pdf_path))
    try:
        try:
            names = doc.embfile_names()
        except Exception as exc:
            raise ValueError(f"Could not read embedded files from {pdf_path}") from exc

        target_name: str | None = None
        for name in names:
            # Get the metadata for the internal ID (e.g., 'l3ef0001')
            info = doc.embfile_info(name)
            # Check both the standard and unicode filename metadata fields
            real_name = info.get("filename", "") or info.get("ufilename", "") or name

            if "content" in real_name.lower() and real_name.lower().endswith(".json"):
                target_name = name
                break

        if target_name is None:
            raise ValueError(
                f"No content.json attachment found in {pdf_path}. "
                f"Available attachments: {names}"
            )

        raw_bytes = doc.embfile_get(target_name)
    finally:
        doc.close()

    try:
        loaded = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("embedded content.json is not an object")
        return cast(dict[str, Any], loaded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse {target_name} from PDF: {exc}") from exc


# =====================================================================
# Content Source: File Discovery
# =====================================================================


def find_content_json(repo_root: Path) -> Path | None:
    """Locate content.json in the repo."""
    for candidate in [
        repo_root / "docs" / "content.json",
        repo_root / "content.json",
    ]:
        if candidate.is_file():
            return candidate
    return None


def load_content(
    pdf_path: Path | None = None,
    content_path: Path | None = None,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Load content.json from an explicit source or the single committed spec PDF."""
    if pdf_path:
        return extract_content_from_pdf(pdf_path)

    if content_path:
        if not content_path.is_file():
            raise FileNotFoundError(f"content.json not found: {content_path}")
        with open(content_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"content.json is not an object: {content_path}")
        return cast(dict[str, Any], loaded)

    spec_pdfs = sorted((repo_root / "docs").glob("tech-spec-v*.pdf"))
    if len(spec_pdfs) == 1:
        return extract_content_from_pdf(spec_pdfs[0])

    if not spec_pdfs:
        raise FileNotFoundError(
            f"No spec PDF found. Expected exactly one match for "
            f"{repo_root}/docs/tech-spec-v*.pdf, or pass --from-pdf/--content."
        )

    matches = ", ".join(str(path.relative_to(repo_root)) for path in spec_pdfs)
    raise FileExistsError(
        f"Expected exactly one docs/tech-spec-v*.pdf match; found {len(spec_pdfs)}: "
        f"{matches}"
    )


# =====================================================================
# Index Entry
# =====================================================================


class _IndexEntry:
    __slots__ = ("title", "content_path", "content_type", "preview", "explicit")

    def __init__(
        self,
        title: str,
        content_path: str,
        content_type: str = "",
        preview: str = "",
        explicit: bool = False,
    ) -> None:
        self.title = title
        self.content_path = content_path
        self.content_type = content_type
        self.preview = preview
        self.explicit = explicit

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content_path": self.content_path,
            "content_type": self.content_type,
            "preview": self.preview,
            "explicit": self.explicit,
        }


# =====================================================================
# Layer 1: Structural Index
# =====================================================================

_ROOT_KEY_TO_SECTION: dict[str, str] = {
    "document_control": "0",
    "conformance_gates": "0.5",
    "system_philosophy": "1",
    "data_flow_pipeline": "2",
    "codebase_architecture": "3",
    "core_modules": "4",
    "data_governance": "5",
    "interface_contracts": "6",
    "math_specifications": "7",
    "llm_prompt": "8",
    "docker_topology": "9",
    "runtime_topology": "9",
    "dependency_matrix": "10",
    "variable_matrix": "11",
    "error_handling": "12",
    "audit_checklist": "13",
    "cloud_control_plane": "14",
}

_SUBSECTION_CONVENTIONS: dict[str, list[tuple[str, str]]] = {
    "document_control": [
        ("purpose_prose", "0.1"),
        ("normative_terms", "0.2"),
        ("canonical_terms", "0.3"),
        ("version_history", "0.4"),
    ],
    "codebase_architecture": [
        ("directory_hierarchy", "3.1"),
        ("build_context_rules", "3.2"),
    ],
    "data_governance": [
        ("vault_parameters", "5.1"),
        ("data_classifications", "5.2"),
        ("anonymization_rule", "5.3"),
    ],
    "interface_contracts": [("schema_definition", "6.1")],
    "llm_prompt": [
        ("inference_parameters", "8.1"),
        ("output_schema", "8.2"),
        ("system_prompt", "8.3"),
    ],
    "docker_topology": [
        ("containers", "9.1"),
        ("volumes", "9.2"),
        ("gpu_allocation_prose", "9.3"),
        ("device_passthrough_prose", "9.4"),
        ("network_prose", "9.5"),
        ("startup_order", "9.6"),
    ],
    "runtime_topology": [
        ("processes", "9.1"),
        ("ipc_contract", "9.2"),
        ("native_subprocess_supervision", "9.3"),
        ("runtime_base", "9.4"),
        ("startup_sequence", "9.5"),
        ("tier2_design_intent", "9.6"),
    ],
    "dependency_matrix": [
        ("system_requirements", "10.1"),
        ("pinned_packages", "10.2"),
    ],
    "audit_checklist": [("items", "13.1")],
}

# Math topic_id -> uppercase letter. Content.json uses 7A/7B/7C, not 7.au12.
_MATH_TOPIC_LETTER: dict[str, str] = {
    "au12": "A",
    "thompson_sampling": "B",
    "comodulation": "C",
}

_LEGACY_DATA_GOVERNANCE_ALIASES: dict[str, tuple[str, str]] = {
    "5.1.6": ("Vault parameter — Key storage location", "data_governance.secret_storage_rule"),
    "5.1.7": ("Vault parameter — Key lifecycle", "data_governance.volatile_memory_controls"),
    "5.1.8": ("Vault parameter — Secure deletion method", "data_governance.volatile_memory_controls"),
    "5.1.9": ("Vault parameter — Retention period", "data_governance.vault_parameters"),
}

_LEGACY_ERROR_HANDLING_ALIASES: dict[str, tuple[str, str]] = {
    "12.1": ("Legacy Module A failure handling alias", "error_handling.matrix"),
    "12.1.1": ("Legacy Module A — Network Disconnection alias", "error_handling.matrix"),
    "12.1.2": ("Legacy Module A — Hardware Device Loss alias", "error_handling.matrix[0]"),
    "12.1.3": ("Legacy Module A — Worker Process Crash alias", "error_handling.matrix[1]"),
    "12.1.4": ("Legacy Module A — Queue Overload alias", "error_handling.matrix"),
    "12.2": ("Legacy Module B failure handling alias", "error_handling.matrix"),
    "12.2.1": ("Legacy Module B — Network Disconnection alias", "error_handling.matrix"),
    "12.2.2": ("Legacy Module B — Hardware Device Loss alias", "error_handling.matrix"),
    "12.2.3": ("Legacy Module B — Worker Process Crash alias", "error_handling.matrix"),
    "12.2.4": ("Legacy Module B — Queue Overload alias", "error_handling.matrix"),
    "12.3": ("Legacy Module C failure handling alias", "error_handling.matrix[module=module_c_orchestrator]"),
    "12.3.1": ("Legacy Module C — Network Disconnection alias", "error_handling.matrix[module=module_c_orchestrator]"),
    "12.3.2": ("Legacy Module C — Hardware Device Loss alias", "error_handling.matrix[module=module_c_orchestrator]"),
    "12.3.3": ("Legacy Module C — Worker Process Crash alias", "error_handling.matrix[module=module_c_orchestrator]"),
    "12.3.4": ("Legacy Module C — Queue Overload alias", "error_handling.matrix[module=module_c_orchestrator]"),
    "12.4": ("Legacy Module D failure handling alias", "error_handling.matrix[module=gpu_ml_worker]"),
    "12.4.1": ("Legacy Module D — Network Disconnection alias", "error_handling.matrix[module=gpu_ml_worker]"),
    "12.4.2": ("Legacy Module D — Hardware Device Loss alias", "error_handling.matrix[module=gpu_ml_worker]"),
    "12.4.3": ("Legacy Module D — Worker Process Crash alias", "error_handling.matrix[module=gpu_ml_worker]"),
    "12.4.4": ("Legacy Module D — Queue Overload alias", "error_handling.matrix[module=gpu_ml_worker]"),
    "12.5": ("Legacy Module E failure handling alias", "error_handling.matrix[module=analytics_state_worker]"),
    "12.5.1": ("Legacy Module E — Network Disconnection alias", "error_handling.matrix[module=analytics_state_worker]"),
    "12.5.2": ("Legacy Module E — Hardware Device Loss alias", "error_handling.matrix[module=analytics_state_worker]"),
    "12.5.3": ("Legacy Module E — Worker Process Crash alias", "error_handling.matrix[module=analytics_state_worker]"),
    "12.5.4": ("Legacy Module E — Queue Overload alias", "error_handling.matrix[module=analytics_state_worker]"),
    "12.6": ("Legacy Module F failure handling alias", "error_handling.matrix[module=cloud_sync_worker]"),
    "12.6.1": ("Legacy Module F — Network Disconnection alias", "error_handling.matrix[module=cloud_sync_worker]"),
    "12.6.2": ("Legacy Module F — Hardware Device Loss alias", "error_handling.matrix[module=cloud_sync_worker]"),
    "12.6.3": ("Legacy Module F — Worker Process Crash alias", "error_handling.matrix[module=cloud_sync_worker]"),
    "12.6.4": ("Legacy Module F — Queue Overload alias", "error_handling.matrix[module=cloud_sync_worker]"),
}

_LEGACY_AUDIT_ALIASES: dict[str, tuple[str, str]] = {
    "13.20": ("Co-modulation determinism", "math_specifications.topics[2]"),
    "13.21": ("Reward pipeline invariance", "math_specifications.topics[1]"),
    "13.22": ("Observational acoustic validity contract", "math_specifications.topics[3]"),
    "13.23": ("Observational acoustic payload/storage conformance", "data_governance.local_state.forbidden_content"),
    "13.24": ("Attribution ledger schema conformance", "interface_contracts.attribution_event_schema"),
    "13.25": ("Stable segment identity", "interface_contracts.schema_definition"),
    "13.26": ("Bandit decision snapshot capture", "interface_contracts.bandit_decision_snapshot_schema"),
    "13.27": ("Semantic scorer shadow/primary separation", "llm_prompt.intro_paragraphs"),
    "13.28": ("Attribution backfill idempotency", "math_specifications.topics[4].derivation_steps[5]"),
    "13.29": ("Outcome linker determinism", "math_specifications.topics[4].derivation_steps[1]"),
    "13.30": ("Derived-only attribution persistence compliance", "data_governance.local_state.forbidden_content"),
    "13.31": ("Attribution reward-path invariance", "math_specifications.topics[4]"),
}


def _preview(obj: Any, max_len: int = 80) -> str:
    if isinstance(obj, str):
        return obj[:max_len]
    if isinstance(obj, list):
        return f"[{len(obj)} items]"
    if isinstance(obj, dict):
        keys = list(obj.keys())[:5]
        return f"{{{', '.join(keys)}, ...}}" if len(obj) > 5 else f"{{{', '.join(keys)}}}"
    return str(obj)[:max_len]


def build_index(spec: dict[str, Any]) -> dict[str, _IndexEntry]:
    """Build the complete spec_ref index from content.json."""
    index: dict[str, _IndexEntry] = {}

    for key, section_num in _ROOT_KEY_TO_SECTION.items():
        section = spec.get(key)
        if section is None:
            continue
        title = (
            section.get("section_title", key.replace("_", " ").title())
            if isinstance(section, dict) else key
        )
        index[section_num] = _IndexEntry(
            title=f"§{section_num} — {title}",
            content_path=key,
            content_type="section",
            preview=_preview(section),
        )

    for key, mappings in _SUBSECTION_CONVENTIONS.items():
        section = spec.get(key)
        if not isinstance(section, dict):
            continue
        for field_name, ref in mappings:
            content = section.get(field_name)
            if content is not None:
                index[ref] = _IndexEntry(
                    title=f"§{ref} — {field_name}",
                    content_path=f"{key}.{field_name}",
                    content_type=type(content).__name__,
                    preview=_preview(content),
                )

    _index_math_section(spec, index)
    _index_core_modules(spec, index)
    _index_error_handling(spec, index)
    _index_data_flow_stages(spec, index)
    _overlay_explicit_refs(spec, index)
    _add_legacy_aliases(index)

    return index


def _index_math_section(spec: dict[str, Any], index: dict[str, _IndexEntry]) -> None:
    """Index math specs using the 7A/7B/7C letter convention."""
    math = spec.get("math_specifications", {})
    if not isinstance(math, dict):
        return

    topics = math.get("topics")
    if topics and isinstance(topics, list):
        for ti, topic in enumerate(topics):
            topic_id = topic.get("topic_id", f"topic_{ti}")
            topic_title = topic.get("topic_title", topic_id)
            letter = _MATH_TOPIC_LETTER.get(topic_id, chr(ord("A") + ti))
            topic_ref = f"7{letter}"

            index[topic_ref] = _IndexEntry(
                title=f"§{topic_ref} — {topic_title}",
                content_path=f"math_specifications.topics[{ti}]",
                content_type="MathTopic",
                preview=topic_title,
            )
            if topic.get("variable_dictionary"):
                index[f"7{letter}.1"] = _IndexEntry(
                    title=f"§7{letter}.1 — Variable Dictionary ({topic_title})",
                    content_path=f"math_specifications.topics[{ti}].variable_dictionary",
                    content_type="list[VariableDictionaryEntry]",
                    preview=f"[{len(topic['variable_dictionary'])} entries]",
                )
            for si, step in enumerate(topic.get("derivation_steps", [])):
                step_ref = f"7{letter}.{si + 2}"
                index[step_ref] = _IndexEntry(
                    title=f"§{step_ref} — {step.get('title', f'Step {si + 1}')}",
                    content_path=f"math_specifications.topics[{ti}].derivation_steps[{si}]",
                    content_type="MathStep",
                    preview=step.get("title", ""),
                )
            if topic.get("reference_implementation"):
                n_steps = len(topic.get("derivation_steps", []))
                impl_ref = f"7{letter}.{n_steps + 2}"
                index[impl_ref] = _IndexEntry(
                    title=f"§{impl_ref} — Reference Implementation ({topic_title})",
                    content_path=f"math_specifications.topics[{ti}].reference_implementation",
                    content_type="CodeBlock",
                    preview="(Python reference implementation)",
                )
        return

    # Legacy v2 flat fields
    steps = math.get("derivation_steps", [])
    if not steps:
        return
    if math.get("variable_dictionary"):
        index["7.1"] = _IndexEntry(
            title="§7.1 — Variable Dictionary",
            content_path="math_specifications.variable_dictionary",
            content_type="list[VariableDictionaryEntry]",
        )
    for i, step in enumerate(steps):
        ref = f"7.{i + 2}"
        index[ref] = _IndexEntry(
            title=f"§{ref} — {step.get('title', f'Step {i + 1}')}",
            content_path=f"math_specifications.derivation_steps[{i}]",
            content_type="MathStep",
            preview=step.get("title", ""),
        )
    if math.get("reference_implementation"):
        ref = f"7.{len(steps) + 2}"
        index[ref] = _IndexEntry(
            title=f"§{ref} — Reference Implementation",
            content_path="math_specifications.reference_implementation",
            content_type="CodeBlock",
        )


def _index_core_modules(spec: dict[str, Any], index: dict[str, _IndexEntry]) -> None:
    modules = spec.get("core_modules", {}).get("modules", [])
    for mod in modules:
        mid = mod.get("module_id", "")
        ref = f"4.{mid}"
        index[ref] = _IndexEntry(
            title=f"§{ref} — {mod.get('module_title', '')}",
            content_path=f"core_modules.modules[module_id={mid}]",
            content_type="ArchitectureModule",
            preview=mod.get("purpose", "")[:100],
        )
        for sub in mod.get("subsections", []):
            sub_num = sub.get("number", "")
            if sub_num:
                index[sub_num] = _IndexEntry(
                    title=f"§{sub_num} — {sub.get('title', '')}",
                    content_path=(
                        f"core_modules.modules[module_id={mid}].subsections[number={sub_num}]"
                    ),
                    content_type="SubSection",
                    preview=_preview(sub),
                )
        if mod.get("contract"):
            index[f"4.{mid}.contract"] = _IndexEntry(
                title=f"§4.{mid}.contract — Module {mid} Formal Contract",
                content_path=f"core_modules.modules[module_id={mid}].contract",
                content_type="ModuleContract",
            )


def _index_error_handling(spec: dict[str, Any], index: dict[str, _IndexEntry]) -> None:
    """Index by module: 12.1=Module A, 12.2=Module B, ... (v3 convention)."""
    modules = ["Module A", "Module B", "Module C", "Module D", "Module E", "Module F"]
    matrix = spec.get("error_handling", {}).get("matrix", [])
    for mi, mod_name in enumerate(modules):
        ref = f"12.{mi + 1}"
        cells = [c for c in matrix if c.get("module") == mod_name]
        if cells:
            index[ref] = _IndexEntry(
                title=f"§{ref} — {mod_name} failure handling",
                content_path=f"error_handling.matrix[module={mod_name}]",
                content_type="ErrorHandlingCell[]",
                preview=f"{len(cells)} failure categories",
            )


def _index_data_flow_stages(spec: dict[str, Any], index: dict[str, _IndexEntry]) -> None:
    for stage in spec.get("data_flow_pipeline", {}).get("stages", []):
        num = stage.get("stage_number")
        if num is not None:
            transition = stage.get("transition", f"Stage {num}")
            index[f"2.{num}"] = _IndexEntry(
                title=f"§2.{num} — {transition}",
                content_path=f"data_flow_pipeline.stages[stage_number={num}]",
                content_type="PipelineStage",
                preview=transition[:80],
            )


def _overlay_explicit_refs(spec: dict[str, Any], index: dict[str, _IndexEntry]) -> None:
    """Explicit SpecRef entries always take precedence."""
    for key in _ROOT_KEY_TO_SECTION:
        section = spec.get(key)
        if not isinstance(section, dict):
            continue
        explicit = section.get("spec_refs")
        if not explicit or not isinstance(explicit, list):
            continue
        for ref_obj in explicit:
            ref = ref_obj.get("ref", "")
            if not ref:
                continue
            rel_path = ref_obj.get("content_path", "")
            full_path = f"{key}.{rel_path}" if rel_path else key
            index[ref] = _IndexEntry(
                title=f"§{ref} — {ref_obj.get('title', '')}",
                content_path=full_path,
                content_type="SpecRef (explicit)",
                preview=ref_obj.get("title", ""),
                explicit=True,
            )


def _add_legacy_aliases(index: dict[str, _IndexEntry]) -> None:
    for ref, (title, content_path) in _LEGACY_DATA_GOVERNANCE_ALIASES.items():
        index[ref] = _IndexEntry(
            title=f"§{ref} — {title}",
            content_path=content_path,
            content_type="SpecRef (legacy alias)",
            preview=title,
            explicit=True,
        )
    for ref, (title, content_path) in _LEGACY_ERROR_HANDLING_ALIASES.items():
        index[ref] = _IndexEntry(
            title=f"§{ref} — {title}",
            content_path=content_path,
            content_type="SpecRef (legacy alias)",
            preview=title,
            explicit=True,
        )
    for ref, (title, content_path) in _LEGACY_AUDIT_ALIASES.items():
        index[ref] = _IndexEntry(
            title=f"§{ref} — {title}",
            content_path=content_path,
            content_type="SpecRef (legacy alias)",
            preview=title,
            explicit=True,
        )


# =====================================================================
# Content Path Resolution / Validation
# =====================================================================

def _split_content_path(path: str) -> list[str]:
    """
    Split a content path on dots that are outside bracket selectors.

    Example:
      core_modules.modules[module_id=B].subsections[number=4.B.2]
    becomes:
      [
        "core_modules",
        "modules[module_id=B]",
        "subsections[number=4.B.2]",
      ]
    """
    if not path:
        return []

    parts: list[str] = []
    buf: list[str] = []
    bracket_depth = 0

    for ch in path:
        if ch == "." and bracket_depth == 0:
            segment = "".join(buf).strip()
            if not segment:
                raise ValueError(f"Invalid empty segment in path '{path}'")
            parts.append(segment)
            buf = []
            continue

        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise ValueError(f"Unmatched ']' in path '{path}'")

        buf.append(ch)

    if bracket_depth != 0:
        raise ValueError(f"Unclosed '[' in path '{path}'")

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    return parts


def resolve_content_path(spec: dict[str, Any], path: str) -> Any:
    """
    Resolve content paths like:
      document_control
      data_flow_pipeline.stages[0]
      data_flow_pipeline.stages[stage_number=8]
      core_modules.modules[module_id=B].subsections[number=4.B.2]
      error_handling.matrix[failure_category=Physiological Ingestion]
    """
    if not path:
        return spec

    cur: Any = spec

    for segment in _split_content_path(path):
        # Extract the leading object key only, stopping before any '['
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$", segment)
        if not m:
            raise ValueError(f"Invalid segment '{segment}' in path '{path}'")

        key, rest = m.groups()

        if not isinstance(cur, dict):
            raise TypeError(
                f"Key '{key}' applied to non-dict object while resolving '{path}'"
            )
        if key not in cur:
            raise KeyError(f"Missing key '{key}' while resolving '{path}'")

        cur = cur[key]

        # Apply zero or more selectors: [0], [module_id=B], etc.
        while rest:
            m = re.match(r"^\[(.*?)\](.*)$", rest)
            if not m:
                raise ValueError(
                    f"Invalid selector syntax '{rest}' in segment '{segment}' of '{path}'"
                )

            selector, rest = m.groups()
            selector = selector.strip()

            # Numeric list index
            if selector.isdigit():
                idx = int(selector)
                if not isinstance(cur, list):
                    raise TypeError(f"Index [{idx}] applied to non-list in '{path}'")
                if idx < 0 or idx >= len(cur):
                    raise IndexError(f"Index [{idx}] out of range in '{path}'")
                cur = cur[idx]
                continue

            # field=value list selector
            if "=" in selector:
                field, value = selector.split("=", 1)
                field = field.strip()
                value = value.strip()

                if not isinstance(cur, list):
                    raise TypeError(
                        f"Selector [{selector}] applied to non-list in '{path}'"
                    )

                matches = [
                    item
                    for item in cur
                    if isinstance(item, dict) and str(item.get(field)) == value
                ]

                if not matches:
                    raise KeyError(
                        f"No match for selector [{selector}] while resolving '{path}'"
                    )

                cur = matches[0] if len(matches) == 1 else matches
                continue

            raise ValueError(f"Unsupported selector '[{selector}]' in '{path}'")

    return cur


def validate_index_targets(
    spec: dict[str, Any],
    index: dict[str, _IndexEntry],
) -> dict[str, str]:
    """Return {ref: error_message} for index entries whose content_path is broken."""
    broken: dict[str, str] = {}

    for ref, entry in index.items():
        try:
            resolved = resolve_content_path(spec, entry.content_path)
            if resolved is None:
                broken[ref] = f"{entry.content_path} :: resolved to null"
            elif isinstance(resolved, list) and len(resolved) == 0:
                broken[ref] = f"{entry.content_path} :: resolved to empty list"
        except Exception as exc:
            broken[ref] = f"{entry.content_path} :: {exc}"

    return broken


# =====================================================================
# Scanner
# =====================================================================

# Captures all v3 ref conventions:
#   7A, 7A.4, 7B.3     (section + letter, optional dot segments)
#   4.A.1, 4.C.4       (dot-separated module subsections)
#   12.3.2, 10.2.14    (numeric multi-level)
#   9, 12              (bare section numbers)
#   §13.17–§13.21      (inclusive simple ranges)
SPEC_REF_PATTERN = re.compile(
    r"§(\d+[A-Z]?(?:\.[A-Za-z0-9_]+)*)(?:[–-]§?(\d+[A-Z]?(?:\.[A-Za-z0-9_]+)*))?"
)

SCAN_GLOBS = ["**/*.py", "**/*.md", "**/*.yml", "**/*.yaml", ".claude/**/*.md"]
SCAN_EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".workspace_index",
    "venv",
    ".venv",
    "data",
}
SCAN_EXCLUDE_FILES = {
    Path("docs/V4_SPEC_DRAFTS.md"),
}


def expand_ref_range(start: str, end: str) -> list[str]:
    """
    Expand simple same-prefix ranges such as:
      13.17 -> 13.21
      7C.2  -> 7C.5
      12    -> 15
      11.5.13 -> 14  (abbreviated end means 11.5.14)

    Falls back to [start, end] when the range shape is unsupported.
    """
    s_parts = start.split(".")
    e_parts = end.split(".")

    if len(e_parts) == 1 and len(s_parts) > 1:
        e_parts = [*s_parts[:-1], e_parts[0]]
    elif len(s_parts) != len(e_parts):
        return [start, end]

    if s_parts[:-1] != e_parts[:-1]:
        return [start, end]

    try:
        s_last = int(s_parts[-1])
        e_last = int(e_parts[-1])
    except ValueError:
        return [start, end]

    if e_last < s_last:
        return [start, end]

    prefix = ".".join(s_parts[:-1])
    if prefix:
        return [f"{prefix}.{i}" for i in range(s_last, e_last + 1)]
    return [str(i) for i in range(s_last, e_last + 1)]


def scan_project_refs(repo_root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for glob_pattern in SCAN_GLOBS:
        for filepath in repo_root.glob(glob_pattern):
            if any(part in SCAN_EXCLUDE_DIRS for part in filepath.parts):
                continue
            if filepath.relative_to(repo_root) in SCAN_EXCLUDE_FILES:
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_num, line in enumerate(text.splitlines(), start=1):
                for match in SPEC_REF_PATTERN.finditer(line):
                    start_ref = match.group(1)
                    end_ref = match.group(2)
                    refs = expand_ref_range(start_ref, end_ref) if end_ref else [start_ref]

                    for ref in refs:
                        findings.append(
                            {
                                "ref": ref,
                                "file": str(filepath.relative_to(repo_root)),
                                "line": line_num,
                                "context": line.strip()[:120],
                            }
                        )
    return findings


# =====================================================================
# Validator
# =====================================================================


def validate_refs(
    index: dict[str, _IndexEntry],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    used_refs: set[str] = set()
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for finding in findings:
        ref = finding["ref"]
        used_refs.add(ref)
        if ref in index:
            resolved.append(
                {**finding, "target": index[ref].content_path, "resolution": "direct"}
            )
        else:
            parent = ref.rsplit(".", 1)[0] if "." in ref else None
            if parent and parent in index:
                resolved.append(
                    {**finding, "target": index[parent].content_path, "resolution": "parent"}
                )
            else:
                unresolved.append(finding)

    unused = {r: e.to_dict() for r, e in index.items() if r not in used_refs}
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "unused_index_entries": unused,
        "stats": {
            "total_refs_found": len(findings),
            "unique_refs": len(used_refs),
            "resolved_direct": sum(
                1 for r in resolved if r.get("resolution") == "direct"
            ),
            "resolved_parent": sum(
                1 for r in resolved if r.get("resolution") == "parent"
            ),
            "unresolved": len(unresolved),
            "index_size": len(index),
            "index_explicit": sum(1 for e in index.values() if e.explicit),
            "index_structural": sum(1 for e in index.values() if not e.explicit),
            "unused_index_entries": len(unused),
        },
    }


# =====================================================================
# CLI
# =====================================================================


def _sort_ref(ref: str) -> tuple[Any, ...]:
    """Sort so that 4.A.1 < 4.B < 7A < 7B.4 < 12.1."""
    parts: list[tuple[int, int, str]] = []
    for seg in ref.split("."):
        m = re.match(r"^(\d+)([A-Z])$", seg)
        if m:
            parts.append((0, int(m.group(1)), ""))
            parts.append((1, ord(m.group(2)), ""))
        else:
            try:
                parts.append((0, int(seg), ""))
            except ValueError:
                parts.append((1, ord(seg[0]) if seg else 0, seg))
    return tuple(parts)


def main() -> int:
    # The script prints section markers (§) and arrow glyphs (→) that are
    # not representable in cp1252. On Windows the default stdout encoding
    # is cp1252, so plain print() raises UnicodeEncodeError when stdout
    # is a console without PYTHONIOENCODING=utf-8 set, or when stdout is
    # redirected to a file. Reconfigure both streams to UTF-8 if the
    # runtime supports it (Python 3.7+).
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass

    parser = argparse.ArgumentParser(
        description="Resolve and validate spec references against content.json",
    )
    parser.add_argument(
        "--from-pdf",
        type=Path,
        default=None,
        help="Extract content.json from this PDF.",
    )
    parser.add_argument(
        "--content",
        type=Path,
        default=None,
        help="Path to standalone content.json.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Repository root (default: cwd).",
    )
    parser.add_argument("--index", action="store_true", help="Print the full spec_ref index.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Scan project and validate all refs.",
    )
    parser.add_argument(
        "--resolve",
        type=str,
        default=None,
        help="Resolve a single ref (e.g., '7A.4').",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format.")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract content.json from the single committed spec PDF to stdout.",
    )
    args = parser.parse_args()

    if not any([args.index, args.validate, args.resolve, args.extract]):
        parser.print_help()
        return 0

    repo_root = args.repo.resolve()

    if args.extract:
        try:
            spec = load_content(args.from_pdf, None, repo_root)
        except (FileNotFoundError, FileExistsError, ValueError, ImportError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        # The extracted spec contains characters outside cp1252 (e.g.
        # U+2011 NON-BREAKING HYPHEN). On Windows the default stdout
        # encoding is cp1252, so a plain print() raises
        # UnicodeEncodeError when stdout is redirected to a file.
        # Write UTF-8 bytes directly so the JSON is portable regardless
        # of the host console encoding.
        payload = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
        try:
            sys.stdout.buffer.write(payload.encode("utf-8"))
        except AttributeError:
            print(payload, end="")
        return 0

    try:
        spec = load_content(args.from_pdf, args.content, repo_root)
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    index = build_index(spec)
    broken_targets = validate_index_targets(spec, index)

    if args.resolve:
        ref = args.resolve.lstrip("§").strip()
        entry = index.get(ref)
        if entry:
            try:
                resolved_obj = resolve_content_path(spec, entry.content_path)
                target_status = "ok"
                target_preview = _preview(resolved_obj)
            except Exception as exc:
                target_status = f"broken: {exc}"
                target_preview = ""

            if args.json:
                print(
                    json.dumps(
                        {
                            ref: {
                                **entry.to_dict(),
                                "target_status": target_status,
                                "target_preview": target_preview,
                            }
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                marker = " (explicit)" if entry.explicit else " (structural)"
                print(f"§{ref} → {entry.content_path}{marker}")
                print(f"  Title:         {entry.title}")
                print(f"  Type:          {entry.content_type}")
                print(f"  Target status: {target_status}")
                if entry.preview:
                    print(f"  Preview:       {entry.preview}")
                if target_preview:
                    print(f"  Target data:   {target_preview}")
            return 0 if target_status == "ok" else 1

        print(f"UNRESOLVED: §{ref} not in index", file=sys.stderr)
        prefix_m = re.match(r"\d+[A-Z]?", ref)
        if prefix_m:
            suggestions = [r for r in index if r.startswith(prefix_m.group(0))]
            if suggestions:
                suggestion_list = ", ".join(
                    "§" + s for s in sorted(suggestions, key=_sort_ref)[:10]
                )
                print(f"  Did you mean: {suggestion_list}", file=sys.stderr)
        return 1

    if args.index:
        if args.json:
            print(
                json.dumps(
                    {
                        r: {
                            **e.to_dict(),
                            "target_status": "broken" if r in broken_targets else "ok",
                        }
                        for r, e in index.items()
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"Spec Reference Index ({len(index)} entries)\n{'=' * 72}")
            for ref in sorted(index, key=_sort_ref):
                entry = index[ref]
                marker = " *" if entry.explicit else ""
                broken_marker = " !" if ref in broken_targets else ""
                print(f"  §{ref:<20} → {entry.content_path}{marker}{broken_marker}")
            explicit_count = sum(1 for e in index.values() if e.explicit)
            print(
                f"\n  {explicit_count} explicit (marked *), "
                f"{len(index) - explicit_count} structural"
            )
            if broken_targets:
                print(f"  {len(broken_targets)} broken targets (marked !)")
            print()

    if args.validate:
        findings = scan_project_refs(repo_root)
        report = validate_refs(index, findings)
        stats = report["stats"]

        if args.json:
            print(
                json.dumps(
                    {
                        "stats": {
                            **stats,
                            "broken_index_targets": len(broken_targets),
                        },
                        "unresolved": report["unresolved"],
                        "broken_index_targets": broken_targets,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 1 if report["unresolved"] or broken_targets else 0

        print(f"Validation Report\n{'=' * 72}")
        print(f"  Refs found in project:    {stats['total_refs_found']} ({stats['unique_refs']} unique)")
        print(f"  Resolved (direct):        {stats['resolved_direct']}")
        print(f"  Resolved (parent):        {stats['resolved_parent']}")
        print(f"  Unresolved:               {stats['unresolved']}")
        print(
            f"  Index size:               {stats['index_size']} "
            f"({stats['index_explicit']} explicit, {stats['index_structural']} structural)"
        )
        print(f"  Broken index targets:     {len(broken_targets)}")
        print(f"  Index entries unused:     {stats['unused_index_entries']}\n")

        if report["unresolved"]:
            print(f"UNRESOLVED REFERENCES\n{'-' * 72}")
            for item in sorted(report["unresolved"], key=lambda x: x["ref"]):
                print(f"  §{item['ref']:<20} {item['file']}:{item['line']}")
                print(f"    {item['context']}")
            print()

        if broken_targets:
            print(f"BROKEN INDEX TARGETS\n{'-' * 72}")
            for ref in sorted(broken_targets, key=_sort_ref):
                print(f"  §{ref:<20} {broken_targets[ref]}")
            print()

        parent_resolved = [r for r in report["resolved"] if r.get("resolution") == "parent"]
        if parent_resolved:
            unique_parents = {r["ref"] for r in parent_resolved}
            print(f"PARENT-RESOLVED REFS ({len(unique_parents)} unique)\n{'-' * 72}")
            for ref in sorted(unique_parents, key=_sort_ref):
                example = next(r for r in parent_resolved if r["ref"] == ref)
                print(f"  §{ref:<20} → {example['target']} (via parent)")
            print()

        return 1 if report["unresolved"] or broken_targets else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())