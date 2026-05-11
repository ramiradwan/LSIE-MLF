from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_DIRS = ("services", "packages")
_EXCLUDED_DIR_NAMES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _production_python_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for dirname in _PRODUCTION_DIRS:
        root = _REPO_ROOT / dirname
        paths.extend(
            path
            for path in root.rglob("*.py")
            if not any(part in _EXCLUDED_DIR_NAMES for part in path.parts)
        )
    return tuple(sorted(paths))


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=_rel(path))


def _calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _name_or_attr_tokens(node: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.add(child.id)
        elif isinstance(child, ast.Attribute):
            tokens.add(child.attr)
    return tokens


def _string_constants(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.add(child.value)
    return values


def _imported_aliases(
    tree: ast.AST,
    *,
    module_name: str,
    symbol_names: Sequence[str],
) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name in symbol_names
            )
    return aliases


def _module_aliases(tree: ast.AST, module_name: str) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname or alias.name.rsplit(".", maxsplit=1)[-1]
                for alias in node.names
                if alias.name == module_name
            )
    return aliases


def test_module_b_ground_truth_ingester_remains_unwired_from_runtime_entrypoints() -> None:
    guarded_symbols = {"GroundTruthIngester", "EulerStreamSigner", "SignatureProvider"}
    implementation_path = "services/worker/pipeline/ground_truth.py"
    offenders: list[str] = []

    for path in _production_python_files():
        if _rel(path) == implementation_path:
            continue
        tree = _parse(path)
        imports = _imported_aliases(
            tree,
            module_name="services.worker.pipeline.ground_truth",
            symbol_names=tuple(guarded_symbols),
        )
        for imported_alias in sorted(imports):
            offenders.append(f"{_rel(path)} imports {imported_alias}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for module_alias in node.names:
                    if module_alias.name == "services.worker.pipeline.ground_truth":
                        offenders.append(f"{_rel(path)} imports {module_alias.name}")

    assert offenders == []


def test_context_enrichment_task_has_no_runtime_producer() -> None:
    implementation_path = "services/worker/tasks/enrichment.py"
    offenders: list[str] = []

    for path in _production_python_files():
        if _rel(path) == implementation_path:
            continue
        tree = _parse(path)
        task_aliases = _imported_aliases(
            tree,
            module_name="services.worker.tasks.enrichment",
            symbol_names=("scrape_context",),
        )
        module_aliases = _module_aliases(tree, "services.worker.tasks.enrichment")
        if task_aliases:
            offenders.append(f"{_rel(path)} imports scrape_context")

        for call in _calls(tree):
            func = call.func
            if isinstance(func, ast.Name) and func.id in task_aliases:
                offenders.append(f"{_rel(path)}:{call.lineno} calls scrape_context directly")
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id in task_aliases:
                    offenders.append(f"{_rel(path)}:{call.lineno} calls scrape_context.{func.attr}")
                if (
                    func.attr == "scrape_context"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in module_aliases
                ):
                    offenders.append(f"{_rel(path)}:{call.lineno} calls enrichment.scrape_context")
                if (
                    func.attr in {"delay", "apply_async"}
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "scrape_context"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id in module_aliases
                ):
                    offenders.append(
                        f"{_rel(path)}:{call.lineno} enqueues enrichment.scrape_context"
                    )

    assert offenders == []


def test_reward_path_signature_excludes_physiology_and_acoustic_inputs() -> None:
    tree = _parse(_REPO_ROOT / "services/worker/pipeline/reward.py")
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward"
    )

    assert [arg.arg for arg in function.args.args] == [
        "au12_series",
        "stimulus_time_s",
        "is_match",
    ]
    forbidden_tokens = {
        token
        for token in _name_or_attr_tokens(function)
        if any(fragment in token.lower() for fragment in ("rmssd", "physiolog", "acoustic"))
    }
    assert forbidden_tokens == set()


def test_attribution_offline_final_replay_has_no_runtime_producer() -> None:
    builder_path = "packages/ml_core/attribution.py"
    offenders: list[str] = []

    for path in _production_python_files():
        if _rel(path) == builder_path:
            continue
        tree = _parse(path)
        for call in _calls(tree):
            if _call_name(call) != "build_attribution_ledger_records":
                continue
            if any(keyword.arg == "finality" for keyword in call.keywords):
                offenders.append(f"{_rel(path)}:{call.lineno} passes finality")
            if "offline_final" in _string_constants(call):
                offenders.append(f"{_rel(path)}:{call.lineno} builds offline_final records")

    assert offenders == []


def test_desktop_cloud_outbox_producers_remain_current_analytics_surface_only() -> None:
    allowed_producer_path = "services/desktop_app/processes/analytics_state_worker.py"
    enqueue_methods = {
        "enqueue_inference_handoff",
        "enqueue_attribution_event",
        "enqueue_posterior_delta",
    }
    offenders: list[str] = []

    for path in (_REPO_ROOT / "services/desktop_app").rglob("*.py"):
        rel_path = _rel(path)
        if rel_path in {allowed_producer_path, "services/desktop_app/cloud/outbox.py"}:
            continue
        tree = _parse(path)
        for call in _calls(tree):
            if _call_name(call) in enqueue_methods:
                offenders.append(f"{rel_path}:{call.lineno} enqueues desktop cloud payloads")

    assert offenders == []


def test_desktop_runtime_processes_do_not_import_retained_or_cloud_route_handlers() -> None:
    allowed_desktop_route_host = "services/desktop_app/processes/operator_api_runtime.py"
    forbidden_prefixes = (
        "services.api.routes",
        "services.cloud_api.routes",
    )
    offenders: list[str] = []

    for path in (_REPO_ROOT / "services/desktop_app/processes").glob("*.py"):
        rel_path = _rel(path)
        if rel_path == allowed_desktop_route_host:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        offenders.append(f"{rel_path} imports {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith(forbidden_prefixes)
            ):
                offenders.append(f"{rel_path} imports from {node.module}")

    assert offenders == []


def test_operator_console_sqlite_write_helpers_remain_cloud_bundle_exception_only() -> None:
    allowed_imports = {
        "services/operator_console/polling.py": {
            "services.desktop_app.cloud.experiment_bundle.ExperimentBundleStore",
            "services.desktop_app.os_adapter.resolve_state_dir",
            "services.desktop_app.processes.cloud_sync_worker.SQLITE_FILENAME",
        }
    }
    offenders: list[str] = []

    for path in (_REPO_ROOT / "services/operator_console").rglob("*.py"):
        rel_path = _rel(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            imported_names = {alias.name for alias in node.names}
            imported_symbols = {f"{node.module}.{name}" for name in imported_names}
            relevant = {
                symbol
                for symbol in imported_symbols
                if symbol.startswith("services.desktop_app.state.")
                or symbol
                in {
                    "services.desktop_app.cloud.experiment_bundle.ExperimentBundleStore",
                    "services.desktop_app.os_adapter.resolve_state_dir",
                    "services.desktop_app.processes.cloud_sync_worker.SQLITE_FILENAME",
                }
            }
            unexpected = relevant.difference(allowed_imports.get(rel_path, set()))
            for symbol in sorted(unexpected):
                offenders.append(f"{rel_path} imports {symbol}")

    assert offenders == []


def test_retained_and_cloud_api_routes_do_not_import_desktop_runtime_state() -> None:
    route_roots = (
        _REPO_ROOT / "services/api/routes",
        _REPO_ROOT / "services/cloud_api/routes",
    )
    forbidden_prefix = "services.desktop_app.state"
    offenders: list[str] = []

    for route_root in route_roots:
        for path in route_root.glob("*.py"):
            rel_path = _rel(path)
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(forbidden_prefix):
                            offenders.append(f"{rel_path} imports {alias.name}")
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.startswith(forbidden_prefix)
                ):
                    offenders.append(f"{rel_path} imports from {node.module}")

    assert offenders == []


def test_ephemeral_vault_cron_is_not_started_by_runtime_code() -> None:
    implementation_path = "services/worker/vault_cron.py"
    offenders: list[str] = []

    for path in _production_python_files():
        if _rel(path) == implementation_path:
            continue
        tree = _parse(path)
        imported = _imported_aliases(
            tree,
            module_name="services.worker.vault_cron",
            symbol_names=("run_vault_cron",),
        )
        if imported:
            offenders.append(f"{_rel(path)} imports run_vault_cron")
        for call in _calls(tree):
            if _call_name(call) in imported:
                offenders.append(f"{_rel(path)}:{call.lineno} calls run_vault_cron")

    assert offenders == []
