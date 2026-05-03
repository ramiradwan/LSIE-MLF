from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIGN_SCRIPT = ROOT / "build" / "sign_windows.ps1"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
GPU_REPLAY_WORKFLOW = ROOT / ".github" / "workflows" / "gpu_replay_parity.yml"
MACOS_DEFERRED = ROOT / "build" / "MACOS_DEFERRED.md"


def test_sign_windows_script_targets_exe_and_dll_with_artifact_signing() -> None:
    script = SIGN_SCRIPT.read_text(encoding="utf-8")

    assert "Azure.CodeSigning.Dlib.dll" in script
    assert "AZURE_ARTIFACT_SIGNING_ENDPOINT" in script
    assert "AZURE_ARTIFACT_SIGNING_ACCOUNT_NAME" in script
    assert "AZURE_ARTIFACT_SIGNING_CERTIFICATE_PROFILE_NAME" in script
    assert "http://timestamp.acs.microsoft.com" in script
    assert 'Where-Object { $_.Extension -in @(".exe", ".dll") }' in script
    assert "/fd SHA256" in script
    assert "/td SHA256" in script
    assert "/dmdf" in script


def test_release_workflow_runs_on_tags_and_signs_release_artifact() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in workflow
    assert "runs-on: windows-2022" in workflow
    assert "uv sync --frozen --group dev" in workflow
    assert "--name LSIE-MLF-Launcher" in workflow
    assert "services/desktop_launcher/ui.py" in workflow
    assert "dist\\LSIE-MLF-Launcher" in workflow
    assert "Bundle desktop app source for hydrated runtime" in workflow
    assert "dist\\LSIE-MLF-Launcher\\app" in workflow
    assert "Copy-Item -Recurse -Force \"${{ github.workspace }}\\packages\"" in workflow
    assert "Copy-Item -Recurse -Force \"${{ github.workspace }}\\services\"" in workflow
    assert "Copy-Item -Force \"${{ github.workspace }}\\pyproject.toml\"" in workflow
    assert "Copy-Item -Force \"${{ github.workspace }}\\uv.lock\"" in workflow
    assert "Smoke packaged launcher runtime handoff" in workflow
    assert 'LSIE_DEV_FORCE_CPU_SPEECH: "1"' in workflow
    assert 'LSIE-MLF-Launcher.exe"' in workflow
    assert "--smoke" in workflow
    assert "uses: azure/artifact-signing-action@v1" in workflow
    assert "azure-tenant-id: ${{ secrets.AZURE_TENANT_ID }}" in workflow
    assert "azure-client-id: ${{ secrets.AZURE_CLIENT_ID }}" in workflow
    assert "azure-client-secret: ${{ secrets.AZURE_CLIENT_SECRET }}" in workflow
    assert "files-folder-filter: exe,dll" in workflow
    assert "files-folder-recurse: true" in workflow
    assert "timestamp-rfc3161: http://timestamp.acs.microsoft.com" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_gpu_replay_workflow_runs_gate0_on_self_hosted_turing_runner() -> None:
    workflow = GPU_REPLAY_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Gate 0 GPU Replay Parity" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, turing]" in workflow
    assert "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader" in workflow
    assert "uv sync --frozen --extra ml_backend --group dev" in workflow
    assert "uv run python scripts/gate0_gpu_replay.py" in workflow
    assert "LSIE_GATE0_WHISPER_MODEL: large-v3" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert '"packages/ml_core/**"' in workflow
    assert '"tests/fixtures/capture/baseline/**"' in workflow


def test_macos_signing_is_documented_as_deferred() -> None:
    note = MACOS_DEFERRED.read_text(encoding="utf-8")

    assert "macOS signing deferred to v4.1" in note
    assert "hardened runtime" in note
    assert "entitlements" in note
    assert "Windows Azure Artifact Signing workflow" in note
