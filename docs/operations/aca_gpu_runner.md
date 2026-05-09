# ACA GPU runner operations

This repo’s Gate 0 GPU replay can run on an Azure Container Apps Jobs self-hosted runner backed by the `Consumption-GPU-NC8as-T4` workload profile in `swedencentral`.

## What is deployed

- Azure Container Registry for the runner and probe images
- User-assigned managed identity with `AcrPull`
- Container Apps environment with workload profiles enabled
- `t4-consumption` workload profile (`Consumption-GPU-NC8as-T4`)
- Manual GPU probe job: `lsie-gpu-probe`
- Event-driven GitHub runner job: `lsie-gpu-runner`

The GitHub runner job is filtered to the repo queue for `ramiradwan/LSIE-MLF` and the labels `gpu,turing,aca-gpu`.

## Why the workflow is split

- `.github/workflows/gpu_replay_parity.yml` runs only on `push` to `main` and `workflow_dispatch`.
- `.github/workflows/gpu_replay_trusted_dispatch.yml` listens on `pull_request_target` and only dispatches the GPU workflow for trusted same-repo PRs (`OWNER`, `MEMBER`, `COLLABORATOR`).
- Fork PRs and untrusted authors never wake the GPU runner automatically.

## Prerequisites

- Azure subscription with serverless GPU quota in `swedencentral`
- Fine-grained GitHub PAT scoped to `ramiradwan/LSIE-MLF`
  - Actions: read-only
  - Administration: read/write
  - Metadata: read-only
- Azure CLI authenticated as a principal allowed to create resource groups, identities, Container Apps resources, and role assignments

## Probe-first rollout

Run the manual probe deployment first:

```powershell
pwsh -File automation/azure/aca_gpu_runner/probe.ps1
```

This creates the environment resources if they do not already exist, builds `probe.Dockerfile` into ACR, deploys `lsie-gpu-probe`, and starts one execution.

Verify the probe execution:

```powershell
az containerapp job execution list --resource-group lsie-mlf-aca-gpu-rg --name lsie-gpu-probe --output table
```

A healthy probe prints `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` output and exits 0 only when at least one GPU reports compute capability `>= 7.5`.

## Deploy the runner job

Once the probe succeeds, deploy the full runner job:

```powershell
$env:GITHUB_PAT = "<fine-grained PAT>"
pwsh -File automation/azure/aca_gpu_runner/deploy.ps1
```

To skip rebuilding images during an update:

```powershell
pwsh -File automation/azure/aca_gpu_runner/deploy.ps1 -SkipImageBuild
```

## Rotate the PAT

```powershell
pwsh -File automation/azure/aca_gpu_runner/update_secret.ps1 -GitHubPat "<new fine-grained PAT>"
```

## Verify end to end

1. Run `gpu_replay_parity.yml` with `workflow_dispatch` on `main`.
2. Confirm a Container Apps execution is created:
   ```powershell
   az containerapp job execution list --resource-group lsie-mlf-aca-gpu-rg --name lsie-gpu-runner --output table
   ```
3. Confirm the workflow lands on `runs-on: [self-hosted, linux, x64, gpu, turing, aca-gpu]` and passes the early CUDA contract checks.
4. Open a same-repo trusted PR touching the watched paths and confirm `gpu_replay_trusted_dispatch.yml` dispatches the GPU workflow.
5. Confirm fork PRs and untrusted PR authors only produce the hosted summary message and do not create ACA executions.

## Constraints

- Azure Container Apps jobs do not support Docker-in-Docker. Workflows scheduled onto this runner must not require Docker commands.
- The runner image is intentionally thin. The repo still hydrates the ML environment with `uv sync --frozen --extra ml_backend --group dev` from `pyproject.toml` and `uv.lock`.
- The runner job is scoped to `ramiradwan/LSIE-MLF`; adjust `GitHubOwner`, `GitHubRepo`, and PAT scope together if the repo moves.
