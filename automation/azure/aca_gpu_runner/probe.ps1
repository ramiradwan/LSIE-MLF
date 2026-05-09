#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ResourceGroup = "lsie-mlf-aca-gpu-rg",

    [Parameter()]
    [string]$Location = "swedencentral",

    [Parameter()]
    [string]$EnvironmentName = "lsie-mlf-aca-gpu-env",

    [Parameter()]
    [string]$WorkspaceName = "lsie-mlf-aca-gpu-law",

    [Parameter()]
    [string]$RegistryName = "",

    [Parameter()]
    [string]$IdentityName = "lsie-mlf-aca-gpu-mi",

    [Parameter()]
    [string]$WorkloadProfileName = "t4-consumption",

    [Parameter()]
    [string]$WorkloadProfileType = "Consumption-GPU-NC8as-T4",

    [Parameter()]
    [string]$ProbeJobName = "lsie-gpu-probe",

    [Parameter()]
    [string]$ProbeImageName = "gpu-probe:aca-gpu",

    [Parameter()]
    [switch]$SkipImageBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$deployScript = Join-Path $scriptRoot "deploy.ps1"

$deployParameters = @{
    ResourceGroup = $ResourceGroup
    Location = $Location
    EnvironmentName = $EnvironmentName
    WorkspaceName = $WorkspaceName
    RegistryName = $RegistryName
    IdentityName = $IdentityName
    WorkloadProfileName = $WorkloadProfileName
    WorkloadProfileType = $WorkloadProfileType
    ProbeJobName = $ProbeJobName
    ProbeImageName = $ProbeImageName
    ProbeOnly = $true
    StartProbe = $true
    SkipImageBuild = $SkipImageBuild
}

& $deployScript @deployParameters

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

az containerapp job execution list --resource-group $ResourceGroup --name $ProbeJobName --output table --query "[].{Status:properties.status,Name:name,StartTime:properties.startTime}"
