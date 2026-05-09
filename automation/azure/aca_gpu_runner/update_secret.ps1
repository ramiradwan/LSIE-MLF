#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ResourceGroup = "lsie-mlf-aca-gpu-rg",

    [Parameter()]
    [string]$RunnerJobName = "lsie-gpu-runner",

    [Parameter(Mandatory = $true)]
    [string]$GitHubPat
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

az containerapp job secret set --resource-group $ResourceGroup --name $RunnerJobName --secrets "personal-access-token=$GitHubPat"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update the GitHub PAT secret on $RunnerJobName"
}

Write-Host "Updated personal-access-token on $RunnerJobName in $ResourceGroup"
