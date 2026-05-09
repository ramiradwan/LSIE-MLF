#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path $_ -PathType Container })]
    [string]$ArtifactRoot,

    [Parameter()]
    [string]$SignToolPath = $env:LSIE_SIGNTOOL_PATH,

    [Parameter()]
    [string]$DlibPath = $env:LSIE_AZURE_CODESIGNING_DLIB_PATH,

    [Parameter()]
    [string]$MetadataPath = $env:LSIE_ARTIFACT_SIGNING_METADATA_PATH,

    [Parameter()]
    [string]$Endpoint = $env:AZURE_ARTIFACT_SIGNING_ENDPOINT,

    [Parameter()]
    [string]$CodeSigningAccountName = $env:AZURE_ARTIFACT_SIGNING_ACCOUNT_NAME,

    [Parameter()]
    [string]$CertificateProfileName = $env:AZURE_ARTIFACT_SIGNING_CERTIFICATE_PROFILE_NAME,

    [Parameter()]
    [string]$CorrelationId = $env:GITHUB_RUN_ID,

    [Parameter()]
    [string]$TimestampUrl = "http://timestamp.acs.microsoft.com"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "$Name is required. Install Microsoft.Azure.ArtifactSigningClientTools or set the matching LSIE_* path."
    }

    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if (-not $item.PSIsContainer) {
        return $item.FullName
    }

    throw "$Name must point to a file: $Path"
}

function Resolve-DefaultSignTool {
    $candidate = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($null -ne $candidate) {
        return $candidate.Source
    }

    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
        "$env:ProgramFiles\Windows Kits\10\bin"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path $_) }

    foreach ($root in $roots) {
        $match = Get-ChildItem -LiteralPath $root -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($null -ne $match) {
            return $match.FullName
        }
    }

    return $null
}

function Resolve-DefaultDlib {
    $roots = @(
        "${env:ProgramFiles(x86)}\Microsoft\ArtifactSigningClientTools\bin",
        "$env:ProgramFiles\Microsoft\ArtifactSigningClientTools\bin"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path $_) }

    foreach ($root in $roots) {
        $matches = Get-ChildItem -LiteralPath $root -Filter Azure.CodeSigning.Dlib.dll -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
        $match = $matches |
            Where-Object { $_.FullName -match "\\x64\\Azure\.CodeSigning\.Dlib\.dll$" } |
            Select-Object -First 1
        if ($null -eq $match) {
            $match = $matches | Select-Object -First 1
        }
        if ($null -ne $match) {
            return $match.FullName
        }
    }

    return $null
}

function New-ArtifactSigningMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $missing = @()
    if ([string]::IsNullOrWhiteSpace($Endpoint)) { $missing += "AZURE_ARTIFACT_SIGNING_ENDPOINT" }
    if ([string]::IsNullOrWhiteSpace($CodeSigningAccountName)) { $missing += "AZURE_ARTIFACT_SIGNING_ACCOUNT_NAME" }
    if ([string]::IsNullOrWhiteSpace($CertificateProfileName)) { $missing += "AZURE_ARTIFACT_SIGNING_CERTIFICATE_PROFILE_NAME" }
    if ($missing.Count -gt 0) {
        throw "Missing Azure Artifact Signing metadata: $($missing -join ', ')"
    }

    $metadata = [ordered]@{
        Endpoint = $Endpoint
        CodeSigningAccountName = $CodeSigningAccountName
        CertificateProfileName = $CertificateProfileName
    }

    if (-not [string]::IsNullOrWhiteSpace($CorrelationId)) {
        $metadata.CorrelationId = $CorrelationId
    }

    $metadata | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $Destination -Encoding UTF8
    return $Destination
}

$resolvedArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path
if ([string]::IsNullOrWhiteSpace($SignToolPath)) {
    $SignToolPath = Resolve-DefaultSignTool
}
if ([string]::IsNullOrWhiteSpace($DlibPath)) {
    $DlibPath = Resolve-DefaultDlib
}

$resolvedSignTool = Resolve-RequiredFile -Path $SignToolPath -Name "SignToolPath"
$resolvedDlib = Resolve-RequiredFile -Path $DlibPath -Name "DlibPath"

$tempMetadata = $null
if ([string]::IsNullOrWhiteSpace($MetadataPath)) {
    $tempMetadata = Join-Path ([System.IO.Path]::GetTempPath()) "lsie-artifact-signing-metadata-$([Guid]::NewGuid()).json"
    $MetadataPath = New-ArtifactSigningMetadata -Destination $tempMetadata
}
$resolvedMetadata = Resolve-RequiredFile -Path $MetadataPath -Name "MetadataPath"

$targets = Get-ChildItem -LiteralPath $resolvedArtifactRoot -Recurse -File |
    Where-Object { $_.Extension -in @(".exe", ".dll") } |
    Sort-Object FullName

if ($targets.Count -eq 0) {
    throw "No .exe or .dll files found under $resolvedArtifactRoot"
}

try {
    foreach ($target in $targets) {
        Write-Host "Signing $($target.FullName)"
        & $resolvedSignTool sign /v /fd SHA256 /tr $TimestampUrl /td SHA256 /dlib $resolvedDlib /dmdf $resolvedMetadata $target.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "SignTool failed with exit code $LASTEXITCODE for $($target.FullName)"
        }
    }
}
finally {
    if ($null -ne $tempMetadata -and (Test-Path -LiteralPath $tempMetadata)) {
        Remove-Item -LiteralPath $tempMetadata -Force
    }
}
