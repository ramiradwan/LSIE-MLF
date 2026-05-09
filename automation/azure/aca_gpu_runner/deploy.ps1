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
    [string]$RunnerJobName = "lsie-gpu-runner",

    [Parameter()]
    [string]$ProbeImageName = "gpu-probe:aca-gpu",

    [Parameter()]
    [string]$RunnerImageName = "github-runner:aca-gpu",

    [Parameter()]
    [string]$GitHubOwner = "ramiradwan",

    [Parameter()]
    [string]$GitHubRepo = "LSIE-MLF",

    [Parameter()]
    [string]$GitHubApiUrl = "https://api.github.com",

    [Parameter()]
    [string]$RunnerLabels = "gpu,turing,aca-gpu",

    [Parameter()]
    [int]$PollingIntervalSeconds = 30,

    [Parameter()]
    [int]$MaxExecutions = 1,

    [Parameter()]
    [string]$GitHubPat = $env:GITHUB_PAT,

    [Parameter()]
    [switch]$SkipImageBuild,

    [Parameter()]
    [switch]$ProbeOnly,

    [Parameter()]
    [switch]$StartProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-AzJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & az @Arguments --output json
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')"
    }
    if ([string]::IsNullOrWhiteSpace($output)) {
        return $null
    }
    return $output | ConvertFrom-Json
}

function Invoke-AzTsv {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & az @Arguments --output tsv
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')"
    }
    return [string]$output
}

function Test-AzSuccess {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & az @Arguments --only-show-errors 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Wait-ForRegistryDns {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter()][int]$TimeoutSeconds = 300
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $addresses = [System.Net.Dns]::GetHostAddresses($HostName)
            if (@($addresses).Count -gt 0) {
                return
            }
        }
        catch {
        }

        Start-Sleep -Seconds 5
    }

    throw "Timed out waiting for DNS to resolve for $HostName"
}

function Ensure-ContainerAppExtension {
    if (-not (Test-AzSuccess -Arguments @("extension", "show", "--name", "containerapp"))) {
        & az extension add --name containerapp --upgrade --allow-preview true | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Azure Container Apps CLI extension"
        }
        return
    }

    & az extension add --name containerapp --upgrade --allow-preview true | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade Azure Container Apps CLI extension"
    }
}

function Ensure-Provider {
    param([Parameter(Mandatory = $true)][string]$Namespace)

    & az provider register --namespace $Namespace | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to register Azure provider $Namespace"
    }
}

function Ensure-ResourceGroup {
    $exists = Invoke-AzTsv @("group", "exists", "--name", $ResourceGroup)
    if ($exists -eq "true") {
        return
    }

    & az group create --name $ResourceGroup --location $Location | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create resource group $ResourceGroup"
    }
}

function Ensure-Workspace {
    if (Test-AzSuccess -Arguments @("monitor", "log-analytics", "workspace", "show", "--resource-group", $ResourceGroup, "--workspace-name", $WorkspaceName)) {
        return
    }

    & az monitor log-analytics workspace create --resource-group $ResourceGroup --workspace-name $WorkspaceName --location $Location | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Log Analytics workspace $WorkspaceName"
    }
}

function Ensure-Registry {
    if (-not (Test-AzSuccess -Arguments @("acr", "show", "--name", $RegistryName, "--resource-group", $ResourceGroup))) {
        & az acr create --name $RegistryName --resource-group $ResourceGroup --location $Location --sku Basic | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Azure Container Registry $RegistryName"
        }
    }

    & az acr config authentication-as-arm update --registry $RegistryName --status enabled | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to enable ARM authentication for Azure Container Registry $RegistryName"
    }
}

function Ensure-Identity {
    if (Test-AzSuccess -Arguments @("identity", "show", "--name", $IdentityName, "--resource-group", $ResourceGroup)) {
        return
    }

    & az identity create --name $IdentityName --resource-group $ResourceGroup --location $Location | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create user-assigned managed identity $IdentityName"
    }
}

function Ensure-AcrPullRole {
    param(
        [Parameter(Mandatory = $true)][string]$PrincipalId,
        [Parameter(Mandatory = $true)][string]$AcrId
    )

    $assignments = Invoke-AzTsv @(
        "role", "assignment", "list",
        "--assignee-object-id", $PrincipalId,
        "--scope", $AcrId,
        "--query", "[?roleDefinitionName=='AcrPull'].id"
    )
    if (-not [string]::IsNullOrWhiteSpace($assignments)) {
        return
    }

    & az role assignment create --assignee-object-id $PrincipalId --assignee-principal-type ServicePrincipal --role AcrPull --scope $AcrId | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to grant AcrPull on $AcrId to principal $PrincipalId"
    }
}

function Ensure-Environment {
    param(
        [Parameter(Mandatory = $true)][string]$WorkspaceId,
        [Parameter(Mandatory = $true)][string]$WorkspaceKey
    )

    if (Test-AzSuccess -Arguments @("containerapp", "env", "show", "--name", $EnvironmentName, "--resource-group", $ResourceGroup)) {
        return
    }

    & az containerapp env create --name $EnvironmentName --resource-group $ResourceGroup --location $Location --enable-workload-profiles --logs-workspace-id $WorkspaceId --logs-workspace-key $WorkspaceKey | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Container Apps environment $EnvironmentName"
    }
}

function Ensure-WorkloadProfile {
    $profiles = Invoke-AzJson @("containerapp", "env", "workload-profile", "list", "--resource-group", $ResourceGroup, "--name", $EnvironmentName)
    foreach ($profile in @($profiles)) {
        if ($profile.name -eq $WorkloadProfileName) {
            return
        }
    }

    & az containerapp env workload-profile add --resource-group $ResourceGroup --name $EnvironmentName --workload-profile-name $WorkloadProfileName --workload-profile-type $WorkloadProfileType | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to add workload profile $WorkloadProfileName to $EnvironmentName"
    }
}

function Build-Image {
    param(
        [Parameter(Mandatory = $true)][string]$Dockerfile,
        [Parameter(Mandatory = $true)][string]$ImageName
    )

    Write-Host "Building $ImageName in Azure Container Registry..."
    & az acr build --registry $RegistryName --image $ImageName --file (Join-Path $scriptRoot $Dockerfile) --no-logs --only-show-errors $scriptRoot | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build image $ImageName with $Dockerfile"
    }
}

function New-TemplateFile {
    param([Parameter(Mandatory = $true)][hashtable]$Template)

    $path = [System.IO.Path]::GetTempFileName()
    $json = $Template | ConvertTo-Json -Depth 100
    Set-Content -LiteralPath $path -Value $json -Encoding UTF8
    return $path
}

function Deploy-ArmTemplate {
    param(
        [Parameter(Mandatory = $true)][string]$DeploymentName,
        [Parameter(Mandatory = $true)][string]$TemplateFile,
        [Parameter(Mandatory = $true)][string[]]$Parameters
    )

    & az deployment group create --resource-group $ResourceGroup --name $DeploymentName --template-file $TemplateFile --parameters @Parameters | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed deployment $DeploymentName"
    }
}

function New-ProbeJobTemplate {
    return @{
        '$schema' = 'https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#'
        contentVersion = '1.0.0.0'
        parameters = @{
            location = @{ type = 'string' }
            environmentId = @{ type = 'string' }
            identityId = @{ type = 'string' }
            registryServer = @{ type = 'string' }
            image = @{ type = 'string' }
            jobName = @{ type = 'string' }
            workloadProfileName = @{ type = 'string' }
        }
        resources = @(
            @{
                type = 'Microsoft.App/jobs'
                apiVersion = '2025-01-01'
                name = "[parameters('jobName')]"
                location = "[parameters('location')]"
                identity = @{
                    type = 'UserAssigned'
                    userAssignedIdentities = @{
                        "[parameters('identityId')]" = @{}
                    }
                }
                properties = @{
                    environmentId = "[parameters('environmentId')]"
                    workloadProfileName = "[parameters('workloadProfileName')]"
                    configuration = @{
                        triggerType = 'Manual'
                        replicaTimeout = 600
                        replicaRetryLimit = 0
                        manualTriggerConfig = @{
                            parallelism = 1
                            replicaCompletionCount = 1
                        }
                        registries = @(
                            @{
                                server = "[parameters('registryServer')]"
                                identity = "[parameters('identityId')]"
                            }
                        )
                    }
                    template = @{
                        containers = @(
                            @{
                                name = 'main'
                                image = "[parameters('image')]"
                                resources = @{
                                    cpu = 8
                                    memory = '56Gi'
                                }
                            }
                        )
                    }
                }
            }
        )
    }
}

function New-RunnerJobTemplate {
    return @{
        '$schema' = 'https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#'
        contentVersion = '1.0.0.0'
        parameters = @{
            location = @{ type = 'string' }
            environmentId = @{ type = 'string' }
            identityId = @{ type = 'string' }
            registryServer = @{ type = 'string' }
            image = @{ type = 'string' }
            jobName = @{ type = 'string' }
            workloadProfileName = @{ type = 'string' }
            githubPat = @{ type = 'secureString' }
            githubApiUrl = @{ type = 'string' }
            githubOwner = @{ type = 'string' }
            githubRepo = @{ type = 'string' }
            runnerLabels = @{ type = 'string' }
            pollingInterval = @{ type = 'int' }
            maxExecutions = @{ type = 'int' }
        }
        resources = @(
            @{
                type = 'Microsoft.App/jobs'
                apiVersion = '2025-01-01'
                name = "[parameters('jobName')]"
                location = "[parameters('location')]"
                identity = @{
                    type = 'UserAssigned'
                    userAssignedIdentities = @{
                        "[parameters('identityId')]" = @{}
                    }
                }
                properties = @{
                    environmentId = "[parameters('environmentId')]"
                    workloadProfileName = "[parameters('workloadProfileName')]"
                    configuration = @{
                        triggerType = 'Event'
                        replicaTimeout = 5400
                        replicaRetryLimit = 0
                        secrets = @(
                            @{
                                name = 'personal-access-token'
                                value = "[parameters('githubPat')]"
                            }
                        )
                        registries = @(
                            @{
                                server = "[parameters('registryServer')]"
                                identity = "[parameters('identityId')]"
                            }
                        )
                        eventTriggerConfig = @{
                            parallelism = 1
                            replicaCompletionCount = 1
                            scale = @{
                                minExecutions = 0
                                maxExecutions = "[parameters('maxExecutions')]"
                                pollingInterval = "[parameters('pollingInterval')]"
                                rules = @(
                                    @{
                                        name = 'github-runner'
                                        type = 'github-runner'
                                        metadata = @{
                                            githubAPIURL = "[parameters('githubApiUrl')]"
                                            owner = "[parameters('githubOwner')]"
                                            runnerScope = 'repo'
                                            repos = "[parameters('githubRepo')]"
                                            labels = "[parameters('runnerLabels')]"
                                            targetWorkflowQueueLength = '1'
                                        }
                                        auth = @(
                                            @{
                                                triggerParameter = 'personalAccessToken'
                                                secretRef = 'personal-access-token'
                                            }
                                        )
                                    }
                                )
                            }
                        }
                    }
                    template = @{
                        containers = @(
                            @{
                                name = 'main'
                                image = "[parameters('image')]"
                                env = @(
                                    @{
                                        name = 'GITHUB_PAT'
                                        secretRef = 'personal-access-token'
                                    }
                                    @{
                                        name = 'GH_URL'
                                        value = "[format('https://github.com/{0}/{1}', parameters('githubOwner'), parameters('githubRepo'))]"
                                    }
                                    @{
                                        name = 'REGISTRATION_TOKEN_API_URL'
                                        value = "[format('{0}/repos/{1}/{2}/actions/runners/registration-token', parameters('githubApiUrl'), parameters('githubOwner'), parameters('githubRepo'))]"
                                    }
                                    @{
                                        name = 'RUNNER_LABELS'
                                        value = "[parameters('runnerLabels')]"
                                    }
                                )
                                resources = @{
                                    cpu = 8
                                    memory = '56Gi'
                                }
                            }
                        )
                    }
                }
            }
        )
    }
}

$account = Invoke-AzJson @("account", "show")
$subscriptionId = [string]$account.id
if ([string]::IsNullOrWhiteSpace($RegistryName)) {
    $suffix = ($subscriptionId -replace '-', '').Substring(0, 8).ToLowerInvariant()
    $RegistryName = "lsiemlfgpu$suffix"
}

Ensure-ContainerAppExtension
Ensure-Provider -Namespace "Microsoft.App"
Ensure-Provider -Namespace "Microsoft.OperationalInsights"
Ensure-Provider -Namespace "Microsoft.ManagedIdentity"
Ensure-Provider -Namespace "Microsoft.ContainerRegistry"
Ensure-ResourceGroup
Ensure-Workspace
Ensure-Registry
Ensure-Identity

$workspaceId = Invoke-AzTsv @("monitor", "log-analytics", "workspace", "show", "--resource-group", $ResourceGroup, "--workspace-name", $WorkspaceName, "--query", "customerId")
$workspaceKey = Invoke-AzTsv @("monitor", "log-analytics", "workspace", "get-shared-keys", "--resource-group", $ResourceGroup, "--workspace-name", $WorkspaceName, "--query", "primarySharedKey")
$identityId = Invoke-AzTsv @("identity", "show", "--name", $IdentityName, "--resource-group", $ResourceGroup, "--query", "id")
$identityPrincipalId = Invoke-AzTsv @("identity", "show", "--name", $IdentityName, "--resource-group", $ResourceGroup, "--query", "principalId")
$acrId = Invoke-AzTsv @("acr", "show", "--name", $RegistryName, "--resource-group", $ResourceGroup, "--query", "id")
$registryServer = Invoke-AzTsv @("acr", "show", "--name", $RegistryName, "--resource-group", $ResourceGroup, "--query", "loginServer")
Wait-ForRegistryDns -HostName $registryServer

Ensure-AcrPullRole -PrincipalId $identityPrincipalId -AcrId $acrId
Ensure-Environment -WorkspaceId $workspaceId -WorkspaceKey $workspaceKey
Ensure-WorkloadProfile

$environmentId = Invoke-AzTsv @("containerapp", "env", "show", "--name", $EnvironmentName, "--resource-group", $ResourceGroup, "--query", "id")
$shouldDeployRunner = (-not $ProbeOnly) -and (-not [string]::IsNullOrWhiteSpace($GitHubPat))
if ((-not $ProbeOnly) -and (-not $shouldDeployRunner)) {
    Write-Warning "GITHUB_PAT was not provided; the runner image build and runner job deployment will be skipped."
}

if (-not $SkipImageBuild) {
    Build-Image -Dockerfile "probe.Dockerfile" -ImageName $ProbeImageName
    if ($shouldDeployRunner) {
        Build-Image -Dockerfile "Dockerfile" -ImageName $RunnerImageName
    }
}

$probeTemplateFile = New-TemplateFile -Template (New-ProbeJobTemplate)
try {
    Deploy-ArmTemplate -DeploymentName "aca-gpu-probe-job" -TemplateFile $probeTemplateFile -Parameters @(
        "location=$Location",
        "environmentId=$environmentId",
        "identityId=$identityId",
        "registryServer=$registryServer",
        "image=$registryServer/$ProbeImageName",
        "jobName=$ProbeJobName",
        "workloadProfileName=$WorkloadProfileName"
    )
}
finally {
    Remove-Item -LiteralPath $probeTemplateFile -Force -ErrorAction SilentlyContinue
}

if ($StartProbe) {
    & az containerapp job start --resource-group $ResourceGroup --name $ProbeJobName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start probe job $ProbeJobName"
    }
}

if ($shouldDeployRunner) {
    $runnerTemplateFile = New-TemplateFile -Template (New-RunnerJobTemplate)
    try {
        Deploy-ArmTemplate -DeploymentName "aca-gpu-runner-job" -TemplateFile $runnerTemplateFile -Parameters @(
            "location=$Location",
            "environmentId=$environmentId",
            "identityId=$identityId",
            "registryServer=$registryServer",
            "image=$registryServer/$RunnerImageName",
            "jobName=$RunnerJobName",
            "workloadProfileName=$WorkloadProfileName",
            "githubPat=$GitHubPat",
            "githubApiUrl=$GitHubApiUrl",
            "githubOwner=$GitHubOwner",
            "githubRepo=$GitHubRepo",
            "runnerLabels=$RunnerLabels",
            "pollingInterval=$PollingIntervalSeconds",
            "maxExecutions=$MaxExecutions"
        )
    }
    finally {
        Remove-Item -LiteralPath $runnerTemplateFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Resource group: $ResourceGroup"
Write-Host "Environment: $EnvironmentName"
Write-Host "Registry: $RegistryName ($registryServer)"
Write-Host "Identity: $IdentityName"
Write-Host "Probe job: $ProbeJobName"
Write-Host "Runner job: $RunnerJobName"
