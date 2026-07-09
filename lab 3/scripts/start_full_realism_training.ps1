$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

Write-Host "[full] codec realism run first"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_codec_realism_run.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Codec realism run failed"
}

Write-Host "[full] diffusion realism run second"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_diffusion_realism_run.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Diffusion realism run failed"
}

Write-Host "[full] all realism-focused training runs completed"
