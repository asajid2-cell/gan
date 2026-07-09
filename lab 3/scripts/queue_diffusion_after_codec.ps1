param([string]$CodecRunName = "run1056", [string]$DiffusionRunName = "run_d004", [int]$PollSeconds = 300)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$statePath = Join-Path $repoRoot ("saves2\lab3_codec_transfer\" + $CodecRunName + "\run_state.json")
$queueLog = Join-Path $repoRoot "saves2\_supervision\queued_diffusion.log"
New-Item -ItemType Directory -Force -Path (Split-Path $queueLog) | Out-Null

Add-Content -Path $queueLog -Value ("[queue] watching codec run " + $CodecRunName + " at " + (Get-Date).ToString("s"))

while ($true) {
    if (Test-Path $statePath) {
        $state = Get-Content $statePath -Raw | ConvertFrom-Json
        $stage1 = [bool]$state.stage1_done
        $stage2 = [bool]$state.stage2_done
        $stage3 = [bool]$state.stage3_done
        $current = [string]$state.current_stage
        Add-Content -Path $queueLog -Value ("[queue] codec current_stage=" + $current + " stage1=" + $stage1 + " stage2=" + $stage2 + " stage3=" + $stage3 + " @" + (Get-Date).ToString("s"))
        if ($current -eq "done" -or ($stage1 -and $stage2 -and $stage3)) {
            break
        }
    } else {
        Add-Content -Path $queueLog -Value ("[queue] waiting for state file @" + (Get-Date).ToString("s"))
    }
    Start-Sleep -Seconds $PollSeconds
}

Add-Content -Path $queueLog -Value ("[queue] codec finished, starting diffusion " + $DiffusionRunName + " @" + (Get-Date).ToString("s"))
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_diffusion_realism_run.ps1") -RunName $DiffusionRunName
if ($LASTEXITCODE -ne 0) {
    throw "Queued diffusion run failed"
}

Add-Content -Path $queueLog -Value ("[queue] diffusion finished @" + (Get-Date).ToString("s"))
