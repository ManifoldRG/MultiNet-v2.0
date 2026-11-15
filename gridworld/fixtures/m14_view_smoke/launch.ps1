# Start the 16 M11 and M14 view smokes. API keys must already be in the environment.
$ErrorActionPreference = "Stop"
$root = Split-Path (Split-Path (Split-Path $PSScriptRoot))
Set-Location $root
if (-not $env:OPENAI_API_KEY -or -not $env:ANTHROPIC_API_KEY) {
    throw "OPENAI_API_KEY and ANTHROPIC_API_KEY must be set"
}
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "artifacts\m14_view_smoke\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pids = @()
Get-ChildItem (Join-Path $PSScriptRoot "run_config.*_*.json") | ForEach-Object {
    $name = $_.BaseName -replace "^run_config\.", ""
    $taskId = switch -Wildcard ($name) {
        "m11_*" { "25x25_dense_kr_sg_kb_0" }
        "mix_all_*" { "25x25_dense_kr_sg_kb_mix_all_0" }
        "mix_commitment_*" { "25x25_dense_kr_sg_kb_mix_commitment_0" }
        "mix_recoverable_*" { "25x25_dense_kr_sg_kb_mix_recoverable_0" }
        default { throw "no task id for $name" }
    }
    $art = Join-Path $root "artifacts\m14_view_smoke\$name"
    $out = Join-Path $logDir "$name.out.txt"
    $err = Join-Path $logDir "$name.err.txt"
    $argLine = "-u -m scripts.run_pipeline --run-config `"$($_.FullName)`" --artifacts-root `"$art`" --run-set-id $taskId --difficulty-max-static-score 2000000"
    $proc = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err -ArgumentList $argLine
    $pids += "$($proc.Id) $name"
    Write-Output "$($proc.Id) $name"
}
$pids | Set-Content (Join-Path $logDir "pids.txt")
