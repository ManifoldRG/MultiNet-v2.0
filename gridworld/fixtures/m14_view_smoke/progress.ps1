# Print the current step of each M14 2D/3D smoke.
$root = Split-Path (Split-Path (Split-Path $PSScriptRoot))
$logDir = Join-Path $root "artifacts\m14_view_smoke\logs"
$alive = @{}
if (Test-Path (Join-Path $logDir "pids.txt")) {
    foreach ($line in Get-Content (Join-Path $logDir "pids.txt")) {
        $id = [int]($line.Split(" ")[0])
        $name = $line.Substring($line.IndexOf(" ") + 1)
        $alive[$name] = $null -ne (Get-Process -Id $id -ErrorAction SilentlyContinue)
    }
}
Get-ChildItem (Join-Path $logDir "*.progress.json") | Sort-Object Name | ForEach-Object {
    $name = $_.BaseName -replace "\.progress$", ""
    $row = Get-Content $_.FullName -Raw | ConvertFrom-Json
    $step = if ($null -ne $row.step) { "$($row.step)/$($row.max_steps)" } else { "-" }
    $up = if ($alive.ContainsKey($name)) { $(if ($alive[$name]) { "up" } else { "down" }) } else { "?" }
    "{0,-28} {1,-4} {2,-8} {3,-8} q={4,-4} {5}" -f $name, $up, $row.status, $step, $row.query, $row.at
}
