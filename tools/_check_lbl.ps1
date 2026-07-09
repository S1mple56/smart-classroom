$root = $PSScriptRoot
$lblTrainDir = Join-Path $root "datasets\bvn\labels\train"
Write-Host "Checking: $lblTrainDir"
Write-Host "Exists: $(Test-Path $lblTrainDir)"
$files = Get-ChildItem $lblTrainDir -ErrorAction SilentlyContinue
Write-Host "File count: $($files.Count)"
$files | Select-Object -First 30 | ForEach-Object { Write-Host "  $($_.Name) ($($_.Length) bytes)" }
