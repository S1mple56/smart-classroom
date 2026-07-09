$root = $PSScriptRoot

Write-Host "===== Top-level folders ====="
Get-ChildItem $root -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 2)
    Write-Host ("  {0,-25} {1,10:N2} MB" -f $_.Name, $sizeMB)
}

Write-Host ""
Write-Host "===== Total ====="
$total = (Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$totalGB = [math]::Round($total / 1GB, 2)
$count = (Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host ("  Total size: {0:N2} GB" -f $totalGB)
Write-Host ("  Total files: {0}" -f $count)

Write-Host ""
Write-Host "===== datasets/bvn sub-folders ====="
$bvn = Join-Path $root "datasets\bvn"
if (Test-Path $bvn) {
    Get-ChildItem $bvn -Directory | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 2)
        $cnt = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host ("  {0,-25} {1,10:N2} MB  ({2} files)" -f $_.Name, $sizeMB, $cnt)
    }
}
