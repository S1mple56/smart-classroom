$root = $PSScriptRoot

$imgTrainDir = Join-Path $root "datasets\bvn\images\train"
$lblTrainDir = Join-Path $root "datasets\bvn\labels\train"
$imgValDir = Join-Path $root "datasets\bvn\images\val"
$lblValDir = Join-Path $root "datasets\bvn\labels\val"

$trainLblNames = @{}
Get-ChildItem $lblTrainDir -File | ForEach-Object { $trainLblNames[$_.BaseName] = $true }
$valLblNames = @{}
Get-ChildItem $lblValDir -File | ForEach-Object { $valLblNames[$_.BaseName] = $true }

$totalSize = 0
$totalCount = 0

Write-Host "===== Train orphan images (will DELETE) ====="
Get-ChildItem $imgTrainDir -File | ForEach-Object {
    if (-not $trainLblNames.ContainsKey($_.BaseName)) {
        $sizeKB = [math]::Round($_.Length/1KB, 1)
        $totalSize += $_.Length
        $totalCount++
        Write-Host "  $($_.Name)  $sizeKB KB"
    }
}

Write-Host ""
Write-Host "===== Val orphan images (will DELETE) ====="
Get-ChildItem $imgValDir -File | ForEach-Object {
    if (-not $valLblNames.ContainsKey($_.BaseName)) {
        $sizeKB = [math]::Round($_.Length/1KB, 1)
        $totalSize += $_.Length
        $totalCount++
        Write-Host "  $($_.Name)  $sizeKB KB"
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "Total to delete: $totalCount images, $([math]::Round($totalSize/1MB,2)) MB"
