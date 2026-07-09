$root = $PSScriptRoot

# Find images in train that have no label in labels/train
$imgTrainDir = Join-Path $root "datasets\bvn\images\train"
$lblTrainDir = Join-Path $root "datasets\bvn\labels\train"
$imgValDir = Join-Path $root "datasets\bvn\images\val"
$lblValDir = Join-Path $root "datasets\bvn\labels\val"

Write-Host "===== Train: images without labels ====="
$trainImgNames = Get-ChildItem $imgTrainDir -File -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
$trainLblNames = Get-ChildItem $lblTrainDir -File -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
$trainOrphans = $trainImgNames | Where-Object { $_ -notin $trainLblNames }
$orphanCount = ($trainOrphans | Measure-Object).Count
Write-Host "Train images: $(($trainImgNames | Measure-Object).Count), Train labels: $(($trainLblNames | Measure-Object).Count)"
Write-Host "Train orphans (img w/o label): $orphanCount"

Write-Host ""
Write-Host "===== Val: images without labels ====="
$valImgNames = Get-ChildItem $imgValDir -File -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
$valLblNames = Get-ChildItem $lblValDir -File -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
$valOrphans = $valImgNames | Where-Object { $_ -notin $valLblNames }
$valOrphanCount = ($valOrphans | Measure-Object).Count
Write-Host "Val images: $(($valImgNames | Measure-Object).Count), Val labels: $(($valLblNames | Measure-Object).Count)"
Write-Host "Val orphans (img w/o label): $valOrphanCount"

Write-Host ""
Write-Host "===== Train: labels without images ====="
$trainRevOrphans = $trainLblNames | Where-Object { $_ -notin $trainImgNames }
$trainRevCount = ($trainRevOrphans | Measure-Object).Count
Write-Host "Train label-only (no img): $trainRevCount"

Write-Host ""
Write-Host "===== Val: labels without images ====="
$valRevOrphans = $valLblNames | Where-Object { $_ -notin $valImgNames }
$valRevCount = ($valRevOrphans | Measure-Object).Count
Write-Host "Val label-only (no img): $valRevCount"

# Calculate total orphan size
Write-Host ""
Write-Host "===== Orphan images total size ====="
$orphanSize = 0
foreach ($name in $trainOrphans) {
    $f = Join-Path $imgTrainDir "$name.jpg"
    if (Test-Path $f) { $orphanSize += (Get-Item $f).Length }
}
foreach ($name in $valOrphans) {
    $f = Join-Path $imgValDir "$name.jpg"
    if (Test-Path $f) { $orphanSize += (Get-Item $f).Length }
}
Write-Host "Total orphan images size: $([math]::Round($orphanSize/1MB,2)) MB"
Write-Host "Total orphan count: $($orphanCount + $valOrphanCount)"
