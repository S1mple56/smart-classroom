$root = $PSScriptRoot
$bvn = Join-Path $root "datasets\bvn"

# 1. Check image sizes to find abnormal ones
Write-Host "===== Top 30 largest images ====="
Get-ChildItem (Join-Path $bvn "images") -Recurse -File | Sort-Object Length -Descending | Select-Object -First 30 | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 0)
    $parent = Split-Path $_.DirectoryName -Leaf
    Write-Host ("  {0}/{1}  {2} KB  {3}" -f $parent, $_.Name, $sizeKB, $_.FullName)
}

Write-Host ""
Write-Host "===== Images with no labels ====="
$imgFiles = Get-ChildItem (Join-Path $bvn "images") -Recurse -File | ForEach-Object { $_.BaseName }
$lblFiles = Get-ChildItem (Join-Path $bvn "labels") -Recurse -File | ForEach-Object { $_.BaseName }
$noLabel = $imgFiles | Where-Object { $_ -notin $lblFiles }
foreach ($n in $noLabel) { Write-Host "  No label: $n" }

Write-Host ""
Write-Host "===== Labels with no images ====="
$noImg = $lblFiles | Where-Object { $_ -notin $imgFiles }
foreach ($n in $noImg) { Write-Host "  No image: $n" }

Write-Host ""
Write-Host "===== Zero-byte or suspicious files ====="
Get-ChildItem (Join-Path $bvn "images") -Recurse -File | Where-Object { $_.Length -eq 0 } | ForEach-Object { Write-Host "  Zero-byte image: $($_.FullName)" }
Get-ChildItem (Join-Path $bvn "labels") -Recurse -File | Where-Object { $_.Length -eq 0 } | ForEach-Object { Write-Host "  Zero-byte label: $($_.FullName)" }

Write-Host ""
Write-Host "===== Image files with unusual extensions ====="
Get-ChildItem (Join-Path $bvn "images") -Recurse -File | Where-Object { $_.Extension -notin @('.jpg','.jpeg','.png','.bmp','.webp') } | ForEach-Object { Write-Host "  $($_.Name)" }

Write-Host ""
Write-Host "===== Train/Val splits ====="
$trainDir = Join-Path $bvn "images\train"
$valDir = Join-Path $bvn "images\val"
if (Test-Path $trainDir) {
    $tCount = (Get-ChildItem $trainDir -File).Count
    $tSize = (Get-ChildItem $trainDir -File | Measure-Object Length -Sum).Sum
    Write-Host "  Train: $tCount files, $([math]::Round($tSize/1MB,2)) MB"
}
if (Test-Path $valDir) {
    $vCount = (Get-ChildItem $valDir -File).Count
    $vSize = (Get-ChildItem $valDir -File | Measure-Object Length -Sum).Sum
    Write-Host "  Val:   $vCount files, $([math]::Round($vSize/1MB,2)) MB"
}

Write-Host ""
Write-Host "===== Large files outside datasets ====="
Get-ChildItem $root -Recurse -File | Where-Object { $_.Length -gt 10MB } | Sort-Object Length -Descending | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  $sizeMB MB  $($_.FullName)"
}
