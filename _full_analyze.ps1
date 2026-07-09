$root = $PSScriptRoot

Write-Host "===== [1] Orphan images (img w/o label) ====="
$lblT = @{}
Get-ChildItem "$root\datasets\bvn\labels\train" -File | ForEach-Object { $lblT[$_.BaseName] = 1 }
$lblV = @{}
Get-ChildItem "$root\datasets\bvn\labels\val" -File | ForEach-Object { $lblV[$_.BaseName] = 1 }
$oSize = 0; $oCount = 0
Get-ChildItem "$root\datasets\bvn\images\train" -File | ForEach-Object { if (-not $lblT[$_.BaseName]) { $oSize += $_.Length; $oCount++ } }
Get-ChildItem "$root\datasets\bvn\images\val" -File | ForEach-Object { if (-not $lblV[$_.BaseName]) { $oSize += $_.Length; $oCount++ } }
Write-Host "  Size: $([math]::Round($oSize/1KB,0)) KB, Files: $oCount"

Write-Host ""
Write-Host "===== [2] __pycache__ ====="
$pyc = Get-ChildItem "$root\__pycache__" -File -ErrorAction SilentlyContinue
$pycSize = ($pyc | Measure-Object Length -Sum).Sum
Write-Host "  Size: $([math]::Round($pycSize/1KB,0)) KB, Files: $($pyc.Count)"

Write-Host ""
Write-Host "===== [3] runs (training artifacts) ====="
$runs = Get-ChildItem "$root\runs" -Recurse -File -ErrorAction SilentlyContinue
$runsSize = ($runs | Measure-Object Length -Sum).Sum
Write-Host "  Size: $([math]::Round($runsSize/1MB,2)) MB, Files: $($runs.Count)"

Write-Host ""
Write-Host "===== [4] model files ====="
Get-ChildItem "$root\model" -File | Sort-Object Length -Descending | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB,2)) MB"
}

Write-Host ""
Write-Host "===== [5] rknn_convert ====="
$rknn = Get-ChildItem "$root\rknn_convert" -Recurse -File -ErrorAction SilentlyContinue
$rknnSize = ($rknn | Measure-Object Length -Sum).Sum
Write-Host "  Size: $([math]::Round($rknnSize/1MB,2)) MB, Files: $($rknn.Count)"

Write-Host ""
Write-Host "===== [6] data zip ====="
Get-ChildItem "$root\data\zipped-eval" -File -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB,2)) MB"
}

Write-Host ""
Write-Host "===== [7] dataset/ (maybe empty) ====="
$ds = Get-ChildItem "$root\dataset" -Recurse -File -ErrorAction SilentlyContinue
$dsSize = ($ds | Measure-Object Length -Sum).Sum
Write-Host "  Size: $([math]::Round($dsSize/1KB,0)) KB, Files: $($ds.Count)"

Write-Host ""
Write-Host "==================== SUMMARY ===================="
Write-Host ("Orphan images:      {0,8:N2} MB  ({1} files)" -f ($oSize/1MB), $oCount)
Write-Host ("__pycache__:        {0,8:N2} MB  ({1} files)" -f ($pycSize/1MB), $pyc.Count)
Write-Host ("runs artifacts:     {0,8:N2} MB  ({1} files)" -f ($runsSize/1MB), $runs.Count)
Write-Host ("rknn_convert:       {0,8:N2} MB  ({1} files)" -f ($rknnSize/1MB), $rknn.Count)
$total = $oSize + $pycSize + $runsSize + $rknnSize
Write-Host ("------------------------------------")
Write-Host ("TOTAL cleanable:    {0,8:N2} MB" -f ($total/1MB))
Write-Host ""
Write-Host "Also in model/: adam.pdopt = 27 MB (optimizer state, not needed for inference)"
