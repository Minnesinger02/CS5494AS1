# Build the submission package: code + results + report source + final PDFs + docs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\package.ps1
#   powershell ... -File .\package.ps1 -Out CS5494_Assignment.zip -WithBackup
#
# The archive of the original package (_archive_original_20261103) is included because the
# comparison tables in the report refer to it. The local rollback copy
# (_backup_20261104_pre_improve, ~97 MB) is excluded by default.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Out        = 'CS5494_Assignment_Package_20261104.zip',
    [string]$Python     = 'D:\HKStudy\.venv\Scripts\python.exe',
    [switch]$WithBackup,
    [switch]$WithData
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$root  = $PSScriptRoot
$stage = Join-Path $root 'CS5494_Assignment_Package'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

# --- files that belong at the package root ------------------------------------------------
$files = @(
    'README.md', 'REPRODUCE.md', 'RESULTS.md', 'HYPERPARAM_SWEEP.md',
    '00_组内对比说明.md', '01_检查记录.md', 'discussion.txt',
    'requirements.txt', 'requirements-lock.txt',
    'sweep_made.csv', 'sweep_made2.csv', 'sweep_made3.csv', 'sweep_made4.csv',
    'assignment.ipynb',
    'run_windows.ps1', 'reproduce.ps1', 'package.ps1'
)
# --- directories --------------------------------------------------------------------------
$dirs = @('cs5494', 'report', 'submission', 'results', 'comparisons', 'logs')
if ($WithData)   { $dirs += 'data' }
if ($WithBackup) { $dirs += '_backup_20261104_pre_improve' }

foreach ($f in $files) {
    if (-not (Test-Path (Join-Path $root $f))) { Write-Warning "missing file: $f"; continue }
    Copy-Item (Join-Path $root $f) $stage -Force
}
foreach ($d in $dirs) {
    if (-not (Test-Path (Join-Path $root $d))) { Write-Warning "missing folder: $d"; continue }
    Copy-Item (Join-Path $root $d) $stage -Recurse -Force
}

# drop build artefacts that are not part of the deliverable
Get-ChildItem $stage -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
foreach ($ext in 'aux', 'bcf', 'blg', 'run.xml', 'out', 'toc') {
    Get-ChildItem $stage -Recurse -File -Filter "*.$ext" | Where-Object { $_.Name -like 'main.*' } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

# --- integrity manifest inside the package ------------------------------------------------
if (Test-Path $Python) {
    & $Python -m cs5494.checks manifest $stage --out (Join-Path $stage 'MANIFEST_SHA256.json')
} else {
    Write-Warning "python not found ($Python); package built without a manifest"
}

# --- zip ----------------------------------------------------------------------------------
$zip = Join-Path $root $Out
if (Test-Path $zip) { Remove-Item -Force $zip }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $stage, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)

$files = Get-ChildItem $stage -Recurse -File
"packaged : {0}" -f $Out
"files    : {0}" -f $files.Count
"raw size : {0:N1} MB" -f (($files | Measure-Object Length -Sum).Sum/1MB)
"zip size : {0:N1} MB" -f ((Get-Item $zip).Length/1MB)
Remove-Item -Recurse -Force $stage
