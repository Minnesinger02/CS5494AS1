# CS5494 pipeline: FVSBN, three MADE members + mixture, MLP-GAN baseline, DCGAN with EMA, LaTeX report.
# Windows / RTX 5080.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_windows.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_windows.ps1 -SkipTests
#
# Writes to results/ by default; existing folders are overwritten on purpose (same seeds).

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Python  = '',
    [string]$Outputs = 'results',
    [int[]]$Seeds    = @(42, 43, 44),
    [int]$EpochsAR   = 400,
    [int]$BatchAR    = 1024,
    [double]$LrAR    = 0.001,
    [int]$EpochsGAN  = 400,
    [int]$BatchGAN   = 512,
    [double]$LrGAN   = 0.0002,
    [double]$LrD     = 0.00002,
    [double]$Ema     = 0.999,
    [int]$Workers    = 4,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$script:TotalWall = 0.0
function Invoke-Step {
    param([string]$Title, [string]$Log, [string[]]$PyArgs)
    Write-Host ''
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ("    python " + ($PyArgs -join ' '))
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @PyArgs 2>&1 | Tee-Object -FilePath $Log | Out-Host
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Host ("    exit={0}  wall={1:N1}s" -f $code, $sw.Elapsed.TotalSeconds) -ForegroundColor Yellow
    if ($code -ne 0) { throw "step failed: $Title (exit $code)" }
    $script:TotalWall += $sw.Elapsed.TotalSeconds
}

# Interpreter resolution: -Python wins, then whatever is on PATH, then the path used on the
# machine that produced the bundled results (so the script also runs on a fresh clone).
if (-not $Python) {
    foreach ($candidate in 'python', 'python3') {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source; break }
    }
}
if (-not $Python -and (Test-Path 'D:\HKStudy\.venv\Scripts\python.exe')) {
    $Python = 'D:\HKStudy\.venv\Scripts\python.exe'
}
if (-not $Python -or -not (Test-Path $Python)) { throw "python not found on PATH (pass -Python <path>)" }
New-Item -ItemType Directory -Force -Path $Outputs, 'logs', 'comparisons' | Out-Null

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "started : $stamp"
Write-Host "outputs : $Outputs"

if (-not $SkipTests) {
    Invoke-Step 'unit checks (causal gradients / shapes)' 'logs\test_models.log' @('-m','cs5494.checks','units')
}

# --- 1. three MADE members, 60k and 10k ------------------------------------------------
foreach ($s in $Seeds) {
    Invoke-Step "MADE 60k seed $s" "logs\made_seed$s.log" @(
        '-m','cs5494.train','--models','made','--samples','60000','--test-samples','10000',
        '--side','28','--hidden','1536','--layers','1',
        '--epochs',"$EpochsAR",'--batch',"$BatchAR",'--lr',"$LrAR",'--scheduler','cosine',
        '--device','cuda','--workers',"$Workers",'--seed',"$s",
        "--output=$Outputs/made_seed$s")
}
foreach ($s in $Seeds) {
    Invoke-Step "MADE 10k seed $s" "logs\made10k_seed$s.log" @(
        '-m','cs5494.train','--models','made','--samples','10000','--test-samples','10000',
        '--side','28','--hidden','1536','--layers','1',
        '--epochs',"$EpochsAR",'--batch',"$BatchAR",'--lr',"$LrAR",'--scheduler','cosine',
        '--device','cuda','--workers',"$Workers",'--seed',"$s",
        "--output=$Outputs/made10k_seed$s")
}

# --- 2. mixture likelihood --------------------------------------------------------------
$mainRuns = $Seeds | ForEach-Object { "$Outputs/made_seed$_" }
$ablaRuns = $Seeds | ForEach-Object { "$Outputs/made10k_seed$_" }
Invoke-Step 'mixture evaluation, 60k' 'logs\ensemble.log' (@('-m','cs5494.evaluate','mixture','--runs') + $mainRuns + @(
    '--out', "$Outputs/ensemble_metrics.json", '--figure', "$Outputs/ensemble_ablation.png"))
Invoke-Step 'mixture evaluation, 10k' 'logs\ensemble_10k.log' (@('-m','cs5494.evaluate','mixture','--runs') + $ablaRuns + @(
    '--out', "$Outputs/ensemble_10k_metrics.json", '--figure', "$Outputs/ensemble_10k_ablation.png"))

# --- 3. DCGAN with EMA ------------------------------------------------------------------
Invoke-Step 'DCGAN with EMA, 60k' 'logs\gan_ema.log' @(
    '-m','cs5494.train','--models','gan','--arch','dcgan',
    '--samples','60000','--test-samples','10000','--epochs',"$EpochsGAN",'--batch',"$BatchGAN",
    '--lr',"$LrGAN",'--lr-d',"$LrD",'--scheduler','cosine','--ema',"$Ema",
    '--device','cuda','--workers',"$Workers",'--output',"$Outputs/gan_ema")

Invoke-Step 'resample GAN (live weights)' 'logs\sample.log' @('-m','cs5494.sample', "$Outputs/gan_ema", 'gan', '--which', 'raw')
Invoke-Step 'resample GAN (moving average)' 'logs\sample.log' @('-m','cs5494.sample', "$Outputs/gan_ema", 'gan', '--which', 'ema')
Invoke-Step 'score GAN (live weights)' 'logs\eval_gan_raw.log' @(
    '-m','cs5494.evaluate','gan','--runs',"$Outputs/gan_ema",'--weights','raw','--out','comparisons/gan_eval_ema.json')
Invoke-Step 'score GAN (moving average)' 'logs\eval_gan_ema.log' @(
    '-m','cs5494.evaluate','gan','--runs',"$Outputs/gan_ema",'--weights','ema','--out','comparisons/gan_eval_ema_tmp.json')

# --- 4. figures used by the report ------------------------------------------------------
Invoke-Step 'per-digit showcase' 'logs\digit_grid.log' @('-m','cs5494.figures','digits', '--run', 'results')
Invoke-Step 'MLP variant comparison figure' 'logs\mlp_variants.log' @('-m','cs5494.figures','variants')

# --- 5. report --------------------------------------------------------------------------
if (Test-Path 'report/build.ps1') {
    Write-Host ''
    Write-Host '=== LaTeX report ===' -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File .\report\build.ps1
    if ($LASTEXITCODE -ne 0) { throw 'report build failed' }
    Invoke-Step 'report number check' 'logs\check_numbers.log' @('-m','cs5494.checks','report')
    Copy-Item report\main.pdf submission\60777841.pdf -Force
}

$info = @"
CS5494 pipeline
started        : $stamp
finished       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
python         : $Python
outputs        : $Outputs
seeds          : $($Seeds -join ',')
epochs (AR/GAN): $EpochsAR / $EpochsGAN
batch (AR/GAN) : $BatchAR / $BatchGAN
lr (AR/GAN/D)  : $LrAR / $LrGAN / $LrD
ema decay      : $Ema
wall seconds   : $([math]::Round($script:TotalWall,1))
"@
[System.IO.File]::WriteAllText((Join-Path $Outputs 'RUN_INFO.txt'), "$info`r`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Host ''
Write-Host $info -ForegroundColor Green
