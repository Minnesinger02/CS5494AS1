# Fast reproduction from the saved checkpoints (no training).
# Recomputes every metric used in the report, redraws every figure and rebuilds the PDF.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1
#   powershell ... -File .\reproduce.ps1 -Python "C:\path\to\python.exe"
#   powershell ... -File .\reproduce.ps1 -SkipReport      # skip LaTeX (no TeX Live needed)
#
# Full retraining is a separate script: .\run_windows.ps1  (~12 min on an RTX 5080).

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Python = '',
    [switch]$SkipReport
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Step {
    param([string]$Title, [string[]]$PyArgs)
    Write-Host ''
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "step failed: $Title (exit $LASTEXITCODE)" }
    Write-Host ("    ok  {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Yellow
}

# Interpreter resolution: -Python wins, then whatever is on PATH, then the path used
# on the machine that produced the results. This keeps the script portable for markers
# who clone the repository and run it on a different machine.
if (-not $Python) {
    foreach ($candidate in 'python', 'python3') {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source; break }
    }
}
if (-not $Python -and (Test-Path 'D:\HKStudy\.venv\Scripts\python.exe')) {
    $Python = 'D:\HKStudy\.venv\Scripts\python.exe'
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "python not found on PATH  (pass -Python <path>)"
}
if (-not (Test-Path 'results/autoregressive/made/model.pt')) {
    throw 'results/ is missing; run run_windows.ps1 to train from scratch'
}

Write-Host "python : $Python"
& $Python -c "import torch,sys;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',sys.version.split()[0])"

Step 'unit checks'            @('-m','cs5494.checks','units')
Step 'resample FVSBN'         @('-m','cs5494.sample','results/autoregressive','fvsbn')
Step 'resample MADE'          @('-m','cs5494.sample','results/autoregressive','made')
Step 'resample DCGAN (raw)'   @('-m','cs5494.sample','results/gan_ema','gan','--which','raw')
Step 'resample DCGAN (EMA)'   @('-m','cs5494.sample','results/gan_ema','gan','--which','ema')
Step 'score GAN variants'     @('-m','cs5494.evaluate','gan','--runs',
                                'results/gan_initial','results/gan_mlp',
                                'results/gan_mlp_shared_lr','results/gan_dcgan_repeat',
                                '--out','comparisons/gan_eval.json')
Step 'score DCGAN raw'        @('-m','cs5494.evaluate','gan','--runs','results/gan_ema','--weights','raw',
                                '--out','comparisons/gan_eval_ema.json')
Step 'score DCGAN EMA'        @('-m','cs5494.evaluate','gan','--runs','results/gan_ema','--weights','ema',
                                '--out','comparisons/gan_eval_ema_tmp.json')
Step 'mixture, 60k'           @('-m','cs5494.evaluate','mixture','--runs','results/made_seed42',
                                'results/made_seed43','results/made_seed44',
                                '--out','results/ensemble_metrics.json',
                                '--figure','results/ensemble_ablation.png')
Step 'mixture, 10k'           @('-m','cs5494.evaluate','mixture','--runs','results/made10k_seed42',
                                'results/made10k_seed43','results/made10k_seed44',
                                '--out','results/ensemble_10k_metrics.json',
                                '--figure','results/ensemble_10k_ablation.png')
Step 'per-digit grids'        @('-m','cs5494.figures','digits')
Step 'MLP variant figure'     @('-m','cs5494.figures','variants')
Step 'NLL re-check'           @('-m','cs5494.checks','nll','--runs','results/made_seed42',
                                'results/made_seed43','results/made_seed44')

if (-not $SkipReport) {
    Write-Host ''
    Write-Host '=== LaTeX report ===' -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File .\report\build.ps1
    if ($LASTEXITCODE -ne 0) { throw 'report build failed' }
    Step 'report numbers'     @('-m','cs5494.checks','report')
    Copy-Item report\main.pdf submission\60777841_61083251_60369937.pdf -Force
    Write-Host '    copied report/main.pdf -> submission/60777841_61083251_60369937.pdf' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Done. See submission/60777841_61083251_60369937.pdf and report/figures/.' -ForegroundColor Green
