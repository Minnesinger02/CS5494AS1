# Build the LaTeX report (XeLaTeX + Biber, as required by the CityUHK template conventions).
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -Clean
param([switch]$Clean)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PATH = "D:\texlive\2026\bin\windows;$env:PATH"
# The path above is where TeX Live lived on the machine that produced the report. It is only
# prepended when it exists, so the script also works with any TeX installation already on PATH.
$texlive = 'D:\texlive\2026\bin\windows'
if (Test-Path $texlive) { $env:PATH = "$texlive;$env:PATH" }
foreach ($tool in 'xelatex', 'biber') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found: install TeX Live or MiKTeX (or add its bin directory to PATH). " +
              'The already compiled PDF is in ../submission/ if you only need the report.'
    }
}

Push-Location $PSScriptRoot
try {
    if ($Clean) {
        Remove-Item main.aux, main.bbl, main.bcf, main.blg, main.log, main.run.xml, main.out, main.toc -ErrorAction SilentlyContinue
    }
    & xelatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'xelatex pass 1 failed (see main.log)' }
    & biber main | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'biber failed (see main.blg)' }
    & xelatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'xelatex pass 2 failed' }
    & xelatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'xelatex pass 3 failed' }

    $log = Get-Content main.log -Raw
    $warn = [regex]::Matches($log, 'Overfull \\hbox \((\d+\.\d+)pt too wide')
    $big = @($warn | Where-Object { [double]$_.Groups[1].Value -gt 5 })
    "compiled: main.pdf ({0:N0} KB)" -f ((Get-Item main.pdf).Length/1KB)
    "overfull > 5pt: {0}" -f $big.Count
    if ($log -match 'LaTeX Warning: Citation .* undefined') { 'WARNING: undefined citations' }
    if ($log -match 'LaTeX Warning: Reference .* undefined') { 'WARNING: undefined references' }
}
finally {
    Pop-Location
}
