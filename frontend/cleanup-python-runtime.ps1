# Python Runtime Cleanup Script for Production Build
# Removes unnecessary packages to reduce size

$ErrorActionPreference = "Stop"

$RUNTIME_DIR = "python-runtime"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Runtime Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $RUNTIME_DIR)) {
    Write-Host "Error: $RUNTIME_DIR not found!" -ForegroundColor Red
    exit 1
}

$sitePackages = Join-Path $RUNTIME_DIR "Lib\site-packages"

if (-not (Test-Path $sitePackages)) {
    Write-Host "Error: site-packages not found!" -ForegroundColor Red
    exit 1
}

# Packages to remove (not needed in production)
$packagesToRemove = @(
    "streamlit",           # 30MB - Web UI framework
    "IPython",             # 5.7MB - Jupyter notebook
    "debugpy",             # 31MB - Debugger
    "pip",                 # 11MB - Package installer
    "setuptools",          # 7.9MB - Build tools
    "Cython",              # 12MB - Compiler
    "jedi",                # 8.1MB - Code completion
    "pygments",            # 8.6MB - Syntax highlighting (for Jupyter)
    "examples",            # 6.7MB - Example files
    "docs",                # 7MB - Documentation
    "jupyterlab_plotly",   # 9MB - Jupyter extension
    "pydeck",              # 14MB - Map visualization (if not used)
    "altair"               # 8.8MB - Alternative visualization (if not used)
)

$totalSaved = 0

foreach ($package in $packagesToRemove) {
    $packagePath = Join-Path $sitePackages $package

    if (Test-Path $packagePath) {
        $size = (Get-ChildItem $packagePath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "Removing: $package ($([math]::Round($size, 2)) MB)" -ForegroundColor Yellow

        Remove-Item -Path $packagePath -Recurse -Force
        $totalSaved += $size
    } else {
        Write-Host "Skipping: $package (not found)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "Total space saved: $([math]::Round($totalSaved, 2)) MB" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Run this AFTER 'setup-python-runtime.ps1'" -ForegroundColor Yellow
Write-Host "      and BEFORE 'npm run electron:build'" -ForegroundColor Yellow
