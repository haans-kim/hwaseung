# Python Embedded Runtime Setup Script for Hwaseung Dashboard
# This script downloads Python 3.10 Embedded and installs required packages

$ErrorActionPreference = "Stop"

$PYTHON_VERSION = "3.10.11"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-embed-amd64.zip"
$RUNTIME_DIR = "python-runtime"
$PYTHON_ZIP = "python-embedded.zip"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Runtime Setup for Hwaseung" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if runtime already exists
if (Test-Path $RUNTIME_DIR) {
    Write-Host "Python runtime directory already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/n)"
    if ($response -ne "y") {
        Write-Host "Skipping setup." -ForegroundColor Yellow
        exit 0
    }
    Write-Host "Removing existing runtime..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $RUNTIME_DIR
}

# Create runtime directory
Write-Host "Creating runtime directory..." -ForegroundColor Green
New-Item -ItemType Directory -Path $RUNTIME_DIR -Force | Out-Null

# Download Python Embedded
Write-Host "Downloading Python $PYTHON_VERSION Embedded..." -ForegroundColor Green
Write-Host "URL: $PYTHON_URL" -ForegroundColor Gray
Invoke-WebRequest -Uri $PYTHON_URL -OutFile $PYTHON_ZIP

# Extract Python
Write-Host "Extracting Python..." -ForegroundColor Green
Expand-Archive -Path $PYTHON_ZIP -DestinationPath $RUNTIME_DIR -Force
Remove-Item $PYTHON_ZIP

# Enable site-packages by uncommenting in python310._pth
Write-Host "Configuring Python paths..." -ForegroundColor Green
$pthFile = Get-ChildItem -Path $RUNTIME_DIR -Filter "python*._pth" | Select-Object -First 1
if ($pthFile) {
    $content = Get-Content $pthFile.FullName
    $content = $content -replace "#import site", "import site"
    $content | Set-Content $pthFile.FullName
    Write-Host "Enabled site-packages in $($pthFile.Name)" -ForegroundColor Gray
}

# Download get-pip.py
Write-Host "Downloading get-pip.py..." -ForegroundColor Green
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$getPipPath = Join-Path $RUNTIME_DIR "get-pip.py"
Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath

# Install pip
Write-Host "Installing pip..." -ForegroundColor Green
$pythonExe = Join-Path $RUNTIME_DIR "python.exe"
& $pythonExe $getPipPath --no-warn-script-location

# Verify pip installation
Write-Host "Verifying pip installation..." -ForegroundColor Green
& $pythonExe -m pip --version

# Install required packages from requirements.txt
Write-Host ""
Write-Host "Installing Python packages from requirements.txt..." -ForegroundColor Green

$requirementsPath = "..\backend\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "Found requirements.txt at: $requirementsPath" -ForegroundColor Gray
    & $pythonExe -m pip install -r $requirementsPath --no-warn-script-location

    if ($LASTEXITCODE -eq 0) {
        Write-Host "All packages installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Some packages failed to install. Check the output above." -ForegroundColor Red
    }
} else {
    Write-Host "WARNING: requirements.txt not found at $requirementsPath" -ForegroundColor Red
    Write-Host "Please install packages manually or check the path." -ForegroundColor Red
}

# Cleanup
Write-Host ""
Write-Host "Cleaning up..." -ForegroundColor Green
Remove-Item $getPipPath -ErrorAction SilentlyContinue

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python runtime location: $RUNTIME_DIR" -ForegroundColor White
Write-Host "Python executable: $pythonExe" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Test the runtime: $pythonExe --version" -ForegroundColor White
Write-Host "2. Run development mode: npm run electron:dev" -ForegroundColor White
Write-Host "3. Build production: npm run electron:build" -ForegroundColor White
Write-Host ""
