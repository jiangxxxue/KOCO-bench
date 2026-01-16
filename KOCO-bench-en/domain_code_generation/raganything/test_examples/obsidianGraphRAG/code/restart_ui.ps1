# Restart Obsidian RAG UI Server
# This script stops any running instances and starts fresh

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Restarting Obsidian RAG UI Server" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop any running Python processes
Write-Host "[1/3] Stopping existing server processes..." -ForegroundColor Yellow

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "   Found $($pythonProcesses.Count) Python process(es)" -ForegroundColor Gray
    $pythonProcesses | Where-Object { $_.CommandLine -like "*run_ui*" -or $_.CommandLine -like "*uvicorn*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "   ✓ Stopped running servers" -ForegroundColor Green
    Start-Sleep -Seconds 2
} else {
    Write-Host "   ✓ No running processes found" -ForegroundColor Green
}

# Step 2: Clean up any port locks
Write-Host ""
Write-Host "[2/3] Checking port 8000..." -ForegroundColor Yellow

$portProcess = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portProcess) {
    Write-Host "   Port 8000 is in use, attempting to free..." -ForegroundColor Gray
    $processId = $portProcess.OwningProcess
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Write-Host "   ✓ Port 8000 freed" -ForegroundColor Green
    Start-Sleep -Seconds 2
} else {
    Write-Host "   ✓ Port 8000 is available" -ForegroundColor Green
}

# Step 3: Start the server
Write-Host ""
Write-Host "[3/3] Starting UI server..." -ForegroundColor Yellow
Write-Host ""

# Check if conda environment is activated
$condaEnv = $env:CONDA_DEFAULT_ENV
if ($condaEnv -ne "turing0.1") {
    Write-Host "WARNING: Not in conda environment 'turing0.1'" -ForegroundColor Red
    Write-Host "Please run: conda activate turing0.1" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host "✓ Conda environment: $condaEnv" -ForegroundColor Green
Write-Host ""

# Start the server
python run_ui.py


