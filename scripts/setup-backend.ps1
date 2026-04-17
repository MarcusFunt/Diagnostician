param(
    [switch]$SkipModelPull,
    [switch]$SkipDocker,
    [string]$Python,
    [string]$OllamaBaseUrl = "http://localhost:11434"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not $Python) {
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
    } else {
        $Python = "python"
    }
}

if (-not (Test-Path (Join-Path $Root ".venv"))) {
    Write-Host "Creating backend virtual environment..."
    & $Python -m venv .venv
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}

Write-Host "Installing backend dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dev,pdf,multicare]"

$env:PYTHONPATH = Join-Path $Root "backend"

Write-Host "Scanning hardware and writing .env..."
$ScanOutput = & $Python -m diagnostician.setup.hardware --write-env --env-path ".env" --ollama-base-url $OllamaBaseUrl --json
$ScanJson = $ScanOutput -join [Environment]::NewLine
$Config = $ScanJson | ConvertFrom-Json

$QwenModel = [string]$Config.selection.case_generator_model
$MedModel = [string]$Config.selection.medical_check_model
$MedEnabled = [bool]$Config.selection.medical_check_enabled
$EmbeddingModel = [string]$Config.selection.embedding_model

Write-Host "Selected Qwen generator: $QwenModel"
if ($MedEnabled) {
    Write-Host "Selected medical checker: $MedModel"
} else {
    Write-Host "Medical checker disabled for limited local hardware."
}
Write-Host "Ollama keep_alive: $($Config.selection.ollama_keep_alive)"
foreach ($Warning in $Config.selection.warnings) {
    Write-Warning $Warning
}

if (-not $SkipModelPull) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama CLI was not found. Install Ollama and rerun this script, or rerun with -SkipModelPull."
    }

    Write-Host "Pulling Ollama models..."
    & ollama pull $QwenModel
    if ($MedEnabled) {
        & ollama pull $MedModel
    }
    & ollama pull $EmbeddingModel
}

if (-not $SkipDocker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI was not found. Install Docker Desktop and rerun this script, or rerun with -SkipDocker."
    }

    Write-Host "Starting backend Docker services..."
    & docker compose --env-file ".env" up --build -d postgres backend
}

Write-Host "Backend setup complete."
