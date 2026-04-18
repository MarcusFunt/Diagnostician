param(
  [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $root
try {
  docker compose up --build -d

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  do {
    try {
      $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
      $cases = Invoke-RestMethod -Uri "http://127.0.0.1:8000/cases" -TimeoutSec 5
      $frontend = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -TimeoutSec 5
      if ($health.ok -and $cases.total_estimate -gt 0 -and $frontend.StatusCode -eq 200) {
        Write-Host "Smoke test passed: backend healthy, cases seeded, frontend reachable."
        exit 0
      }
    } catch {
      Start-Sleep -Seconds 3
    }
  } while ((Get-Date) -lt $deadline)

  throw "Smoke test timed out before backend, seeded cases, and frontend were all reachable."
} finally {
  docker compose down
  Pop-Location
}
