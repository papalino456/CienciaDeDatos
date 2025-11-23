# Create venv and install requirements
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workspace

$venvPython = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    & py -3 -m venv ".venv"
  } else {
    & python -m venv ".venv"
  }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r "requirements.txt" --disable-pip-version-check

Write-Output "Virtual environment ready at: $venvPython"
