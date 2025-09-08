<#  create_conda.ps1 ─ Install Miniconda (if needed) and create the ‘worker’ env  #>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ─── Settings ──────────────────────────────────────────────────
$ENV_NAME      = 'worker'               # <─ conda env name
$PY_VERSION    = '3.13.2'               # create …python=$PY_VERSION
$MINICONDA_URL = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
$MINICONDA_DIR = Join-Path $env:LOCALAPPDATA 'Miniconda3'
# ───────────────────────────────────────────────────────────────

$WORK_DIR = $PSScriptRoot
$condaExe = Join-Path $MINICONDA_DIR 'Scripts\conda.exe'

# --- 1. Ensure Miniconda is present ------------------------------------------
if (-not (Test-Path $condaExe)) {
    Write-Host "→ Installing Miniconda to $MINICONDA_DIR …"
    $miniExe = Join-Path $env:TEMP 'miniconda.exe'
    Invoke-WebRequest $MINICONDA_URL -OutFile $miniExe
    Start-Process $miniExe -ArgumentList '/InstallationType=JustMe','/AddToPath=1','/RegisterPython=0','/S',"/D=$MINICONDA_DIR" -Wait
    Remove-Item $miniExe
}

# --- 2. Enable ‘conda’ inside PowerShell -------------------------------------
& $condaExe shell.powershell hook | Out-String | Invoke-Expression
& $condaExe config --set auto_activate false

# --- 3. Create / update the ‘worker’ env -------------------------------------
$envExists = & $condaExe env list | Select-String -Quiet -Pattern "^\s*${ENV_NAME}\s"
if (-not $envExists) {
    Write-Host "→ Creating conda env '$ENV_NAME' (Python $PY_VERSION)…"
    & $condaExe create -y -n $ENV_NAME python=$PY_VERSION
}

Write-Host "→ Installing requirements into '$ENV_NAME'…"
& $condaExe run -n $ENV_NAME python -m pip install --upgrade pip
& $condaExe run -n $ENV_NAME python -m pip install --no-cache-dir -r "$WORK_DIR\requirements.txt"

