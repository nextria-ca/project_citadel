<#  create_conda.ps1 ─ Install/Update Miniconda + create & hydrate a conda env  #>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ─── Settings ──────────────────────────────────────────────────────────────
$PY_VERSION     = '3.13.2'                                   # Python version
$ENV_NAME       = 'master'                                   # Conda env name
$MINICONDA_URL  = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
$MINICONDA_DIR  = Join-Path $env:LOCALAPPDATA 'Miniconda3'
# ───────────────────────────────────────────────────────────────────────────

$WORK_DIR  = $PSScriptRoot
$REQ_FILE  = Join-Path $WORK_DIR 'requirements.txt'
$condaExe  = Join-Path $MINICONDA_DIR 'Scripts\conda.exe'

# ─── Miniconda install / update ───────────────────────────────────────────
$condaOK = $false
if (Test-Path $condaExe) {
    try { & $condaExe --version | Out-Null; $condaOK = $true }
    catch { Write-Warning "Miniconda detected but not functional; will reinstall." }
}

if (-not $condaOK) {
    Write-Host "Installing Miniconda to $MINICONDA_DIR …"
    $miniExe = Join-Path $env:TEMP 'miniconda.exe'
    Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $miniExe
    Start-Process $miniExe -ArgumentList `
        '/InstallationType=JustMe', '/AddToPath=1', '/RegisterPython=0', '/S', "/D=$MINICONDA_DIR" `
        -NoNewWindow -Wait
    Remove-Item $miniExe
}

# Ensure current session sees conda
$env:PATH = "$MINICONDA_DIR;$MINICONDA_DIR\Scripts;$MINICONDA_DIR\Library\bin;$env:PATH"

# ─── (Re)create & hydrate env ─────────────────────────────────────────────
Write-Host "Creating/Updating conda env '$ENV_NAME' with Python $PY_VERSION …"

# Remove the env only if it already exists (avoid first-run failure)
$exists = & $condaExe env list | Select-String -SimpleMatch " $ENV_NAME "
if ($exists) {
    Write-Host "Removing existing env '$ENV_NAME' …"
    & $condaExe env remove -y -n $ENV_NAME | Out-Null
}

& $condaExe create -y -n $ENV_NAME "python=$PY_VERSION"

Write-Host "Installing requirements …"
if (Test-Path $REQ_FILE) {
    & $condaExe run -n $ENV_NAME python -m pip install --upgrade pip
    & $condaExe run -n $ENV_NAME pip install --no-cache-dir -r $REQ_FILE
}

Write-Host "`n✅ Env '$ENV_NAME' is ready."
