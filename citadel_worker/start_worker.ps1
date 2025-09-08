<#  start_master.ps1 ─ Start engine server in the ‘master’ conda env (no admin)  #>

[CmdletBinding()]
param(
    [string]$EnvName = 'worker'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Where this script lives
$ScriptDir = $PSScriptRoot

# Conda binaries
$MINICONDA_DIR = Join-Path $env:LOCALAPPDATA 'Miniconda3'
$condaExe      = Join-Path $MINICONDA_DIR 'Scripts\conda.exe'
Write-Host "Miniconda directory: $MINICONDA_DIR , conda executable: $condaExe"
if (-not (Test-Path $condaExe)) {
    Write-Error "Conda not found. Run create_conda.ps1 first.";  exit 1
    
}

# Code that the new window will execute
$inner = @"
& `"$condaExe`" shell.powershell hook | Out-String | Invoke-Expression   # enable 'conda activate'
conda activate $EnvName                                                  # switch to env
Set-Location -LiteralPath `"$ScriptDir`"                                 # cd to project dir
`$env:PYTHONPATH = '..;..\proto\python'                                  # ← EXACT assignment
python .\start_worker_server.py                                          # run server
"@

Write-Host "Opening new PowerShell in '$ScriptDir' (env '$EnvName') …"
Start-Process powershell.exe `
    -WorkingDirectory $ScriptDir `
    -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-Command',$inner
