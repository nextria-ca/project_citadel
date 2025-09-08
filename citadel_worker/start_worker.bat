@echo off
setlocal enabledelayedexpansion
 
REM Set the name of the Conda environment
set ENVNAME=worker
 
REM Set the path to your Miniconda installation
set "MINICONDA_DIR=%LOCALAPPDATA%\Miniconda3"
 
REM Conda activation script path (not conda.exe)
set "CONDA_BAT=%MINICONDA_DIR%\Scripts\activate.bat"
 
REM Debug: Show paths
echo Miniconda directory: %MINICONDA_DIR%
echo Conda activation script: %CONDA_BAT%
 
REM Ensure conda.bat exists
if not exist "%CONDA_BAT%" (
    echo Conda not found at "%CONDA_BAT%"
    exit /b 1
)
 
REM Set working directory to the script location
cd /d "%~dp0"
 
REM Set PYTHONPATH (exact assignment)
set PYTHONPATH=..\;..\proto\python
 
REM Call conda.bat to activate environment and run python script
call "%CONDA_BAT%" %ENVNAME%
python start_worker_server.py
