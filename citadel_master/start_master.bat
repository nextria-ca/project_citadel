@echo off
setlocal enabledelayedexpansion
 
REM Set the name of the Conda environment
set ENVNAME=master
 
REM Set the path to your Miniconda installation
set "MINICONDA_DIR=%LOCALAPPDATA%\Miniconda3"
 
REM Conda script path (NOT conda.exe)
set "CONDA_BAT=%MINICONDA_DIR%\Scripts\activate.bat"
 
REM Ensure conda.bat exists
if not exist "%CONDA_BAT%" (
    echo Conda not found at "%CONDA_BAT%"
    exit /b 1
)
 
REM Set working directory to the script location
cd /d "%~dp0"
 
REM Set PYTHONPATH environment variable
set PYTHONPATH=..\;..\proto\python
 
REM Call conda.bat to activate environment and run python script
call "%CONDA_BAT%" %ENVNAME%
python start_master_server.py
