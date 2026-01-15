@echo off
REM This script runs all Python unit tests in the current directory.

REM Activate the conda environment
call conda activate smolagents-test

REM Check if activation was successful
if %errorlevel% neq 0 (
    echo Failed to activate conda environment 'smolagents-test'.
    echo Please make sure conda is installed and the environment exists.
    exit /b 1
)

REM Run the test runner Python script
python "%~dp0\run_all_tests.py"

REM Deactivate the environment
call conda deactivate