@echo off
setlocal
REM Run all tests in one command using conda environment smolagents-test, executed in current directory (no cd required)
set ENV=smolagents-test
set SCRIPT_DIR=%~dp0

echo Running tests with conda environment: %ENV%
conda run -n %ENV% python "%SCRIPT_DIR%run_all_tests.py"
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo Tests failed with exit code %EXITCODE%
) else (
  echo Tests completed successfully.
)

exit /b %EXITCODE%