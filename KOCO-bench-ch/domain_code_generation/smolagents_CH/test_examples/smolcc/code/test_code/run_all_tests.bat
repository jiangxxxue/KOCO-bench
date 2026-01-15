@echo off
setlocal
REM 一键运行所有测试，使用 conda 环境 smolagents-test，当前目录执行无需 cd
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