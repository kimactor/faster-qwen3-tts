@echo off

set "ROOT_DIR=%~dp0.."
set "PYTHON_EXE="

if exist "%ROOT_DIR%\.venv\Scripts\python.exe" set "PYTHON_EXE=%ROOT_DIR%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%ROOT_DIR%\python_env\python.exe" set "PYTHON_EXE=%ROOT_DIR%\python_env\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

pushd "%ROOT_DIR%" >nul
"%PYTHON_EXE%" "tools\build_portable_bundle.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
