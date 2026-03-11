@echo off

set "ROOT_DIR=%~dp0.."

pushd "%ROOT_DIR%" >nul
if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    "%ROOT_DIR%\.venv\Scripts\python.exe" "tools\build_pyinstaller_bundle.py" %*
) else (
    where conda >nul 2>&1
    if not errorlevel 1 (
        conda run -n qwen-voice python "tools\build_pyinstaller_bundle.py" %*
    ) else (
        python "tools\build_pyinstaller_bundle.py" %*
    )
)
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
