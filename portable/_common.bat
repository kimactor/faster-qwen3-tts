@echo off

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "BASE_DIR=%%~fI"
set "APP_ROOT=%BASE_DIR%"
if exist "%BASE_DIR%\app\examples\openai_server.py" set "APP_ROOT=%BASE_DIR%\app"

set "ENV_FILE=%SCRIPT_DIR%runtime.env"
if not exist "%ENV_FILE%" if exist "%SCRIPT_DIR%runtime.env.sample" set "ENV_FILE=%SCRIPT_DIR%runtime.env.sample"
if exist "%ENV_FILE%" call :load_env "%ENV_FILE%"

set "PYTHON_MODE="
set "PYTHON_CMD="

if defined QWEN_PORTABLE_PYTHON if exist "%QWEN_PORTABLE_PYTHON%" (
    set "PYTHON_MODE=exe"
    set "PYTHON_CMD=%QWEN_PORTABLE_PYTHON%"
)

if not defined PYTHON_CMD if exist "%BASE_DIR%\python_env\python.exe" (
    set "PYTHON_MODE=exe"
    set "PYTHON_CMD=%BASE_DIR%\python_env\python.exe"
)

if not defined PYTHON_CMD if exist "%APP_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_MODE=exe"
    set "PYTHON_CMD=%APP_ROOT%\.venv\Scripts\python.exe"
)

if not defined PYTHON_MODE (
    if not defined QWEN_CONDA_ENV set "QWEN_CONDA_ENV=qwen-voice"
    where conda >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_MODE=conda"
    )
)

if not defined PYTHON_CMD if not defined PYTHON_MODE (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_MODE=exe"
        set "PYTHON_CMD=python"
    )
)

if defined PYTHON_MODE exit /b 0

echo ERROR: Python runtime not found.
echo 1. Extract python_env.tar.gz and run portable\install_packed_env.bat, or
echo 2. Set QWEN_PORTABLE_PYTHON in portable\runtime.env, or
echo 3. Install / activate the %QWEN_CONDA_ENV% conda environment.
exit /b 1

:load_env
for /f "usebackq eol=# tokens=1* delims==" %%A in ("%~1") do (
    if not "%%~A"=="" set "%%~A=%%~B"
)
exit /b 0
