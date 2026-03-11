@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "BASE_DIR=%%~fI"
set "ARCHIVE=%BASE_DIR%\python_env.tar.gz"
set "TARGET_DIR=%BASE_DIR%\python_env"

if not exist "%ARCHIVE%" (
    echo ERROR: %ARCHIVE% not found.
    echo Run portable\build_portable_bundle.bat first, or copy python_env.tar.gz into the bundle root.
    exit /b 1
)

if exist "%TARGET_DIR%\python.exe" (
    echo python_env already exists at %TARGET_DIR%.
    exit /b 0
)

echo [1/3] Creating target directory...
mkdir "%TARGET_DIR%" >nul 2>&1
echo [2/3] Extracting python_env.tar.gz...
echo This can take a few minutes on Windows. Please wait...
tar -xf "%ARCHIVE%" -C "%TARGET_DIR%"
if errorlevel 1 (
    echo ERROR: failed to extract %ARCHIVE%
    exit /b 1
)

if not exist "%TARGET_DIR%\python.exe" (
    echo ERROR: extraction finished but python.exe was not found in %TARGET_DIR%
    exit /b 1
)

echo [3/3] Extraction complete.
echo Packed environment extracted to %TARGET_DIR%
exit /b 0
