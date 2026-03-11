@echo off
call "%~dp0_common.bat" || exit /b 1

pushd "%APP_ROOT%" >nul
if /i "%PYTHON_MODE%"=="conda" (
    conda run -n "%QWEN_CONDA_ENV%" python "examples\openai_server.py" %*
) else (
    "%PYTHON_CMD%" "examples\openai_server.py" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
