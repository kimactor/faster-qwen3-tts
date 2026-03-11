@echo off
setlocal EnableDelayedExpansion
call "%~dp0_common.bat" || exit /b 1

set "ARGS="

if defined QWEN_ASR_PATH set "ARGS=!ARGS! --asr-path ""%QWEN_ASR_PATH%"""
if defined QWEN_LLM_PATH set "ARGS=!ARGS! --llm-path ""%QWEN_LLM_PATH%"""
if defined QWEN_ASSISTANT_TTS_MODEL set "ARGS=!ARGS! --tts-model %QWEN_ASSISTANT_TTS_MODEL%"
if defined QWEN_ASSISTANT_TTS_PATH set "ARGS=!ARGS! --tts-path ""%QWEN_ASSISTANT_TTS_PATH%"""
if defined QWEN_TTS_VOICE_ANCHOR set "ARGS=!ARGS! --voice-anchor ""%QWEN_TTS_VOICE_ANCHOR%"""
if defined QWEN_TTS_REF_AUDIO set "ARGS=!ARGS! --ref-audio ""%QWEN_TTS_REF_AUDIO%"""
if defined QWEN_TTS_REF_TEXT set "ARGS=!ARGS! --ref-text ""%QWEN_TTS_REF_TEXT%"""
if defined QWEN_TTS_LANGUAGE set "ARGS=!ARGS! --language ""%QWEN_TTS_LANGUAGE%"""
if defined QWEN_INPUT_DEVICE_HINT set "ARGS=!ARGS! --input-device-hint ""%QWEN_INPUT_DEVICE_HINT%"""
if defined QWEN_OUTPUT_DEVICE_HINT set "ARGS=!ARGS! --output-device-hint ""%QWEN_OUTPUT_DEVICE_HINT%"""
if defined QWEN_ASSISTANT_CHUNK_SIZE set "ARGS=!ARGS! --tts-chunk-size %QWEN_ASSISTANT_CHUNK_SIZE%"
if defined QWEN_PRONUNCIATION_LEXICON set "ARGS=!ARGS! --pronunciation-lexicon ""%QWEN_PRONUNCIATION_LEXICON%"""
if /i "%QWEN_ASSISTANT_XVECTOR_ONLY%"=="1" set "ARGS=!ARGS! --xvector-only"
if /i "%QWEN_ASSISTANT_ICL%"=="1" set "ARGS=!ARGS! --icl"
if /i "%QWEN_ASSISTANT_SERVE_WEB%"=="1" set "ARGS=!ARGS! --serve-web"
if defined QWEN_ASSISTANT_WEB_HOST set "ARGS=!ARGS! --web-host ""%QWEN_ASSISTANT_WEB_HOST%"""
if defined QWEN_ASSISTANT_WEB_PORT set "ARGS=!ARGS! --web-port %QWEN_ASSISTANT_WEB_PORT%"
if defined QWEN_RAG_BACKEND set "ARGS=!ARGS! --rag-backend %QWEN_RAG_BACKEND%"
if defined QWEN_RAG_SOURCE set "ARGS=!ARGS! --rag-source ""%QWEN_RAG_SOURCE%"""
if defined QWEN_RAG_INDEX set "ARGS=!ARGS! --rag-index ""%QWEN_RAG_INDEX%"""
if defined QWEN_RAG_EMBEDDING_MODEL set "ARGS=!ARGS! --rag-embedding-model ""%QWEN_RAG_EMBEDDING_MODEL%"""
if defined QWEN_RAG_COLLECTION set "ARGS=!ARGS! --rag-collection ""%QWEN_RAG_COLLECTION%"""
if /i "%QWEN_RAG_DEBUG%"=="1" set "ARGS=!ARGS! --rag-debug"

pushd "%APP_ROOT%" >nul
if /i "%PYTHON_MODE%"=="conda" (
    call conda run -n "%QWEN_CONDA_ENV%" python examples\realtime_voice_assistant.py !ARGS! %*
) else (
    call "%PYTHON_CMD%" examples\realtime_voice_assistant.py !ARGS! %*
)
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
endlocal & exit /b %EXIT_CODE%
