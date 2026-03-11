from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

try:
    from tools.pyi_runtime import apply_runtime_env, bool_env, install_runtime_shims
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.pyi_runtime import apply_runtime_env, bool_env, install_runtime_shims


def _extend_argv(default_args: list[str], extra_args: list[str]) -> None:
    sys.argv = [sys.argv[0], *default_args, *extra_args]


def _assistant_defaults() -> list[str]:
    args: list[str] = []
    mapping = [
        ("QWEN_ASR_PATH", "--asr-path"),
        ("QWEN_LLM_PATH", "--llm-path"),
        ("QWEN_ASSISTANT_TTS_MODEL", "--tts-model"),
        ("QWEN_ASSISTANT_TTS_PATH", "--tts-path"),
        ("QWEN_TTS_VOICE_ANCHOR", "--voice-anchor"),
        ("QWEN_TTS_REF_AUDIO", "--ref-audio"),
        ("QWEN_TTS_REF_TEXT", "--ref-text"),
        ("QWEN_TTS_LANGUAGE", "--language"),
        ("QWEN_INPUT_DEVICE_HINT", "--input-device-hint"),
        ("QWEN_OUTPUT_DEVICE_HINT", "--output-device-hint"),
        ("QWEN_ASSISTANT_CHUNK_SIZE", "--tts-chunk-size"),
        ("QWEN_PRONUNCIATION_LEXICON", "--pronunciation-lexicon"),
        ("QWEN_ASSISTANT_WEB_HOST", "--web-host"),
        ("QWEN_ASSISTANT_WEB_PORT", "--web-port"),
        ("QWEN_RAG_BACKEND", "--rag-backend"),
        ("QWEN_RAG_SOURCE", "--rag-source"),
        ("QWEN_RAG_INDEX", "--rag-index"),
        ("QWEN_RAG_EMBEDDING_MODEL", "--rag-embedding-model"),
        ("QWEN_RAG_COLLECTION", "--rag-collection"),
    ]
    for env_name, flag in mapping:
        value = os.environ.get(env_name, "").strip()
        if value:
            args.extend([flag, value])
    if bool_env("QWEN_ASSISTANT_XVECTOR_ONLY"):
        args.append("--xvector-only")
    if bool_env("QWEN_ASSISTANT_ICL"):
        args.append("--icl")
    if bool_env("QWEN_ASSISTANT_SERVE_WEB"):
        args.append("--serve-web")
    if bool_env("QWEN_RAG_DEBUG"):
        args.append("--rag-debug")
    return args


def run_assistant(extra_args: list[str]) -> int:
    from examples.realtime_voice_assistant import VoiceAssistant, build_runtime_config, run_websocket_server

    _extend_argv(_assistant_defaults(), extra_args)
    cfg = build_runtime_config()
    if cfg.serve_web:
        run_websocket_server(cfg)
    else:
        VoiceAssistant(cfg).run()
    return 0


def run_openai_server(extra_args: list[str]) -> int:
    from examples import openai_server

    defaults: list[str] = []
    _extend_argv(defaults, extra_args)
    openai_server.main()
    return 0


def run_demo(extra_args: list[str]) -> int:
    from demo import server as demo_server

    _extend_argv([], extra_args)
    demo_server.main()
    return 0


def run_cli(extra_args: list[str]) -> int:
    from faster_qwen3_tts.cli import main as cli_main

    _extend_argv([], extra_args)
    cli_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyInstaller launcher for Faster Qwen3 TTS bundle")
    parser.add_argument("command", choices=["assistant", "openai-server", "demo", "cli"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    apply_runtime_env()
    install_runtime_shims()
    parser = build_parser()
    ns = parser.parse_args()

    commands: dict[str, Callable[[list[str]], int]] = {
        "assistant": run_assistant,
        "openai-server": run_openai_server,
        "demo": run_demo,
        "cli": run_cli,
    }
    return commands[ns.command](ns.args)


if __name__ == "__main__":
    raise SystemExit(main())
