# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


ROOT = Path.cwd()
ENTRY = str(ROOT / "tools" / "pyi_launcher.py")

datas = [
    (str(ROOT / "config"), "config"),
    (str(ROOT / "demo"), "demo"),
]

for package in [
    "transformers",
    "huggingface_hub",
    "qwen_tts",
    "qwen_asr",
    "nano_parakeet",
    "nagisa",
    "tokenizers",
    "sentence_transformers",
]:
    datas += collect_data_files(package, include_py_files=False)

datas += copy_metadata("faster-qwen3-tts")
for package in [
    "transformers",
    "huggingface_hub",
    "qwen-tts",
    "torch",
    "torchaudio",
    "sounddevice",
    "soundfile",
    "sentence-transformers",
]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

binaries = []
for package in ["torch", "torchaudio", "sounddevice", "soundfile", "tokenizers"]:
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

hiddenimports = [
    "demo.server",
    "examples.openai_server",
    "examples.rag_backends",
    "examples.realtime_voice_assistant",
    "faster_qwen3_tts.cli",
    "prepro",
    "model",
    "tagger",
    "mecab_system_eval",
]

for package in [
    "qwen_tts",
    "qwen_asr",
    "nano_parakeet",
    "nagisa",
    "transformers",
    "torchaudio",
    "sounddevice",
]:
    hiddenimports += collect_submodules(package)


a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="faster-qwen3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="faster-qwen3",
)
