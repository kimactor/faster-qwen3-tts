# PyInstaller deployment

This build path packages the code into a Windows `onedir` bundle with PyInstaller.

## Build

```bat
portable\build_pyinstaller_bundle.bat ^
  --copy-model D:\models\Qwen3-TTS-12Hz-1.7B-Base ^
  --copy-model D:\models\Qwen3-TTS-12Hz-0.6B-Base ^
  --copy-model D:\models\Qwen3-ASR-1.7B ^
  --copy-model D:\models\Qwen3-4B-Instruct-2507 ^
  --include-path ref_voice.wav
```

The output is written to `dist\pyinstaller_bundle`.

## Bundle layout

- `faster-qwen3.exe`: unified launcher
- `run_assistant.bat`: launches `faster-qwen3.exe assistant`
- `run_openai_server.bat`: launches `faster-qwen3.exe openai-server`
- `run_demo.bat`: launches `faster-qwen3.exe demo`
- `run_cli.bat`: launches `faster-qwen3.exe cli`
- `runtime.env`: runtime configuration loaded by the launcher
- `models\`: optional copied local models
- `_internal\`: PyInstaller runtime files

## Notes

- This strategy no longer needs a copied Python environment on the target machine.
- The bundle now defaults the assistant to `1.7b`, and only keeps the `0.6B-Base` / `1.7B-Base` TTS models.
- On first assistant start, if no model-matched voice anchor is found, it will generate `<ref_audio_stem>.<tts-model>.anchor.json` from `ref_audio` and `ref_text`.
- Models are still external assets; for offline deployment, copy them with `--copy-model`.
- Relative paths in `runtime.env` are resolved relative to the bundle root.
- CUDA, GPU driver, and VC runtime compatibility still need to match the target machine.
