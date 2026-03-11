# Portable deployment

This repo now includes a portable packaging flow for Windows:

1. Edit `portable/runtime.env` from `portable/runtime.env.sample`.
2. Build a lean bundle:

```bat
portable\build_portable_bundle.bat --copy-model D:\models\Qwen3-TTS-12Hz-1.7B-Base
```

3. Copy the generated `dist\portable_bundle` folder to the target machine.
4. If `python_env.tar.gz` is present, run `portable\install_packed_env.bat` once on the target machine.
5. Start the service with one of:

```bat
portable\run_openai_server.bat
portable\run_demo.bat
portable\run_assistant.bat
portable\run_cli.bat clone --text "hello" --output outputs\hello.wav
```

Notes:

- `build_portable_bundle.py` copies only runtime-related code and excludes test/cache artifacts, so the package is smaller than copying the whole repo.
- The build exports `environment.yml`, `conda-explicit.txt`, and `requirements.txt` for the `qwen-voice` environment. If `conda-pack` is installed, it also creates `python_env.tar.gz`.
- For fully offline deployment, copy the local model directories with `--copy-model` and set their paths in `portable/runtime.env`.
- CUDA / driver compatibility must still match the packed PyTorch environment. If the target GPU or CUDA stack changes a lot, rebuild the environment on a similar machine.
- Voice assets such as `voice_anchor`, `ref_audio`, or `voices.json` should be copied alongside the bundle and referenced from `portable/runtime.env`.
