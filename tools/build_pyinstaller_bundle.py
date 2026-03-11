from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "pyinstaller_bundle"
SUPPORTED_TTS_BUNDLE_MODELS = {
    "Qwen3-TTS-12Hz-0.6B-Base": "0.6b",
    "Qwen3-TTS-12Hz-1.7B-Base": "1.7b",
}


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _validate_bundle_model(src: Path) -> None:
    name = src.name
    if name.startswith("Qwen3-TTS-12Hz-") and name not in SUPPORTED_TTS_BUNDLE_MODELS:
        allowed = ", ".join(SUPPORTED_TTS_BUNDLE_MODELS)
        raise ValueError(f"Unsupported TTS model for PyInstaller bundle: {name}. Allowed: {allowed}")


def copy_runtime_resources(bundle_dir: Path) -> None:
    copy_tree(REPO_ROOT / "config", bundle_dir / "config")


def build_pyinstaller(work_dir: Path) -> Path:
    dist_dir = work_dir / "dist"
    build_dir = work_dir / "build"
    spec_path = REPO_ROOT / "tools" / "faster_qwen3.spec"
    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(build_dir),
            str(spec_path),
        ]
    )
    return dist_dir / "faster-qwen3"


def copy_optional_paths(bundle_dir: Path, include_paths: list[str], model_paths: list[str]) -> dict[str, list[str]]:
    included: list[str] = []
    copied_models: list[str] = []

    for raw_path in include_paths:
        src = (REPO_ROOT / raw_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Include path not found: {raw_path}")
        dst = bundle_dir / src.relative_to(REPO_ROOT)
        if src.is_dir():
            copy_tree(src, dst)
        else:
            copy_file(src, dst)
        included.append(str(dst))

    models_dir = bundle_dir / "models"
    for raw_path in model_paths:
        src = Path(raw_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Model path not found: {raw_path}")
        _validate_bundle_model(src)
        dst = models_dir / src.name
        copy_tree(src, dst)
        copied_models.append(str(dst))

    return {"included_paths": included, "copied_models": copied_models}


def write_runtime_files(bundle_dir: Path) -> None:
    runtime_env = REPO_ROOT / "portable" / "runtime.env"
    runtime_sample = REPO_ROOT / "portable" / "runtime.env.sample"
    target_env = bundle_dir / "runtime.env"
    if runtime_env.exists():
        copy_file(runtime_env, target_env)
    else:
        copy_file(runtime_sample, target_env)
    copy_file(runtime_sample, bundle_dir / "runtime.env.sample")


def _rewrite_runtime_value(raw_value: str, bundle_dir: Path) -> str | None:
    if not raw_value:
        return None

    if "," in raw_value:
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
        rewritten_parts: list[str] = []
        changed = False
        for part in parts:
            basename = Path(part).name
            model_match = bundle_dir / "models" / basename
            bundle_match = bundle_dir / basename
            config_match = bundle_dir / "config" / basename
            if model_match.exists():
                rewritten_parts.append(str(Path("models") / basename))
                changed = True
            elif bundle_match.exists():
                rewritten_parts.append(basename)
                changed = True
            elif config_match.exists():
                rewritten_parts.append(str(Path("config") / basename))
                changed = True
            else:
                rewritten_parts.append(part)
        if changed:
            return ",".join(rewritten_parts)
        return None

    candidate = Path(raw_value)
    basename = candidate.name
    model_match = bundle_dir / "models" / basename
    bundle_match = bundle_dir / basename
    config_match = bundle_dir / "config" / basename

    if model_match.exists():
        return str(Path("models") / basename)
    if bundle_match.exists():
        return basename
    if config_match.exists():
        return str(Path("config") / basename)
    return None


def rewrite_runtime_env(bundle_dir: Path) -> None:
    target_env = bundle_dir / "runtime.env"
    if not target_env.exists():
        return

    models_dir = bundle_dir / "models"
    available_models = {
        path.name: str(Path("models") / path.name)
        for path in models_dir.iterdir()
        if path.exists()
    } if models_dir.exists() else {}
    active_tts_models = [
        available_models[name]
        for name in ("Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-Base")
        if name in available_models
    ]
    preferred_tts = available_models.get("Qwen3-TTS-12Hz-1.7B-Base") or available_models.get("Qwen3-TTS-12Hz-0.6B-Base", "")
    preferred_tts_key = "1.7b" if "Qwen3-TTS-12Hz-1.7B-Base" in available_models else "0.6b"
    forced_values = {
        "QWEN_TTS_MODEL": preferred_tts,
        "QWEN_TTS_MODEL_06B": available_models.get("Qwen3-TTS-12Hz-0.6B-Base", ""),
        "QWEN_TTS_MODEL_17B": available_models.get("Qwen3-TTS-12Hz-1.7B-Base", ""),
        "QWEN_TTS_MODEL_CUSTOM_06B": "",
        "QWEN_TTS_MODEL_CUSTOM_17B": "",
        "QWEN_TTS_MODEL_DESIGN_17B": "",
        "QWEN_TTS_DEMO_MODEL": preferred_tts,
        "QWEN_TTS_VOICE_ANCHOR": "",
        "QWEN_ASSISTANT_TTS_MODEL": preferred_tts_key if preferred_tts else "",
        "QWEN_ASSISTANT_TTS_PATH": preferred_tts,
        "ACTIVE_MODELS": ",".join(active_tts_models),
    }

    lines = target_env.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    seen_keys: set[str] = set()
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            rewritten.append(line)
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        raw_value = value.strip()
        seen_keys.add(key)
        if key in forced_values:
            rewritten.append(f"{key}={forced_values[key]}")
            continue

        if not raw_value:
            rewritten.append(f"{key}=")
            continue

        replacement = _rewrite_runtime_value(raw_value, bundle_dir)
        rewritten.append(f"{key}={replacement}" if replacement is not None else f"{key}={raw_value}")

    for key, value in forced_values.items():
        if key not in seen_keys:
            rewritten.append(f"{key}={value}")

    target_env.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def write_wrapper_bats(bundle_dir: Path) -> None:
    wrappers = {
        "run_assistant.bat": "@echo off\r\n\"%~dp0faster-qwen3.exe\" assistant %*\r\n",
        "run_openai_server.bat": "@echo off\r\n\"%~dp0faster-qwen3.exe\" openai-server %*\r\n",
        "run_demo.bat": "@echo off\r\n\"%~dp0faster-qwen3.exe\" demo %*\r\n",
        "run_cli.bat": "@echo off\r\n\"%~dp0faster-qwen3.exe\" cli %*\r\n",
    }
    for name, content in wrappers.items():
        (bundle_dir / name).write_text(content, encoding="ascii")


def write_manifest(bundle_dir: Path, copied: dict[str, list[str]]) -> None:
    manifest = {
        "bundle_root": str(bundle_dir),
        "launcher": str(bundle_dir / "faster-qwen3.exe"),
        **copied,
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PyInstaller bundle for Faster Qwen3 TTS")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output bundle directory")
    parser.add_argument("--copy-model", action="append", default=[], help="Copy local model directory into bundle/models")
    parser.add_argument("--include-path", action="append", default=[], help="Copy extra repo-relative file or directory")
    parser.add_argument("--keep-build", action="store_true", help="Keep intermediate PyInstaller build folders")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    work_dir = output_dir.parent / f"{output_dir.name}_build"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pyinstaller_dir = build_pyinstaller(work_dir)
    copy_tree(pyinstaller_dir, output_dir)
    copy_runtime_resources(output_dir)
    copied = copy_optional_paths(output_dir, args.include_path, args.copy_model)
    write_runtime_files(output_dir)
    rewrite_runtime_env(output_dir)
    write_wrapper_bats(output_dir)
    write_manifest(output_dir, copied)

    if not args.keep_build and work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"PyInstaller bundle ready: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
