from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = REPO_ROOT / "dist" / "portable_bundle"
APP_DIRS = [
    "faster_qwen3_tts",
    "demo",
    "config",
]
APP_FILES = [
    "LICENSE",
    "README.md",
    "WINDOWS_SETUP_GUIDE.md",
    "docs/PORTABLE_DEPLOY.md",
    "examples/openai_server.py",
    "examples/quick_start.py",
    "examples/realtime_voice_assistant.py",
    "examples/rag_backends.py",
    "examples/build_rag_index.py",
    "examples/evaluate_rag.py",
    "pyproject.toml",
]
IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def run_command(command: list[str], output_path: Path | None = None) -> bool:
    print("+", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=output_path is not None,
            text=True,
        )
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        return False

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(completed.stdout, encoding="utf-8")
    return True


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=IGNORE_PATTERNS)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_application(bundle_dir: Path, include_paths: list[str]) -> Path:
    app_root = bundle_dir / "app"
    if app_root.exists():
        shutil.rmtree(app_root)
    app_root.mkdir(parents=True, exist_ok=True)

    portable_root = bundle_dir / "portable"
    if portable_root.exists():
        shutil.rmtree(portable_root)
    copy_tree(REPO_ROOT / "portable", portable_root)

    for relative in APP_DIRS:
        src = REPO_ROOT / relative
        if src.exists():
            copy_tree(src, app_root / relative)

    for relative in APP_FILES:
        src = REPO_ROOT / relative
        if src.exists():
            copy_file(src, app_root / relative)

    for relative in include_paths:
        src = (REPO_ROOT / relative).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Include path not found: {relative}")
        try:
            rel = src.relative_to(REPO_ROOT)
        except ValueError:
            rel = Path("extra") / src.name
        dst = app_root / rel
        if src.is_dir():
            copy_tree(src, dst)
        else:
            copy_file(src, dst)

    runtime_env = REPO_ROOT / "portable" / "runtime.env"
    runtime_sample = portable_root / "runtime.env.sample"
    runtime_target = portable_root / "runtime.env"
    if runtime_env.exists():
        copy_file(runtime_env, runtime_target)
    elif runtime_sample.exists():
        copy_file(runtime_sample, runtime_target)

    return app_root


def copy_models(bundle_dir: Path, model_paths: list[str]) -> list[str]:
    copied = []
    models_dir = bundle_dir / "models"
    for raw_path in model_paths:
        src = Path(raw_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Model path not found: {raw_path}")
        dst = models_dir / src.name
        print(f"Copying model: {src} -> {dst}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(str(dst))
    return copied


def export_environment(bundle_dir: Path, env_name: str, skip_pack: bool) -> dict[str, str | bool]:
    env_dir = bundle_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str | bool] = {
        "env_name": env_name,
        "packed": False,
    }

    run_command(["conda", "env", "export", "-n", env_name, "--no-builds"], env_dir / "environment.yml")
    run_command(["conda", "list", "-n", env_name, "--explicit"], env_dir / "conda-explicit.txt")
    run_command(
        ["conda", "run", "-n", env_name, "python", "-m", "pip", "freeze"],
        env_dir / "requirements.txt",
    )
    run_command(
        ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
        env_dir / "python-path.txt",
    )

    if skip_pack:
        return results

    archive_path = bundle_dir / "python_env.tar.gz"
    packed = run_command(
        [
            "conda-pack",
            "-n",
            env_name,
            "-o",
            str(archive_path),
            "--ignore-editable-packages",
            "--ignore-missing-files",
        ]
    )
    results["packed"] = packed
    if not packed:
        print("WARNING: conda-pack not found or packing failed. Environment metadata was still exported.")
    return results


def write_manifest(bundle_dir: Path, copied_models: list[str], env_result: dict[str, str | bool]) -> None:
    manifest = {
        "bundle_root": str(bundle_dir),
        "copied_models": copied_models,
        "environment": env_result,
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable Faster Qwen3-TTS bundle")
    parser.add_argument("--output", default=str(DEFAULT_BUNDLE), help="Bundle output directory")
    parser.add_argument("--env-name", default="qwen-voice", help="Conda environment name to export")
    parser.add_argument("--skip-env", action="store_true", help="Skip conda environment export")
    parser.add_argument("--skip-env-pack", action="store_true", help="Export env metadata only, do not create python_env.tar.gz")
    parser.add_argument("--copy-model", action="append", default=[], help="Local model directory to copy into bundle/models")
    parser.add_argument("--include-path", action="append", default=[], help="Extra repo-relative file or directory to include")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.output).expanduser().resolve()

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    stage_application(bundle_dir, args.include_path)
    copied_models = copy_models(bundle_dir, args.copy_model)

    env_result: dict[str, str | bool] = {"env_name": args.env_name, "packed": False}
    if not args.skip_env:
        env_result = export_environment(bundle_dir, args.env_name, args.skip_env_pack)

    for relative in ["logs", "outputs"]:
        (bundle_dir / relative).mkdir(exist_ok=True)

    write_manifest(bundle_dir, copied_models, env_result)
    print(f"Portable bundle ready: {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
