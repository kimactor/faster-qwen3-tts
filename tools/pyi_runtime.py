from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np


PATH_ENV_KEYS = (
    "QWEN_TTS_MODEL",
    "QWEN_TTS_MODEL_06B",
    "QWEN_TTS_MODEL_17B",
    "QWEN_TTS_MODEL_CUSTOM_06B",
    "QWEN_TTS_MODEL_CUSTOM_17B",
    "QWEN_TTS_MODEL_DESIGN_17B",
    "QWEN_TTS_DEMO_MODEL",
    "QWEN_TTS_VOICES",
    "QWEN_TTS_VOICE_ANCHOR",
    "QWEN_TTS_REF_AUDIO",
    "QWEN_PRONUNCIATION_LEXICON",
    "QWEN_ASR_PATH",
    "QWEN_LLM_PATH",
    "QWEN_ASSISTANT_TTS_PATH",
    "QWEN_RAG_SOURCE",
    "QWEN_RAG_INDEX",
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def runtime_env_candidates() -> list[Path]:
    root = bundle_root()
    return [
        root / "runtime.env",
        root / "portable" / "runtime.env",
        root / "runtime.env.sample",
        root / "portable" / "runtime.env.sample",
    ]


def load_runtime_env() -> tuple[dict[str, str], Path | None]:
    for path in runtime_env_candidates():
        if not path.exists():
            continue
        data: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
        return data, path
    return {}, None


def data_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()
    return bundle_root()


def _resolve_path(value: str, base_dir: Path) -> str:
    text = (value or "").strip()
    if not text:
        return text
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    primary = (base_dir / candidate).resolve()
    if primary.exists():
        return str(primary)
    fallback = (data_root() / candidate).resolve()
    if fallback.exists():
        return str(fallback)
    return str(primary)


def _install_sox_shim() -> None:
    if "sox" in sys.modules:
        return

    class SoxError(RuntimeError):
        pass

    class SoxiError(RuntimeError):
        pass

    class Transformer:
        def __init__(self) -> None:
            self._norm_db_level: float | None = None

        def norm(self, db_level: float = -6) -> "Transformer":
            self._norm_db_level = float(db_level)
            return self

        def build_array(self, input_array, sample_rate_in: int | None = None):
            audio = np.asarray(input_array, dtype=np.float32).copy()
            if self._norm_db_level is None or audio.size == 0:
                return audio
            peak = float(np.max(np.abs(audio)))
            if peak <= 0.0:
                return audio
            target_peak = float(10.0 ** (self._norm_db_level / 20.0))
            audio *= target_peak / peak
            return audio

    class Combiner:
        def __init__(self, *args, **kwargs) -> None:
            raise SoxError("Portable bundle uses an internal sox shim; Combiner is unavailable.")

    sox_mod = types.ModuleType("sox")
    sox_mod.NO_SOX = True
    sox_mod.Transformer = Transformer
    sox_mod.Combiner = Combiner
    sox_mod.SoxError = SoxError
    sox_mod.SoxiError = SoxiError
    sox_mod.__version__ = "portable-shim"

    sox_core = types.ModuleType("sox.core")
    sox_core.SoxError = SoxError
    sox_core.SoxiError = SoxiError

    sox_transform = types.ModuleType("sox.transform")
    sox_transform.Transformer = Transformer

    sox_combine = types.ModuleType("sox.combine")
    sox_combine.Combiner = Combiner

    sox_file_info = types.ModuleType("sox.file_info")
    sox_version = types.ModuleType("sox.version")
    sox_version.version = sox_mod.__version__
    sox_log = types.ModuleType("sox.log")

    sys.modules["sox"] = sox_mod
    sys.modules["sox.core"] = sox_core
    sys.modules["sox.transform"] = sox_transform
    sys.modules["sox.combine"] = sox_combine
    sys.modules["sox.file_info"] = sox_file_info
    sys.modules["sox.version"] = sox_version
    sys.modules["sox.log"] = sox_log


def install_runtime_shims() -> None:
    if is_frozen():
        _install_sox_shim()


def apply_runtime_env() -> tuple[dict[str, str], Path | None]:
    values, env_path = load_runtime_env()
    base_dir = env_path.parent if env_path is not None else bundle_root()
    for key, value in values.items():
        os.environ.setdefault(key, value)
    for key in PATH_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            os.environ[key] = _resolve_path(value, base_dir)
    return values, env_path


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
