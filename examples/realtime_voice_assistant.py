#!/usr/bin/env python3
"""
Realtime ASR + LLM + FasterQwen3TTS voice assistant.

This ports the local microphone conversation loop from the sibling
QWen3-ASR-LLM-TTS workspace, but uses FasterQwen3TTS for the speech
generation stage so the assistant speaks through the CUDA-graph path
provided by this repository.
"""

import argparse
import asyncio
import base64
import importlib.util
import io
import os
import queue
import re
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import sounddevice as sd
except ImportError:
    sd = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.text_processing import TextNormalizer, default_pronunciation_lexicon_path
from rag_backends import build_rag_backend, format_rag_context, parse_metadata_filters


def _flash_attn_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _attn_impl(use_flash: bool) -> Optional[str]:
    if not use_flash:
        return None
    return "flash_attention_2" if _flash_attn_available() else None


@dataclass
class RuntimeConfig:
    asr_path: str = "D:/work/QWen3/Qwen3-ASR-1.7B"
    llm_path: str = "D:/work/QWen3/Qwen3-4B-Instruct-2507"
    tts_model_key: str = "0.6b"
    tts_path_06b: str = "D:/work/QWen3/Qwen3-TTS-12Hz-0.6B-Base"
    tts_path_17b: str = "D:/work/QWen3/Qwen3-TTS-12Hz-1.7B-Base"
    tts_path: str = "D:/work/QWen3/Qwen3-TTS-12Hz-0.6B-Base"
    ref_audio: str = "ref_voice.wav"
    voice_anchor: str = ""
    ref_text: str = ""
    ref_text_source: str = ""
    xvector_only_mode: bool = True
    pronunciation_lexicon_path: str = str(default_pronunciation_lexicon_path())
    rag_backend: str = "none"
    rag_source: str = ""
    rag_index: str = ""
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_collection: str = ""
    rag_filters: Dict[str, Tuple[str, ...]] | None = None
    rag_top_k: int = 3
    rag_context_chars: int = 1600
    rag_debug: bool = False

    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 1024
    input_device_hint: str = ""
    language: str = "Chinese"

    rms_threshold: float = 0.015
    pre_roll_sec: float = 0.25
    min_utterance_sec: float = 0.35
    silence_sec: float = 0.45
    max_utterance_sec: float = 12.0

    asr_min_valid_chars: int = 2
    asr_ignore_fillers: Tuple[str, ...] = ("\u55ef", "\u554a", "\u5443", "\u5582", "\u989d")

    interrupt_enable: bool = True
    interrupt_require_wake_word_when_busy: bool = True
    interrupt_wake_words: Tuple[str, ...] = ("\u6253\u65ad", "\u5c0fQ")
    interrupt_min_query_chars: int = 3

    system_prompt: str = (
        "\u4f60\u662f\u8bed\u97f3\u52a9\u624b\u3002"
        "\u8bf7\u76f4\u63a5\u56de\u7b54\u7528\u6237\u95ee\u9898\uff0c"
        "\u56de\u7b54\u7b80\u6d01\u6e05\u6670\u3002"
    )
    max_new_tokens: int = 256
    top_p: float = 0.9
    temperature: float = 0.7
    max_history_turns: int = 4

    enable_tts: bool = True
    tts_attn_implementation: str = "eager"
    tts_non_streaming_mode: bool = False
    tts_max_new_tokens: int = 512
    tts_do_sample: bool = True
    tts_top_k: int = 50
    tts_top_p: float = 1.0
    tts_temperature: float = 0.9
    tts_repetition_penalty: float = 1.05
    tts_stream_chunk_size: int = 8
    tts_chunk_max_len: int = 140
    tts_batch_wait_ms: int = 25
    tts_stream_soft_chars: int = 18
    tts_stream_force_chars: int = 44

    asr_pause_during_tts: bool = True

    enable_flash_attn: bool = True
    asr_use_flash_attn: bool = True
    llm_use_flash_attn: bool = True
    serve_web: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8011


def _normalize_tts_model_key(value: str) -> str:
    key = (value or "").strip().lower()
    mapping = {
        "0.6": "0.6b",
        "0.6b": "0.6b",
        "0b6": "0.6b",
        "06b": "0.6b",
        "1.7": "1.7b",
        "1.7b": "1.7b",
        "1b7": "1.7b",
        "17b": "1.7b",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported TTS model preset: {value}")
    return mapping[key]


def _read_text_file(path: str) -> str:
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return Path(path).read_text(encoding=encoding).strip()
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return ""


def _resolve_ref_text(
    ref_audio: str,
    ref_text: Optional[str] = None,
    ref_text_file: Optional[str] = None,
) -> Tuple[str, str]:
    if ref_text is not None:
        return ref_text.strip(), "inline"

    if ref_text_file:
        return _read_text_file(ref_text_file), ref_text_file

    audio_path = (ref_audio or "").strip()
    if audio_path:
        stem, _ = os.path.splitext(audio_path)
        for ext in (".txt", ".lab"):
            candidate = stem + ext
            if os.path.exists(candidate):
                return _read_text_file(candidate), candidate

    return "", ""


def _resolve_voice_anchor_path(
    explicit_anchor: Optional[str],
    ref_audio: Optional[str],
) -> str:
    if explicit_anchor:
        return explicit_anchor

    candidates: List[Path] = []
    if ref_audio:
        ref_path = Path(ref_audio)
        candidates.append(ref_path.with_suffix(".anchor.json"))
        candidates.append(ref_path.parent / "narrator.anchor.json")
    candidates.append(Path("narrator.anchor.json"))

    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return str(candidate)

    return ""


def apply_tts_model_preset(cfg: RuntimeConfig, model_key: str) -> RuntimeConfig:
    cfg.tts_model_key = _normalize_tts_model_key(model_key)
    if cfg.tts_model_key == "0.6b":
        cfg.tts_path = cfg.tts_path_06b
        cfg.tts_attn_implementation = "eager"
        cfg.tts_do_sample = True
        cfg.tts_top_k = 50
        cfg.tts_top_p = 1.0
        cfg.tts_temperature = 0.9
        cfg.tts_repetition_penalty = 1.05
        cfg.tts_non_streaming_mode = False
        cfg.tts_max_new_tokens = 512
    else:
        cfg.tts_path = cfg.tts_path_17b
        cfg.tts_attn_implementation = "eager"
        cfg.tts_do_sample = False
        cfg.tts_top_k = 1
        cfg.tts_top_p = 1.0
        cfg.tts_temperature = 1.0
        cfg.tts_repetition_penalty = 1.05
        cfg.tts_non_streaming_mode = False
        cfg.tts_max_new_tokens = 512
    return cfg


def build_runtime_config() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Realtime ASR + LLM + FasterQwen3TTS assistant")
    parser.add_argument("--asr-path", default=None, help="ASR model path")
    parser.add_argument("--llm-path", default=None, help="LLM model path")
    parser.add_argument("--tts-model", default="0.6b", choices=["0.6b", "1.7b"], help="TTS preset")
    parser.add_argument("--tts-path", default=None, help="Override TTS model path")
    parser.add_argument("--tts-attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--ref-audio", default=None, help="Reference audio path for voice clone")
    parser.add_argument("--voice-anchor", default=None, help="Saved voice anchor JSON for TTS")
    parser.add_argument("--ref-text", default=None, help="Reference transcript for ICL mode")
    parser.add_argument("--ref-text-file", default=None, help="Load reference transcript from file")
    parser.add_argument("--language", default=None, help="ASR/TTS language")
    parser.add_argument("--input-device-hint", default=None, help="Substring matched against microphone name")
    parser.add_argument("--tts-chunk-size", type=int, default=None, help="Streaming TTS codec chunk size")
    parser.add_argument("--pronunciation-lexicon", default=None, help="Pronunciation/silent-symbol override file")
    parser.add_argument("--rag-backend", default="none", choices=["none", "json-keyword", "vector-index"], help="RAG backend for assistant context injection")
    parser.add_argument("--rag-source", default=None, help="Path to a RAG knowledge JSON file")
    parser.add_argument("--rag-index", default=None, help="Path to a prebuilt vector RAG index (.npz)")
    parser.add_argument("--rag-embedding-model", default=None, help="Sentence-transformers model used to build/query the vector RAG index")
    parser.add_argument("--rag-collection", default=None, help="Collection id inside a multi-knowledge-base JSON file")
    parser.add_argument("--rag-filter", action="append", default=None, help="Metadata filter in the form key=value1,value2 (repeatable)")
    parser.add_argument("--rag-top-k", type=int, default=None, help="How many retrieved chunks to inject")
    parser.add_argument("--rag-context-chars", type=int, default=None, help="Max chars of retrieved context injected into prompt")
    parser.add_argument("--rag-debug", action="store_true", help="Print retrieved RAG chunks for each turn")
    parser.add_argument("--xvector-only", action="store_true", help="Force x-vector voice clone mode")
    parser.add_argument("--icl", action="store_true", help="Force ICL voice clone mode")
    parser.add_argument("--tts-non-streaming", action="store_true", help="Use non-streaming text prompt layout for TTS")
    parser.add_argument("--pre-roll-sec", type=float, default=None, help="Speech detector pre-roll seconds")
    parser.add_argument("--min-utterance-sec", type=float, default=None, help="Minimum utterance duration before auto-stop")
    parser.add_argument("--silence-sec", type=float, default=None, help="Trailing silence needed to end an utterance")
    parser.add_argument("--serve-web", action="store_true", help="Run a WebSocket server for the browser digital-human frontend")
    parser.add_argument("--web-host", default=None, help="Web server host")
    parser.add_argument("--web-port", type=int, default=None, help="Web server port")
    args = parser.parse_args()

    cfg = apply_tts_model_preset(RuntimeConfig(), args.tts_model)
    if args.asr_path:
        cfg.asr_path = args.asr_path
    if args.llm_path:
        cfg.llm_path = args.llm_path
    if args.tts_path:
        cfg.tts_path = args.tts_path
    if args.tts_attn_implementation:
        cfg.tts_attn_implementation = args.tts_attn_implementation
    if args.ref_audio:
        cfg.ref_audio = args.ref_audio
    cfg.voice_anchor = _resolve_voice_anchor_path(args.voice_anchor, cfg.ref_audio)
    if args.language:
        cfg.language = args.language
    if args.input_device_hint is not None:
        cfg.input_device_hint = args.input_device_hint
    if args.tts_chunk_size is not None:
        cfg.tts_stream_chunk_size = max(1, int(args.tts_chunk_size))
    if args.pronunciation_lexicon is not None:
        cfg.pronunciation_lexicon_path = args.pronunciation_lexicon
    if args.rag_source is not None:
        cfg.rag_source = args.rag_source
    if args.rag_index is not None:
        cfg.rag_index = args.rag_index
    if args.rag_embedding_model is not None:
        cfg.rag_embedding_model = args.rag_embedding_model
    if args.rag_collection is not None:
        cfg.rag_collection = args.rag_collection
    if args.rag_filter:
        cfg.rag_filters = parse_metadata_filters(args.rag_filter)
    if args.rag_top_k is not None:
        cfg.rag_top_k = max(1, int(args.rag_top_k))
    if args.rag_context_chars is not None:
        cfg.rag_context_chars = max(200, int(args.rag_context_chars))
    if args.rag_debug:
        cfg.rag_debug = True
    cfg.rag_backend = args.rag_backend
    if args.pre_roll_sec is not None:
        cfg.pre_roll_sec = max(0.05, float(args.pre_roll_sec))
    if args.min_utterance_sec is not None:
        cfg.min_utterance_sec = max(0.1, float(args.min_utterance_sec))
    if args.silence_sec is not None:
        cfg.silence_sec = max(0.1, float(args.silence_sec))
    if args.serve_web:
        cfg.serve_web = True
    if args.web_host is not None:
        cfg.web_host = args.web_host
    if args.web_port is not None:
        cfg.web_port = int(args.web_port)

    cfg.ref_text, cfg.ref_text_source = _resolve_ref_text(
        ref_audio=cfg.ref_audio,
        ref_text=args.ref_text,
        ref_text_file=args.ref_text_file,
    )

    if args.tts_non_streaming:
        cfg.tts_non_streaming_mode = True

    if args.xvector_only and args.icl:
        raise ValueError("Choose either --xvector-only or --icl, not both")

    if args.xvector_only:
        cfg.xvector_only_mode = True
    elif args.icl:
        cfg.xvector_only_mode = False
    else:
        cfg.xvector_only_mode = not bool((cfg.ref_text or "").strip())

    if not cfg.voice_anchor and not cfg.xvector_only_mode and not (cfg.ref_text or "").strip():
        raise ValueError("ICL mode requires --ref-text or a matching <ref_audio>.txt/<ref_audio>.lab file")

    return cfg


def split_for_tts(text: str, max_len: int = 140) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"([\u3002\uff01\uff1f!?])", text)
    merged: List[str] = []
    for i in range(0, len(parts), 2):
        base = parts[i].strip()
        if not base:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        merged.append(base + punct)

    if not merged:
        return [text]

    out: List[str] = []
    buf = ""
    for seg in merged:
        if len(seg) > max_len:
            if buf:
                out.append(buf)
                buf = ""
            for j in range(0, len(seg), max_len):
                chunk = seg[j:j + max_len].strip()
                if chunk:
                    out.append(chunk)
            continue

        if not buf:
            buf = seg
        elif len(buf) + len(seg) <= max_len:
            buf += seg
        else:
            out.append(buf)
            buf = seg

    if buf:
        out.append(buf)
    return out


def _audio_to_pcm16_b64(audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = np.where(pcm < 0.0, pcm * 32768.0, pcm * 32767.0).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if src_sr <= 0 or dst_sr <= 0 or arr.size == 0 or src_sr == dst_sr:
        return arr
    duration = arr.shape[0] / float(src_sr)
    target_len = max(1, int(round(duration * dst_sr)))
    xp = np.linspace(0.0, duration, num=arr.shape[0], endpoint=False)
    fp = arr.astype(np.float32)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, xp, fp).astype(np.float32)


def _decode_uploaded_audio(content: bytes, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return _resample_audio(arr, int(sr), int(target_sr))


def _decode_pcm16_chunk(content: bytes, src_sr: int, dst_sr: int) -> np.ndarray:
    if not content:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
    return _resample_audio(audio, int(src_sr), int(dst_sr))


def _build_segmenter(
    cfg: RuntimeConfig,
    *,
    pre_roll_sec: float | None = None,
    min_utterance_sec: float | None = None,
    silence_sec: float | None = None,
) -> "AudioSegmenter":
    local_cfg = replace(cfg)
    if pre_roll_sec is not None:
        local_cfg.pre_roll_sec = max(0.05, float(pre_roll_sec))
    if min_utterance_sec is not None:
        local_cfg.min_utterance_sec = max(0.1, float(min_utterance_sec))
    if silence_sec is not None:
        local_cfg.silence_sec = max(0.1, float(silence_sec))
    return AudioSegmenter(local_cfg)


class AudioSegmenter:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.active = False
        self.frames: List[np.ndarray] = []
        self.total_samples = 0
        self.silence_samples = 0

        pre_roll_blocks = max(1, int((cfg.pre_roll_sec * cfg.sample_rate) / cfg.block_size))
        self.pre_roll = deque(maxlen=pre_roll_blocks)

        self.min_samples = int(cfg.min_utterance_sec * cfg.sample_rate)
        self.max_samples = int(cfg.max_utterance_sec * cfg.sample_rate)
        self.silence_limit = int(cfg.silence_sec * cfg.sample_rate)

    def reset(self) -> None:
        self.active = False
        self.frames = []
        self.total_samples = 0
        self.silence_samples = 0

    def force_flush(self) -> Optional[np.ndarray]:
        if not self.frames:
            self.reset()
            return None
        audio = np.concatenate(self.frames)
        self.reset()
        return audio if audio.size > 0 else None

    def push(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(chunk, dtype=np.float32).flatten()
        if x.size == 0:
            return None

        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        voiced = rms >= self.cfg.rms_threshold
        self.pre_roll.append(x)

        if not self.active and voiced:
            self.active = True
            self.frames = list(self.pre_roll)
            self.total_samples = sum(f.shape[0] for f in self.frames)
            self.silence_samples = 0
            return None

        if not self.active:
            return None

        self.frames.append(x)
        self.total_samples += x.shape[0]

        if voiced:
            self.silence_samples = 0
        else:
            self.silence_samples += x.shape[0]

        if self.total_samples >= self.max_samples:
            audio = np.concatenate(self.frames)
            self.reset()
            return audio

        if self.silence_samples >= self.silence_limit and self.total_samples >= self.min_samples:
            audio = np.concatenate(self.frames)
            self.reset()
            return audio

        return None


class StreamingASR:
    def __init__(self, cfg: RuntimeConfig):
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ImportError("qwen_asr is required for realtime ASR") from exc

        self.cfg = cfg
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        use_flash = cfg.enable_flash_attn and cfg.asr_use_flash_attn
        attn = _attn_impl(use_flash)

        load_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if attn:
            load_kwargs["attn_implementation"] = attn

        try:
            self.model = Qwen3ASRModel.from_pretrained(cfg.asr_path, **load_kwargs)
            print(f"[Init] ASR attn_implementation={load_kwargs.get('attn_implementation', 'default')}")
        except Exception as exc:
            if "attn_implementation" in load_kwargs:
                print(f"[Warn] ASR flash-attn failed, fallback to default: {exc}")
                load_kwargs.pop("attn_implementation", None)
                self.model = Qwen3ASRModel.from_pretrained(cfg.asr_path, **load_kwargs)
                print("[Init] ASR attn_implementation=default")
            else:
                raise

    @staticmethod
    def _extract_text(result) -> str:
        if result is None:
            return ""
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        if isinstance(result, list):
            buf: List[str] = []
            for item in result:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                else:
                    text = str(getattr(item, "text", item)).strip()
                if text:
                    buf.append(text)
            return " ".join(buf).strip()
        return str(getattr(result, "text", result)).strip()

    def recognize(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_path = fp.name
        try:
            sf.write(tmp_path, np.asarray(audio, dtype=np.float32), self.cfg.sample_rate)
            result = self.model.transcribe(audio=tmp_path, language=self.cfg.language)
            text = self._extract_text(result)
            match = re.search(r"text='([^']*)'", text)
            if match:
                text = match.group(1)
            return re.sub(r"\s+", "", text).strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class StreamingLLM:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        use_flash = cfg.enable_flash_attn and cfg.llm_use_flash_attn
        attn = _attn_impl(use_flash)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path, trust_remote_code=True)
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if attn:
            load_kwargs["attn_implementation"] = attn

        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_path, **load_kwargs).eval()
            print(f"[Init] LLM attn_implementation={load_kwargs.get('attn_implementation', 'default')}")
        except Exception as exc:
            if "attn_implementation" in load_kwargs:
                print(f"[Warn] LLM flash-attn failed, fallback to default: {exc}")
                load_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_path, **load_kwargs).eval()
                print("[Init] LLM attn_implementation=default")
            else:
                raise

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def stream_chat(self, messages: List[dict]):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = dict(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            top_p=self.cfg.top_p,
            temperature=self.cfg.temperature,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True).start()
        for piece in streamer:
            yield piece
class StreamingTTS:
    def __init__(self, cfg: RuntimeConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("FasterQwen3TTS requires a CUDA device")

        self.cfg = cfg
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = FasterQwen3TTS.from_pretrained(
            cfg.tts_path,
            device="cuda",
            dtype=dtype,
            attn_implementation=cfg.tts_attn_implementation,
        )
        print(f"[Init] TTS attn_implementation={cfg.tts_attn_implementation}")
        print(f"[Init] TTS dtype={dtype}")
        print(f"[Init] TTS sample_rate={self.model.sample_rate}")

        if cfg.voice_anchor:
            if not os.path.exists(cfg.voice_anchor):
                raise FileNotFoundError(f"voice anchor not found: {cfg.voice_anchor}")
        else:
            if not os.path.exists(cfg.ref_audio):
                raise FileNotFoundError(f"ref audio not found: {cfg.ref_audio}")
        if not cfg.voice_anchor and not cfg.xvector_only_mode and not (cfg.ref_text or "").strip():
            raise ValueError("ref_text is required when xvector_only_mode is False")

    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        text = (text or "").strip()
        if not text:
            return None

        max_tokens = min(self.cfg.tts_max_new_tokens, max(180, len(text) * 20))
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=self.cfg.language,
            ref_audio=self.cfg.ref_audio,
            ref_text=self.cfg.ref_text,
            voice_anchor=self.cfg.voice_anchor or None,
            max_new_tokens=max_tokens,
            do_sample=self.cfg.tts_do_sample,
            top_k=self.cfg.tts_top_k,
            top_p=self.cfg.tts_top_p,
            temperature=self.cfg.tts_temperature,
            repetition_penalty=self.cfg.tts_repetition_penalty,
            xvec_only=self.cfg.xvector_only_mode,
            non_streaming_mode=self.cfg.tts_non_streaming_mode,
        )
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        return np.asarray(audio, dtype=np.float32).squeeze(), int(sr)

    def synthesize_stream(self, text: str):
        text = (text or "").strip()
        if not text:
            return

        max_tokens = min(self.cfg.tts_max_new_tokens, max(180, len(text) * 20))
        for audio_chunk, sr, timing in self.model.generate_voice_clone_streaming(
            text=text,
            language=self.cfg.language,
            ref_audio=self.cfg.ref_audio,
            ref_text=self.cfg.ref_text,
            voice_anchor=self.cfg.voice_anchor or None,
            max_new_tokens=max_tokens,
            do_sample=self.cfg.tts_do_sample,
            top_k=self.cfg.tts_top_k,
            top_p=self.cfg.tts_top_p,
            temperature=self.cfg.tts_temperature,
            repetition_penalty=self.cfg.tts_repetition_penalty,
            xvec_only=self.cfg.xvector_only_mode,
            non_streaming_mode=self.cfg.tts_non_streaming_mode,
            chunk_size=self.cfg.tts_stream_chunk_size,
        ):
            yield np.asarray(audio_chunk, dtype=np.float32).squeeze(), int(sr), timing

    @staticmethod
    def speak(audio: np.ndarray, sr: int) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for local speaker playback")
        sd.play(audio, sr)
        sd.wait()


class AudioChunkPlayer:
    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._drain = threading.Condition(self._lock)
        self._queued: deque[np.ndarray] = deque()
        self._current = np.zeros(0, dtype=np.float32)
        self._current_pos = 0
        self._pending_samples = 0
        self._stream = None

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for local speaker playback")
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def enqueue(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).flatten()
        if chunk.size == 0:
            return
        with self._lock:
            self._queued.append(chunk.copy())
            self._pending_samples += int(chunk.size)
            self._drain.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._queued.clear()
            self._current = np.zeros(0, dtype=np.float32)
            self._current_pos = 0
            self._pending_samples = 0
            self._drain.notify_all()

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            return self._drain.wait_for(lambda: self._pending_samples <= 0, timeout=timeout)

    def pending_seconds(self) -> float:
        with self._lock:
            return float(self._pending_samples) / float(self.sample_rate)

    def _callback(self, outdata, frames, _time_info, status) -> None:
        if status:
            print(f"[AudioOut] status: {status}")
        out = outdata[:, 0]
        out.fill(0)

        with self._lock:
            i = 0
            while i < frames:
                if self._current_pos >= self._current.size:
                    if self._queued:
                        self._current = self._queued.popleft()
                        self._current_pos = 0
                    else:
                        self._current = np.zeros(0, dtype=np.float32)
                        break

                take = min(frames - i, self._current.size - self._current_pos)
                if take <= 0:
                    break
                out[i:i + take] = self._current[self._current_pos:self._current_pos + take]
                self._current_pos += take
                self._pending_samples -= take
                i += take

            if self._pending_samples <= 0 and not self._queued and self._current_pos >= self._current.size:
                self._pending_samples = 0
                self._drain.notify_all()


class VoiceAssistant:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.segmenter = AudioSegmenter(cfg)
        self.text_normalizer = TextNormalizer(cfg.pronunciation_lexicon_path)
        self.rag = build_rag_backend(
            cfg.rag_backend,
            cfg.rag_source,
            index_path=cfg.rag_index,
            embedding_model=cfg.rag_embedding_model,
            collection_id=cfg.rag_collection,
            filters=cfg.rag_filters,
        )

        print("[Init] loading ASR...")
        self.asr = StreamingASR(cfg)
        print("[Init] loading LLM...")
        self.llm = StreamingLLM(cfg)
        print("[Init] loading TTS...")
        self.tts = StreamingTTS(cfg) if cfg.enable_tts else None

        self.asr_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self.text_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=8)
        self.tts_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=32)
        self.tts_ready_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=64)
        self.audio_player = AudioChunkPlayer(self.tts.model.sample_rate) if self.tts else None
        self.stop_event = threading.Event()
        self.speaking_event = threading.Event()
        self.llm_generating_event = threading.Event()

        self.epoch_lock = threading.Lock()
        self.active_epoch = 0

        self.history: List[dict] = []
        self.history_lock = threading.Lock()

        self.metrics_lock = threading.Lock()
        self.turn_counter = 0
        self.turn_metrics: Dict[int, Dict[str, Any]] = {}
        self.dialog_lock = threading.Lock()

    def _resolve_input_device(self) -> Optional[int]:
        if sd is None:
            return None
        hint = (self.cfg.input_device_hint or "").strip().lower()
        if not hint:
            return None
        try:
            devices = sd.query_devices()
        except Exception as exc:
            print(f"[Warn] query devices failed: {exc}")
            return None

        for idx, dev in enumerate(devices):
            name = str(dev.get("name", ""))
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0 and hint in name.lower():
                print(f"[Audio] matched input device: {name}")
                return idx
        return None

    def _build_messages(self, user_text: str) -> List[dict]:
        rag_chunks = self.rag.retrieve(user_text, top_k=self.cfg.rag_top_k, filters=self.cfg.rag_filters)
        rag_context = format_rag_context(rag_chunks, max_chars=self.cfg.rag_context_chars)
        if self.cfg.rag_debug and rag_context:
            print("[RAG]")
            print(rag_context)
        with self.history_lock:
            messages = [{"role": "system", "content": self.cfg.system_prompt}]
            if rag_context:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "以下是与当前问题最相关的知识库片段。"
                            "如果内容相关，优先基于这些片段回答；如果无关，不要强行引用。\n\n"
                            f"{rag_context}"
                        ),
                    }
                )
            messages.extend(self.history)
            messages.append({"role": "user", "content": user_text})
            return messages

    def _append_history(self, user_text: str, assistant_text: str) -> None:
        with self.history_lock:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": assistant_text})
            keep = self.cfg.max_history_turns * 2
            if len(self.history) > keep:
                self.history = self.history[-keep:]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[\uff0c\u3002\uff01\uff1f!?\uff1b;\u3001,.\s]", "", (text or "")).strip()

    def _is_valid_user_text(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        normalized = self._normalize_text(raw)
        if not normalized:
            return False
        if normalized in set(self.cfg.asr_ignore_fillers):
            return False
        return len(normalized) >= self.cfg.asr_min_valid_chars

    def _get_epoch(self) -> int:
        with self.epoch_lock:
            return self.active_epoch

    def _bump_epoch(self) -> int:
        with self.epoch_lock:
            self.active_epoch += 1
            return self.active_epoch

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _is_assistant_busy(self) -> bool:
        return (
            self.llm_generating_event.is_set()
            or self.speaking_event.is_set()
            or (not self.text_queue.empty())
            or (not self.tts_queue.empty())
            or (not self.tts_ready_queue.empty())
            or (self.audio_player is not None and self.audio_player.pending_seconds() > 0.05)
        )

    def _extract_interrupt_query(self, text: str) -> Optional[str]:
        raw = (text or "").strip()
        if not raw:
            return None
        if not self.cfg.interrupt_require_wake_word_when_busy:
            return raw if self._is_valid_user_text(raw) else None

        normalized = self._normalize_text(raw)
        hit = False
        cleaned = raw
        for wake in self.cfg.interrupt_wake_words:
            wake_word = (wake or "").strip()
            if not wake_word:
                continue
            if wake_word in normalized or wake_word in cleaned:
                hit = True
                cleaned = cleaned.replace(wake_word, " ")

        if not hit:
            return None

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \uff0c\u3002\uff01\uff1f!?\uff1b;\u3001,.")
        if len(self._normalize_text(cleaned)) < int(self.cfg.interrupt_min_query_chars):
            return None
        return cleaned

    def _interrupt_current_response(self) -> None:
        epoch = self._bump_epoch()
        self._clear_queue(self.asr_queue)
        self._clear_queue(self.text_queue)
        self._clear_queue(self.tts_queue)
        self._clear_queue(self.tts_ready_queue)
        if self.audio_player is not None:
            self.audio_player.clear()
        self.segmenter.reset()
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        self.speaking_event.clear()
        print(f"[Interrupt] cancel current response (epoch={epoch})")

    def _mark(self, turn_id: int, key: str, value: float, only_if_none: bool = True) -> None:
        with self.metrics_lock:
            metrics = self.turn_metrics.get(turn_id)
            if not metrics:
                return
            if only_if_none and metrics.get(key) is not None:
                return
            metrics[key] = value

    def _add(self, turn_id: int, key: str, delta: float) -> None:
        with self.metrics_lock:
            metrics = self.turn_metrics.get(turn_id)
            if not metrics:
                return
            metrics[key] = float(metrics.get(key, 0.0)) + float(delta)

    def _snapshot(self, turn_id: int) -> Optional[Dict[str, Any]]:
        with self.metrics_lock:
            metrics = self.turn_metrics.get(turn_id)
            return dict(metrics) if metrics else None

    def _print_turn_metrics(self, turn_id: int) -> None:
        metrics = self._snapshot(turn_id)
        if not metrics:
            return

        t0 = metrics.get("ts_start")
        t1 = metrics.get("ts_llm_first_token")
        t2 = metrics.get("ts_llm_done")
        t3 = metrics.get("ts_tts_enqueue_first")
        t4 = metrics.get("ts_tts_play_first")

        parts = [f"[Metrics][turn={turn_id}]"]
        if t0 and t1:
            parts.append(f"first_token={t1 - t0:.3f}s")
        if t0 and t2:
            parts.append(f"llm_done={t2 - t0:.3f}s")
        if t1 and t4:
            parts.append(f"llm_to_tts_play={t4 - t1:.3f}s")
        if t3 and t4:
            parts.append(f"text_to_tts_play={t4 - t3:.3f}s")

        synth = float(metrics.get("tts_synth_total", 0.0))
        audio = float(metrics.get("tts_audio_total", 0.0))
        if audio > 1e-6:
            parts.append(f"rtf={synth / audio:.3f}")

        print(" ".join(parts))

    def _metrics_payload(self, turn_id: int) -> Dict[str, Any]:
        metrics = self._snapshot(turn_id) or {}
        t0 = metrics.get("ts_start")
        t1 = metrics.get("ts_llm_first_token")
        t2 = metrics.get("ts_llm_done")
        t4 = metrics.get("ts_tts_play_first")
        synth = float(metrics.get("tts_synth_total", 0.0))
        audio = float(metrics.get("tts_audio_total", 0.0))

        payload: Dict[str, Any] = {
            "turn_id": turn_id,
            "llm_first_token_ms": int(round((t1 - t0) * 1000.0)) if t0 and t1 else None,
            "llm_done_ms": int(round((t2 - t0) * 1000.0)) if t0 and t2 else None,
            "ttfa_ms": int(round((t4 - t0) * 1000.0)) if t0 and t4 else None,
            "tts_audio_s": round(audio, 3),
            "tts_synth_s": round(synth, 3),
            "rtf": round((audio / synth), 3) if synth > 1e-6 else None,
        }
        return payload

    def transcribe_audio_array(self, audio: np.ndarray) -> str:
        text = self.asr.recognize(np.asarray(audio, dtype=np.float32))
        return re.sub(r"\s+", "", text).strip()

    def transcribe_audio_bytes(self, content: bytes) -> str:
        audio = _decode_uploaded_audio(content, self.cfg.sample_rate)
        return self.transcribe_audio_array(audio)

    def _stream_tts_events(self, turn_id: int, epoch: int, text: str):
        if not self.tts:
            return
        normalized_text = self.text_normalizer.normalize_for_tts(text)
        if not normalized_text:
            return

        for segment in split_for_tts(normalized_text, max_len=self.cfg.tts_chunk_max_len):
            self._mark(turn_id, "ts_tts_enqueue_first", time.monotonic())
            first_chunk = True
            try:
                self.speaking_event.set()
                for audio_chunk, sr, timing in self.tts.synthesize_stream(segment):
                    if epoch != self._get_epoch():
                        return
                    if first_chunk:
                        self._mark(turn_id, "ts_tts_play_first", time.monotonic())
                        first_chunk = False
                    duration_s = (len(audio_chunk) / float(sr)) if sr > 0 else 0.0
                    synth_time = (timing.get("prefill_ms", 0.0) + timing.get("decode_ms", 0.0)) / 1000.0
                    self._add(turn_id, "tts_audio_total", duration_s)
                    self._add(turn_id, "tts_synth_total", synth_time)
                    yield {
                        "type": "audio_chunk",
                        "turn_id": turn_id,
                        "audio_pcm16_b64": _audio_to_pcm16_b64(audio_chunk),
                        "sample_rate": int(sr),
                        "duration_s": round(duration_s, 3),
                        "timing": timing,
                    }
            finally:
                self.speaking_event.clear()

    def stream_dialog(self, user_text: str):
        user_text = (user_text or "").strip()
        if not self._is_valid_user_text(user_text):
            raise ValueError("Empty or invalid user text")

        with self.dialog_lock:
            turn = self._new_turn(user_text)
            turn_id = int(turn["turn_id"])
            epoch = int(turn["epoch"])
            messages = self._build_messages(user_text)
            response = ""
            tts_buf = ""
            first_token_marked = False
            interrupted = False
            tts_event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
            tts_queue: queue.Queue[Optional[str]] = queue.Queue()
            tts_done = threading.Event()

            def drain_tts_events(wait: bool = False):
                timeout = 0.01 if wait else 0.0
                while True:
                    try:
                        event = tts_event_queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    else:
                        yield event
                        timeout = 0.0

            def tts_worker() -> None:
                try:
                    while True:
                        item = tts_queue.get()
                        if item is None:
                            break
                        for event in self._stream_tts_events(turn_id, epoch, item) or ():
                            tts_event_queue.put(event)
                finally:
                    tts_done.set()

            yield {"type": "user_text", "turn_id": turn_id, "text": user_text}
            self.llm_generating_event.set()
            tts_thread: Optional[threading.Thread] = None
            if self.tts:
                tts_thread = threading.Thread(target=tts_worker, daemon=True)
                tts_thread.start()
            try:
                for chunk in self.llm.stream_chat(messages):
                    if epoch != self._get_epoch():
                        interrupted = True
                        break

                    now = time.monotonic()
                    if not first_token_marked:
                        self._mark(turn_id, "ts_llm_first_token", now)
                        first_token_marked = True

                    response += chunk
                    tts_buf += chunk
                    yield {
                        "type": "assistant_text",
                        "turn_id": turn_id,
                        "delta": chunk,
                        "text": response,
                    }

                    if self.tts:
                        norm_len = len(self._normalize_text(tts_buf))
                        hard_stop = re.search(r"[\u3002\uff01\uff1f!?]\s*$", tts_buf) is not None
                        soft_stop = re.search(r"[\uff0c,\u3001\uff1b;\uff1a:\n]\s*$", tts_buf) is not None
                        force_flush = norm_len >= int(self.cfg.tts_stream_force_chars)
                        soft_flush = norm_len >= int(self.cfg.tts_stream_soft_chars) and soft_stop
                        if hard_stop or force_flush or soft_flush:
                            tts_queue.put(tts_buf)
                            tts_buf = ""

                    for event in drain_tts_events(wait=False):
                        yield event

                self._mark(turn_id, "ts_llm_done", time.monotonic(), only_if_none=True)
                if interrupted:
                    if self.tts:
                        tts_queue.put(None)
                        while not tts_done.is_set() or not tts_event_queue.empty():
                            for event in drain_tts_events(wait=True):
                                yield event
                    yield {
                        "type": "interrupted",
                        "turn_id": turn_id,
                        "metrics": self._metrics_payload(turn_id),
                    }
                    return

                if self.tts and tts_buf.strip():
                    tts_queue.put(tts_buf)

                if self.tts:
                    tts_queue.put(None)
                    while not tts_done.is_set() or not tts_event_queue.empty():
                        for event in drain_tts_events(wait=True):
                            yield event

                self._append_history(user_text, response.strip())
                yield {
                    "type": "done",
                    "turn_id": turn_id,
                    "text": response.strip(),
                    "metrics": self._metrics_payload(turn_id),
                }
            finally:
                self.llm_generating_event.clear()

    def _new_turn(self, user_text: str) -> Dict[str, Any]:
        epoch = self._get_epoch()
        with self.metrics_lock:
            self.turn_counter += 1
            turn_id = self.turn_counter
            self.turn_metrics[turn_id] = {
                "user_text": user_text,
                "ts_start": time.monotonic(),
                "ts_llm_first_token": None,
                "ts_llm_done": None,
                "ts_tts_enqueue_first": None,
                "ts_tts_play_first": None,
                "tts_synth_total": 0.0,
                "tts_audio_total": 0.0,
            }
        return {"turn_id": turn_id, "epoch": epoch, "text": user_text}

    def _enqueue_tts_segments(self, turn_id: int, epoch: int, text: str, seq: int) -> int:
        normalized_text = self.text_normalizer.normalize_for_tts(text)
        if not normalized_text:
            return seq
        for segment in split_for_tts(normalized_text, max_len=self.cfg.tts_chunk_max_len):
            self._mark(turn_id, "ts_tts_enqueue_first", time.monotonic())
            self.tts_queue.put({"turn_id": turn_id, "epoch": epoch, "seq": seq, "text": segment})
            seq += 1
        return seq

    def audio_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            print(f"[Audio] status: {status}")
        if self.stop_event.is_set():
            return
        if self.cfg.asr_pause_during_tts and self.speaking_event.is_set():
            self.segmenter.reset()
            return

        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        segment = self.segmenter.push(chunk)
        if segment is not None:
            try:
                self.asr_queue.put_nowait(segment)
            except queue.Full:
                pass

    def asr_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio = self.asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = self.asr.recognize(audio)
                if not self._is_valid_user_text(text):
                    continue

                if self.cfg.interrupt_enable and self._is_assistant_busy():
                    interrupt_query = self._extract_interrupt_query(text)
                    if not interrupt_query:
                        continue
                    self._interrupt_current_response()
                    print(f"\n[User][Interrupt] {interrupt_query}")
                    self.text_queue.put(self._new_turn(interrupt_query))
                    continue

                print(f"\n[User] {text}")
                self.text_queue.put(self._new_turn(text))
            except Exception as exc:
                print(f"[Error] ASR: {exc}")

    def llm_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            turn_id = int(item["turn_id"])
            epoch = int(item.get("epoch", -1))
            user_text = str(item["text"])

            try:
                if epoch != self._get_epoch():
                    continue
                messages = self._build_messages(user_text)
                print("[Assistant] ", end="", flush=True)
                self.llm_generating_event.set()

                response = ""
                tts_buf = ""
                tts_seq = 0
                first_token_marked = False
                interrupted = False

                for chunk in self.llm.stream_chat(messages):
                    if epoch != self._get_epoch():
                        interrupted = True
                        break

                    now = time.monotonic()
                    if not first_token_marked:
                        self._mark(turn_id, "ts_llm_first_token", now)
                        first_token_marked = True

                    print(chunk, end="", flush=True)
                    response += chunk
                    tts_buf += chunk

                    if self.tts:
                        norm_len = len(self._normalize_text(tts_buf))
                        hard_stop = re.search(r"[\u3002\uff01\uff1f!?]\s*$", tts_buf) is not None
                        soft_stop = re.search(r"[\uff0c,\u3001\uff1b;\uff1a:\n]\s*$", tts_buf) is not None
                        force_flush = norm_len >= int(self.cfg.tts_stream_force_chars)
                        soft_flush = norm_len >= int(self.cfg.tts_stream_soft_chars) and soft_stop
                        if hard_stop or force_flush or soft_flush:
                            tts_seq = self._enqueue_tts_segments(turn_id, epoch, tts_buf, tts_seq)
                            tts_buf = ""

                self._mark(turn_id, "ts_llm_done", time.monotonic(), only_if_none=True)
                print()

                if interrupted:
                    self._print_turn_metrics(turn_id)
                    continue

                if self.tts and tts_buf.strip():
                    tts_seq = self._enqueue_tts_segments(turn_id, epoch, tts_buf, tts_seq)

                if self.tts:
                    self.tts_queue.put({"turn_id": turn_id, "epoch": epoch, "eot": True, "eot_seq": tts_seq})

                self._append_history(user_text, response.strip())
                self._print_turn_metrics(turn_id)
            except Exception as exc:
                print(f"\n[Error] LLM: {exc}")
            finally:
                self.llm_generating_event.clear()

    def tts_synth_worker(self) -> None:
        if not self.tts:
            return

        turn_states: Dict[int, Dict[str, Any]] = {}
        while not self.stop_event.is_set():
            cur_epoch = self._get_epoch()

            for _ in range(24):
                try:
                    item = self.tts_queue.get_nowait()
                except queue.Empty:
                    break

                item_epoch = int(item.get("epoch", -1))
                if item_epoch != cur_epoch:
                    continue

                turn_id = int(item["turn_id"])
                state = turn_states.setdefault(
                    turn_id,
                    {
                        "epoch": item_epoch,
                        "next_synth_seq": 0,
                        "eot_seq": None,
                        "pending_text": {},
                        "last_enqueue_ts": 0.0,
                        "eot_sent": False,
                    },
                )

                if item.get("eot"):
                    state["eot_seq"] = int(item.get("eot_seq", state["next_synth_seq"]))
                    continue

                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                seq = int(item.get("seq", 0))
                state["pending_text"][seq] = text
                state["last_enqueue_ts"] = time.monotonic()

            stale_turns = [tid for tid, st in turn_states.items() if int(st.get("epoch", -1)) != cur_epoch]
            for turn_id in stale_turns:
                turn_states.pop(turn_id, None)

            did_work = False
            synth_target = None

            for turn_id in sorted(turn_states.keys()):
                state = turn_states[turn_id]
                seq = int(state["next_synth_seq"])
                text = state["pending_text"].pop(seq, None)
                if text is None:
                    continue

                if seq > 0 and state.get("eot_seq") is None:
                    wait_s = max(0.0, float(self.cfg.tts_batch_wait_ms) / 1000.0)
                    if (time.monotonic() - float(state.get("last_enqueue_ts", 0.0))) < wait_s:
                        state["pending_text"][seq] = text
                        continue

                state["next_synth_seq"] = seq + 1
                synth_target = (turn_id, int(state["epoch"]), seq, text)
                break

            if synth_target is not None:
                turn_id, target_epoch, seq, text = synth_target
                t0 = time.monotonic()
                packet = self.tts.synthesize(text)
                t1 = time.monotonic()

                if packet is not None and target_epoch == self._get_epoch() and not self.stop_event.is_set():
                    audio, sr = packet
                    self._add(turn_id, "tts_synth_total", t1 - t0)
                    self._add(turn_id, "tts_audio_total", (len(audio) / float(sr)) if sr > 0 else 0.0)
                    try:
                        self.tts_ready_queue.put_nowait(
                            {
                                "turn_id": turn_id,
                                "epoch": target_epoch,
                                "seq": seq,
                                "audio": audio,
                                "sr": sr,
                            }
                        )
                    except queue.Full:
                        pass
                did_work = True

            for turn_id in sorted(list(turn_states.keys())):
                state = turn_states.get(turn_id)
                if state is None:
                    continue
                eot_seq = state.get("eot_seq")
                if (
                    eot_seq is not None
                    and not state.get("eot_sent", False)
                    and int(state["next_synth_seq"]) >= int(eot_seq)
                    and (not state["pending_text"])
                ):
                    try:
                        self.tts_ready_queue.put_nowait(
                            {
                                "turn_id": turn_id,
                                "epoch": int(state["epoch"]),
                                "eot": True,
                                "eot_seq": int(eot_seq),
                            }
                        )
                    except queue.Full:
                        pass
                    state["eot_sent"] = True
                    turn_states.pop(turn_id, None)
                    did_work = True

            if not did_work:
                time.sleep(0.005)

    def tts_play_worker(self) -> None:
        if not self.tts:
            return

        play_states: Dict[int, Dict[str, Any]] = {}
        while not self.stop_event.is_set():
            cur_epoch = self._get_epoch()

            for _ in range(24):
                try:
                    item = self.tts_ready_queue.get_nowait()
                except queue.Empty:
                    break

                item_epoch = int(item.get("epoch", -1))
                if item_epoch != cur_epoch:
                    continue

                turn_id = int(item["turn_id"])
                state = play_states.setdefault(
                    turn_id,
                    {"epoch": item_epoch, "next_play_seq": 0, "eot_seq": None, "ready": {}},
                )

                if item.get("eot"):
                    state["eot_seq"] = int(item.get("eot_seq", state["next_play_seq"]))
                    continue

                seq = int(item.get("seq", 0))
                state["ready"][seq] = (item["audio"], int(item["sr"]))

            stale_turns = [tid for tid, st in play_states.items() if int(st.get("epoch", -1)) != cur_epoch]
            for turn_id in stale_turns:
                play_states.pop(turn_id, None)

            did_work = False
            while play_states:
                turn_id = min(play_states.keys())
                state = play_states[turn_id]
                next_seq = int(state["next_play_seq"])
                packet = state["ready"].pop(next_seq, None)
                if packet is None:
                    break

                audio, sr = packet
                try:
                    self.speaking_event.set()
                    self._mark(turn_id, "ts_tts_play_first", time.monotonic())
                    self.tts.speak(audio, sr)
                except Exception as exc:
                    print(f"[Error] TTS: {exc}")
                finally:
                    self.speaking_event.clear()

                state["next_play_seq"] = next_seq + 1
                did_work = True

                eot_seq = state.get("eot_seq")
                if eot_seq is not None and int(state["next_play_seq"]) >= int(eot_seq) and (not state["ready"]):
                    self._print_turn_metrics(turn_id)
                    play_states.pop(turn_id, None)

            if not did_work:
                time.sleep(0.005)

    def tts_stream_worker(self) -> None:
        if not self.tts or self.audio_player is None:
            return

        while not self.stop_event.is_set():
            try:
                item = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            item_epoch = int(item.get("epoch", -1))
            if item_epoch != self._get_epoch():
                continue

            turn_id = int(item["turn_id"])
            if item.get("eot"):
                self.audio_player.wait_until_idle(timeout=30.0)
                self.speaking_event.clear()
                self._print_turn_metrics(turn_id)
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            try:
                self.speaking_event.set()
                first_chunk = True
                for audio_chunk, sr, timing in self.tts.synthesize_stream(text):
                    if item_epoch != self._get_epoch() or self.stop_event.is_set():
                        break
                    if first_chunk:
                        self._mark(turn_id, "ts_tts_play_first", time.monotonic())
                        first_chunk = False

                    self.audio_player.enqueue(audio_chunk)
                    self._add(turn_id, "tts_audio_total", (len(audio_chunk) / float(sr)) if sr > 0 else 0.0)
                    synth_time = (timing.get("prefill_ms", 0.0) + timing.get("decode_ms", 0.0)) / 1000.0
                    self._add(turn_id, "tts_synth_total", synth_time)
            except Exception as exc:
                print(f"[Error] TTS stream: {exc}")

    def run(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for microphone/speaker realtime mode")
        workers = [
            threading.Thread(target=self.asr_worker, daemon=True),
            threading.Thread(target=self.llm_worker, daemon=True),
        ]
        if self.tts:
            workers.append(threading.Thread(target=self.tts_stream_worker, daemon=True))
        for worker in workers:
            worker.start()

        input_device = self._resolve_input_device()

        print("====== Realtime ASR-LLM-FasterQwen3TTS Started ======")
        print("Speak naturally. Input q + Enter to quit.")
        print(
            f"[Config] flash_attn_pkg={_flash_attn_available()} "
            f"asr={self.cfg.asr_use_flash_attn} llm={self.cfg.llm_use_flash_attn}"
        )
        print(f"[Config] tts_model={self.cfg.tts_model_key} tts_path={self.cfg.tts_path}")
        print(f"[Config] llm_path={self.cfg.llm_path}")
        print(
            f"[Config] xvector_only_mode={self.cfg.xvector_only_mode} "
            f"ref_text_len={len((self.cfg.ref_text or '').strip())}"
        )
        print(f"[Config] pronunciation_lexicon={self.cfg.pronunciation_lexicon_path}")
        print(f"[Config] rag_backend={self.cfg.rag_backend} rag_source={self.cfg.rag_source or '-'}")
        if self.cfg.rag_index:
            print(f"[Config] rag_index={self.cfg.rag_index}")
        if self.cfg.rag_backend == "vector-index":
            print(f"[Config] rag_embedding_model={self.cfg.rag_embedding_model}")
        if self.cfg.rag_collection:
            print(f"[Config] rag_collection={self.cfg.rag_collection}")
        if self.cfg.rag_filters:
            print(f"[Config] rag_filters={self.cfg.rag_filters}")
        if self.cfg.voice_anchor:
            print(f"[Config] voice_anchor={self.cfg.voice_anchor}")
        if self.cfg.ref_text_source:
            print(f"[Config] ref_text_source={self.cfg.ref_text_source}")
        print(f"[Config] tts_non_streaming_mode={self.cfg.tts_non_streaming_mode}")
        print(
            f"[Config] tts_stream_chunk_size={self.cfg.tts_stream_chunk_size} "
            f"tts_chunk_max_len={self.cfg.tts_chunk_max_len} "
            f"batch_wait_ms={self.cfg.tts_batch_wait_ms}"
        )
        print(
            f"[Config] interrupt_enable={self.cfg.interrupt_enable} "
            f"require_wake={self.cfg.interrupt_require_wake_word_when_busy} "
            f"wake_words={self.cfg.interrupt_wake_words}"
        )
        print(f"[Config] asr_pause_during_tts={self.cfg.asr_pause_during_tts}")

        try:
            if self.audio_player is not None:
                self.audio_player.start()
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                blocksize=self.cfg.block_size,
                dtype="float32",
                device=input_device,
                callback=self.audio_callback,
            ):
                while True:
                    cmd = input().strip().lower()
                    if cmd in {"q", "quit", "exit"}:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            if self.audio_player is not None:
                self.audio_player.clear()
                self.audio_player.stop()
            time.sleep(0.5)
            print("Assistant stopped.")


def run_websocket_server(cfg: RuntimeConfig) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError("Web mode requires fastapi and uvicorn. Install with: pip install -e .[demo]") from exc

    assistant = VoiceAssistant(cfg)
    app = FastAPI(title="Realtime Voice Assistant")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    root = Path(__file__).resolve().parent.parent
    digital_human_page = root / "demo" / "digital_human.html"
    app.mount("/assets", StaticFiles(directory=root / "demo"), name="assets")

    @app.get("/")
    async def home():
        return FileResponse(digital_human_page)

    @app.get("/digital-human")
    async def digital_human():
        return FileResponse(digital_human_page)

    @app.get("/status")
    async def status():
        return {
            "loaded": True,
            "mode": "realtime_voice_assistant_websocket",
            "asr_path": cfg.asr_path,
            "llm_path": cfg.llm_path,
            "tts_path": cfg.tts_path,
            "language": cfg.language,
            "voice_anchor": cfg.voice_anchor,
            "pronunciation_lexicon": cfg.pronunciation_lexicon_path,
            "rag_backend": cfg.rag_backend,
            "rag_source": cfg.rag_source,
            "rag_collection": cfg.rag_collection,
            "rag_filters": cfg.rag_filters,
            "pre_roll_sec": cfg.pre_roll_sec,
            "min_utterance_sec": cfg.min_utterance_sec,
            "silence_sec": cfg.silence_sec,
        }

    @app.post("/text/normalize")
    async def normalize_text_preview(text: str = Form(...)):
        normalized = assistant.text_normalizer.normalize_for_tts(text)
        return {"original": text, "normalized": normalized}

    @app.websocket("/ws/assistant")
    async def assistant_ws(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        outgoing: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        worker_lock = threading.Lock()
        worker_holder: Dict[str, Optional[threading.Thread]] = {"thread": None}
        audio_state: Dict[str, Any] = {
            "segmenter": _build_segmenter(cfg),
            "language": cfg.language,
            "chunk_size": cfg.tts_stream_chunk_size,
            "streaming": False,
            "pre_roll_sec": cfg.pre_roll_sec,
            "min_utterance_sec": cfg.min_utterance_sec,
            "silence_sec": cfg.silence_sec,
        }

        def emit(event: Dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(outgoing.put(event), loop)

        def interrupt_current() -> None:
            assistant._interrupt_current_response()
            emit({"type": "interrupted"})

        def start_worker(user_text: str, payload_meta: Dict[str, Any], transcript: str | None = None) -> None:
            def worker() -> None:
                old_language = assistant.cfg.language
                old_chunk_size = assistant.cfg.tts_stream_chunk_size
                try:
                    if payload_meta.get("language"):
                        assistant.cfg.language = str(payload_meta.get("language"))
                    if payload_meta.get("chunk_size"):
                        assistant.cfg.tts_stream_chunk_size = max(1, int(payload_meta.get("chunk_size")))
                    if transcript is not None:
                        emit({"type": "transcript", "text": transcript})
                    for event in assistant.stream_dialog(user_text):
                        emit(event)
                except Exception as exc:
                    emit({"type": "error", "message": str(exc)})
                finally:
                    assistant.cfg.language = old_language
                    assistant.cfg.tts_stream_chunk_size = old_chunk_size
                    emit({"type": "request_done"})
                    with worker_lock:
                        worker_holder["thread"] = None

            with worker_lock:
                thread = worker_holder.get("thread")
                if thread is not None and thread.is_alive():
                    interrupt_current()
                    thread.join(timeout=1.5)
                thread = threading.Thread(target=worker, daemon=True)
                worker_holder["thread"] = thread
                thread.start()

        def handle_audio_segment(audio: np.ndarray, payload_meta: Dict[str, Any]) -> None:
            text = assistant.transcribe_audio_array(audio)
            if not assistant._is_valid_user_text(text):
                emit({"type": "transcript", "text": ""})
                return
            start_worker(text, payload_meta, transcript=text)

        def process_audio_chunk(payload: Dict[str, Any]) -> None:
            chunk_b64 = str(payload.get("audio_chunk_b64", "")).strip()
            if not chunk_b64:
                raise ValueError("audio_chunk_b64 is required")
            src_sr = int(payload.get("sample_rate") or cfg.sample_rate)
            audio_chunk = _decode_pcm16_chunk(base64.b64decode(chunk_b64), src_sr, cfg.sample_rate)
            segmenter = audio_state["segmenter"]
            was_active = bool(segmenter.active)
            segment = segmenter.push(audio_chunk)
            if (not was_active) and segmenter.active:
                emit({"type": "listening", "state": "speech_start"})
            if segment is not None:
                emit({"type": "listening", "state": "speech_end"})
                handle_audio_segment(
                    segment,
                    {
                        "language": audio_state.get("language"),
                        "chunk_size": audio_state.get("chunk_size"),
                    },
                )

        def flush_audio_stream() -> None:
            segmenter = audio_state["segmenter"]
            flushed = segmenter.force_flush()
            if flushed is not None and flushed.size > 0:
                emit({"type": "listening", "state": "speech_end"})
                handle_audio_segment(
                    flushed,
                    {
                        "language": audio_state.get("language"),
                        "chunk_size": audio_state.get("chunk_size"),
                    },
                )

        async def sender_loop() -> None:
            while True:
                event = await outgoing.get()
                await websocket.send_json(event)

        sender_task = asyncio.create_task(sender_loop())
        await outgoing.put(
            {
                "type": "ready",
                "message": "assistant websocket ready",
                "status": {
                    "language": cfg.language,
                    "rag_backend": cfg.rag_backend,
                    "rag_collection": cfg.rag_collection,
                },
            }
        )

        try:
            while True:
                payload = await websocket.receive_json()
                action = str(payload.get("type", "")).strip()
                if action == "text":
                    start_worker(
                        str(payload.get("text", "")).strip(),
                        payload,
                    )
                elif action == "audio":
                    audio_b64 = str(payload.get("audio_b64", "")).strip()
                    if not audio_b64:
                        await outgoing.put({"type": "error", "message": "audio_b64 is required"})
                        continue
                    try:
                        text = assistant.transcribe_audio_bytes(base64.b64decode(audio_b64))
                    except Exception as exc:
                        await outgoing.put({"type": "error", "message": str(exc)})
                        continue
                    start_worker(text, payload, transcript=text)
                elif action == "audio_start":
                    audio_state["pre_roll_sec"] = max(0.05, float(payload.get("pre_roll_sec") or cfg.pre_roll_sec))
                    audio_state["min_utterance_sec"] = max(
                        0.1,
                        float(payload.get("min_utterance_sec") or cfg.min_utterance_sec),
                    )
                    audio_state["silence_sec"] = max(0.1, float(payload.get("silence_sec") or cfg.silence_sec))
                    audio_state["segmenter"] = _build_segmenter(
                        cfg,
                        pre_roll_sec=audio_state["pre_roll_sec"],
                        min_utterance_sec=audio_state["min_utterance_sec"],
                        silence_sec=audio_state["silence_sec"],
                    )
                    audio_state["language"] = str(payload.get("language") or cfg.language)
                    audio_state["chunk_size"] = max(1, int(payload.get("chunk_size") or cfg.tts_stream_chunk_size))
                    audio_state["streaming"] = True
                    await outgoing.put(
                        {
                            "type": "listening",
                            "state": "armed",
                            "silence_sec": audio_state["silence_sec"],
                            "min_utterance_sec": audio_state["min_utterance_sec"],
                        }
                    )
                elif action == "audio_chunk":
                    try:
                        process_audio_chunk(payload)
                    except Exception as exc:
                        await outgoing.put({"type": "error", "message": str(exc)})
                elif action == "audio_end":
                    audio_state["streaming"] = False
                    flush_audio_stream()
                    await outgoing.put({"type": "listening", "state": "idle"})
                elif action == "interrupt":
                    interrupt_current()
                elif action == "normalize":
                    normalized = assistant.text_normalizer.normalize_for_tts(str(payload.get("text", "")))
                    await outgoing.put({"type": "normalized", "normalized": normalized})
                elif action == "ping":
                    await outgoing.put({"type": "pong"})
                else:
                    await outgoing.put({"type": "error", "message": f"Unknown action: {action}"})
        except WebSocketDisconnect:
            interrupt_current()
        finally:
            sender_task.cancel()

    print(f"[Web] serving digital human on http://{cfg.web_host}:{cfg.web_port}/digital-human")
    uvicorn.run(app, host=cfg.web_host, port=int(cfg.web_port), log_level="info")


if __name__ == "__main__":
    runtime_cfg = build_runtime_config()
    if runtime_cfg.serve_web:
        run_websocket_server(runtime_cfg)
    else:
        VoiceAssistant(runtime_cfg).run()
