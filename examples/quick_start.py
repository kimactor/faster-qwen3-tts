import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


MODEL_PATH = "D:/work/QWen3/Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO = "ref_voice.wav"
REF_TEXT = (
    "I am using this reference recording to clone the speaker voice for streaming synthesis."
)
TEXT = "This benchmark-style example measures streaming TTFA and end-to-end RTF."
CHUNK_SIZE = 8
OUTPUT_DIR = Path("stream_chunks")


def save_chunk(audio_chunk, sample_rate, chunk_index):
    OUTPUT_DIR.mkdir(exist_ok=True)
    chunk_path = OUTPUT_DIR / f"chunk_{chunk_index:03d}.wav"
    sf.write(chunk_path, audio_chunk, sample_rate)
    print(f"saved {chunk_path} ({len(audio_chunk)} samples @ {sample_rate} Hz)")


def format_timing(timing):
    chunk_steps = timing.get("chunk_steps", timing.get("steps", 0))
    decode_ms = timing.get("decode_ms")
    ms_per_step = timing.get("ms_per_step")

    if ms_per_step is None and decode_ms is not None and chunk_steps:
        ms_per_step = decode_ms / chunk_steps

    parts = [f"{chunk_steps} steps"]
    if decode_ms is not None:
        parts.append(f"{decode_ms:.0f} ms decode")
    if ms_per_step is not None:
        parts.append(f"{ms_per_step:.1f} ms/step")
    return ", ".join(parts)


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = FasterQwen3TTS.from_pretrained(
        MODEL_PATH,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Warmup once so CUDA graph capture is not included in the reported TTFA.
    print("\nWarmup run (excluded from metrics)...")
    warmup_start = time.perf_counter()
    _audio_list, _sr = model.generate_voice_clone(
        text="Warmup run.",
        language="English",
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        max_new_tokens=20,
    )
    torch.cuda.synchronize()
    print(f"Warmup finished in {time.perf_counter() - warmup_start:.3f}s")

    print(f"\nStreaming benchmark (chunk_size={CHUNK_SIZE})...")
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    total_save_time = 0.0
    chunks = []
    ttfa = None
    sr = None

    generator = model.generate_voice_clone_streaming(
        text=TEXT,
        language="English",
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        chunk_size=CHUNK_SIZE,
    )

    for chunk_index, (audio_chunk, sr, timing) in enumerate(generator, start=1):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if ttfa is None:
            ttfa = now - start_total
            print(f"TTFA (warm run): {ttfa * 1000:.0f} ms")

        chunks.append(audio_chunk)

        save_start = time.perf_counter()
        save_chunk(audio_chunk, sr, chunk_index)
        total_save_time += time.perf_counter() - save_start

        print(f"chunk {chunk_index}: {format_timing(timing)}")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_total

    if not chunks or sr is None:
        print("No audio generated.")
        return

    full_audio = np.concatenate(chunks)
    audio_duration = len(full_audio) / sr
    gen_time = total_time - total_save_time
    rtf_excl_save = audio_duration / gen_time if gen_time > 0 else 0.0
    rtf_incl_save = audio_duration / total_time if total_time > 0 else 0.0

    sf.write("quick_start_full.wav", full_audio, sr)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total time (including save): {total_time:.3f} s")
    print(f"Total save time: {total_save_time:.3f} s")
    print(f"Total generation time (excluding save): {gen_time:.3f} s")
    print(f"Total audio duration: {audio_duration:.3f} s")
    print(f"RTF (excluding save, benchmark style): {rtf_excl_save:.3f}")
    print(f"RTF (including save): {rtf_incl_save:.3f}")
    print("saved quick_start_full.wav")


if __name__ == "__main__":
    main()
