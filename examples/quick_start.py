import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


def save_chunk(audio_chunk, sample_rate, chunk_index, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    chunk_path = output_dir / f"chunk_{chunk_index:03d}.wav"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming quick-start for FasterQwen3TTS")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        help="Model id or local path",
    )
    parser.add_argument("--ref-audio", default="ref_voice.wav", help="Reference audio path")
    parser.add_argument(
        "--ref-text",
        default="I am using this reference recording to clone the speaker voice for streaming synthesis.",
        help="Reference transcript",
    )
    parser.add_argument(
        "--text",
        default="This benchmark-style example measures streaming TTFA and end-to-end RTF.",
        help="Text to synthesize",
    )
    parser.add_argument("--chunk-size", type=int, default=8, help="Streaming chunk size")
    parser.add_argument("--device", default=os.environ.get("QWEN_TTS_DEVICE", "cuda"), help="Torch device")
    parser.add_argument("--output-dir", default="stream_chunks", help="Directory for chunk wav files")
    parser.add_argument("--full-output", default="quick_start_full.wav", help="Merged output wav path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading model from {args.model} ...")
    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print("\nWarmup run (excluded from metrics)...")
    warmup_start = time.perf_counter()
    _audio_list, _sr = model.generate_voice_clone(
        text="Warmup run.",
        language="English",
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        max_new_tokens=20,
    )
    torch.cuda.synchronize()
    print(f"Warmup finished in {time.perf_counter() - warmup_start:.3f}s")

    print(f"\nStreaming benchmark (chunk_size={args.chunk_size})...")
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    total_save_time = 0.0
    chunks = []
    ttfa = None
    sr = None

    generator = model.generate_voice_clone_streaming(
        text=args.text,
        language="English",
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        chunk_size=args.chunk_size,
    )

    for chunk_index, (audio_chunk, sr, timing) in enumerate(generator, start=1):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if ttfa is None:
            ttfa = now - start_total
            print(f"TTFA (warm run): {ttfa * 1000:.0f} ms")

        chunks.append(audio_chunk)

        save_start = time.perf_counter()
        save_chunk(audio_chunk, sr, chunk_index, output_dir)
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

    sf.write(args.full_output, full_audio, sr)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total time (including save): {total_time:.3f} s")
    print(f"Total save time: {total_save_time:.3f} s")
    print(f"Total generation time (excluding save): {gen_time:.3f} s")
    print(f"Total audio duration: {audio_duration:.3f} s")
    print(f"RTF (excluding save, benchmark style): {rtf_excl_save:.3f}")
    print(f"RTF (including save): {rtf_incl_save:.3f}")
    print(f"saved {args.full_output}")


if __name__ == "__main__":
    main()
