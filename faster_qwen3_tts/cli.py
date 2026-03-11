#!/usr/bin/env python3
"""CLI for FasterQwen3TTS."""
import argparse
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


def _load_model(model_id: str, device: str, dtype: str):
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    return FasterQwen3TTS.from_pretrained(
        model_id,
        device=device,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        max_seq_len=2048,
    )


def _write_audio(out_path: str, audio: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio, sr)


def _stream_to_audio(gen):
    chunks = []
    sr = None
    for audio_chunk, sr, _ in gen:
        chunks.append(audio_chunk)
    if not chunks:
        return np.zeros(1, dtype=np.float32), 24000
    return np.concatenate(chunks), sr


def cmd_clone(args):
    model = _load_model(args.model, args.device, args.dtype)
    ref_audio = args.ref_audio
    ref_text = args.ref_text or ""
    voice_anchor = args.voice_anchor

    if not voice_anchor and not ref_audio:
        print("ERROR: provide --voice-anchor or --ref-audio")
        sys.exit(2)
    if not voice_anchor and not args.xvec_only and not ref_text:
        print("ERROR: --ref-text is required unless --xvec-only is set")
        sys.exit(2)

    if args.streaming:
        start = time.perf_counter()
        gen = model.generate_voice_clone_streaming(
            text=args.text,
            language=args.language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_anchor=voice_anchor,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
            xvec_only=args.xvec_only,
            non_streaming_mode=args.non_streaming_mode,
        )
        audio, sr = _stream_to_audio(gen)
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0
    else:
        start = time.perf_counter()
        audio_list, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_anchor=voice_anchor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
            xvec_only=args.xvec_only,
            non_streaming_mode=args.non_streaming_mode,
        )
        audio = audio_list[0]
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0

    _write_audio(args.output, audio, sr)
    print(f"Wrote {args.output} (dur {audio_dur:.2f}s, RTF {rtf:.2f})")


def cmd_anchor(args):
    model = _load_model(args.model, args.device, args.dtype)
    if not args.xvec_only and not args.ref_text:
        print("ERROR: --ref-text is required unless --xvec-only is set")
        sys.exit(2)
    model.save_voice_anchor(
        args.output,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text or "",
        xvec_only=args.xvec_only,
    )
    print(f"Wrote voice anchor {args.output}")


def cmd_custom(args):
    model = _load_model(args.model, args.device, args.dtype)

    if args.list_speakers:
        speakers = model.model.get_supported_speakers() or []
        print("\n".join(speakers))
        return

    if not args.speaker:
        print("ERROR: --speaker is required (use --list-speakers)")
        sys.exit(2)

    if args.streaming:
        start = time.perf_counter()
        gen = model.generate_custom_voice_streaming(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
        )
        audio, sr = _stream_to_audio(gen)
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0
    else:
        start = time.perf_counter()
        audio_list, sr = model.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
        )
        audio = audio_list[0]
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0

    _write_audio(args.output, audio, sr)
    print(f"Wrote {args.output} (dur {audio_dur:.2f}s, RTF {rtf:.2f})")


def cmd_design(args):
    model = _load_model(args.model, args.device, args.dtype)

    if args.streaming:
        start = time.perf_counter()
        gen = model.generate_voice_design_streaming(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
        )
        audio, sr = _stream_to_audio(gen)
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0
    else:
        start = time.perf_counter()
        audio_list, sr = model.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.sub_greedy,
            subtalker_top_k=args.sub_top_k,
            subtalker_top_p=args.sub_top_p,
            subtalker_temperature=args.sub_temperature,
            seed=args.seed,
            subtalker_seed=args.sub_seed,
        )
        audio = audio_list[0]
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0

    _write_audio(args.output, audio, sr)
    print(f"Wrote {args.output} (dur {audio_dur:.2f}s, RTF {rtf:.2f})")


def cmd_serve(args):
    model = _load_model(args.model, args.device, args.dtype)

    if args.mode == "clone":
        if not args.voice_anchor and not args.ref_audio:
            print("ERROR: clone mode requires --voice-anchor or --ref-audio")
            sys.exit(2)
        if not args.voice_anchor and not args.xvec_only and not args.ref_text:
            print("ERROR: --ref-text is required for clone mode unless --xvec-only is set")
            sys.exit(2)
    if args.mode == "custom" and not args.speaker:
        print("ERROR: --speaker is required for custom mode")
        sys.exit(2)
    if args.mode == "design" and not args.instruct:
        print("ERROR: --instruct is required for design mode")
        sys.exit(2)

    print("Server started. Enter text per line. Type 'exit' or 'quit' to stop.")
    idx = 1
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "stop"):
            break

        out_path = os.path.join(args.output_dir, f"out_{idx:04d}.wav")
        idx += 1

        start = time.perf_counter()

        if args.mode == "clone":
            if args.streaming:
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    voice_anchor=args.voice_anchor,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                    xvec_only=args.xvec_only,
                    non_streaming_mode=args.non_streaming_mode,
                )
                audio, sr = _stream_to_audio(gen)
            else:
                audio_list, sr = model.generate_voice_clone(
                    text=text,
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    voice_anchor=args.voice_anchor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                    xvec_only=args.xvec_only,
                    non_streaming_mode=args.non_streaming_mode,
                )
                audio = audio_list[0]
        elif args.mode == "custom":
            if args.streaming:
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=args.speaker,
                    language=args.language,
                    instruct=args.instruct,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                )
                audio, sr = _stream_to_audio(gen)
            else:
                audio_list, sr = model.generate_custom_voice(
                    text=text,
                    speaker=args.speaker,
                    language=args.language,
                    instruct=args.instruct,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                )
                audio = audio_list[0]
        else:
            if args.streaming:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=args.instruct,
                    language=args.language,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                )
                audio, sr = _stream_to_audio(gen)
            else:
                audio_list, sr = model.generate_voice_design(
                    text=text,
                    instruct=args.instruct,
                    language=args.language,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not args.sub_greedy,
                    subtalker_top_k=args.sub_top_k,
                    subtalker_top_p=args.sub_top_p,
                    subtalker_temperature=args.sub_temperature,
                    seed=args.seed,
                    subtalker_seed=args.sub_seed,
                )
                audio = audio_list[0]

        _write_audio(out_path, audio, sr)
        total_time = time.perf_counter() - start
        audio_dur = len(audio) / sr if sr else 0.0
        rtf = audio_dur / total_time if total_time > 0 else 0.0
        print(f"Wrote {out_path} (dur {audio_dur:.2f}s, RTF {rtf:.2f})")


def build_parser():
    p = argparse.ArgumentParser(prog="faster-qwen3-tts", description="FasterQwen3TTS CLI")
    default_model = os.environ.get("QWEN_TTS_MODEL")
    p.add_argument("--device", default=os.environ.get("QWEN_TTS_DEVICE", "cuda"), help="Device (cuda or cpu)")
    p.add_argument(
        "--dtype",
        default=os.environ.get("QWEN_TTS_DTYPE", "bf16"),
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--text", required=True, help="Text to synthesize")
        sp.add_argument("--language", default="Auto", help="Language (Auto, English, French, ...)" )
        sp.add_argument("--output", required=True, help="Output wav path")
        sp.add_argument(
            "--model",
            default=default_model,
            required=default_model is None,
            help="Model id or local path",
        )
        sp.add_argument("--max-new-tokens", type=int, default=2048)
        sp.add_argument("--temperature", type=float, default=0.9)
        sp.add_argument("--top-k", type=int, default=50)
        sp.add_argument("--top-p", type=float, default=1.0)
        sp.add_argument("--repetition-penalty", type=float, default=1.05)
        sp.add_argument("--seed", type=int, default=None, help="Talker sampling seed")
        sp.add_argument("--sub-temperature", type=float, default=None, help="Predictor temperature")
        sp.add_argument("--sub-top-k", type=int, default=None, help="Predictor top-k")
        sp.add_argument("--sub-top-p", type=float, default=None, help="Predictor top-p")
        sp.add_argument("--sub-seed", type=int, default=None, help="Predictor sampling seed")
        sp.add_argument("--greedy", action="store_true", help="Disable sampling")
        sp.add_argument("--sub-greedy", action="store_true", help="Disable predictor sampling")
        sp.add_argument("--streaming", action="store_true", help="Use streaming generation")
        nsm_group = sp.add_mutually_exclusive_group()
        nsm_group.add_argument(
            "--non-streaming-mode",
            dest="non_streaming_mode",
            action="store_true",
            help="Prefill full text for non-streaming quality",
        )
        nsm_group.add_argument(
            "--no-non-streaming-mode",
            dest="non_streaming_mode",
            action="store_false",
            help="Disable full-text prefill (match upstream non-streaming layout)",
        )
        sp.set_defaults(non_streaming_mode=True)
        sp.add_argument("--chunk-size", type=int, default=8, help="Streaming chunk size")

    sp = sub.add_parser("clone", help="Voice cloning (reference audio)")
    add_common(sp)
    sp.add_argument("--ref-audio", help="Reference audio path")
    sp.add_argument("--ref-text", help="Reference transcript")
    sp.add_argument("--voice-anchor", help="Path to a saved voice anchor JSON")
    sp.add_argument("--xvec-only", action="store_true", help="Use speaker embedding only")
    sp.set_defaults(fn=cmd_clone)

    sp = sub.add_parser("anchor", help="Export a reusable voice anchor JSON")
    sp.add_argument(
        "--model",
        default=default_model,
        required=default_model is None,
        help="Model id or local path",
    )
    sp.add_argument("--ref-audio", required=True, help="Reference audio path")
    sp.add_argument("--ref-text", default="", help="Reference transcript (required for ICL)")
    sp.add_argument("--output", required=True, help="Output anchor JSON path")
    sp.add_argument("--xvec-only", action="store_true", help="Export x-vector-only anchor")
    sp.set_defaults(fn=cmd_anchor)

    sp = sub.add_parser("custom", help="CustomVoice model (speaker IDs)")
    add_common(sp)
    sp.add_argument("--speaker", help="Speaker ID")
    sp.add_argument("--instruct", default="", help="Optional instruction")
    sp.add_argument("--list-speakers", action="store_true", help="List available speaker IDs")
    sp.set_defaults(fn=cmd_custom)

    sp = sub.add_parser("design", help="VoiceDesign model (instruction-based)")
    add_common(sp)
    sp.add_argument("--instruct", required=True, help="Voice/style instruction")
    sp.set_defaults(fn=cmd_design)

    sp = sub.add_parser("serve", help="Keep model hot and generate multiple requests from stdin")
    sp.add_argument("--mode", required=True, choices=["clone", "custom", "design"])
    sp.add_argument(
        "--model",
        default=default_model,
        required=default_model is None,
        help="Model id or local path",
    )
    sp.add_argument("--language", default="Auto", help="Language (Auto, English, French, ...)")
    sp.add_argument("--ref-audio", help="Reference audio path (clone)")
    sp.add_argument("--ref-text", help="Reference transcript (clone)")
    sp.add_argument("--voice-anchor", help="Saved voice anchor JSON (clone)")
    sp.add_argument("--xvec-only", action="store_true", help="Use speaker embedding only in clone mode")
    sp.add_argument("--speaker", help="Speaker ID (custom)")
    sp.add_argument("--instruct", default="", help="Instruction (custom/design)")
    sp.add_argument("--streaming", action="store_true", help="Use streaming generation")
    nsm_group = sp.add_mutually_exclusive_group()
    nsm_group.add_argument(
        "--non-streaming-mode",
        dest="non_streaming_mode",
        action="store_true",
        help="Prefill full text for non-streaming quality",
    )
    nsm_group.add_argument(
        "--no-non-streaming-mode",
        dest="non_streaming_mode",
        action="store_false",
        help="Disable full-text prefill (match upstream non-streaming layout)",
    )
    sp.set_defaults(non_streaming_mode=True)
    sp.add_argument("--chunk-size", type=int, default=8, help="Streaming chunk size")
    sp.add_argument("--max-new-tokens", type=int, default=2048)
    sp.add_argument("--temperature", type=float, default=0.9)
    sp.add_argument("--top-k", type=int, default=50)
    sp.add_argument("--top-p", type=float, default=1.0)
    sp.add_argument("--repetition-penalty", type=float, default=1.05)
    sp.add_argument("--seed", type=int, default=None, help="Talker sampling seed")
    sp.add_argument("--sub-temperature", type=float, default=None, help="Predictor temperature")
    sp.add_argument("--sub-top-k", type=int, default=None, help="Predictor top-k")
    sp.add_argument("--sub-top-p", type=float, default=None, help="Predictor top-p")
    sp.add_argument("--sub-seed", type=int, default=None, help="Predictor sampling seed")
    sp.add_argument("--greedy", action="store_true", help="Disable sampling")
    sp.add_argument("--sub-greedy", action="store_true", help="Disable predictor sampling")
    sp.add_argument("--output-dir", default="outputs", help="Directory for output wavs")
    sp.set_defaults(fn=cmd_serve)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
