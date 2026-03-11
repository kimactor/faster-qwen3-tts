import time

from examples.realtime_voice_assistant import RuntimeConfig, StreamingTTS, apply_tts_model_preset


cfg = apply_tts_model_preset(RuntimeConfig(), "1.7b")
cfg.tts_path = r"D:\work\QWen3\Qwen3-TTS-12Hz-1.7B-Base"
cfg.ref_audio = r"D:\work\QWen3\faster-qwen3-tts\dist\current_pyinstaller_bundle_fixed3\ref_voice.wav"
cfg.voice_anchor = r"D:\work\QWen3\faster-qwen3-tts\dist\current_pyinstaller_bundle_fixed3\ref_voice.anchor.json"
cfg.language = "Chinese"
cfg.tts_stream_chunk_size = 8
cfg.xvector_only_mode = False

text = "你好，这是一次一点七B模型的流式语音测试。"

start = time.perf_counter()
tts = StreamingTTS(cfg)
init_done = time.perf_counter()
gen = tts.synthesize_stream(text)
first_chunk, sr, timing = next(gen)
first_done = time.perf_counter()

chunk_samples = int(getattr(first_chunk, "shape", [len(first_chunk)])[0])
print(
    {
        "init_sec": round(init_done - start, 3),
        "ttfa_sec": round(first_done - init_done, 3),
        "sr": sr,
        "chunk_samples": chunk_samples,
        "timing": timing,
    }
)
