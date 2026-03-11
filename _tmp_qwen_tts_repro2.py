import traceback
from examples.realtime_voice_assistant import RuntimeConfig, apply_tts_model_preset, StreamingTTS

cfg = apply_tts_model_preset(RuntimeConfig(), '1.7b')
cfg.tts_path = r'D:\work\QWen3\Qwen3-TTS-12Hz-1.7B-Base'
cfg.ref_audio = r'D:\work\QWen3\faster-qwen3-tts\ref_voice.wav'
cfg.voice_anchor = ''
cfg.xvector_only_mode = True
cfg.language = 'Chinese'
cfg.tts_stream_chunk_size = 8

try:
    tts = StreamingTTS(cfg)
    gen = tts.synthesize_stream('你好。')
    first = next(gen)
    print('OK', type(first[0]), getattr(first[0], 'shape', None), first[1], first[2])
except Exception:
    traceback.print_exc()
