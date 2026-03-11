import traceback
from examples.realtime_voice_assistant import RuntimeConfig, apply_tts_model_preset, StreamingTTS

cfg = apply_tts_model_preset(RuntimeConfig(), '1.7b')
cfg.tts_path = r'D:\work\QWen3\Qwen3-TTS-12Hz-1.7B-Base'
cfg.ref_audio = r'D:\work\QWen3\faster-qwen3-tts\ref_voice.wav'
cfg.voice_anchor = r'D:\work\QWen3\faster-qwen3-tts\narrator.anchor.json'
cfg.xvector_only_mode = True
cfg.language = 'Chinese'

try:
    StreamingTTS(cfg)
except Exception:
    traceback.print_exc()
