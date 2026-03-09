import types

from faster_qwen3_tts.model import FasterQwen3TTS


def _dummy_graph():
    return object()


def test_uses_speech_tokenizer_sample_rate_when_available():
    base_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            speech_tokenizer=types.SimpleNamespace(sample_rate=24000),
        )
    )
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph())
    assert model.sample_rate == 24000


def test_falls_back_to_base_model_sample_rate():
    base_model = types.SimpleNamespace(sample_rate=22050)
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph())
    assert model.sample_rate == 22050


def test_defaults_to_24khz_when_sample_rate_unavailable():
    base_model = types.SimpleNamespace(model=types.SimpleNamespace())
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph())
    assert model.sample_rate == 24000


def test_uses_model_config_sample_rate_when_available():
    base_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            config=types.SimpleNamespace(sample_rate=24000),
        )
    )
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph())
    assert model.sample_rate == 24000


def test_prefers_dtype_kwarg_when_available():
    def from_pretrained(*, dtype=None, torch_dtype=None):
        return None

    assert FasterQwen3TTS._dtype_kwarg_name(from_pretrained) == "dtype"


def test_falls_back_to_torch_dtype_kwarg_for_older_versions():
    def from_pretrained(*, torch_dtype=None):
        return None

    assert FasterQwen3TTS._dtype_kwarg_name(from_pretrained) == "torch_dtype"
