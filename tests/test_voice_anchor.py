import types
import sys

import numpy as np
import torch

sys.modules.setdefault("soundfile", types.SimpleNamespace())

from faster_qwen3_tts.model import FasterQwen3TTS


def _dummy_graph():
    return object()


def _make_base_model():
    talker = types.SimpleNamespace(device="cpu", rope_deltas=None)
    model = types.SimpleNamespace(
        talker=talker,
        config=types.SimpleNamespace(talker_config=types.SimpleNamespace(), sample_rate=24000),
        speech_tokenizer=types.SimpleNamespace(sample_rate=24000),
    )
    return types.SimpleNamespace(
        model=model,
        sample_rate=24000,
        _build_assistant_text=lambda text: text,
        _build_ref_text=lambda text: f"REF::{text}",
        _tokenize_texts=lambda texts: [
            torch.tensor([[101, 102, 103, 104]], dtype=torch.long) for _ in texts
        ],
    )


def test_array_payload_roundtrip_preserves_shape_and_dtype():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)

    payload = FasterQwen3TTS._encode_array_payload(source)
    restored = FasterQwen3TTS._decode_array_payload(payload)

    assert restored is not None
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert np.array_equal(restored, source)


def test_encode_array_payload_supports_bfloat16_tensor():
    source = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)

    payload = FasterQwen3TTS._encode_array_payload(source)
    restored = FasterQwen3TTS._decode_array_payload(payload)

    assert restored is not None
    assert restored.dtype == np.float32
    assert np.allclose(restored, np.array([0.25, -0.5], dtype=np.float32), atol=1e-3)


def test_resolve_subtalker_sampling_overrides_independently():
    result = FasterQwen3TTS._resolve_subtalker_sampling(
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        subtalker_dosample=False,
        subtalker_top_k=12,
        subtalker_top_p=0.7,
        subtalker_temperature=0.4,
    )

    assert result == (False, 12, 0.7, 0.4)


def test_save_and_load_voice_anchor_roundtrip(tmp_path):
    base_model = _make_base_model()
    calls = []

    def create_voice_clone_prompt(*, ref_audio, ref_text, x_vector_only_mode=False):
        calls.append((ref_audio, ref_text, x_vector_only_mode))
        return [
            types.SimpleNamespace(
                ref_spk_embedding=torch.tensor([0.25, -0.5], dtype=torch.float32),
                ref_code=None,
                ref_text="",
            )
        ]

    base_model.create_voice_clone_prompt = create_voice_clone_prompt
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph(), device="cpu", dtype=torch.float32)

    anchor_path = tmp_path / "speaker.anchor.json"
    saved = model.save_voice_anchor(
        anchor_path,
        ref_audio="speaker.wav",
        xvec_only=True,
        metadata={"speaker": "demo"},
    )
    loaded = model.load_voice_anchor(anchor_path)

    assert calls == [("speaker.wav", "", True)]
    assert loaded == saved
    assert loaded["format"] == "faster_qwen3_tts.voice_anchor.v1"
    assert loaded["xvec_only"] is True
    assert loaded["metadata"] == {"speaker": "demo"}


def test_prepare_generation_uses_anchor_mode_to_return_ref_codes():
    base_model = _make_base_model()
    model = FasterQwen3TTS(base_model, _dummy_graph(), _dummy_graph(), device="cpu", dtype=torch.float32)
    model._warmed_up = True
    model._build_talker_inputs_local = lambda **kwargs: (
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, dtype=torch.long),
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 4),
    )

    ref_codes = np.arange(32, dtype=np.int64).reshape(2, 16)
    voice_anchor = {
        "format": "faster_qwen3_tts.voice_anchor.v1",
        "xvec_only": False,
        "ref_text": "reference transcript",
        "speaker_embedding": FasterQwen3TTS._encode_array_payload(
            np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ),
        "ref_codes": FasterQwen3TTS._encode_array_payload(ref_codes),
        "metadata": {},
    }

    *_, resolved_ref_codes = model._prepare_generation(
        text="hello",
        ref_audio=None,
        ref_text="",
        language="English",
        xvec_only=True,
        voice_anchor=voice_anchor,
    )

    assert resolved_ref_codes is not None
    assert torch.equal(resolved_ref_codes, torch.from_numpy(ref_codes))
