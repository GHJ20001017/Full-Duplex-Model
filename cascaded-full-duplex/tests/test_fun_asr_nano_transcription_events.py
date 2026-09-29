from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT.fun_asr_nano_handler import FunASRNanoSTTHandler

MODEL_NAME = "FunAudioLLM/Fun-ASR-Nano-2512"


class FakeModel:
    def __init__(self, generate_result=None):
        self.generate_result = [{"text": "ok"}] if generate_result is None else generate_result
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.generate_result


def install_fake_funasr(monkeypatch, model, constructors):
    def auto_model(**kwargs):
        constructors.append(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=auto_model))


def hotwords_file(tmp_path, content="\ufeff# heading\nhello world\n"):
    path = tmp_path / "hotwords.txt"
    path.write_text(content, encoding="utf-8")
    return path


def vad(audio, *, mode="progressive", turn_id="turn", revision=1, created=12.5):
    return VADAudio(
        audio=np.asarray(audio, dtype=np.float32),
        mode=mode,
        turn_id=turn_id,
        turn_revision=revision,
        created_at_s=created,
    )


def new_handler():
    return object.__new__(FunASRNanoSTTHandler)


def configured_handler(monkeypatch, tmp_path, model=None, **setup_kwargs):
    model = model or FakeModel()
    constructors = []
    install_fake_funasr(monkeypatch, model, constructors)
    content = setup_kwargs.pop("content", "\ufeff# heading\nhello world\n")
    handler = new_handler()
    handler.setup(hotwords_file=str(hotwords_file(tmp_path, content)), **setup_kwargs)
    return handler, model, constructors


def test_setup_loads_full_model_path_and_fun_asr_options(monkeypatch, tmp_path):
    handler, model, constructors = configured_handler(
        monkeypatch,
        tmp_path,
        model=FakeModel(),
        model_name=MODEL_NAME,
        device="cpu",
        hub="modelscope",
    )

    assert handler.model is model
    assert constructors == [
        {
            "model": MODEL_NAME,
            "device": "cpu",
            "hub": "modelscope",
            "trust_remote_code": True,
        }
    ]


@pytest.mark.parametrize(
    "content, error",
    [
        (None, FileNotFoundError),
        ("", ValueError),
        ("# only a comment\n  \n# another\n", ValueError),
    ],
)
def test_hotwords_are_validated_before_model_load(monkeypatch, tmp_path, content, error):
    constructors = []
    install_fake_funasr(monkeypatch, FakeModel(), constructors)
    handler = new_handler()
    path = tmp_path / "hotwords.txt"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        handler.setup(hotwords_file=str(path))
    assert constructors == []


def test_missing_hotwords_argument_fails_before_model_load(monkeypatch):
    constructors = []
    install_fake_funasr(monkeypatch, FakeModel(), constructors)
    with pytest.raises(ValueError, match="requires --fun_asr_nano_stt_hotwords_file"):
        new_handler().setup()
    assert constructors == []


def test_unreadable_hotwords_are_rejected_before_model_load(monkeypatch, tmp_path):
    constructors = []
    install_fake_funasr(monkeypatch, FakeModel(), constructors)
    path = tmp_path / "hotwords.txt"
    path.write_text("hello\n", encoding="utf-8")
    original_read_text = __import__("pathlib").Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == path:
            raise PermissionError("unreadable hotwords")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.read_text", unreadable)
    with pytest.raises(PermissionError):
        new_handler().setup(hotwords_file=str(path))
    assert constructors == []


def test_hotwords_support_bom_comments_dedup_and_multiword_phrases(monkeypatch, tmp_path):
    handler, _, _ = configured_handler(
        monkeypatch,
        tmp_path,
        content="\ufeff# comment\n hello world \nhello world\n\n# ignored\nNew York City\n",
    )

    assert handler.hotwords == ["hello world", "New York City"]


def test_process_redecodes_each_cumulative_snapshot_with_fresh_cache(monkeypatch, tmp_path):
    model = FakeModel()
    model.generate_result = [{"text": "  hello world  "}]
    handler, _, _ = configured_handler(monkeypatch, tmp_path, model=model)

    first = list(handler.process(vad([1], turn_id="a", revision=1)))[0]
    second = list(handler.process(vad([1, 2], turn_id="a", revision=2)))[0]
    third = list(handler.process(vad([9], turn_id="b", revision=1)))[0]

    assert [first.text, second.text, third.text] == ["hello world"] * 3
    assert [call["input"][0].tolist() for call in model.generate_calls] == [[1.0], [1.0, 2.0], [9.0]]
    assert all(call["cache"] == {} for call in model.generate_calls)
    assert all(call["input"][0] is not model.generate_calls[0]["input"][0] for call in model.generate_calls[1:])


def test_process_propagates_hotwords_each_call_and_preserves_english_spaces(monkeypatch, tmp_path):
    model = FakeModel()
    model.generate_result = [{"text": "  turn left now  "}]
    handler, _, _ = configured_handler(
        monkeypatch,
        tmp_path,
        model=model,
        language="英文",
        itn=False,
        content="left turn\nturn left now\n",
    )

    outputs = [list(handler.process(vad([1])))[0], list(handler.process(vad([2])))[0]]

    assert [out.text for out in outputs] == ["turn left now", "turn left now"]
    assert all(call["hotwords"] == ["left turn", "turn left now"] for call in model.generate_calls)
    assert all(call["hotwords"] is not handler.hotwords for call in model.generate_calls)
    assert all(call["language"] == "英文" and call["itn"] is False for call in model.generate_calls)


def test_final_transcription_maps_language_and_metadata(monkeypatch, tmp_path):
    handler, model, _ = configured_handler(monkeypatch, tmp_path, language="日文")
    result = list(handler.process(vad([1], mode="final", turn_id="final", revision=7, created=42.25)))[0]

    assert isinstance(result, Transcription)
    assert result.text == "ok"
    assert result.language_code == "ja"
    assert result.turn_id == "final"
    assert result.turn_revision == 7
    assert result.speech_stopped_at_s == 42.25
    assert len(model.generate_calls) == 1


@pytest.mark.parametrize("malformed", [[], [{}], [{"text": None}], ["text"], [{"other": "x"}]])
def test_malformed_generate_results_raise_runtime_error(monkeypatch, tmp_path, malformed):
    model = FakeModel(generate_result=malformed)
    handler, _, _ = configured_handler(monkeypatch, tmp_path, model=model)

    with pytest.raises(RuntimeError, match="no valid transcription text"):
        list(handler.process(vad([1])))


def test_empty_audio_emits_empty_event_without_model_generate(monkeypatch, tmp_path):
    handler, model, _ = configured_handler(monkeypatch, tmp_path)

    partial = list(handler.process(vad([], mode="progressive")))[0]
    final = list(handler.process(vad([], mode="final", created=9.0)))[0]

    assert isinstance(partial, PartialTranscription)
    assert partial.text == ""
    assert isinstance(final, Transcription)
    assert final.text == ""
    assert final.language_code == "zh"
    assert model.generate_calls == []


def test_invalid_language_is_rejected_before_model_load(monkeypatch, tmp_path):
    constructors = []
    install_fake_funasr(monkeypatch, FakeModel(), constructors)
    with pytest.raises(ValueError, match="language"):
        new_handler().setup(hotwords_file=str(hotwords_file(tmp_path)), language="French")
    assert constructors == []
