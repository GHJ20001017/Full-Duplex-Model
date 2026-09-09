from speech_to_speech.api.openai_realtime.wake_word import WakeWordGate


class FakeDetector:
    def __init__(self):
        self.detect = False

    def process(self, _pcm):
        detected, self.detect = self.detect, False
        return detected


def test_gate_requires_wake_word_then_speech(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.api.openai_realtime.wake_word.time.monotonic", lambda: clock[0])
    detector = FakeDetector()
    gate = WakeWordGate(detector, timeout_s=5.0, sample_rate=16000, speech_rms_threshold=10.0)
    silence = b"\x00\x00" * 160
    speech = b"\x64\x00" * 160

    assert gate.process(speech) == []
    detector.detect = True
    assert gate.process(silence) == []
    assert gate.consume_wake() is True
    assert gate.consume_wake() is False
    detector.detect = True
    assert gate.process(silence) == []
    assert gate.process(speech) == [silence, speech]


def test_gate_expires_without_speech(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.api.openai_realtime.wake_word.time.monotonic", lambda: clock[0])
    detector = FakeDetector()
    gate = WakeWordGate(detector, timeout_s=5.0, sample_rate=16000, speech_rms_threshold=10.0)
    detector.detect = True
    assert gate.process(b"\x00\x00" * 160) == []
    clock[0] = 106.0
    assert gate.process(b"\x00\x00" * 160) == []
