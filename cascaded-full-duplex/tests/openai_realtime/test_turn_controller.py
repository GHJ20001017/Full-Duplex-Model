from speech_to_speech.api.openai_realtime.turn_controller import TurnController, TurnDecision


def test_backchannel_does_not_cancel():
    controller = TurnController()
    assert controller.decide("嗯", speech_duration_ms=900) is TurnDecision.BACKCHANNEL
    assert controller.decide("对", speech_duration_ms=900) is TurnDecision.BACKCHANNEL


def test_short_or_empty_transcript_waits():
    controller = TurnController()
    assert controller.decide("", speech_duration_ms=900) is TurnDecision.WAIT
    assert controller.decide("我", speech_duration_ms=900) is TurnDecision.WAIT
    assert controller.decide("天气", speech_duration_ms=100) is TurnDecision.WAIT


def test_explicit_interruption_cancels():
    controller = TurnController()
    assert controller.decide("等一下", speech_duration_ms=150) is TurnDecision.CANCEL
    assert controller.decide("不是这个意思", speech_duration_ms=150) is TurnDecision.CANCEL


def test_meaningful_question_cancels_after_minimum_duration():
    controller = TurnController()
    assert controller.decide("明天杭州天气怎么样", speech_duration_ms=600) is TurnDecision.CANCEL
    assert controller.decide("明天杭州天气怎么样", speech_duration_ms=600, assistant_is_speaking=False) is TurnDecision.NORMAL_TURN


def test_final_non_backchannel_turn_cancels_even_if_partial_was_short():
    controller = TurnController()
    assert controller.decide("不", speech_duration_ms=100, final=True) is TurnDecision.CANCEL
