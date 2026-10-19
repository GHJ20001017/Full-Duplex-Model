"""Regression for the actual robot DSML incident; all providers/tools are fake."""
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseOutputItemDoneEvent, ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk
from tests.test_responses_api_language_model import (
    _make_handler,
    _make_request,
    _make_response,
    _make_stream,
    _make_text_delta_event,
)

DSML = ('<DSML calls>\n<DSML invoke name="robot_action">\n'
        '<DSML parameter name="action" string="true">起立</DSML parameter>\n'
        '</DSML invoke>\n</DSML calls>')
TOOL = {"type": "function", "name": "robot_action", "parameters": {
    "type": "object", "properties": {"action": {"type": "string", "enum": ["起立"]}},
    "required": ["action"], "additionalProperties": False,
}}


def drive(text=DSML, *, parts=None, stream=True, tools=None, choice="auto"):
    handler = _make_handler(stream=stream)
    handler.max_output_tokens = 1024
    message = ResponseOutputMessage(id="msg_test", type="message", role="assistant", status="completed",
                                    content=[ResponseOutputText(type="output_text", text=text, annotations=[])])
    raw = _make_stream([
        *(_make_text_delta_event(p) for p in (parts if parts is not None else [text])),
        ResponseOutputItemDoneEvent(type="response.output_item.done", item=message,
                                    output_index=0, sequence_number=100),
    ]) if stream else _make_response([message])
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: raw))
    req = _make_request("站起来")
    req.runtime_config.session.tools = [TOOL] if tools is None else tools
    req.runtime_config.session.tool_choice = choice
    result = list(handler.process(req))
    return result, req.runtime_config.chat, raw


@pytest.mark.parametrize("split", range(len(DSML) + 1))
def test_incident_every_stream_split(split):
    outputs, chat, raw = drive(parts=[DSML[:split], DSML[split:]])
    chunks = [o for o in outputs if isinstance(o, LLMResponseChunk)]
    assert not any(o.text for o in chunks)
    calls = [t for o in chunks for t in (o.tools or [])]
    assert len(calls) == 1
    assert calls[0].name == "robot_action"
    assert json.loads(calls[0].arguments) == {"action": "起立"}
    assert not next(o for o in outputs if isinstance(o, EndOfResponse)).error
    assert "DSML" not in str(chat.buffer)
    raw.close.assert_called()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", ["ascii", "unicode"])
def test_both_protocols(stream, dialect):
    text = DSML if dialect == "ascii" else DSML.replace("DSML ", "｜DSML｜").replace("calls>", "tool_calls>")
    outputs, _, _ = drive(text, stream=stream)
    assert len([t for o in outputs if isinstance(o, LLMResponseChunk) for t in (o.tools or [])]) == 1
    assert not any(o.text for o in outputs if isinstance(o, LLMResponseChunk))


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("text,options", [
    (DSML[:-1], {}), ("<DSML", {}), (DSML + "extra", {}),
    (DSML.replace("robot_action", "unknown"), {}),
    (DSML.replace("起立", "飞行"), {}),
    (DSML.replace('name="action"', 'name="other"'), {}),
    (DSML, {"tools": []}), (DSML, {"choice": "none"}),
    (DSML, {"choice": {"type": "function", "name": "other"}}),
    ("说明：" + DSML, {}), ('"' + DSML + '"', {}), ("```xml\n" + DSML + "\n```", {}),
])
def test_invalid_or_embedded_protocol_never_spoken_or_executed(stream, text, options):
    outputs, chat, _ = drive(text, stream=stream, parts=list(text), **options)
    chunks = [o for o in outputs if isinstance(o, LLMResponseChunk)]
    assert not any(o.tools for o in chunks)
    assert "DSML" not in "".join(o.text or "" for o in chunks)
    assert next(o for o in outputs if isinstance(o, EndOfResponse)).error
    assert len(chat.buffer) == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("text", ["你好。", "a < b", "<", "<DIV>说明</DIV>", "正在读取。稍后回复。"])
def test_ordinary_text_preserved(stream, text):
    outputs, _, _ = drive(text, stream=stream, parts=list(text))
    assert not next(o for o in outputs if isinstance(o, EndOfResponse)).error
    assert not any(o.tools for o in outputs if isinstance(o, LLMResponseChunk))


@pytest.mark.parametrize("cut", range(2, len(DSML)))
def test_all_truncated_ascii_prefixes_fail_closed(cut):
    outputs, _, _ = drive(DSML[:cut], parts=list(DSML[:cut]))
    assert not any(o.tools for o in outputs if isinstance(o, LLMResponseChunk))
    assert "DSML" not in "".join(o.text or "" for o in outputs if isinstance(o, LLMResponseChunk))
    assert next(o for o in outputs if isinstance(o, EndOfResponse)).error


def test_guard_normal_text_does_not_wait_for_eof():
    from speech_to_speech.LLM.base_openai_compatible_language_model import TextDelta
    from speech_to_speech.LLM.dsml_speech_guard import normalize_speech_events

    def source():
        yield TextDelta(text="你好。")
        pytest.fail("normal text must stream before requesting the next event")

    assert next(normalize_speech_events(source(), {})).text == "你好。"


def test_guard_usage_and_cancellation_during_buffering():
    from speech_to_speech.LLM.base_openai_compatible_language_model import TextDelta, Usage
    from speech_to_speech.LLM.dsml_speech_guard import normalize_speech_events

    stopped = False

    def source():
        nonlocal stopped
        yield TextDelta(text="<DSML")
        stopped = True
        yield Usage(input_tokens=5, output_tokens=6)
        yield TextDelta(text=" calls>")
        pytest.fail("cancelled DSML must stop consuming")

    result = list(normalize_speech_events(source(), {"tools": [TOOL]}, cancelled=lambda: stopped))
    assert len(result) == 1
    assert result[0] == Usage(input_tokens=5, output_tokens=6)


def test_guard_rejects_ambiguous_native_after_dsml_without_releasing_call():
    from openai.types.responses import ResponseFunctionToolCall

    from speech_to_speech.LLM.base_openai_compatible_language_model import TextDelta, ToolCall
    from speech_to_speech.LLM.dsml_speech_guard import normalize_speech_events

    source = iter([TextDelta(text=DSML), ToolCall(item=ResponseFunctionToolCall(
        type="function_call", call_id="native", name="robot_action", arguments='{"action":"起立"}',
    ))])
    emitted = []
    with pytest.raises(ValueError, match="DSML"):
        for item in normalize_speech_events(source, {"tools": [TOOL]}):
            emitted.append(item)
    assert emitted == []


@pytest.mark.parametrize("split", range(len("说明：" + DSML) + 1))
def test_marker_after_prose_never_reaches_text_output(split):
    from speech_to_speech.LLM.base_openai_compatible_language_model import TextDelta
    from speech_to_speech.LLM.dsml_speech_guard import normalize_speech_events

    text = "说明：" + DSML
    emitted = []
    with pytest.raises(ValueError, match="DSML"):
        for event in normalize_speech_events(iter([TextDelta(text=text[:split]), TextDelta(text=text[split:])]), {}):
            emitted.append(event)
    assert "DSML" not in "".join(event.text for event in emitted)
    assert not any("<" in event.text for event in emitted)


@pytest.mark.parametrize("nested", [True, False])
def test_parser_checks_request_schema_and_forced_choice(nested):
    from speech_to_speech.LLM.deepseek_dsml import _parse

    tools = [{"type": "function", "function": TOOL}] if nested else [TOOL]
    name = {"name": "robot_action"}
    choice = {"type": "function", **({"function": name} if nested else name)}
    options = {"tools": tools, "tool_choice": choice}
    assert json.loads(_parse(DSML, options)[0]["function"]["arguments"]) == {"action": "起立"}
    for invalid in [DSML.replace("起立", "飞行"), DSML.replace('name="action"', 'name="oops"'),
                    DSML.replace('string="true">起立', 'string="false">1')]:
        with pytest.raises(ValueError, match="DSML"):
            _parse(invalid, options)


def test_parser_validates_all_calls_before_exposing_any():
    from speech_to_speech.LLM.deepseek_dsml import _parse

    invocation = DSML[DSML.index("<DSML invoke"):DSML.index("</DSML calls>")]
    two = DSML.replace("</DSML calls>", invocation + "</DSML calls>")
    assert len(_parse(two, {"tools": [TOOL]})) == 2
    with pytest.raises(ValueError, match="DSML"):
        _parse(two, {"tools": [TOOL], "parallel_tool_calls": False})
    with pytest.raises(ValueError, match="DSML"):
        _parse(two.replace("起立", "飞行", 1), {"tools": [TOOL]})
