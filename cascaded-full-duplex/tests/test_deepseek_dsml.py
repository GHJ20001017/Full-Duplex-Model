"""DSML fallback exercised through the real handler request/event hooks."""

import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from speech_to_speech.LLM.base_openai_compatible_language_model import TextDelta, ToolCall, Usage
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.deepseek_dsml import normalize_stream
from tests.test_chat_completions_backend import _drive, _FakeStream, _make_handler

TOOLS = {"tools": [{"type": "function", "function": {"name": "spawn_thinking"}}]}
RAW = '  raw <tag> &amp; & "quotes"\n中文  '
VALUES = {"objective": RAW, "n": 5, "flag": True, "nil": None, "arr": [1, "x"], "obj": {"x": 2}}
PARAMS = "\n".join(
    f'<｜DSML｜parameter name="{name}" string="{"true" if isinstance(value, str) else "false"}">'
    f'{value if isinstance(value, str) else json.dumps(value)}</｜DSML｜parameter>'
    for name, value in VALUES.items()
)
DSML = f'<｜DSML｜tool_calls>\n<｜DSML｜invoke name="spawn_thinking">\n{PARAMS}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
NATIVE = {"id": "native", "type": "function", "function": {"name": "spawn_thinking", "arguments": '{"n":5}'}}


def chunk(content=None, **delta):
    return ChatCompletionChunk.model_validate({
        "id": "r", "object": "chat.completion.chunk", "created": 0, "model": "test",
        "choices": [{"index": 0, "delta": {"content": content, **delta}, "finish_reason": None}],
    })


def request(text=DSML, *, stream=True, parts=None, options=None, native=False):
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    if stream:
        response = [chunk(reasoning_content="think"), *(chunk(p) for p in (parts if parts is not None else [text]))]
        if native:
            response.append(chunk(tool_calls=[{"index": 0, **NATIVE}]))
        response.append(ChatCompletionChunk.model_validate({
            "id": "r", "object": "chat.completion.chunk", "created": 0, "model": "test", "choices": [], "usage": usage,
        }))
    else:
        response = ChatCompletion.model_validate({
            "id": "r", "object": "chat.completion", "created": 0, "model": "test", "usage": usage,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": text, "reasoning_content": "think",
                "tool_calls": [NATIVE] if native else None,
            }}],
        })
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler.client = NS(chat=NS(completions=NS(create=lambda **kwargs: response)))
    handler.model_name = "test"
    handler.stream = stream
    handler._extra_body = None
    handler.request_timeout = 10
    return handler._iter_events(handler._request([], TOOLS if options is None else options))


XML_QUESTION = "95服务器的地址或连接方式是什么？"
XML_CALL = f"<tool_call>\n<function=ask_user>\n<parameter=question>\n{XML_QUESTION}\n</parameter>\n</function>\n</tool_call>"
XML_TOOLS = {"tools": [{"type": "function", "function": {
    "name": "ask_user", "parameters": {
        "type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"],
    },
}}]}


@pytest.mark.parametrize("split", range(len(XML_CALL) + 1))
def test_xml_screenshot_every_split(split):
    events = list(request(parts=[XML_CALL[:split], XML_CALL[split:]], options=XML_TOOLS))
    assert not any(isinstance(e, TextDelta) for e in events)
    call, = [e for e in events if isinstance(e, ToolCall)]
    assert call.item.name == "ask_user"
    assert json.loads(call.item.arguments) == {"question": XML_QUESTION}
    assert call.reasoning_content == "think"


@pytest.mark.parametrize("stream", [True, False])
def test_xml_multiple_calls(stream):
    events = list(request(XML_CALL + "\n" + XML_CALL, stream=stream, options=XML_TOOLS))
    assert not any(isinstance(e, TextDelta) for e in events)
    assert len([e for e in events if isinstance(e, ToolCall)]) == 2


@pytest.mark.parametrize("cut", range(len("<tool_"), len(XML_CALL)))
def test_xml_truncation_does_not_leak(cut):
    with pytest.raises(ValueError):
        list(request(parts=list(XML_CALL[:cut]), options=XML_TOOLS))


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("text", [
    XML_CALL + "extra", XML_CALL.replace("ask_user", "unknown"),
    XML_CALL.replace("question>", "unknown>"), XML_CALL.replace("</function>", ""),
    XML_CALL.replace("<parameter=question>", "<parameter=question>x</parameter><parameter=question>"),
    "<tool_call><function=ask_user></function></tool_call>",
    XML_CALL + XML_CALL.replace("ask_user", "unknown"),
])
def test_xml_invalid_response_is_atomic(stream, text):
    events = []
    with pytest.raises(ValueError):
        for event in request(text, stream=stream, options=XML_TOOLS):
            events.append(event)
    assert not any(isinstance(e, (TextDelta, ToolCall)) for e in events)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "other"}}])
def test_xml_tool_choice_guard(stream, choice):
    with pytest.raises(ValueError):
        list(request(XML_CALL, stream=stream, options={**XML_TOOLS, "tool_choice": choice}))


@pytest.mark.parametrize("stream", [True, False])
def test_xml_mixed_native_rejected(stream):
    with pytest.raises(ValueError):
        list(request(XML_CALL, stream=stream, options=XML_TOOLS, native=True))


@pytest.mark.parametrize("text", ["Example: " + XML_CALL, "```xml\n" + XML_CALL + "\n```", "<table>example</table>"])
def test_xml_examples_remain_text(text):
    events = list(request(parts=list(text), options=XML_TOOLS))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == text
    assert not any(isinstance(e, ToolCall) for e in events)


@pytest.mark.parametrize("kind,value,expected", [
    ("string", "123", "123"), ("integer", "123", 123), ("boolean", "true", True),
    ("array", '[1,"x"]', [1, "x"]), ("object", '{"a":2}', {"a": 2}), ("null", "null", None),
])
def test_xml_schema_argument_types(kind, value, expected):
    options = {"tools": [{"type": "function", "function": {
        "name": "ask_user", "parameters": {"properties": {"question": {"type": kind}}},
    }}]}
    events = list(request(XML_CALL.replace(XML_QUESTION, value), options=options))
    call, = [e for e in events if isinstance(e, ToolCall)]
    assert json.loads(call.item.arguments) == {"question": expected}


def test_xml_process_emits_call_without_markup():
    handler = _make_handler()
    handler.client.chat.completions.next_result = _FakeStream([chunk(XML_CALL)])
    text, calls, _, _, end = _drive(handler, tools=[{"type": "function", **XML_TOOLS["tools"][0]["function"]}])
    assert text == ""
    assert len(calls) == 1
    assert calls[0].name == "ask_user"
    assert end is not None and not end.error


@pytest.mark.parametrize("split", range(len(DSML) + 1))
def test_every_split(split):
    events = list(request(parts=[DSML[:split], DSML[split:]]))
    assert not any(isinstance(e, TextDelta) for e in events)
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(calls) == 1
    assert calls[0].item.name == "spawn_thinking"
    assert json.loads(calls[0].item.arguments) == VALUES
    assert calls[0].reasoning_content == "think"
    assert [(e.input_tokens, e.output_tokens) for e in events if isinstance(e, Usage)] == [(11, 7)]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("container", ["tool_calls", "function_calls"])
def test_containers_and_nonstream(stream, container):
    events = list(request(" \n" + DSML.replace("tool_calls", container), stream=stream))
    assert not any(isinstance(e, TextDelta) for e in events)
    call, = [e for e in events if isinstance(e, ToolCall)]
    assert json.loads(call.item.arguments) == VALUES
    assert call.reasoning_content == "think"


@pytest.mark.parametrize("cut", range(2, len(DSML)))
def test_every_truncated_prefix_fails_closed(cut):
    events = []
    with pytest.raises(ValueError, match="DSML"):
        for event in request(parts=list(DSML[:cut])):
            events.append(event)
    assert not any(isinstance(e, (TextDelta, ToolCall)) for e in events)


def test_zero_arguments():
    events = list(request(DSML.replace(PARAMS, "")))
    call, = [e for e in events if isinstance(e, ToolCall)]
    assert json.loads(call.item.arguments) == {}


def test_character_chunks():
    assert len([e for e in request(parts=list(DSML)) if isinstance(e, ToolCall)]) == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("text", ["Hello", "", "  ", "<", " \n<", "<tag>plain</tag>", "Example: " + DSML, '"' + DSML + '"', "```xml\n" + DSML + "\n```"])
def test_plain_text_and_examples(stream, text):
    events = list(request(text, stream=stream))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == text
    assert not any(isinstance(e, ToolCall) for e in events)


@pytest.mark.parametrize("stream", [True, False])
def test_native_calls_unchanged(stream):
    events = list(request("", stream=stream, native=True))
    call, = [e for e in events if isinstance(e, ToolCall)]
    assert json.loads(call.item.arguments) == {"n": 5}
    assert call.reasoning_content == "think"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("text", [
    "<｜DSML", DSML[:-1], DSML + "extra", DSML.replace("spawn_thinking", "unknown"),
    DSML.replace('string="true"', 'string="bad"'), DSML.replace(">5</", ">oops</"),
    DSML.replace('name="n"', 'name="objective"'), '<｜DSML｜tool_calls></｜DSML｜tool_calls>',
    DSML.replace(">5</", ">NaN</"), DSML.replace("</｜DSML｜invoke>", ""),
])
def test_invalid_calls_fail_without_text_or_partial_calls(stream, text):
    events = []
    with pytest.raises(ValueError, match="DSML"):
        for event in request(text, stream=stream):
            events.append(event)
    assert not any(isinstance(e, (TextDelta, ToolCall)) for e in events)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("options", [{}, {**TOOLS, "tool_choice": "none"}, {**TOOLS, "tool_choice": {"type": "function", "function": {"name": "other"}}}])
def test_request_guards(stream, options):
    with pytest.raises(ValueError, match="DSML"):
        list(request(stream=stream, options=options))


@pytest.mark.parametrize("stream", [True, False])
def test_mixed_calls_rejected(stream):
    with pytest.raises(ValueError, match="DSML"):
        list(request(stream=stream, native=True))


def test_multiple_calls_and_invalid_second_call():
    invocation = DSML[DSML.index("<｜DSML｜invoke"):DSML.index("</｜DSML｜tool_calls>")]
    two = DSML.replace("</｜DSML｜tool_calls>", invocation + "</｜DSML｜tool_calls>")
    assert len([e for e in request(two) if isinstance(e, ToolCall)]) == 2
    with pytest.raises(ValueError, match="DSML"):
        list(request(two.replace(invocation, invocation.replace("spawn_thinking", "unknown"), 1)))


def test_request_options_are_not_shared():
    first = request()
    second = request(options={})
    assert any(isinstance(e, ToolCall) for e in first)
    with pytest.raises(ValueError, match="DSML"):
        list(second)


@pytest.mark.parametrize("mode", ["unstarted", "early", "normal", "invalid"])
def test_stream_closes_upstream_once(mode):
    class Stream:
        closed = 0

        def __iter__(self):
            yield chunk(DSML[:-1] if mode == "invalid" else "hello")

        def close(self):
            self.closed += 1

    upstream = Stream()
    stream = normalize_stream(upstream, TOOLS)
    if mode == "early":
        next(stream)
    elif mode == "normal":
        list(stream)
        assert upstream.closed == 1
    elif mode == "invalid":
        with pytest.raises(ValueError, match="DSML"):
            list(stream)
        assert upstream.closed == 1
    stream.close()
    stream.close()
    assert upstream.closed == 1


@pytest.mark.parametrize("valid", [True, False])
def test_process_emits_tool_or_error_without_dsml_text(valid):
    handler = _make_handler()
    handler.client.chat.completions.next_result = _FakeStream([chunk(DSML if valid else DSML[:-1])])
    text, calls, _, _, end = _drive(handler, tools=[{"type": "function", "name": "spawn_thinking"}])
    assert "DSML" not in text
    assert end is not None
    if valid:
        assert text == ""
        assert len(calls) == 1
        assert calls[0].name == "spawn_thinking"
        assert not end.error
    else:
        assert not calls
        assert end.error
