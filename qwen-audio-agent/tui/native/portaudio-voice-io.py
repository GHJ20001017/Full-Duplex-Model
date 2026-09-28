#!/usr/bin/env python3
"""Line-delimited JSON bridge for the minimal TUI's PortAudio modes."""

import argparse
import base64
from collections import deque
import json
import queue
import sys
import threading
import time

try:
    import sounddevice as sd
except ImportError:
    sys.stderr.write(
        "缺少 Python sounddevice；请运行 pip install sounddevice\n"
    )
    raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-rate", type=int, default=16000)
    parser.add_argument("--playback-rate", type=int, default=24000)
    parser.add_argument("--mode", choices=("half", "full"), default="half")
    args = parser.parse_args()

    playback_lock = threading.Lock()
    capture_enabled = threading.Event()
    playback = deque()
    started_responses = set()
    events = queue.SimpleQueue()
    last_status = {"message": "", "time": 0.0}

    def write_events():
        while True:
            event = events.get()
            if event is None:
                return
            sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    writer = threading.Thread(target=write_events, daemon=True)
    writer.start()

    def emit(event):
        events.put(event)

    def report_status(status):
        if status:
            message = str(status)
            now = time.monotonic()
            if (message != last_status["message"]
                    or now - last_status["time"] >= 1.0):
                last_status["message"] = message
                last_status["time"] = now
                emit({"type": "error", "message": message})

    def capture_callback(indata, _frames, _time_info, status):
        report_status(status)
        if capture_enabled.is_set():
            emit({
                "type": "audio",
                "audio": base64.b64encode(bytes(indata)).decode("ascii"),
            })

    def playback_callback(outdata, _frames, _time_info, status):
        report_status(status)
        size = len(outdata)
        rendered = bytearray(size)
        cursor = 0
        signals = []
        consumed_responses = set()
        with playback_lock:
            while cursor < size and playback:
                entry = playback[0]
                response_id = entry["response_id"]
                if entry["type"] == "done":
                    # Defer completion until the callback after the final
                    # samples were submitted to PortAudio. This avoids
                    # reporting playback completion while the final block is
                    # still being rendered.
                    if response_id in consumed_responses:
                        break
                    playback.popleft()
                    started_responses.discard(response_id)
                    signals.append({
                        "type": "playback.ended",
                        "responseId": response_id,
                    })
                    continue

                if response_id and response_id not in started_responses:
                    started_responses.add(response_id)
                    signals.append({
                        "type": "playback.started",
                        "responseId": response_id,
                    })
                data = entry["data"]
                offset = entry["offset"]
                count = min(size - cursor, len(data) - offset)
                rendered[cursor:cursor + count] = data[offset:offset + count]
                cursor += count
                entry["offset"] += count
                if response_id:
                    consumed_responses.add(response_id)
                if entry["offset"] >= len(data):
                    playback.popleft()
        outdata[:] = rendered
        for signal in signals:
            emit(signal)

    input_stream = None
    output_stream = None

    def close_stream(stream):
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def ensure_input_stream():
        nonlocal input_stream
        if input_stream is not None:
            return
        input_stream = sd.RawInputStream(
            samplerate=args.capture_rate,
            channels=1,
            dtype="int16",
            blocksize=max(160, args.capture_rate // 50),
            callback=capture_callback,
        )
        input_stream.start()

    def ensure_output_stream():
        nonlocal output_stream
        if output_stream is not None:
            return
        output_stream = sd.RawOutputStream(
            samplerate=args.playback_rate,
            channels=1,
            dtype="int16",
            blocksize=max(240, args.playback_rate // 50),
            callback=playback_callback,
        )
        output_stream.start()

    def use_capture_stream(enabled):
        nonlocal input_stream, output_stream
        if enabled:
            capture_enabled.set()
        else:
            capture_enabled.clear()
        if args.mode == "full":
            return
        if enabled:
            close_stream(output_stream)
            output_stream = None
            ensure_input_stream()
        else:
            close_stream(input_stream)
            input_stream = None

    def use_playback_stream():
        nonlocal input_stream
        if args.mode == "full":
            return
        capture_enabled.clear()
        close_stream(input_stream)
        input_stream = None
        ensure_output_stream()

    exit_code = 0
    try:
        ensure_input_stream()
        if args.mode == "full":
            ensure_output_stream()
        emit({"type": "ready"})
        for line in sys.stdin:
            try:
                event = json.loads(line)
            except Exception as error:
                emit({"type": "error", "message": str(error)})
                continue
            try:
                event_type = event.get("type")
                if event_type == "capture":
                    use_capture_stream(bool(event.get("enabled")))
                elif event_type == "play":
                    audio = base64.b64decode(event.get("audio", ""))
                    response_id = str(event.get("responseId", ""))
                    if audio:
                        with playback_lock:
                            playback.append({
                                "type": "audio",
                                "response_id": response_id,
                                "data": audio,
                                "offset": 0,
                            })
                        use_playback_stream()
                elif event_type == "done":
                    response_id = str(event.get("responseId", ""))
                    if response_id:
                        with playback_lock:
                            playback.append({
                                "type": "done",
                                "response_id": response_id,
                            })
                elif event_type == "clear":
                    with playback_lock:
                        playback.clear()
                        started_responses.clear()
                elif event_type == "close":
                    break
            except Exception as error:
                sys.stderr.write(f"PortAudio 音频流切换失败：{error}\n")
                exit_code = 1
                break
    except Exception as error:
        sys.stderr.write(f"PortAudio 初始化失败：{error}\n")
        exit_code = 1
    finally:
        try:
            close_stream(input_stream)
        except Exception:
            pass
        try:
            close_stream(output_stream)
        except Exception:
            pass
        events.put(None)
        writer.join(timeout=1)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
