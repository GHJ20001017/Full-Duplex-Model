import asyncio

import pytest

from examples import realtime_weather_tool


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, responses, captured, **_kwargs):
        self.responses = iter(responses)
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.captured.append((url, kwargs))
        return next(self.responses)


def test_weather_returns_forecast(monkeypatch):
    captured = []
    responses = [
        FakeResponse({"results": [{"name": "杭州", "latitude": 30.27, "longitude": 120.15}]}),
        FakeResponse({
            "daily": {
                "weather_code": [61],
                "temperature_2m_max": [31.0],
                "temperature_2m_min": [24.0],
                "precipitation_probability_max": [70],
            }
        }),
    ]
    monkeypatch.setattr(
        realtime_weather_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, captured, **kwargs),
    )

    result = asyncio.run(
        realtime_weather_tool.execute_tool("get_weather", {"city": "杭州", "date": "2026-09-03"})
    )

    assert result.output["weather"] == "小雨"
    assert result.output["temperature_max_c"] == 31.0
    assert captured[0][0] == realtime_weather_tool.GEOCODING_URL
    assert captured[1][0] == realtime_weather_tool.FORECAST_URL


def test_parse_relative_date(monkeypatch):
    monkeypatch.setattr(realtime_weather_tool, "datetime", type("FixedDateTime", (), {
        "now": staticmethod(lambda _tz: __import__("datetime").datetime(2026, 9, 2))
    }))
    assert realtime_weather_tool._parse_date("明天").isoformat() == "2026-09-03"
