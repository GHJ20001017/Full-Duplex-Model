"""Open-Meteo weather tool for the packaged Realtime audio client."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from speech_to_speech.api.openai_realtime.audio_client import ToolResult

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Shanghai"

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    95: "雷雨",
    96: "雷雨伴有冰雹",
    99: "强雷雨伴有冰雹",
}

TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "查询指定城市指定日期的实时天气预报。用户询问天气、温度、降雨、下雪或是否需要带伞时使用。"
            "调用前先用一句简短口语说明你要去查（用用户当前语言，例如“我看一下明天的天气，稍等”），"
            "随后立即调用工具；拿到结果后再给出最终答复。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如杭州、上海。"},
                "date": {
                    "type": "string",
                    "description": "日期，使用 YYYY-MM-DD，或使用今天、明天、后天。",
                },
            },
            "required": ["city", "date"],
            "additionalProperties": False,
        },
    }
]


def _parse_date(value: str) -> date:
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    normalized = value.strip().lower()
    relative_days = {"今天": 0, "today": 0, "明天": 1, "tomorrow": 1, "后天": 2}
    if normalized in relative_days:
        return today + timedelta(days=relative_days[normalized])
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD, 今天, 明天, or 后天") from exc


async def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Resolve a city and return its daily forecast to the assistant."""

    if name != "get_weather":
        raise ValueError(f"Unknown tool: {name}")
    city = str(arguments.get("city", "")).strip()
    if not city:
        return ToolResult({"error": "city must be a non-empty string"})
    try:
        target_date = _parse_date(str(arguments.get("date", "")).strip())
    except ValueError as exc:
        return ToolResult({"city": city, "error": str(exc)})

    if target_date < date.today() or target_date > date.today() + timedelta(days=16):
        return ToolResult({"city": city, "date": target_date.isoformat(), "error": "天气预报仅支持未来 16 天内"})

    logger.info("get_weather city=%s date=%s", city, target_date)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            geo_response = await client.get(
                GEOCODING_URL,
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            locations = geo_data.get("results") or []
            if not locations:
                return ToolResult({"city": city, "error": "找不到这个城市"})
            location = locations[0]
            forecast_response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": TIMEZONE,
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                },
            )
            forecast_response.raise_for_status()
            daily = forecast_response.json().get("daily") or {}
    except (httpx.HTTPError, httpx.RequestError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Weather provider failed for %s: %s", city, exc)
        return ToolResult({"city": city, "date": target_date.isoformat(), "error": "天气服务暂时不可用"})

    try:
        code = int(daily["weather_code"][0])
        result = {
            "city": location.get("name", city),
            "date": target_date.isoformat(),
            "weather": WEATHER_CODES.get(code, "未知天气"),
            "temperature_max_c": daily["temperature_2m_max"][0],
            "temperature_min_c": daily["temperature_2m_min"][0],
            "precipitation_probability_percent": daily["precipitation_probability_max"][0],
            "source": "Open-Meteo",
        }
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Unexpected weather response for %s: %s", city, exc)
        return ToolResult({"city": city, "date": target_date.isoformat(), "error": "天气服务返回数据不完整"})
    return ToolResult(result)
