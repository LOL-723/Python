from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Asia/Shanghai"

TIMEZONE_ALIASES: dict[str, str] = {
    "中国": "Asia/Shanghai",
    "中国大陆": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "香港": "Asia/Hong_Kong",
    "台湾": "Asia/Taipei",
    "台北": "Asia/Taipei",
    "日本": "Asia/Tokyo",
    "东京": "Asia/Tokyo",
    "韩国": "Asia/Seoul",
    "首尔": "Asia/Seoul",
    "新加坡": "Asia/Singapore",
    "英国": "Europe/London",
    "伦敦": "Europe/London",
    "法国": "Europe/Paris",
    "巴黎": "Europe/Paris",
    "德国": "Europe/Berlin",
    "柏林": "Europe/Berlin",
    "美国东部": "America/New_York",
    "纽约": "America/New_York",
    "华盛顿": "America/New_York",
    "美国中部": "America/Chicago",
    "芝加哥": "America/Chicago",
    "美国山区": "America/Denver",
    "丹佛": "America/Denver",
    "美国西部": "America/Los_Angeles",
    "洛杉矶": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
    "澳大利亚": "Australia/Sydney",
    "悉尼": "Australia/Sydney",
    "俄罗斯": "Europe/Moscow",
    "莫斯科": "Europe/Moscow",
    "印度": "Asia/Kolkata",
    "新德里": "Asia/Kolkata",
    "迪拜": "Asia/Dubai",
    "utc": "UTC",
    "gmt": "UTC",
}

FIXED_TIMEZONES: dict[str, timezone] = {
    "UTC": UTC,
    "Asia/Shanghai": timezone(timedelta(hours=8), "CST"),
    "Asia/Hong_Kong": timezone(timedelta(hours=8), "HKT"),
    "Asia/Taipei": timezone(timedelta(hours=8), "CST"),
    "Asia/Tokyo": timezone(timedelta(hours=9), "JST"),
    "Asia/Seoul": timezone(timedelta(hours=9), "KST"),
    "Asia/Singapore": timezone(timedelta(hours=8), "SGT"),
    "Asia/Kolkata": timezone(timedelta(hours=5, minutes=30), "IST"),
    "Asia/Dubai": timezone(timedelta(hours=4), "GST"),
}


def get_current_time(
    timezone_name: str | None = None,
    location: str | None = None,
    timezone: str | None = None,
    **_: Any,
) -> dict[str, str]:
    """Get the current time for a region or timezone."""
    resolved_timezone = _resolve_timezone(timezone_name or timezone, location)
    tzinfo = _load_timezone(resolved_timezone)
    now = datetime.now(tzinfo)

    return {
        "location": location or resolved_timezone,
        "timezone": resolved_timezone,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "iso_timestamp": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
    }


def _resolve_timezone(timezone_name: str | None, location: str | None) -> str:
    for value in (timezone_name, location):
        if not value:
            continue

        normalized_value = value.strip()
        if not normalized_value:
            continue

        alias = TIMEZONE_ALIASES.get(normalized_value)
        if alias:
            return alias

        lower_value = normalized_value.lower()
        alias = TIMEZONE_ALIASES.get(lower_value)
        if alias:
            return alias

        if "/" in normalized_value or normalized_value.upper() == "UTC":
            return normalized_value

    return DEFAULT_TIMEZONE


def _load_timezone(timezone_name: str):
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        if timezone_name in FIXED_TIMEZONES:
            return FIXED_TIMEZONES[timezone_name]
        return FIXED_TIMEZONES[DEFAULT_TIMEZONE]


TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_current_time": "获取当前时间，返回包含年月日时分秒的完整时间戳。默认返回北京时间；如果用户询问其他国家、城市或地区时间，传入 location 或 IANA timezone_name。",
}

TOOL_ARGUMENTS: dict[str, dict[str, str]] = {
    "get_current_time": {
        "location": "用户询问的国家、城市或地区名称，例如：北京、日本、纽约、伦敦。默认可省略。",
        "timezone_name": "可选 IANA 时区名，例如：Asia/Shanghai、Asia/Tokyo、America/New_York。",
    },
}

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
}
