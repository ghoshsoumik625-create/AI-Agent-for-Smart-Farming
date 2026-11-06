"""
weather_tool.py — Fetch weather forecasts via OpenWeatherMap API.
Falls back to mock data when the API key is not configured.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Mock data (used when API key is absent)
# ---------------------------------------------------------------------------
_MOCK_WEATHER = {
    "location": "Demo Location",
    "temperature_c": 28,
    "humidity_pct": 65,
    "description": "Partly cloudy",
    "wind_speed_kmh": 12,
    "rainfall_mm": 2,
    "forecast_note": "[DEMO DATA — configure OPENWEATHER_API_KEY for live data]",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_weather(location: str) -> dict[str, Any]:
    """
    Return current weather for a location.
    location: city name or 'lat,lon' string.
    """
    api_key = settings.openweather_api_key
    if not api_key or api_key.startswith("your_"):
        logger.warning("OpenWeatherMap API key not set — returning mock weather data.")
        return {**_MOCK_WEATHER, "location": location}

    try:
        # Determine if location is lat/lon or city name
        if "," in location and all(
            part.replace(".", "").replace("-", "").isdigit()
            for part in location.split(",")
        ):
            lat, lon = location.split(",")
            params = {"lat": lat.strip(), "lon": lon.strip()}
        else:
            params = {"q": location}

        params.update(
            {"appid": api_key, "units": "metric", "lang": "en"}
        )

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.openweather_base_url}/weather", params=params
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "location": data.get("name", location),
            "country": data.get("sys", {}).get("country", ""),
            "temperature_c": round(data["main"]["temp"], 1),
            "feels_like_c": round(data["main"]["feels_like"], 1),
            "humidity_pct": data["main"]["humidity"],
            "description": data["weather"][0]["description"].capitalize(),
            "wind_speed_kmh": round(data["wind"]["speed"] * 3.6, 1),
            "rainfall_mm": data.get("rain", {}).get("1h", 0),
            "visibility_km": round(data.get("visibility", 10000) / 1000, 1),
        }
    except httpx.HTTPStatusError as e:
        logger.error("Weather API HTTP error: %s", e)
        return {"error": f"Weather service error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Weather API error: %s", e)
        return {"error": str(e)}


async def get_weather_forecast(location: str, days: int = 5) -> dict[str, Any]:
    """Return a multi-day weather forecast."""
    api_key = settings.openweather_api_key
    if not api_key or api_key.startswith("your_"):
        return {
            "location": location,
            "forecast": [
                {
                    "day": i + 1,
                    "temp_max_c": 32 - i,
                    "temp_min_c": 22 - i,
                    "description": "Partly cloudy",
                    "rainfall_mm": 2 * i,
                }
                for i in range(days)
            ],
            "note": "[DEMO DATA — configure OPENWEATHER_API_KEY for live data]",
        }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.openweather_base_url}/forecast",
                params={
                    "q": location,
                    "appid": api_key,
                    "units": "metric",
                    "cnt": days * 8,  # 3-hour intervals
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Group by day (take midday reading per day)
        daily: dict[str, dict] = {}
        for entry in data["list"]:
            day = entry["dt_txt"][:10]
            if day not in daily or "12:00:00" in entry["dt_txt"]:
                daily[day] = {
                    "date": day,
                    "temp_max_c": round(entry["main"]["temp_max"], 1),
                    "temp_min_c": round(entry["main"]["temp_min"], 1),
                    "description": entry["weather"][0]["description"].capitalize(),
                    "rainfall_mm": entry.get("rain", {}).get("3h", 0),
                }

        return {"location": data["city"]["name"], "forecast": list(daily.values())[:days]}
    except Exception as e:
        logger.error("Forecast API error: %s", e)
        return {"error": str(e)}


def format_weather_summary(weather: dict[str, Any]) -> str:
    """Convert a weather dict to a human-readable paragraph."""
    if "error" in weather:
        return f"Could not retrieve weather data: {weather['error']}"
    return (
        f"Current weather in {weather.get('location', 'your area')}: "
        f"{weather.get('description', 'N/A')}, "
        f"temperature {weather.get('temperature_c', 'N/A')}°C, "
        f"humidity {weather.get('humidity_pct', 'N/A')}%, "
        f"wind {weather.get('wind_speed_kmh', 'N/A')} km/h, "
        f"rainfall {weather.get('rainfall_mm', 0)} mm."
    )
