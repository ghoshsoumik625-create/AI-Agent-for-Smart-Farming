"""
tests/test_tools.py — Unit tests for agricultural tool modules.
Run with: pytest tests/ -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Crop tool tests
# ---------------------------------------------------------------------------
def test_crop_recommendations_kharif():
    from src.tools.crop_tool import recommend_crops
    results = recommend_crops(ph=6.5, temperature_c=30, season="kharif")
    assert len(results) > 0
    names = [r["crop"].lower() for r in results]
    # Rice and maize should appear for kharif season
    assert any(c in names for c in ["rice", "maize", "cotton", "soybean"])


def test_crop_recommendations_rabi():
    from src.tools.crop_tool import recommend_crops
    results = recommend_crops(ph=6.8, temperature_c=18, season="rabi")
    assert len(results) > 0
    names = [r["crop"].lower() for r in results]
    assert any(c in names for c in ["wheat", "mustard", "potato"])


def test_crop_detail_found():
    from src.tools.crop_tool import get_crop_details
    detail = get_crop_details("wheat")
    assert "crop" in detail
    assert "error" not in detail
    assert detail["crop"] == "Wheat"


def test_crop_detail_not_found():
    from src.tools.crop_tool import get_crop_details
    result = get_crop_details("unknowncrop123")
    assert "error" in result


def test_pest_advice():
    from src.tools.crop_tool import get_pest_advice
    advice = get_pest_advice("cotton")
    assert "crop" in advice
    assert len(advice["common_pests"]) > 0
    assert len(advice["diseases"]) > 0


def test_season_detection():
    from src.tools.crop_tool import detect_season_from_month
    assert detect_season_from_month(7) == "kharif"
    assert detect_season_from_month(12) == "rabi"
    assert detect_season_from_month(4) == "zaid"


def test_ph_interpretation():
    from src.tools.soil_tool import _interpret_ph
    assert "acidic" in _interpret_ph(5.0).lower()
    assert "neutral" in _interpret_ph(7.0).lower()
    assert "alkaline" in _interpret_ph(8.8).lower()


def test_crops_for_ph():
    from src.tools.soil_tool import _crops_for_ph
    crops = _crops_for_ph(6.5)
    assert len(crops) > 0


# ---------------------------------------------------------------------------
# Market tool tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mandi_price_demo_mode():
    """When API key is absent (demo mode), should return mock data."""
    from src.tools.market_tool import get_mandi_price
    result = await get_mandi_price("tomato")
    assert "commodity" in result
    assert "modal" in result
    assert result["commodity"].lower() == "tomato"


@pytest.mark.asyncio
async def test_mandi_price_unknown_commodity():
    from src.tools.market_tool import get_mandi_price
    result = await get_mandi_price("xyzunknown123")
    # Should return an error or empty gracefully
    assert "commodity" in result


def test_format_price_summary():
    from src.tools.market_tool import format_price_summary
    data = {"commodity": "Wheat", "market": "Pune", "min": 2000, "max": 2400, "modal": 2200, "date": "2025-01-01"}
    summary = format_price_summary(data)
    assert "Wheat" in summary
    assert "2200" in summary


# ---------------------------------------------------------------------------
# Weather tool tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_weather_demo_mode():
    """Without API key, should return mock data."""
    from src.tools.weather_tool import get_weather
    result = await get_weather("Mumbai")
    assert "temperature_c" in result or "error" in result


def test_format_weather_summary():
    from src.tools.weather_tool import format_weather_summary
    data = {
        "location": "Delhi",
        "description": "Sunny",
        "temperature_c": 35,
        "humidity_pct": 40,
        "wind_speed_kmh": 10,
        "rainfall_mm": 0,
    }
    summary = format_weather_summary(data)
    assert "Delhi" in summary
    assert "35" in summary


# ---------------------------------------------------------------------------
# Translator tests
# ---------------------------------------------------------------------------
def test_detect_english():
    from src.tools.translator import detect_language
    lang = detect_language("What is the best crop for this season?")
    assert lang == "en"


def test_translate_response_english_passthrough():
    from src.tools.translator import translate_response
    # English to English — should return as-is
    result = translate_response("Hello farmer", "en")
    assert result == "Hello farmer"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
def test_settings_loaded():
    from src.config import get_settings
    settings = get_settings()
    assert settings.app_port == 8000
    assert "granite" in settings.ibm_granite_model_id.lower()
