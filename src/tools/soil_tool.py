"""
soil_tool.py — Retrieve soil health data for a given location.

Primary source: ISRIC SoilGrids REST API (free, no API key required).
Falls back to mock data when the network is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Soil property descriptions (human-readable)
# ---------------------------------------------------------------------------
SOIL_PROPERTIES = {
    "phh2o": "Soil pH (water)",
    "nitrogen": "Total Nitrogen (cg/kg)",
    "soc": "Soil Organic Carbon (dg/kg)",
    "clay": "Clay content (g/kg)",
    "sand": "Sand content (g/kg)",
    "silt": "Silt content (g/kg)",
    "bdod": "Bulk Density (cg/cm³)",
    "cec": "Cation Exchange Capacity (mmol(c)/kg)",
}

# Crop-pH suitability reference
PH_CROP_GUIDE: list[dict] = [
    {"ph_min": 5.5, "ph_max": 6.5, "crops": ["rice", "potato", "sweet potato", "blueberry"]},
    {"ph_min": 6.0, "ph_max": 7.0, "crops": ["wheat", "maize", "soybean", "cotton", "sunflower"]},
    {"ph_min": 6.5, "ph_max": 7.5, "crops": ["onion", "garlic", "tomato", "pepper", "carrot"]},
    {"ph_min": 6.0, "ph_max": 8.0, "crops": ["sugarcane", "mustard", "barley"]},
    {"ph_min": 5.0, "ph_max": 6.5, "crops": ["groundnut", "mango", "banana"]},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_soil_data(lat: float, lon: float) -> dict[str, Any]:
    """
    Fetch soil properties for a GPS coordinate from the SoilGrids API.
    Returns a dict with pH, nitrogen, organic carbon, texture, etc.
    """
    url = f"{settings.soilgrids_base_url}/properties/query"
    properties = list(SOIL_PROPERTIES.keys())

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                url,
                params={
                    "lon": lon,
                    "lat": lat,
                    "property": properties,
                    "depth": "0-5cm",
                    "value": "mean",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        layers = data.get("properties", {}).get("layers", [])
        soil: dict[str, Any] = {"latitude": lat, "longitude": lon}

        for layer in layers:
            prop_name = layer.get("name", "")
            unit_measure = layer.get("unit_measure", {})
            depths = layer.get("depths", [])
            if depths:
                mean_val = depths[0].get("values", {}).get("mean")
                if mean_val is not None:
                    # Convert using d_factor if provided
                    d_factor = unit_measure.get("d_factor", 1)
                    soil[prop_name] = round(mean_val / d_factor, 2)

        # Add interpretation
        ph = soil.get("phh2o")
        if ph:
            soil["ph_interpretation"] = _interpret_ph(ph)
            soil["suitable_crops"] = _crops_for_ph(ph)

        nitrogen = soil.get("nitrogen")
        if nitrogen is not None:
            soil["nitrogen_level"] = _interpret_nitrogen(nitrogen)

        soc = soil.get("soc")
        if soc is not None:
            soil["organic_matter_level"] = _interpret_soc(soc)

        return soil

    except httpx.HTTPStatusError as e:
        logger.error("SoilGrids API HTTP error: %s", e)
        return _mock_soil_data(lat, lon, note="SoilGrids API HTTP error — using mock data.")
    except Exception as e:
        logger.error("SoilGrids API error: %s", e)
        return _mock_soil_data(lat, lon, note=f"SoilGrids API error: {e}")


def _mock_soil_data(lat: float, lon: float, note: str = "[DEMO DATA]") -> dict[str, Any]:
    """Return representative mock soil data."""
    return {
        "latitude": lat,
        "longitude": lon,
        "phh2o": 6.8,
        "nitrogen": 1.2,
        "soc": 12.5,
        "clay": 250,
        "sand": 450,
        "silt": 300,
        "bdod": 1.35,
        "ph_interpretation": "Slightly acidic to neutral — good for most crops.",
        "suitable_crops": ["wheat", "maize", "soybean", "cotton", "sunflower", "tomato"],
        "nitrogen_level": "Medium",
        "organic_matter_level": "Moderate",
        "note": note,
    }


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _interpret_ph(ph: float) -> str:
    if ph < 5.5:
        return "Strongly acidic — lime application recommended."
    elif ph < 6.5:
        return "Slightly acidic — suitable for many crops."
    elif ph <= 7.5:
        return "Neutral — optimal for most crops."
    elif ph <= 8.5:
        return "Slightly alkaline — sulphur application may help."
    return "Strongly alkaline — gypsum treatment recommended."


def _crops_for_ph(ph: float) -> list[str]:
    suitable: list[str] = []
    for entry in PH_CROP_GUIDE:
        if entry["ph_min"] <= ph <= entry["ph_max"]:
            suitable.extend(entry["crops"])
    return list(dict.fromkeys(suitable))  # deduplicate, preserve order


def _interpret_nitrogen(n: float) -> str:
    if n < 0.5:
        return "Low — nitrogen fertiliser recommended."
    elif n < 1.5:
        return "Medium — monitor and supplement as needed."
    return "High — reduce nitrogen fertiliser application."


def _interpret_soc(soc: float) -> str:
    if soc < 5:
        return "Low organic matter — add compost or green manure."
    elif soc < 20:
        return "Moderate organic matter — good soil health."
    return "High organic matter — excellent soil fertility."


def format_soil_summary(soil: dict[str, Any]) -> str:
    """Convert soil data dict to a human-readable paragraph."""
    if "error" in soil:
        return f"Could not retrieve soil data: {soil['error']}"
    crops = ", ".join(soil.get("suitable_crops", [])[:5]) or "various crops"
    return (
        f"Soil at ({soil.get('latitude')}, {soil.get('longitude')}): "
        f"pH {soil.get('phh2o', 'N/A')} — {soil.get('ph_interpretation', '')} "
        f"Nitrogen: {soil.get('nitrogen_level', 'N/A')}. "
        f"Organic matter: {soil.get('organic_matter_level', 'N/A')}. "
        f"Recommended crops: {crops}."
    )
