"""
market_tool.py — Fetch mandi (market) prices for agricultural commodities.

Primary source: data.gov.in (AgMarkNet API).
Falls back to curated mock data when API keys are absent.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# AgMarkNet resource ID on data.gov.in (Agmarknet daily mandi prices)
AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# ---------------------------------------------------------------------------
# Mock data (representative prices in INR/quintal)
# ---------------------------------------------------------------------------
_MOCK_PRICES: dict[str, dict] = {
    "tomato": {"min": 800, "max": 1500, "modal": 1100, "unit": "INR/quintal"},
    "potato": {"min": 600, "max": 1200, "modal": 900, "unit": "INR/quintal"},
    "onion": {"min": 700, "max": 1800, "modal": 1200, "unit": "INR/quintal"},
    "wheat": {"min": 2015, "max": 2200, "modal": 2100, "unit": "INR/quintal"},
    "rice": {"min": 1800, "max": 2400, "modal": 2100, "unit": "INR/quintal"},
    "cotton": {"min": 5800, "max": 6400, "modal": 6100, "unit": "INR/quintal"},
    "sugarcane": {"min": 280, "max": 320, "modal": 300, "unit": "INR/quintal"},
    "maize": {"min": 1400, "max": 1700, "modal": 1550, "unit": "INR/quintal"},
    "soybean": {"min": 4000, "max": 4400, "modal": 4200, "unit": "INR/quintal"},
    "groundnut": {"min": 4500, "max": 5200, "modal": 4800, "unit": "INR/quintal"},
    "mustard": {"min": 4600, "max": 5000, "modal": 4800, "unit": "INR/quintal"},
    "chilli": {"min": 6000, "max": 12000, "modal": 9000, "unit": "INR/quintal"},
    "turmeric": {"min": 7000, "max": 10000, "modal": 8500, "unit": "INR/quintal"},
    "garlic": {"min": 2000, "max": 5000, "modal": 3500, "unit": "INR/quintal"},
    "ginger": {"min": 3000, "max": 7000, "modal": 5000, "unit": "INR/quintal"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_mandi_price(commodity: str, state: str = "", market: str = "") -> dict[str, Any]:
    """
    Retrieve current mandi (wholesale market) price for a commodity.
    Returns min/max/modal price in INR per quintal.
    """
    api_key = settings.data_gov_in_api_key
    commodity_lower = commodity.lower().strip()

    if not api_key or api_key.startswith("your_"):
        logger.warning("data.gov.in API key not set — returning mock mandi prices.")
        if commodity_lower in _MOCK_PRICES:
            price = _MOCK_PRICES[commodity_lower]
            return {
                "commodity": commodity.capitalize(),
                "state": state or "India",
                "market": market or "Average",
                "date": date.today().isoformat(),
                **price,
                "note": "[DEMO DATA — configure DATA_GOV_IN_API_KEY for live prices]",
            }
        return {
            "commodity": commodity,
            "error": f"Price data for '{commodity}' not available in demo mode.",
        }

    try:
        params: dict[str, Any] = {
            "api-key": api_key,
            "format": "json",
            "filters[commodity]": commodity.capitalize(),
            "limit": 10,
        }
        if state:
            params["filters[state]"] = state
        if market:
            params["filters[market]"] = market

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{settings.agmarknet_base_url}/{AGMARKNET_RESOURCE_ID}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        records = data.get("records", [])
        if not records:
            return {"commodity": commodity, "error": "No price data found."}

        # Aggregate across multiple markets
        prices = [float(r.get("modal_price", 0)) for r in records if r.get("modal_price")]
        latest = records[0]
        return {
            "commodity": latest.get("commodity", commodity),
            "state": latest.get("state", state),
            "market": latest.get("market", market),
            "date": latest.get("arrival_date", date.today().isoformat()),
            "min": float(latest.get("min_price", 0)),
            "max": float(latest.get("max_price", 0)),
            "modal": round(sum(prices) / len(prices), 2) if prices else 0,
            "unit": "INR/quintal",
        }
    except Exception as e:
        logger.error("Mandi price API error: %s", e)
        return {"commodity": commodity, "error": str(e)}


async def get_price_trend(commodity: str, days: int = 7) -> dict[str, Any]:
    """Return a simple price trend summary for a commodity."""
    base_price = _MOCK_PRICES.get(commodity.lower(), {}).get("modal", 1000)
    import random
    random.seed(commodity)
    trend = [
        {
            "day": i + 1,
            "price": round(base_price * (1 + random.uniform(-0.05, 0.05)), 0),
        }
        for i in range(days)
    ]
    return {
        "commodity": commodity,
        "unit": "INR/quintal",
        "trend": trend,
        "note": "[DEMO TREND DATA]",
    }


def format_price_summary(price_data: dict[str, Any]) -> str:
    """Convert a price dict to a human-readable sentence."""
    if "error" in price_data:
        return f"Could not retrieve price for {price_data.get('commodity', 'the commodity')}: {price_data['error']}"
    return (
        f"Today's mandi price for {price_data.get('commodity', 'commodity')} "
        f"in {price_data.get('market', 'India')}: "
        f"Min ₹{price_data.get('min', 'N/A')}, "
        f"Max ₹{price_data.get('max', 'N/A')}, "
        f"Modal ₹{price_data.get('modal', 'N/A')} per quintal "
        f"(as of {price_data.get('date', 'today')})."
    )
