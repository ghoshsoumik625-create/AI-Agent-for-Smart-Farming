"""
crop_tool.py — Crop recommendation engine.

Combines soil pH, weather (temperature + rainfall), and season to recommend
the best crops. Includes pest/disease guidance and farming best practices.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Crop knowledge base (static — augmented by RAG at agent level)
# ---------------------------------------------------------------------------

CROP_DB: dict[str, dict] = {
    "rice": {
        "seasons": ["kharif"],
        "ph_range": (5.5, 7.0),
        "temp_range_c": (20, 38),
        "rainfall_mm": (1000, 2500),
        "water_requirement": "High",
        "duration_days": (90, 150),
        "common_pests": ["Brown Plant Hopper", "Stem Borer", "Leaf Folder"],
        "diseases": ["Blast", "Sheath Blight", "Bacterial Leaf Blight"],
        "fertilizer": "NPK 120:60:60 kg/ha; split urea application.",
        "best_states": ["West Bengal", "Punjab", "Andhra Pradesh", "Tamil Nadu"],
    },
    "wheat": {
        "seasons": ["rabi"],
        "ph_range": (6.0, 7.5),
        "temp_range_c": (10, 25),
        "rainfall_mm": (300, 600),
        "water_requirement": "Medium",
        "duration_days": (110, 140),
        "common_pests": ["Aphids", "Termites", "Armyworm"],
        "diseases": ["Rust", "Powdery Mildew", "Karnal Bunt"],
        "fertilizer": "NPK 120:60:40 kg/ha.",
        "best_states": ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh"],
    },
    "maize": {
        "seasons": ["kharif", "rabi"],
        "ph_range": (5.8, 7.0),
        "temp_range_c": (18, 35),
        "rainfall_mm": (500, 900),
        "water_requirement": "Medium",
        "duration_days": (80, 110),
        "common_pests": ["Fall Armyworm", "Stem Borer", "Aphids"],
        "diseases": ["Northern Leaf Blight", "Common Rust", "Downy Mildew"],
        "fertilizer": "NPK 150:75:75 kg/ha.",
        "best_states": ["Karnataka", "Andhra Pradesh", "Maharashtra"],
    },
    "cotton": {
        "seasons": ["kharif"],
        "ph_range": (6.0, 8.0),
        "temp_range_c": (25, 40),
        "rainfall_mm": (600, 1200),
        "water_requirement": "Medium-High",
        "duration_days": (150, 180),
        "common_pests": ["Bollworm", "Whitefly", "Thrips", "Jassids"],
        "diseases": ["Bacterial Blight", "Alternaria Leaf Spot", "Wilt"],
        "fertilizer": "NPK 100:50:50 kg/ha; micronutrients (Zn, B).",
        "best_states": ["Gujarat", "Maharashtra", "Telangana", "Punjab"],
    },
    "tomato": {
        "seasons": ["kharif", "rabi", "zaid"],
        "ph_range": (6.0, 7.0),
        "temp_range_c": (18, 32),
        "rainfall_mm": (400, 1200),
        "water_requirement": "Medium",
        "duration_days": (60, 90),
        "common_pests": ["Fruit Borer", "Whitefly", "Mites"],
        "diseases": ["Early Blight", "Late Blight", "Fusarium Wilt"],
        "fertilizer": "NPK 100:50:50 kg/ha; calcium and boron spray.",
        "best_states": ["Andhra Pradesh", "Karnataka", "Maharashtra"],
    },
    "potato": {
        "seasons": ["rabi"],
        "ph_range": (5.5, 6.5),
        "temp_range_c": (15, 25),
        "rainfall_mm": (400, 700),
        "water_requirement": "Medium",
        "duration_days": (75, 120),
        "common_pests": ["Potato Tuber Moth", "Aphids", "Colorado Beetle"],
        "diseases": ["Late Blight", "Early Blight", "Common Scab"],
        "fertilizer": "NPK 180:80:100 kg/ha; potassium critical.",
        "best_states": ["Uttar Pradesh", "West Bengal", "Punjab", "Bihar"],
    },
    "soybean": {
        "seasons": ["kharif"],
        "ph_range": (6.0, 7.0),
        "temp_range_c": (20, 32),
        "rainfall_mm": (600, 1000),
        "water_requirement": "Medium",
        "duration_days": (90, 120),
        "common_pests": ["Stem Fly", "Girdle Beetle", "Leaf Eating Caterpillar"],
        "diseases": ["Yellow Mosaic", "Charcoal Rot", "Bacterial Pustule"],
        "fertilizer": "NPK 20:80:40 kg/ha; Rhizobium seed treatment.",
        "best_states": ["Madhya Pradesh", "Maharashtra", "Rajasthan"],
    },
    "mustard": {
        "seasons": ["rabi"],
        "ph_range": (6.0, 8.0),
        "temp_range_c": (10, 25),
        "rainfall_mm": (250, 500),
        "water_requirement": "Low",
        "duration_days": (100, 130),
        "common_pests": ["Aphids", "Painted Bug", "Sawfly"],
        "diseases": ["Alternaria Blight", "White Rust", "Downy Mildew"],
        "fertilizer": "NPK 80:40:40 kg/ha; sulphur 30 kg/ha.",
        "best_states": ["Rajasthan", "Uttar Pradesh", "Haryana", "Madhya Pradesh"],
    },
    "onion": {
        "seasons": ["rabi", "kharif"],
        "ph_range": (6.0, 7.5),
        "temp_range_c": (13, 28),
        "rainfall_mm": (300, 700),
        "water_requirement": "Medium",
        "duration_days": (90, 140),
        "common_pests": ["Thrips", "Onion Fly", "Mites"],
        "diseases": ["Purple Blotch", "Stemphylium Blight", "Basal Rot"],
        "fertilizer": "NPK 110:50:60 kg/ha.",
        "best_states": ["Maharashtra", "Karnataka", "Madhya Pradesh", "Gujarat"],
    },
}

# Seasonal mapping (India)
SEASON_MONTHS: dict[str, list[int]] = {
    "kharif": [6, 7, 8, 9, 10],    # June – October
    "rabi": [11, 12, 1, 2, 3],     # November – March
    "zaid": [3, 4, 5, 6],          # March – June (summer)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_crops(
    ph: float | None = None,
    temperature_c: float | None = None,
    rainfall_mm: float | None = None,
    season: str | None = None,
    state: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return a ranked list of suitable crops given soil/weather conditions.
    Any parameter can be None (ignored in scoring if absent).
    """
    recommendations: list[dict[str, Any]] = []

    for crop_name, crop in CROP_DB.items():
        score = 0
        reasons: list[str] = []

        # pH match
        if ph is not None:
            ph_min, ph_max = crop["ph_range"]
            if ph_min <= ph <= ph_max:
                score += 3
                reasons.append(f"pH {ph} is in optimal range ({ph_min}–{ph_max}).")
            elif abs(ph - ph_min) <= 0.5 or abs(ph - ph_max) <= 0.5:
                score += 1

        # Temperature match
        if temperature_c is not None:
            t_min, t_max = crop["temp_range_c"]
            if t_min <= temperature_c <= t_max:
                score += 3
                reasons.append(f"Temperature {temperature_c}°C is suitable ({t_min}–{t_max}°C).")
            elif abs(temperature_c - t_min) <= 5 or abs(temperature_c - t_max) <= 5:
                score += 1

        # Rainfall match
        if rainfall_mm is not None:
            r_min, r_max = crop["rainfall_mm"]
            if r_min <= rainfall_mm <= r_max:
                score += 2
                reasons.append(f"Rainfall {rainfall_mm} mm is suitable.")

        # Season match
        if season and season.lower() in crop["seasons"]:
            score += 3
            reasons.append(f"Suitable for {season.capitalize()} season.")

        # State preference
        if state and any(state.lower() in s.lower() for s in crop.get("best_states", [])):
            score += 1
            reasons.append(f"Commonly grown in {state}.")

        if score > 0:
            recommendations.append(
                {
                    "crop": crop_name.capitalize(),
                    "score": score,
                    "reasons": reasons,
                    "seasons": crop["seasons"],
                    "water_requirement": crop["water_requirement"],
                    "duration_days": f"{crop['duration_days'][0]}–{crop['duration_days'][1]} days",
                    "fertilizer_guide": crop["fertilizer"],
                }
            )

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:6]  # Return top 6


def get_crop_details(crop_name: str) -> dict[str, Any]:
    """Return detailed information for a specific crop."""
    crop = CROP_DB.get(crop_name.lower())
    if not crop:
        return {"error": f"No data found for crop '{crop_name}'."}
    return {
        "crop": crop_name.capitalize(),
        **crop,
        "ph_range": f"{crop['ph_range'][0]}–{crop['ph_range'][1]}",
        "temp_range_c": f"{crop['temp_range_c'][0]}–{crop['temp_range_c'][1]}°C",
        "rainfall_mm": f"{crop['rainfall_mm'][0]}–{crop['rainfall_mm'][1]} mm",
        "duration_days": f"{crop['duration_days'][0]}–{crop['duration_days'][1]} days",
    }


def get_pest_advice(crop_name: str) -> dict[str, Any]:
    """Return pest and disease management advice for a crop."""
    crop = CROP_DB.get(crop_name.lower())
    if not crop:
        return {"error": f"No pest data found for crop '{crop_name}'."}
    return {
        "crop": crop_name.capitalize(),
        "common_pests": crop["common_pests"],
        "diseases": crop["diseases"],
        "general_ipm_tips": [
            "Use certified disease-free seeds.",
            "Practice crop rotation to break pest cycles.",
            "Apply neem-based biopesticides (Neem oil 2–3%) for early infestations.",
            "Use pheromone traps to monitor and trap adult insects.",
            "Follow state agricultural department spray schedules.",
            "Avoid excessive nitrogen — it increases pest susceptibility.",
            "Consult your nearest KVK (Krishi Vigyan Kendra) for local guidance.",
        ],
    }


def detect_season_from_month(month: int) -> str:
    """Infer the Indian agricultural season from calendar month."""
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return "rabi"


def format_crop_recommendations(crops: list[dict]) -> str:
    """Format crop recommendations as a readable text."""
    if not crops:
        return "No suitable crops found for the given conditions."
    lines = ["Top crop recommendations for your conditions:\n"]
    for i, c in enumerate(crops, 1):
        lines.append(
            f"{i}. {c['crop']} — Season: {', '.join(c['seasons']).capitalize()}, "
            f"Water: {c['water_requirement']}, Duration: {c['duration_days']}"
        )
        if c["reasons"]:
            lines.append(f"   ✓ {c['reasons'][0]}")
    return "\n".join(lines)
