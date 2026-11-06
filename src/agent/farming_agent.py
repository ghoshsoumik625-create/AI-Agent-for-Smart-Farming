"""
farming_agent.py — Smart Farming AI Agent.

Orchestrates:
  1. Language detection & translation (input → English)
  2. Intent classification (what is the farmer asking?)
  3. Tool invocation (weather / soil / crop / market)
  4. RAG context retrieval (knowledge base lookup)
  5. IBM Granite LLM response generation
  6. Response translation back to farmer's language
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import get_settings
from src.rag.rag_pipeline import retrieve_context
from src.tools.crop_tool import (
    detect_season_from_month,
    format_crop_recommendations,
    get_crop_details,
    get_pest_advice,
    recommend_crops,
)
from src.tools.market_tool import format_price_summary, get_mandi_price
from src.tools.soil_tool import format_soil_summary, get_soil_data
from src.tools.translator import (
    translate_response,
    translate_to_english,
)
from src.tools.weather_tool import format_weather_summary, get_weather

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert agricultural advisor AI, helping small-scale farmers 
in India make smart, data-driven decisions. You have access to:
- Real-time weather data for any location
- Soil health data (pH, nitrogen, organic matter) via GPS coordinates
- Crop recommendation engine based on soil + weather conditions
- Live mandi (wholesale market) prices for commodities
- A comprehensive agricultural knowledge base (via RAG)

Guidelines:
1. Always be practical, specific, and actionable. Avoid vague advice.
2. Use the retrieved context and tool outputs as primary facts; supplement with your knowledge.
3. Mention the source of data (weather API, soil data, mandi prices) to build trust.
4. If you lack data for a specific region, provide general best practices.
5. Structure responses clearly: short summary first, then details.
6. For pest/disease questions, always recommend consulting the local KVK (Krishi Vigyan Kendra).
7. Be empathetic to the farmer's situation and resource constraints.
8. Responses should be concise — target 200–400 words unless detail is specifically requested.

Context from agricultural knowledge base:
{rag_context}

Current date and season: {current_date} ({current_season} season)
"""

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, list[str]] = {
    "weather": ["weather", "rain", "rainfall", "temperature", "forecast", "climate", "monsoon", "drought"],
    "soil": ["soil", "ph", "nitrogen", "organic", "fertility", "land", "loam", "clay", "sandy"],
    "crop_recommendation": ["best crop", "which crop", "what to grow", "crop recommend", "suitable crop", "this season"],
    "crop_detail": ["how to grow", "cultivation", "sowing", "irrigation", "fertiliz", "harvest"],
    "pest": ["pest", "insect", "disease", "fungus", "blight", "worm", "spray", "pesticide"],
    "market": ["price", "mandi", "market", "rate", "sell", "cost", "today", "quintal"],
    "general": [],
}


def classify_intent(query: str) -> str:
    """Return the primary intent of the farmer's query."""
    query_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Tool orchestration
# ---------------------------------------------------------------------------

async def gather_tool_context(
    intent: str,
    query: str,
    location: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    commodity: str | None = None,
    crop: str | None = None,
) -> dict[str, Any]:
    """Run the relevant tools based on detected intent. Returns a dict of results."""
    results: dict[str, Any] = {}

    if intent == "weather" and location:
        results["weather"] = await get_weather(location)

    elif intent == "soil" and lat is not None and lon is not None:
        results["soil"] = await get_soil_data(lat, lon)

    elif intent == "crop_recommendation":
        # Gather soil + weather together, then recommend
        tasks = {}
        if lat is not None and lon is not None:
            tasks["soil"] = get_soil_data(lat, lon)
        if location:
            tasks["weather"] = get_weather(location)

        if tasks:
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), task_results):
                if not isinstance(result, Exception):
                    results[key] = result

        # Extract values for recommendation
        ph = results.get("soil", {}).get("phh2o")
        temp = results.get("weather", {}).get("temperature_c")
        month = datetime.now().month
        season = detect_season_from_month(month)
        recommendations = recommend_crops(ph=ph, temperature_c=temp, season=season)
        results["crop_recommendations"] = recommendations
        results["season"] = season

    elif intent == "crop_detail" and crop:
        results["crop_detail"] = get_crop_details(crop)

    elif intent == "pest" and crop:
        results["pest_advice"] = get_pest_advice(crop)

    elif intent == "market":
        if commodity:
            results["mandi_price"] = await get_mandi_price(commodity)
        else:
            # Try to extract commodity from query (simple heuristic)
            from src.tools.market_tool import _MOCK_PRICES

            for c in _MOCK_PRICES:
                if c in query.lower():
                    results["mandi_price"] = await get_mandi_price(c)
                    break

    return results


def build_tool_summary(tool_results: dict[str, Any]) -> str:
    """Convert tool result dict to a formatted summary for the LLM prompt."""
    parts: list[str] = []

    if "weather" in tool_results:
        parts.append("=== Weather Data ===\n" + format_weather_summary(tool_results["weather"]))

    if "soil" in tool_results:
        parts.append("=== Soil Data ===\n" + format_soil_summary(tool_results["soil"]))

    if "crop_recommendations" in tool_results:
        parts.append(
            f"=== Crop Recommendations ({tool_results.get('season', 'current').capitalize()} season) ===\n"
            + format_crop_recommendations(tool_results["crop_recommendations"])
        )

    if "crop_detail" in tool_results:
        d = tool_results["crop_detail"]
        parts.append(
            f"=== {d.get('crop', '')} Cultivation Guide ===\n"
            f"Season: {', '.join(d.get('seasons', []))}\n"
            f"pH Range: {d.get('ph_range', '')}\n"
            f"Temperature: {d.get('temp_range_c', '')}\n"
            f"Water Requirement: {d.get('water_requirement', '')}\n"
            f"Duration: {d.get('duration_days', '')}\n"
            f"Fertilizer: {d.get('fertilizer', '')}"
        )

    if "pest_advice" in tool_results:
        p = tool_results["pest_advice"]
        parts.append(
            f"=== {p.get('crop', '')} Pest & Disease Guide ===\n"
            f"Common Pests: {', '.join(p.get('common_pests', []))}\n"
            f"Diseases: {', '.join(p.get('diseases', []))}\n"
            f"IPM Tips: {'; '.join(p.get('general_ipm_tips', [])[:4])}"
        )

    if "mandi_price" in tool_results:
        parts.append("=== Market Price ===\n" + format_price_summary(tool_results["mandi_price"]))

    return "\n\n".join(parts) if parts else "No specific tool data available."


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

async def run_agent(
    user_query: str,
    location: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    commodity: str | None = None,
    crop: str | None = None,
    preferred_language: str | None = None,
) -> dict[str, Any]:
    """
    Process a farmer's query end-to-end.

    Returns:
        {
            "response": str,          # Final answer in farmer's language
            "intent": str,            # Detected intent
            "detected_language": str, # Detected input language code
            "tool_data": dict,        # Raw tool outputs
        }
    """
    # Step 1: Translate to English if needed
    english_query, detected_lang = translate_to_english(user_query)
    output_lang = preferred_language or detected_lang
    logger.info("Query: '%s' | Detected lang: %s | Intent query: '%s'",
                user_query[:80], detected_lang, english_query[:80])

    # Step 2: Classify intent
    intent = classify_intent(english_query)
    logger.info("Intent: %s", intent)

    # Step 3: Gather tool context
    tool_results = await gather_tool_context(
        intent=intent,
        query=english_query,
        location=location,
        lat=lat,
        lon=lon,
        commodity=commodity,
        crop=crop,
    )

    # Step 4: Retrieve RAG context
    rag_context = retrieve_context(english_query)

    # Step 5: Build prompt and call Granite
    tool_summary = build_tool_summary(tool_results)
    current_date = datetime.now().strftime("%B %d, %Y")
    month = datetime.now().month
    current_season = detect_season_from_month(month).capitalize()

    filled_system = SYSTEM_PROMPT.format(
        rag_context=rag_context,
        current_date=current_date,
        current_season=current_season,
    )

    full_user_message = (
        f"Farmer's question: {english_query}\n\n"
        f"Available data from tools:\n{tool_summary}"
    )

    try:
        from src.agent.granite_llm import get_llm

        llm = get_llm()
        messages = [
            SystemMessage(content=filled_system),
            HumanMessage(content=full_user_message),
        ]
        ai_response = llm.invoke(messages)
        english_response = ai_response.content.strip()
    except Exception as e:
        logger.error("Granite LLM call failed: %s", e)
        # Graceful fallback using tool data only
        english_response = (
            f"Based on available data:\n\n{tool_summary}\n\n"
            "Please consult your local Krishi Vigyan Kendra (KVK) for personalised advice."
        )

    # Step 6: Translate response back to farmer's language
    final_response = translate_response(english_response, output_lang)

    return {
        "response": final_response,
        "intent": intent,
        "detected_language": detected_lang,
        "tool_data": tool_results,
    }
