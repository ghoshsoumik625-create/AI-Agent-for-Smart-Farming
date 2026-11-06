"""
main.py — FastAPI application entry point.
Run with: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("smart_farming")
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan: warm up vector store on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart Farming AI Agent …")
    try:
        from src.rag.rag_pipeline import get_vector_store
        get_vector_store()
        logger.info("Vector store initialised.")
    except Exception as e:
        logger.warning("Vector store init failed (non-fatal): %s", e)
    yield
    logger.info("Shutting down Smart Farming AI Agent.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Agent for Smart Farming Advice",
    description=(
        "RAG-powered AI assistant delivering real-time agricultural guidance "
        "to small-scale farmers. Powered by IBM Granite (watsonx.ai)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Farmer's question")
    location: Optional[str] = Field(None, description="City name or region")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    commodity: Optional[str] = Field(None, description="Crop/commodity for market price")
    crop: Optional[str] = Field(None, description="Specific crop for details or pest advice")
    preferred_language: Optional[str] = Field(None, description="ISO 639-1 language code")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What crop should I grow this season?",
                "location": "Pune",
                "latitude": 18.52,
                "longitude": 73.86,
                "preferred_language": "hi",
            }
        }


class QueryResponse(BaseModel):
    response: str
    intent: str
    detected_language: str
    tool_data: dict[str, Any] = {}
    latency_ms: float


class WeatherRequest(BaseModel):
    location: str


class MarketRequest(BaseModel):
    commodity: str
    state: Optional[str] = None
    market: Optional[str] = None


class SoilRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main farmer chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "AI Agent for Smart Farming Advice",
        "model": settings.ibm_granite_model_id,
        "version": "1.0.0",
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest):
    """
    Main endpoint — processes a farmer's question and returns an AI-generated answer.
    Supports multi-language input and optional location/commodity context.
    """
    start = time.perf_counter()
    try:
        from src.agent.farming_agent import run_agent

        result = await run_agent(
            user_query=req.query,
            location=req.location,
            lat=req.latitude,
            lon=req.longitude,
            commodity=req.commodity,
            crop=req.crop,
            preferred_language=req.preferred_language,
        )
        latency = round((time.perf_counter() - start) * 1000, 1)
        return QueryResponse(latency_ms=latency, **result)
    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/weather")
async def weather_endpoint(req: WeatherRequest):
    """Get current weather for a location."""
    from src.tools.weather_tool import get_weather
    return await get_weather(req.location)


@app.post("/api/market")
async def market_endpoint(req: MarketRequest):
    """Get current mandi price for a commodity."""
    from src.tools.market_tool import get_mandi_price
    return await get_mandi_price(req.commodity, req.state or "", req.market or "")


@app.post("/api/soil")
async def soil_endpoint(req: SoilRequest):
    """Get soil health data for GPS coordinates."""
    from src.tools.soil_tool import get_soil_data
    return await get_soil_data(req.latitude, req.longitude)


@app.get("/api/crops")
async def list_crops():
    """Return the list of crops in the recommendation database."""
    from src.tools.crop_tool import CROP_DB
    return {
        "crops": [
            {
                "name": name.capitalize(),
                "seasons": data["seasons"],
                "water_requirement": data["water_requirement"],
            }
            for name, data in CROP_DB.items()
        ]
    }


@app.get("/api/crop/{crop_name}")
async def crop_detail(crop_name: str):
    """Get detailed information for a specific crop."""
    from src.tools.crop_tool import get_crop_details
    result = get_crop_details(crop_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/pest/{crop_name}")
async def pest_advice(crop_name: str):
    """Get pest and disease management advice for a crop."""
    from src.tools.crop_tool import get_pest_advice
    result = get_pest_advice(crop_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/ingest")
async def ingest_documents():
    """
    Trigger re-ingestion of the knowledge base documents.
    Useful after adding new PDF/TXT files to data/knowledge_base/.
    """
    try:
        from src.rag.rag_pipeline import ingest_knowledge_base
        count = ingest_knowledge_base()
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found."})


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."},
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.app_log_level.lower(),
    )
