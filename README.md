# 🌾 KisanMitra — AI Agent for Smart Farming Advice

> **Problem Statement #9** — RAG-powered AI agent delivering real-time, localized agricultural guidance to small-scale farmers using **IBM Granite** (watsonx.ai) and IBM Cloud Lite services.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [IBM Services Used](#-ibm-services-used)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration (.env)](#-configuration-env)
- [API Reference](#-api-reference)
- [Knowledge Base](#-knowledge-base)
- [Running Tests](#-running-tests)

---

## 🌱 Overview

KisanMitra ("Farmer's Friend") is an AI-powered agricultural advisor that:

| Feature | Description |
|---|---|
| **Multi-language Input** | Understands Hindi, Telugu, Tamil, Kannada, Marathi, Bengali, Punjabi, Gujarati, Malayalam + English |
| **Weather Forecasts** | Real-time weather via OpenWeatherMap API |
| **Crop Recommendations** | Soil + weather + season-aware crop selection engine |
| **Mandi Market Prices** | Live wholesale prices via data.gov.in / AgMarkNet API |
| **Soil Health Analysis** | GPS-based soil data via ISRIC SoilGrids API |
| **Pest & Disease Control** | Integrated Pest Management (IPM) advice |
| **RAG Knowledge Base** | Retrieves from agricultural guides using ChromaDB + IBM Granite embeddings |
| **IBM Granite LLM** | Response generation via `ibm/granite-13b-instruct-v2` on watsonx.ai |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Farmer (Browser / Mobile)                    │
│                    Any local language input                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Application (main.py)                │
│                    POST /api/query  │  GET /api/crops  │  etc.      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Smart Farming AI Agent                           │
│             (src/agent/farming_agent.py)                            │
│                                                                     │
│  1. Language Detection (langdetect)                                 │
│  2. Translation → English (IBM Watson Translator / Google)          │
│  3. Intent Classification (weather/crop/market/soil/pest)           │
│  4. Tool Invocation (parallel async)                                │
│  5. RAG Context Retrieval (ChromaDB + Granite Embeddings)           │
│  6. Prompt Build + IBM Granite LLM Call (watsonx.ai)                │
│  7. Response Translation → Farmer's Language                        │
└──────────┬────────────┬──────────────┬──────────────┬──────────────┘
           │            │              │              │
           ▼            ▼              ▼              ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
    │ Weather  │ │  Mandi   │ │    Soil       │ │    Crop      │
    │   API    │ │  Price   │ │  (SoilGrids)  │ │ Recommender  │
    │(OWM API) │ │(data.gov)│ │   REST API   │ │  (built-in)  │
    └──────────┘ └──────────┘ └──────────────┘ └──────────────┘

           ┌─────────────────────────────────────────┐
           │         IBM Cloud Services               │
           │  ┌──────────────────────────────────┐   │
           │  │  IBM watsonx.ai (Granite LLM)    │   │
           │  │  ibm/granite-13b-instruct-v2     │   │
           │  │  ibm/slate-125m-english-rtrvr    │   │
           │  └──────────────────────────────────┘   │
           │  ┌──────────────────────────────────┐   │
           │  │  IBM Watson Language Translator  │   │
           │  └──────────────────────────────────┘   │
           │  ┌──────────────────────────────────┐   │
           │  │  IBM Cloud Object Storage (COS)  │   │
           │  │  (optional — knowledge base docs)│   │
           │  └──────────────────────────────────┘   │
           └─────────────────────────────────────────┘
```

---

## 🔵 IBM Services Used

| Service | Purpose | IBM Cloud Lite? |
|---|---|---|
| **IBM watsonx.ai** | Granite LLM inference (`ibm/granite-13b-instruct-v2`) | ✅ Yes (50,000 tokens/month free) |
| **IBM Granite Embeddings** | `ibm/slate-125m-english-rtrvr` — RAG vector embeddings | ✅ Yes |
| **IBM Watson Language Translator** | Translate farmer queries to English; translate responses back | ✅ Yes (1M chars/month free) |
| **IBM Cloud Object Storage** | Store knowledge base documents (optional) | ✅ Yes (25 GB free) |

---

## 📁 Project Structure

```
AI Agent for Smart Farming Advice/
├── main.py                         # FastAPI app entry point
├── requirements.txt                # Python dependencies
├── env.example                     # Environment variable template
│
├── src/
│   ├── config.py                   # Pydantic settings (reads .env)
│   ├── agent/
│   │   ├── farming_agent.py        # Main agent orchestration
│   │   └── granite_llm.py          # IBM Granite LLM + Embedder wrappers
│   ├── rag/
│   │   └── rag_pipeline.py         # ChromaDB vector store + retrieval
│   └── tools/
│       ├── weather_tool.py         # OpenWeatherMap API
│       ├── market_tool.py          # Mandi prices (data.gov.in)
│       ├── soil_tool.py            # Soil data (ISRIC SoilGrids)
│       ├── crop_tool.py            # Crop recommendation engine
│       └── translator.py           # Language detection & translation
│
├── data/
│   ├── knowledge_base/             # Agricultural PDF/TXT documents for RAG
│   │   ├── kharif_crop_guide.txt
│   │   ├── rabi_crop_guide.txt
│   │   ├── soil_fertilizer_guide.txt
│   │   └── market_prices_guide.txt
│   └── chroma_db/                  # Auto-created vector store (gitignored)
│
├── templates/
│   └── index.html                  # Farmer chat UI (Jinja2)
│
├── static/
│   ├── css/style.css               # Application styles
│   └── js/app.js                   # Frontend chat logic
│
├── scripts/
│   └── ingest.py                   # CLI: build/rebuild vector store
│
└── tests/
    └── test_tools.py               # pytest unit tests
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
cd "AI Agent for Smart Farming Advice"
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy env.example .env          # Windows
# cp env.example .env          # Linux/macOS
```

Edit `.env` with your credentials (see [Configuration](#-configuration-env) section).

### 3. Ingest the knowledge base

```bash
python scripts/ingest.py
```

### 4. Run the application

```bash
python main.py
# OR
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## ⚙️ Configuration (.env)

Copy `env.example` to `.env` and fill in your values:

```env
# REQUIRED — IBM watsonx.ai (Granite LLM)
IBM_WATSONX_API_KEY=your_ibm_cloud_api_key_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
IBM_WATSONX_PROJECT_ID=your_watsonx_project_id_here

# OPTIONAL — For live weather data
OPENWEATHER_API_KEY=your_openweathermap_api_key_here

# OPTIONAL — For live mandi prices
DATA_GOV_IN_API_KEY=your_data_gov_in_api_key_here
```

> 💡 **The app works in demo mode with mock data when API keys are absent.**
> Only `IBM_WATSONX_API_KEY` + `IBM_WATSONX_PROJECT_ID` are needed for live LLM responses.

### Getting IBM watsonx.ai Credentials

1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (free Lite account)
2. Create a **watsonx.ai** service instance
3. Create a **Project** in watsonx.ai Studio
4. Generate an **API key** in IBM Cloud → Manage → Access → API Keys
5. Copy the **Project ID** from the project settings page

---

## 📡 API Reference

### `POST /api/query` — Main Agent Endpoint

```json
{
  "query": "टमाटर का आज का मंडी भाव क्या है?",
  "location": "Pune",
  "latitude": 18.52,
  "longitude": 73.86,
  "preferred_language": "hi",
  "commodity": "tomato",
  "crop": null
}
```

**Response:**
```json
{
  "response": "पुणे में आज टमाटर का मंडी भाव: न्यूनतम ₹800, अधिकतम ₹1500, मोडल ₹1100 प्रति क्विंटल।",
  "intent": "market",
  "detected_language": "hi",
  "tool_data": { "mandi_price": { ... } },
  "latency_ms": 1245.3
}
```

### Other Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Farmer chat UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/weather` | Current weather for location |
| `POST` | `/api/market` | Mandi price for commodity |
| `POST` | `/api/soil` | Soil data for GPS coordinates |
| `GET` | `/api/crops` | List all crops in recommendation DB |
| `GET` | `/api/crop/{name}` | Detailed crop information |
| `GET` | `/api/pest/{crop}` | Pest & disease management guide |
| `POST` | `/api/ingest` | Rebuild the RAG vector store |

---

## 📚 Knowledge Base

Add agricultural documents to `data/knowledge_base/` and re-ingest:

```bash
# Add your PDF/TXT/DOCX files to data/knowledge_base/
python scripts/ingest.py --reset
```

Supported formats: `.pdf`, `.txt`, `.docx`

Good sources for agricultural documents:
- [ICAR Publications](https://icar.org.in)
- [State Agriculture Departments](https://agricoop.gov.in)
- [KVK Bulletins](https://kvk.icar.gov.in)
- [FAO Crop Production Guides](https://fao.org)

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 🌐 Supported Languages

Hindi (hi) · Telugu (te) · Tamil (ta) · Kannada (kn) · Marathi (mr) · Bengali (bn) · Punjabi (pa) · Gujarati (gu) · Malayalam (ml) · English (en)

---

## 📞 Farmer Help Resources

- **Kisan Call Centre**: 1800-180-1551 (toll-free)
- **KVK Locator**: [kvk.icar.gov.in](https://kvk.icar.gov.in)
- **PM-KISAN**: [pmkisan.gov.in](https://pmkisan.gov.in)
- **eNAM (Market)**: [enam.gov.in](https://enam.gov.in)
- **AgMarkNet (Prices)**: [agmarknet.gov.in](https://agmarknet.gov.in)

---

*Built with ❤️ for India's farming community using IBM Granite AI and IBM Cloud Lite.*
