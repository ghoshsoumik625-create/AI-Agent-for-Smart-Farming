"""
config.py — Centralised settings loaded from environment variables via Pydantic.
All modules import from here; never read os.environ directly elsewhere.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # IBM watsonx.ai / Granite
    # ------------------------------------------------------------------
    ibm_watsonx_api_key: str = Field(default="", alias="IBM_WATSONX_API_KEY")
    ibm_watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com", alias="IBM_WATSONX_URL"
    )
    ibm_watsonx_project_id: str = Field(default="", alias="IBM_WATSONX_PROJECT_ID")
    ibm_granite_model_id: str = Field(
        default="ibm/granite-13b-instruct-v2", alias="IBM_GRANITE_MODEL_ID"
    )
    ibm_granite_embed_model_id: str = Field(
        default="ibm/slate-125m-english-rtrvr", alias="IBM_GRANITE_EMBED_MODEL_ID"
    )

    # ------------------------------------------------------------------
    # IBM Cloud Object Storage
    # ------------------------------------------------------------------
    ibm_cos_api_key: str = Field(default="", alias="IBM_COS_API_KEY")
    ibm_cos_instance_id: str = Field(default="", alias="IBM_COS_INSTANCE_ID")
    ibm_cos_endpoint: str = Field(
        default="https://s3.us-south.cloud-object-storage.appdomain.cloud",
        alias="IBM_COS_ENDPOINT",
    )
    ibm_cos_bucket_name: str = Field(
        default="smart-farming-knowledge-base", alias="IBM_COS_BUCKET_NAME"
    )

    # ------------------------------------------------------------------
    # IBM Watson Translator
    # ------------------------------------------------------------------
    ibm_translator_api_key: str = Field(default="", alias="IBM_TRANSLATOR_API_KEY")
    ibm_translator_url: str = Field(
        default="https://api.us-south.language-translator.watson.cloud.ibm.com",
        alias="IBM_TRANSLATOR_URL",
    )

    # ------------------------------------------------------------------
    # External APIs
    # ------------------------------------------------------------------
    openweather_api_key: str = Field(default="", alias="OPENWEATHER_API_KEY")
    openweather_base_url: str = Field(
        default="https://api.openweathermap.org/data/2.5",
        alias="OPENWEATHER_BASE_URL",
    )
    agmarknet_api_key: str = Field(default="", alias="AGMARKNET_API_KEY")
    agmarknet_base_url: str = Field(
        default="https://api.data.gov.in/resource", alias="AGMARKNET_BASE_URL"
    )
    data_gov_in_api_key: str = Field(default="", alias="DATA_GOV_IN_API_KEY")
    soilgrids_base_url: str = Field(
        default="https://rest.isric.org/soilgrids/v2.0", alias="SOILGRIDS_BASE_URL"
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    app_secret_key: str = Field(default="dev-secret-key", alias="APP_SECRET_KEY")
    supported_languages: str = Field(
        default="en,hi,te,ta,kn,mr,bn,pa,gu,ml", alias="SUPPORTED_LANGUAGES"
    )
    default_language: str = Field(default="en", alias="DEFAULT_LANGUAGE")

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir: str = Field(default="./logs", alias="LOG_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance (cached after first call)."""
    return Settings()
