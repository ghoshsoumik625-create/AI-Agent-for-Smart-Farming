"""
granite_llm.py — IBM Granite LLM wrapper via ibm-watsonx-ai SDK.

Provides:
  - GraniteLLM   : LangChain-compatible chat/completion wrapper
  - GraniteEmbedder : LangChain Embeddings wrapper using IBM Slate model
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_credentials() -> Credentials:
    return Credentials(
        url=settings.ibm_watsonx_url,
        api_key=settings.ibm_watsonx_api_key,
    )


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Convert LangChain message list to a Granite-compatible prompt string."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"<|system|>\n{msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"<|user|>\n{msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"<|assistant|>\n{msg.content}")
        else:
            parts.append(str(msg.content))
    parts.append("<|assistant|>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class GraniteLLM(BaseChatModel):
    """
    LangChain BaseChatModel backed by IBM Granite via watsonx.ai.
    Compatible with LangChain chains, agents, and RAG pipelines.
    """

    model_id: str = settings.ibm_granite_model_id
    project_id: str = settings.ibm_watsonx_project_id
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    decoding_method: str = "greedy"

    # Internal — excluded from model schema
    _client: Optional[ModelInference] = None

    def _get_client(self) -> ModelInference:
        if self._client is None:
            credentials = _build_credentials()
            decoding = (
                DecodingMethods.GREEDY
                if self.decoding_method == "greedy"
                else DecodingMethods.SAMPLE
            )
            self._client = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
                params={
                    "decoding_method": decoding,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "repetition_penalty": 1.1,
                },
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        logger.debug("Granite prompt (first 200 chars): %s", prompt[:200])
        client = self._get_client()
        response = client.generate_text(prompt=prompt, guardrails=False)
        text = response if isinstance(response, str) else str(response)
        logger.debug("Granite response (first 200 chars): %s", text[:200])
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text.strip()))]
        )

    @property
    def _llm_type(self) -> str:
        return "ibm-granite"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class GraniteEmbedder(Embeddings):
    """
    LangChain Embeddings using IBM Slate (available via watsonx.ai).
    Falls back to sentence-transformers locally if IBM credentials are absent.
    """

    model_id: str = settings.ibm_granite_embed_model_id
    project_id: str = settings.ibm_watsonx_project_id
    _use_local: bool = False
    _local_model: Any = None
    _ibm_client: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not settings.ibm_watsonx_api_key or settings.ibm_watsonx_api_key.startswith(
            "your_"
        ):
            logger.warning(
                "IBM watsonx credentials not set — falling back to local "
                "sentence-transformers (all-MiniLM-L6-v2) for embeddings."
            )
            self._use_local = True
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_ibm_client(self) -> Any:
        if self._ibm_client is None:
            credentials = _build_credentials()
            self._ibm_client = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
            )
        return self._ibm_client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._use_local:
            return self._local_model.encode(texts, convert_to_numpy=True).tolist()
        client = self._get_ibm_client()
        result = client.generate_text_embedding(texts=texts)
        return [r["embedding"] for r in result["results"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_llm(
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> GraniteLLM:
    """Return a configured GraniteLLM instance."""
    return GraniteLLM(max_new_tokens=max_new_tokens, temperature=temperature)


def get_embedder() -> GraniteEmbedder:
    """Return a configured GraniteEmbedder instance."""
    return GraniteEmbedder()
