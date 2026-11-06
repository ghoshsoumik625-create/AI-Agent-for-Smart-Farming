"""
rag_pipeline.py — RAG (Retrieval-Augmented Generation) pipeline.

Steps:
  1. Load agricultural documents from the knowledge base directory.
  2. Split them into chunks.
  3. Embed and store in ChromaDB (local persistent vector store).
  4. At query time: retrieve top-k relevant chunks + pass to Granite.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Vector store singleton
# ---------------------------------------------------------------------------
_vector_store: Chroma | None = None


def _get_embedder():
    """Lazy import to avoid circular dependency at module load."""
    from src.agent.granite_llm import get_embedder

    return get_embedder()


def get_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Return (or lazily create) the ChromaDB vector store.
    If the persist directory already contains data and force_rebuild=False,
    the existing index is loaded — no re-embedding required.
    """
    global _vector_store

    if _vector_store is not None and not force_rebuild:
        return _vector_store

    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)

    embedder = _get_embedder()
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    _vector_store = Chroma(
        client=chroma_client,
        collection_name="smart_farming_kb",
        embedding_function=embedder,
    )

    # If collection is empty, ingest the knowledge base automatically
    if _vector_store._collection.count() == 0:
        logger.info("Vector store empty — ingesting knowledge base documents …")
        ingest_knowledge_base()

    logger.info(
        "Vector store ready — %d documents indexed.",
        _vector_store._collection.count(),
    )
    return _vector_store


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def load_documents(knowledge_base_dir: str = "./data/knowledge_base") -> list[Document]:
    """Load PDF, TXT, and DOCX files from the knowledge base directory."""
    kb_path = Path(knowledge_base_dir)
    if not kb_path.exists():
        logger.warning("Knowledge base directory not found: %s", kb_path)
        return []

    documents: list[Document] = []

    # PDF files
    for pdf_file in kb_path.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            logger.debug("Loaded PDF: %s", pdf_file)
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", pdf_file, e)

    # Text files
    for txt_file in kb_path.rglob("*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents.extend(loader.load())
            logger.debug("Loaded TXT: %s", txt_file)
        except Exception as e:
            logger.error("Failed to load TXT %s: %s", txt_file, e)

    # Word documents
    for docx_file in kb_path.rglob("*.docx"):
        try:
            loader = UnstructuredWordDocumentLoader(str(docx_file))
            documents.extend(loader.load())
            logger.debug("Loaded DOCX: %s", docx_file)
        except Exception as e:
            logger.error("Failed to load DOCX %s: %s", docx_file, e)

    logger.info("Loaded %d raw documents from %s", len(documents), kb_path)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks (size=%d, overlap=%d).",
                len(chunks), settings.rag_chunk_size, settings.rag_chunk_overlap)
    return chunks


def ingest_knowledge_base(knowledge_base_dir: str = "./data/knowledge_base") -> int:
    """
    Load, split, embed and store all documents in the knowledge base directory.
    Returns the number of chunks ingested.
    """
    documents = load_documents(knowledge_base_dir)
    if not documents:
        logger.warning("No documents found to ingest.")
        return 0

    chunks = split_documents(documents)
    store = get_vector_store()
    store.add_documents(chunks)
    store.persist()
    logger.info("Ingested %d chunks into the vector store.", len(chunks))
    return len(chunks)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def get_retriever(top_k: int | None = None) -> VectorStoreRetriever:
    """Return a retriever that fetches the top-k most relevant chunks."""
    k = top_k or settings.rag_top_k
    store = get_vector_store()
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def retrieve_context(query: str, top_k: int | None = None) -> str:
    """
    Retrieve relevant documents for a query and return them as a
    formatted context string ready to be injected into an LLM prompt.
    """
    retriever = get_retriever(top_k)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant context found in the knowledge base."

    sections: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(sections)
