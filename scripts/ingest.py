"""
ingest.py — CLI script to build/rebuild the ChromaDB knowledge base.

Usage:
  python scripts/ingest.py                        # ingest ./data/knowledge_base
  python scripts/ingest.py --dir path/to/docs     # custom directory
  python scripts/ingest.py --reset                # wipe & re-ingest
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.rag.rag_pipeline import ingest_knowledge_base, get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument(
        "--dir",
        default="./data/knowledge_base",
        help="Directory containing PDF/TXT/DOCX documents to ingest.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store before ingesting.",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.reset:
        persist_dir = Path(settings.chroma_persist_dir)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            logger.info("Deleted existing vector store at %s", persist_dir)

    logger.info("Starting document ingestion from: %s", args.dir)
    count = ingest_knowledge_base(args.dir)
    logger.info("✅ Done! Ingested %d chunks.", count)

    # Quick verification
    store = get_vector_store()
    total = store._collection.count()
    logger.info("Vector store now contains %d indexed chunks.", total)


if __name__ == "__main__":
    main()
