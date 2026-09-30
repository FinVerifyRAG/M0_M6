"""Streamlit-safe cached pipeline loader and system status helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DATASET_DIR,
    OLLAMA_HOST,
    OLLAMA_MODEL_NAME,
    PERSIST_DIRECTORY,
    SPARSE_INDEX_PATH,
)
from main import build_pipeline  # noqa: E402
from rag_module.ingestion import DatasetScanner  # noqa: E402


@st.cache_resource(show_spinner="Loading regulatory RAG pipeline…")
def get_cached_pipeline():
    """Load vector store, BM25, hybrid retriever, and generator once per process."""
    return build_pipeline()


def clear_pipeline_cache() -> None:
    """Force reload of indexes and models after ingest."""
    get_cached_pipeline.clear()


def _chroma_count_lightweight() -> int:
    """Read Chroma collection count without loading embedding models."""
    if not PERSIST_DIRECTORY.exists():
        return 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
        collection = client.get_or_create_collection(name="rbi_sebi_guidelines")
        return int(collection.count())
    except Exception:
        return 0


def get_system_status() -> Dict[str, Any]:
    """Status chips for Overview — prefers filesystem / light Chroma read."""
    pdf_count = 0
    if DATASET_DIR.exists():
        pdf_count = len(list(DATASET_DIR.rglob("*.pdf")))

    chroma_count = _chroma_count_lightweight()
    bm25_ok = SPARSE_INDEX_PATH.exists() and SPARSE_INDEX_PATH.stat().st_size > 0

    return {
        "pdf_count": pdf_count,
        "dataset_dir": str(DATASET_DIR),
        "chroma_count": chroma_count,
        "chroma_ok": chroma_count > 0 or PERSIST_DIRECTORY.exists(),
        "bm25_ok": bm25_ok,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL_NAME,
        "sparse_index_path": str(SPARSE_INDEX_PATH),
        "persist_directory": str(PERSIST_DIRECTORY),
    }


def scan_documents(max_docs: Optional[int] = None):
    scanner = DatasetScanner(root_dir=DATASET_DIR)
    return scanner.scan_documents(max_docs=max_docs)


def issuer_filter_dict(issuer: str) -> Optional[Dict[str, Any]]:
    """Map UI issuer choice to Chroma metadata filter."""
    if issuer and issuer != "All":
        return {"issuer": issuer}
    return None


def build_metadata_filter(
    issuer: str = "All",
    year: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compose Chroma where-filter for issuer and/or temporal year."""
    clauses: list = []
    if issuer and issuer != "All":
        clauses.append({"issuer": issuer})
    if year and str(year).strip().isdigit():
        clauses.append({"year": str(year).strip()})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
