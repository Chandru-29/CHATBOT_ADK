"""
chromadb.py — ChromaDB persistent client initialization and collection management.

Initializes a local persistent Chroma DB client storing data in `./chroma_data`
and exposes the two primary vector collections:
  1. `table_schemas`   — Stores database table embedding vectors for RAG table selection.
  2. `semantic_cache`  — Stores query embedding vectors and responses for semantic caching.
"""

# ── MODULE TAG: ChromaDB Vector Client Manager ──
import os
import chromadb
from chromadb.config import Settings

from core.config.settings import PROJECT_ROOT
from core.config.logger import get_logger

log = get_logger(__name__)

CHROMA_DATA_DIR = os.path.join(PROJECT_ROOT, "chroma_data")

try:
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DATA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    log.info(f"ChromaDB: Initialized persistent client at '{CHROMA_DATA_DIR}'")
except Exception as e:
    log.error(f"ChromaDB: Failed to initialize persistent client: {e}")
    raise RuntimeError(f"ChromaDB initialization error: {e}")

_COLLECTION_METADATA = {"hnsw:space": "cosine"}

try:
    table_schemas_collection = chroma_client.get_or_create_collection(
        name="table_schemas",
        metadata=_COLLECTION_METADATA
    )

    semantic_cache_collection = chroma_client.get_or_create_collection(
        name="semantic_cache",
        metadata=_COLLECTION_METADATA
    )

    log.info("ChromaDB: Successfully initialized 'table_schemas' and 'semantic_cache' collections")
except Exception as e:
    log.error(f"ChromaDB: Failed to create collections: {e}")
    raise RuntimeError(f"ChromaDB collection creation error: {e}")


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the active persistent ChromaDB client instance."""
    return chroma_client


def get_table_schemas_collection():
    """Return the 'table_schemas' Chroma collection."""
    return table_schemas_collection


def get_semantic_cache_collection():
    """Return the 'semantic_cache' Chroma collection."""
    return semantic_cache_collection


def reset_semantic_cache_collection():
    """Delete and recreate the 'semantic_cache' collection to handle dimension changes or schema wipes."""
    global semantic_cache_collection
    try:
        chroma_client.delete_collection("semantic_cache")
    except Exception:
        pass
    semantic_cache_collection = chroma_client.get_or_create_collection(
        name="semantic_cache",
        metadata=_COLLECTION_METADATA
    )
    return semantic_cache_collection


def reset_table_schemas_collection():
    """Delete and recreate the 'table_schemas' collection to handle dimension changes."""
    global table_schemas_collection
    try:
        chroma_client.delete_collection("table_schemas")
    except Exception:
        pass
    table_schemas_collection = chroma_client.get_or_create_collection(
        name="table_schemas",
        metadata=_COLLECTION_METADATA
    )
    return table_schemas_collection

