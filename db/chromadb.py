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

    fewshot_exemplars_collection = chroma_client.get_or_create_collection(
        name="fewshot_exemplars",
        metadata=_COLLECTION_METADATA
    )

    log.info("ChromaDB: Successfully initialized 'table_schemas', 'semantic_cache', and 'fewshot_exemplars' collections")
except Exception as e:
    log.error(f"ChromaDB: Failed to create collections: {e}")
    raise RuntimeError(f"ChromaDB collection creation error: {e}")


def get_chroma_client() -> chromadb.PersistentClient:
    """Get the active local vector database client.

    Returns:
        chromadb.PersistentClient: The active local vector database client object.
    """
    return chroma_client


def get_table_schemas_collection():
    """Get the vector database folder that stores database table descriptions.

    Returns:
        Collection: The active table schema collection.
    """
    try:
        return chroma_client.get_or_create_collection(
            name="table_schemas",
            metadata=_COLLECTION_METADATA
        )
    except Exception:
        return table_schemas_collection


def get_semantic_cache_collection():
    """Get the vector database folder that stores past answers for semantic caching.

    Returns:
        Collection: The active semantic cache collection.
    """
    try:
        return chroma_client.get_or_create_collection(
            name="semantic_cache",
            metadata=_COLLECTION_METADATA
        )
    except Exception:
        return semantic_cache_collection


def get_fewshot_exemplars_collection():
    """Get the vector database collection that stores few-shot Q/A query exemplars.

    Returns:
        Collection: The active fewshot_exemplars collection.
    """
    try:
        return chroma_client.get_or_create_collection(
            name="fewshot_exemplars",
            metadata=_COLLECTION_METADATA
        )
    except Exception:
        return fewshot_exemplars_collection


def reset_semantic_cache_collection():
    """Wipe and recreate the semantic cache folder in the local vector database.

    Returns:
        Collection: The fresh, empty semantic cache collection object.
    """
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
    """Wipe and recreate the table schema folder in the local vector database.

    Returns:
        Collection: The fresh, empty table schema collection object.
    """
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


def reset_fewshot_exemplars_collection(metadata: dict | None = None):
    """Wipe and recreate the fewshot_exemplars collection in the local vector database.

    Args:
        metadata (dict | None, optional): Custom metadata dict (e.g. SHA-256 hash). Defaults to None.

    Returns:
        Collection: The fresh, empty fewshot_exemplars collection object.
    """
    global fewshot_exemplars_collection
    try:
        chroma_client.delete_collection("fewshot_exemplars")
    except Exception:
        pass
    meta = dict(_COLLECTION_METADATA)
    if metadata:
        meta.update(metadata)
    fewshot_exemplars_collection = chroma_client.get_or_create_collection(
        name="fewshot_exemplars",
        metadata=meta
    )
    return fewshot_exemplars_collection

