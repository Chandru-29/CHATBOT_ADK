"""
indexer.py — Embeds WMS table metadata into ChromaDB.
"""

# ── MODULE TAG: Table Schema Vector Indexer ──
from db.chromadb import get_table_schemas_collection
from db.schema import get_schema
from db.aliases import TABLE_ALIASES, COLUMN_ALIASES
from core.config.logger import get_logger

log = get_logger(__name__)


def build_rich_table_doc(table_name: str, schema_str: str) -> str:
    """Combine table name, column definitions, and synonyms into a rich text document for embedding."""
    block_header = f"Table: [{table_name}]" if table_name.lower() == "user" else f"Table: {table_name}"
    cols_text = ""

    for line in schema_str.split("\n\n"):
        if line.startswith(block_header):
            cols_text = line
            break

    tbl_syns = TABLE_ALIASES.get(table_name, [])
    col_syn_parts = []
    for col, syns in COLUMN_ALIASES.get(table_name, {}).items():
        col_syn_parts.append(f"{col}: {', '.join(syns)}")

    doc = (
        f"Table Name: {table_name}\n"
        f"Table Synonyms: {', '.join(tbl_syns)}\n"
        f"{cols_text}\n"
        f"Column Synonyms:\n" + "\n".join(col_syn_parts)
    )
    return doc.strip()


def index_tables(candidate_tables: set, engine, embedder) -> None:
    """Index candidate tables into ChromaDB table_schemas collection."""
    schema_str = get_schema()
    collection = get_table_schemas_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for tbl in candidate_tables:
        doc = build_rich_table_doc(tbl, schema_str)
        vec = embedder.embed(doc)

        ids.append(tbl)
        embeddings.append(vec)
        documents.append(doc)
        metadatas.append({"table_name": tbl})

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        log.info(f"Indexer: Upserted {len(ids)} tables into ChromaDB 'table_schemas'")
