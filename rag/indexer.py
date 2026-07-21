"""
indexer.py — Queries the database columns and builds table representations to index.
"""

from sqlalchemy import text as sa_text
from config.logger import get_logger
from rag.aliases import TABLE_ALIASES

log = get_logger(__name__)


def describe_table(table: str, engine) -> str:
    """
    Construct a text description of a database table (name, columns, aliases)
    to feed into the embedding model.
    """
    parts = [f"Table: {table}"]

    # Get column names
    try:
        with engine.connect() as conn:
            rows = conn.execute(sa_text(
                "SELECT column_name "
                "FROM information_schema.columns "
                "WHERE table_schema = DATABASE() "
                "  AND table_name   = :tbl "
                "ORDER BY ordinal_position"
            ), {"tbl": table}).fetchall()
        col_names = [r[0] for r in rows]
        if col_names:
            parts.append(f"Columns: {' '.join(col_names)}")
    except Exception as e:
        log.warning(f"Indexer: column lookup failed for table '{table}': {e}")

    # Append manual semantic aliases
    aliases = TABLE_ALIASES.get(table, [])
    if aliases:
        parts.append(f"Aliases: {' '.join(aliases)}")

    # Append static description if available
    from rag.aliases import STATIC_DESCRIPTIONS
    desc = STATIC_DESCRIPTIONS.get(table)
    if desc:
        parts.append(f"Description: {desc}")

    return "\n".join(parts)


def index_tables(
    candidate_tables: set,
    engine,
    embedder,
) -> dict[str, list[float]]:
    """
    Generate embedding vectors for each candidate table to build the registry.
    """
    log.info(f"Indexer: indexing tables {sorted(candidate_tables)}...")
    registry: dict[str, list[float]] = {}
    for table in candidate_tables:
        table_text      = describe_table(table, engine)
        registry[table] = embedder.embed(table_text)

    log.info(f"Indexer: successfully indexed {len(registry)} tables")
    return registry


# Backward-compatibility aliases
build_table_text = describe_table
build_registry = index_tables
