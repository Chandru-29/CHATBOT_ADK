"""
schema.py — Reads table and column metadata from MySQL and returns a
formatted text description suitable for injecting into LLM prompts.

The result is cached for SCHEMA_CACHE_TTL seconds (default 10 min) so
repeated calls within a session don't hit the database every time.
"""


# ── MODULE TAG: Database Schema Metadata Extraction ──
from sqlalchemy import text
from cachetools import cached, TTLCache
from cachetools.keys import hashkey

from config.settings import DB_NAME, SCHEMA_CACHE_TTL
from database.engine import engine
from config.logger import get_logger

log = get_logger(__name__)

# TTL cache — keyed by the frozenset of included table names (or None for all)
_schema_cache = TTLCache(maxsize=10, ttl=SCHEMA_CACHE_TTL)


def _schema_key(include_tables=None):
    # Always convert to frozenset so plain sets (returned by retrieve_schemas)
    # don't cause an 'unhashable type: set' error in the cache.
    return hashkey(frozenset(include_tables) if include_tables is not None else None)


@cached(cache=_schema_cache, key=_schema_key)
def get_schema(include_tables=None) -> str:
    """
    Read table + column info from MySQL and return a clean text description
    ready to paste into an LLM prompt.

    Args:
        include_tables: Optional set or frozenset of table names to limit the
                        output. If None, all tables in the database are returned.

    Returns:
        Multi-line string with one block per table listing columns and row count.

    Raises:
        RuntimeError: If the database cannot be reached.
    """
    try:
        with engine.connect() as conn:
            # ── SYSTEM SCHEMA EXTRACTION QUERY ─────────────────────────────────────────────
            col_query = text("""
                SELECT table_name, column_name, data_type, column_key, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = :db_name
                ORDER BY table_name, ordinal_position
            """)
            all_cols = conn.execute(col_query, {"db_name": DB_NAME}).fetchall()

            if not all_cols:
                return "No tables found in database."

            # ── TABLE COLUMNS GROUPING & PARSING ───────────────────────────────────────────
            tables: dict[str, list[dict]] = {}
            for row in all_cols:
                table, col_name, dtype, col_key, is_null, default_val = row
                if include_tables is not None and table not in include_tables:
                    continue
                if table not in tables:
                    tables[table] = []
                tables[table].append({
                    "name":     col_name,
                    "type":     dtype,
                    "pk":       col_key == "PRI",
                    "nullable": is_null == "YES",
                    "default":  default_val,
                })

            # ── FORMATTING TABLE SCHEMAS FOR LLM PROMPT STITCHING ───────────────────────
            schema_lines = []
            for table, cols in tables.items():

                col_lines = []
                for col in cols:
                    parts = [f"  {col['name']} {col['type'].upper()}"]
                    if col["pk"]:
                        parts.append("PRIMARY KEY")
                    if not col["nullable"]:
                        parts.append("NOT NULL")
                    if col["default"] is not None:
                        parts.append(f"DEFAULT {col['default']}")
                    col_lines.append(" ".join(parts))

                try:
                    row_count = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{table}`")
                    ).scalar()
                    count_hint = f"  -- {row_count} rows"
                except Exception:
                    count_hint = ""

                schema_lines.append(
                    f"Table: {table}{count_hint}\n"
                    f"Columns:\n" + "\n".join(col_lines)
                )

            # ── Metadata header: always list ALL tables in the database ────────
            # This ensures the LLM knows the full DB structure even when the
            # schema below is filtered to specific tables by RAG.
            header = ""
            try:
                all_tables_result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :db_name ORDER BY table_name"
                ), {"db_name": DB_NAME}).fetchall()
                all_table_names = [r[0] for r in all_tables_result]
                total_count = len(all_table_names)
                header = (
                    f"Database: {DB_NAME}\n"
                    f"Total tables: {total_count}\n"
                    f"All tables: {', '.join(all_table_names)}\n"
                    f"---\n"
                    f"Detailed schema for relevant tables below:"
                )
            except Exception:
                pass

            if header:
                return header + "\n\n" + "\n\n".join(schema_lines)
            return "\n\n".join(schema_lines)

    except Exception as e:
        log.error(f"Schema extraction failed: {e}")
        raise RuntimeError(f"Cannot read database: {e}")
