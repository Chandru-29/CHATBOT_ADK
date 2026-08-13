"""
schema.py — Reads table and column metadata from the WMS HTTP API and returns
a formatted text description suitable for injecting into LLM prompts.
"""

# ── MODULE TAG: Database Schema Metadata Extraction ──
import json
import os
import requests
from cachetools import cached, TTLCache
from cachetools.keys import hashkey

from core.config.settings import SCHEMA_CACHE_TTL, WMS_API_URL, WMS_API_TIMEOUT
from core.config.logger import get_logger

log = get_logger(__name__)

_API_URL: str = WMS_API_URL
_API_TIMEOUT: int = WMS_API_TIMEOUT

_schema_cache = TTLCache(maxsize=10, ttl=SCHEMA_CACHE_TTL)

_WMS_ALL_TABLES = [
    "ITEM", "SKUITEM", "SULOCATION", "LOCATION", "PICKLIST",
    "PICKLISTITEM", "PICKLISTVIEW", "GRN", "FGMODEL",
    "ITEMLOCACNMAP", "FGTRANSACTION", "SUIDACTIVITYLOG",
    "WAREHOUSE", "user",
]


def _schema_key(include_tables=None):
    """Generate a unique lookup key for saving database schema details in cache.

    Args:
        include_tables (Optional[Iterable[str]], optional): List of table names to include. Defaults to None.

    Returns:
        hashkey: A unique key object for cache lookups.
    """
    return hashkey(frozenset(include_tables) if include_tables is not None else None)


def _execute_api(sql: str) -> list[dict]:
    """Send a SQL query to the database web service and return the rows as a list.

    Args:
        sql (str): The SELECT SQL query string to run.

    Returns:
        list[dict]: A list of row dictionaries containing the data.

    Raises:
        RuntimeError: Raised if the web request or response reading fails.
    """
    try:
        resp = requests.post(
            _API_URL,
            json={"query": sql},
            timeout=_API_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("result", "[]")
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        raise RuntimeError(f"WMS API call failed: {e}")


@cached(cache=_schema_cache, key=_schema_key)
def get_schema(include_tables=None) -> str:
    """Fetch table names and column structures from the database to show to the AI.

    Args:
        include_tables (Optional[Iterable[str]], optional): Optional list of table names to filter. Defaults to None.

    Returns:
        str: A clean, formatted text summary of database tables and columns.

    Raises:
        RuntimeError: Raised if reading table structure fails.
    """
    try:
        if include_tables is not None:
            target_tables = [t for t in _WMS_ALL_TABLES if t in include_tables]
        else:
            target_tables = list(_WMS_ALL_TABLES)

        if not target_tables:
            return "No matching WMS tables found."

        clean_tables = [t.strip("[]") for t in target_tables]
        in_clause = ", ".join(f"'{t}'" for t in clean_tables)

        col_sql = f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME IN ({in_clause})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """

        all_cols = _execute_api(col_sql)

        if not all_cols:
            return "No column metadata found for WMS tables."

        tables: dict[str, list[dict]] = {}
        for row in all_cols:
            tbl  = row.get("TABLE_NAME", "")
            col  = row.get("COLUMN_NAME", "")
            dtype = row.get("DATA_TYPE", "").upper()
            nullable = row.get("IS_NULLABLE", "YES") == "YES"

            if tbl not in tables:
                tables[tbl] = []
            tables[tbl].append({
                "name":     col,
                "type":     dtype,
                "nullable": nullable,
            })

        schema_lines = []
        all_table_names = list(tables.keys())

        for table, cols in tables.items():
            col_specs = []
            for c in cols:
                spec = f"{c['name']}:{c['type']}"
                if not c["nullable"]:
                    spec += " NOT NULL"
                col_specs.append(spec)

            display_name = f"[{table}]" if table.lower() == "user" else table
            schema_lines.append(f"Table: {display_name}({', '.join(col_specs)})")

        total_count = len(_WMS_ALL_TABLES)
        header = (
            f"Database: WMS (Warehouse Management System)\n"
            f"Total tables: {total_count}\n"
            f"All tables: {', '.join(_WMS_ALL_TABLES)}\n"
            f"---\n"
            f"Detailed schema for relevant tables below:"
        )

        log.info(f"Schema: loaded {len(tables)} tables via WMS API "
                 f"({', '.join(all_table_names)})")

        return header + "\n\n" + "\n\n".join(schema_lines)

    except Exception as e:
        log.error(f"Schema extraction failed: {e}")
        raise RuntimeError(f"Cannot read WMS schema: {e}")


def clear_schema_cache() -> None:
    """Wipe saved database table structures from memory."""
    _schema_cache.clear()
    log.info("Schema: cache cleared.")
