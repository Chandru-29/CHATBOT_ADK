"""
server.py — FastMCP server that executes read-only SQL queries
via the external WMS HTTP API (WMS_API_URL in .env).
"""

# ── MODULE TAG: MCP SQL Executor Subprocess ──
import sys
import os
import json

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.guardrails import is_safe_sql_query

import requests
from cachetools import cached, TTLCache
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv(override=True)

from core.config.settings import WMS_API_URL, WMS_API_TIMEOUT, SQL_CACHE_TTL


mcp = FastMCP("wms-api-mcp-server")

API_URL: str = WMS_API_URL
API_TIMEOUT: int = WMS_API_TIMEOUT

_SQL_CACHE_TTL: int = SQL_CACHE_TTL
_sql_cache: TTLCache = TTLCache(maxsize=50, ttl=_SQL_CACHE_TTL)


def _call_api(sql_query: str) -> str:
    """Post SQL query to WMS HTTP API and convert JSON response into formatted text table.

    Args:
        sql_query (str): SQL SELECT query string to execute.

    Returns:
        str: Formatted plain-text columnar result string or error message.
    """
    try:
        response = requests.post(
            API_URL,
            json={"query": sql_query},
            timeout=API_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return f"Error executing query: Request to WMS API timed out after {API_TIMEOUT}s"
    except requests.exceptions.ConnectionError as e:
        return f"Error executing query: Cannot connect to WMS API — {str(e)}"
    except requests.exceptions.HTTPError as e:
        return f"Error executing query: WMS API returned HTTP {response.status_code} — {str(e)}"
    except Exception as e:
        return f"Error executing query: {str(e)}"

    try:
        data = response.json()
    except Exception:
        return f"Error executing query: Invalid JSON response from WMS API: {response.text[:200]}"

    if "error" in data and data["error"]:
        return f"Error executing query: {data['error']}"

    raw_result = data.get("result", "[]")
    try:
        rows_list: list[dict] = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except (json.JSONDecodeError, TypeError):
        return f"Error executing query: Could not parse API result: {str(raw_result)[:200]}"

    if not rows_list:
        return "No rows returned."

    columns = list(rows_list[0].keys())
    rows = rows_list[:50]

    result_str = f"Columns: {', '.join(columns)}\nRows (up to 100):\n"
    for row in rows:
        vals = [str(row.get(col, "")) for col in columns]
        result_str += f"- {', '.join(vals)}\n"

    if len(rows_list) > 100:
        result_str += "... (and more rows exist)"

    return result_str


@mcp.tool()
def execute_read_only_query(sql_query: str) -> str:
    """Safely execute a read-only SELECT SQL query via the WMS HTTP API.

    Args:
        sql_query (str): SQL SELECT query string to validate and execute.

    Returns:
        str: Query result data table string or guardrail/execution error message.
    """
    is_safe, error_msg = is_safe_sql_query(sql_query)
    if not is_safe:
        return f"Error: {error_msg}"

    cache_key = sql_query.strip().lower()
    if cache_key in _sql_cache:
        return _sql_cache[cache_key]

    result = _call_api(sql_query)

    if not result.startswith("Error"):
        _sql_cache[cache_key] = result

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
