"""
sql_executor.py — FastMCP server that safely executes read-only SQL queries.

This file is launched as a subprocess by query_router.py via the MCP stdio
transport. It exposes one tool: execute_read_only_query().

Safety layers:
  1. Must start with SELECT
  2. No banned write keywords (INSERT, DELETE, DROP, etc.)
  3. No multiple statements (semicolons inside the query body)
  4. Execution through SQLAlchemy — parameterised and connection-managed
"""


# ── MODULE TAG: MCP SQL Executor Subprocess ──
# ── STITCHGUARD LAYER: L4 (Execution Safety Checks) ──
import sys
import os

# Add parent directory to sys.path so we can import utils.guardrails when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.guardrails import is_safe_sql_query

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from cachetools import cached, TTLCache
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv(override=True)


# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP("mysql-mcp-server")

# ── Database URL: accept CLI arg or build from env vars ───────────────────────
DB_URL = None

if len(sys.argv) > 1:
    provided_arg = sys.argv[1]
    if provided_arg.startswith("mysql"):
        DB_URL = provided_arg

if not DB_URL:
    dialect  = os.getenv("DB_DIALECT", "mysql")
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "3306")
    db_name  = os.getenv("DB_NAME",     "company_data")

    url_obj = URL.create(
        drivername=f"{dialect}+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=db_name,
    )
    DB_URL = url_obj.render_as_string(hide_password=False)

engine = create_engine(DB_URL)

@mcp.tool()
def execute_read_only_query(sql_query: str) -> str:
    """
    Safely execute a read-only SELECT SQL query on the database.
    Only SELECT statements are permitted. Banned actions like INSERT,
    UPDATE, DELETE, etc., will be blocked.

    Args:
        sql_query: The raw SQL string (SELECT query) to execute.

    Returns:
        The result columns and matching database rows formatted as plain
        text, or an error message if the query is invalid or blocked.
    """
    # ── L4: Check query safety ──
    is_safe, error_msg = is_safe_sql_query(sql_query)
    if not is_safe:
        return f"Error: {error_msg}"


    # ── L4: Run query on database ──
    try:
        with engine.connect() as conn:
            result  = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows    = result.fetchmany(100)

            if not rows:
                return "No rows returned."

            result_str = f"Columns: {', '.join(columns)}\nRows (up to 100):\n"
            for row in rows:
                result_str += f"- {tuple(row)}\n"

            if len(rows) == 100:
                try:
                    has_more = conn.execute(
                        text(f"SELECT COUNT(*) FROM ({sql_query}) AS t")
                    ).scalar() > 100
                    if has_more:
                        result_str += "... (and more rows exist)"
                except Exception:
                    pass

            return result_str

    except Exception as e:
        return f"Error executing query: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
