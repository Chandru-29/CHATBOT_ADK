"""
tools.py — Defines the run_select_query() function stub that Ollama
reads as a tool definition during the agent loop.
"""

# ── OLLAMA TOOL DEFINITION: Read-Only SQL Selector Stub ──
def run_select_query(sql_query: str) -> str:
    """
    Safely execute a read-only SELECT SQL query on the database.
    Only SELECT statements are allowed. INSERT, UPDATE, DELETE are blocked.

    Args:
        sql_query: The SQL query to execute.
    """
    pass
