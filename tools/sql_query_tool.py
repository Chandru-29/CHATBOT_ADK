"""
sql_query_tool.py — Defines the run_select_query() function stub that Ollama
reads as a tool definition during the agent loop.

This is NOT executed directly. Ollama reads the function signature and
docstring to understand what the tool does, then generates a tool_call
with the correct arguments. The actual SQL execution happens in
mcp_server/sql_executor.py via the MCP protocol.
"""


def run_select_query(sql_query: str) -> str:
    """
    Safely execute a read-only SELECT SQL query on the database.
    Only SELECT statements are allowed. INSERT, UPDATE, DELETE are blocked.

    Args:
        sql_query: The SQL query to execute.
    """
    pass  # Ollama uses this as a tool definition only — not called directly
