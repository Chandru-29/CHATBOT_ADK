"""
session_manager.py — MCP Persistent Subprocess Session & Query Execution Manager.
"""

# ── MODULE TAG: MCP Session Manager ──
import re
from typing import Optional, Tuple, Union, Any
from fastapi.responses import StreamingResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config.logger import get_logger
from agents.sql_agent import (
    run_sql_agent,
    format_simple_result,
    stream_answer_tokens,
    parse_db_result,
)
from security.guardrails import (
    is_db_error,
    redact_db_output_string,
)
from core.cache.cache_manager import get_api_cache

log = get_logger(__name__)


def make_server_params() -> StdioServerParameters:
    """Build StdioServerParameters for launching MCP sql_executor subprocess."""
    from api.app_factory import build_server_params
    return build_server_params()


def get_persistent_session() -> Optional[ClientSession]:
    """Return active persistent MCP ClientSession or None if not ready."""
    try:
        from api.app_factory import get_mcp_session
        return get_mcp_session()
    except Exception:
        return None


async def execute_sql_with_session(session: ClientSession, sql: str) -> str:
    """Execute a read-only SQL query via the provided ClientSession tool call."""
    result = await session.call_tool("execute_read_only_query", {"sql_query": sql})
    return result.content[0].text if result.content else ""


async def execute_sql_per_request(sql: str) -> str:
    """Open a temporary stdio subprocess to execute a SQL query (fallback path)."""
    server_params = make_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()
            return await execute_sql_with_session(session, sql)


async def execute_and_format_cached_query(
    sql: str,
    intent: str,
    question: str,
    agent_name: str,
    stream: bool,
    hit_source: str = "semantic",
    timer: Any = None,
) -> Tuple[bool, Union[dict, StreamingResponse]]:
    """Execute a cached SQL query on the DB and format the response."""
    db_output = None
    persistent_session = get_persistent_session()

    if persistent_session is not None:
        try:
            db_output = await execute_sql_with_session(persistent_session, sql)
        except Exception as e:
            log.warning(f"  Persistent MCP session failed during cached query execution ({e}). Restarting…")
            try:
                from api.app_factory import restart_mcp
                fresh_session = await restart_mcp()
                db_output = await execute_sql_with_session(fresh_session, sql)
            except Exception as e2:
                log.error(f"  MCP restart also failed: {e2}. Falling back to per-request spawn.")

    if db_output is None:
        try:
            db_output = await execute_sql_per_request(sql)
        except Exception as e:
            log.error(f"  Per-request MCP session failed for cached query: {e}")
            return False, {}

    if is_db_error(db_output):
        log.warning(f"  Cached query execution returned database error: {db_output}")
        return False, {}

    db_output = redact_db_output_string(db_output)

    if stream:
        async def _token_gen():
            async for token in stream_answer_tokens(question, db_output):
                yield token

        sql_flat = " ".join(sql.split())
        safe_sql = sql_flat.encode("ascii", "ignore").decode("ascii")
        headers = {
            "x-agent-name": agent_name,
            "x-sql-query": safe_sql,
            "x-cache-hit": hit_source,
        }
        return True, StreamingResponse(_token_gen(), media_type="text/event-stream", headers=headers)

    final_answer = format_simple_result(db_output, question)
    final_answer = re.sub(r'\n{3,}', '\n\n', final_answer or "")
    columns, rows = parse_db_result(db_output)

    result = {
        "sql":            sql,
        "columns":        columns,
        "rows":           rows,
        "natural_answer": final_answer,
        "error":          None,
        "attempts":       1,
        "agent_name":     agent_name,
        "cache_hit":      hit_source,
    }
    return True, result


async def run_with_session(
    session: ClientSession,
    messages: list,
    system_prompt: str,
    question: str,
    intent: str,
    agent_display_name: str,
    cache_key: str,
    stream: bool,
    timer: Any = None,
) -> Any:
    """Run the SQL agent loop with an open ClientSession."""
    result = await run_sql_agent(
        session=session,
        messages=messages,
        system_prompt=system_prompt,
        question=question,
        agent_name=agent_display_name,
        cache_key=cache_key,
        api_cache=get_api_cache(),
        intent=intent,
        stream=stream,
    )
    return result


async def run_per_request_session(
    messages: list,
    system_prompt: str,
    question: str,
    intent: str,
    agent_display_name: str,
    cache_key: str,
    stream: bool,
    timer: Any = None,
) -> Any:
    """Open a fresh stdio subprocess for this request (fallback path)."""
    server_params = make_server_params()
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()
                return await run_with_session(
                    session, messages, system_prompt, question, intent,
                    agent_display_name, cache_key, stream, timer,
                )
    except Exception as e:
        log.error(f"  Per-request MCP session failed: {e}")
        return {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": None,
            "error":          f"Failed to connect to MCP Server: {str(e)}",
            "attempts":       0,
            "agent_name":     agent_display_name,
        }
