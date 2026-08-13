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
    """Build configuration settings needed to start the database connection helper.

    Returns:
        StdioServerParameters: Configuration object for starting the background database process.
    """
    from api.app_factory import build_server_params
    return build_server_params()


def get_persistent_session() -> None:
    """Old helper function kept so old code doesn't break. Always returns None.

    Returns:
        None: Always None because database connections are managed by a pool now.
    """
    return None


async def execute_sql_with_session(session, sql: str) -> str:
    """Run a read-only SQL query on the database using an active connection.

    Args:
        session: Active connection session to the database helper.
        sql (str): The SELECT SQL query string to run.

    Returns:
        str: The raw text result returned from the database.
    """
    result = await session.call_tool("execute_read_only_query", {"sql_query": sql})
    return result.content[0].text if result.content else ""


async def _execute_sql_subprocess_fallback(sql: str) -> str:
    """Run a SQL query by opening a temporary background process if the main pool is busy.

    Args:
        sql (str): The SELECT SQL query to run.

    Returns:
        str: The result text from the temporary process.
    """
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
    """Run a cached SQL query on the database and format the final answer for the user.

    Args:
        sql (str): The SQL query string to execute.
        intent (str): The category or type of user question.
        question (str): The original question typed by the user.
        agent_name (str): Name of the AI agent handling the question.
        stream (bool): True to stream answer piece by piece, False for a full response.
        hit_source (str, optional): Where the cache hit came from (e.g. "semantic"). Defaults to "semantic".
        timer (Any, optional): Timer used to track execution speed. Defaults to None.

    Returns:
        Tuple[bool, Union[dict, StreamingResponse]]: True/False success flag, and the formatted answer object or stream.
    """
    from mcp_service.mcp_session_pool import mcp_pool

    db_output = None

    if mcp_pool.is_ready:
        try:
            async with mcp_pool.acquire(timeout=4.0) as session:
                db_output = await execute_sql_with_session(session, sql)
        except Exception as e:
            log.warning(f"  MCPPool session failed during cached query ({e}). Falling back to subprocess.")

    if db_output is None:
        try:
            db_output = await _execute_sql_subprocess_fallback(sql)
        except Exception as e:
            log.error(f"  Subprocess fallback failed for cached query: {e}")
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
    session,
    messages: list,
    system_prompt: str,
    question: str,
    intent: str,
    agent_display_name: str,
    cache_key: str,
    stream: bool,
    timer: Any = None,
) -> Any:
    """Run the AI SQL generator loop using an active database connection.

    Args:
        session: Active connection session to the database helper.
        messages (list): List of previous chat messages.
        system_prompt (str): Instructions and rules for the AI model.
        question (str): The question typed by the user.
        intent (str): Category of the user query.
        agent_display_name (str): Display name for the active agent.
        cache_key (str): Unique key used for caching the result.
        stream (bool): True to stream answer piece by piece, False for full response.
        timer (Any, optional): Timer tracking performance. Defaults to None.

    Returns:
        Any: Final answer payload dict or streaming response.
    """
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
    """Run the AI SQL agent by borrowing a database connection from the connection pool.

    Args:
        messages (list): Chat history messages list.
        system_prompt (str): Rules and instructions for the AI.
        question (str): User question text.
        intent (str): Intent category string.
        agent_display_name (str): Display name of the agent.
        cache_key (str): Unique key used to store or fetch from cache.
        stream (bool): True for streaming tokens piece by piece, False otherwise.
        timer (Any, optional): Speed timer object. Defaults to None.

    Returns:
        Any: Execution result dictionary or streaming response.
    """
    from mcp_service.mcp_session_pool import mcp_pool

    if mcp_pool.is_ready:
        try:
            async with mcp_pool.acquire(timeout=4.0) as session:
                return await run_with_session(
                    session, messages, system_prompt, question, intent,
                    agent_display_name, cache_key, stream, timer,
                )
        except RuntimeError as pool_err:
            log.warning(f"  MCPPool exhausted: {pool_err}. Falling back to subprocess.")
        except Exception as e:
            log.warning(f"  MCPPool session error: {e}. Falling back to subprocess.")

    log.warning("  MCPPool not ready — using per-request subprocess spawn (high latency fallback).")
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
