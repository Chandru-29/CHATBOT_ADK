"""
query_router.py — POST /query endpoint.

Pipeline steps:

  Step 1  — Semantic cache check: cosine-similarity lookup skips the entire
             pipeline on paraphrase hits. Falls back to exact-text TTLCache
             when Ollama embeddings are unavailable.

  Step 2  — Exact-text TTLCache check.

  Step 3  — rephrase_and_route(): single LLM call handles rephrasing
             and intent classification. Rule-based regex pre-filter handles
             common greetings for free.

  Step 4  — General chat handled without DB access.

  Step 5  — Vector-RAG table selection → Schema fetch (sequential):
             VectorRAG narrows the candidate tables, then schema is fetched
             for the focused table set.

  Step 6  — Persistent MCP session: get_mcp_session() returns the subprocess
             started at startup. Falls back to per-request spawn if unavailable.

  Streaming: add ?stream=true to get a StreamingResponse of tokens instead
             of a blocking JSON dict.

"""


# ── MODULE TAG: FastAPI Request Router ──
# ── STITCHGUARD LAYER: L1 (Input Gate) & L5 (Output Redaction) Pipeline Hooks ──
import os
import re
import sys
import time
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from cachetools import TTLCache
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import ollama
from config.settings import (
    HR_TABLES, SALES_TABLES, API_CACHE_TTL,
    MODEL_NAME, ROUTER_MODEL_NAME, FORMATTER_MODEL_NAME
)
from config.logger import get_logger
from database.engine import DB_URL, engine
from database.schema import get_schema
from agents.router_agent import rephrase_and_route
from agents.general_chat import handle_general_chat
from prompts.prompt_loader import prompt_loader
from rag.table_selector import TableSelector
from agents.sql_agent import (
    run_sql_agent,
    extract_tables_from_sql,
    format_simple_result,
    stream_answer_tokens,
)
from api.models import ChatRequest
from cache.semantic_cache import SemanticCache
from rag.embedder import TextEmbedder
from config.settings import DEFAULT_EMBED_MODEL
from utils.guardrails import (
    is_safe_prompt,
    redact_pii_from_input,
    redact_db_output_string,
)

log = get_logger(__name__)
router = APIRouter()

# ── Exact-text response cache (secondary fast-path) ───────────────────────────
api_cache: dict = TTLCache(maxsize=100, ttl=API_CACHE_TTL)

# ── Semantic cache (primary fast-path — paraphrase-aware) ────────────────────
_embedder  = TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)
_sem_cache = SemanticCache(embedder=_embedder)

# ── Vector Schema RAG engine ─────────────────────────────────────────────────
_table_selector = TableSelector(use_local_ollama=True)


# ── Timing helper ─────────────────────────────────────────────────────────────
class _Timer:
    """Lightweight wall-clock stopwatch for pipeline step timing."""

    def __init__(self, label: str):
        self.label = label
        self._t0   = time.perf_counter()

    def lap(self, name: str) -> float:
        """Log and return the elapsed ms since the last lap (or start)."""
        now = time.perf_counter()
        ms  = (now - self._t0) * 1000
        log.info(f"  ⏱  [{self.label}] {name}: {ms:.0f} ms")
        self._t0 = now
        return ms

    def total(self, name: str, t_start: float) -> None:
        """Log total elapsed time from t_start."""
        ms = (time.perf_counter() - t_start) * 1000
        log.info(f"  ⏱  [{self.label}] ── TOTAL {name}: {ms:.0f} ms ──")


def _make_server_params() -> StdioServerParameters:
    # ── Load shared parameters from app_factory ──
    from api.app_factory import build_server_params
    return build_server_params()


async def _execute_sql_with_session(session: ClientSession, sql: str) -> str:
    """Execute SQL query using the provided session and return output string."""
    result = await session.call_tool("execute_read_only_query", {"sql_query": sql})
    return result.content[0].text if result.content else ""


async def _execute_sql_per_request(sql: str) -> str:
    """Open a fresh subprocess to execute SQL query (fallback path)."""
    server_params = _make_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()
            return await _execute_sql_with_session(session, sql)


async def _execute_and_format_cached_query(
    sql: str,
    intent: str,
    question: str,
    agent_name: str,
    stream: bool,
    timer: _Timer,
) -> tuple[bool, dict | StreamingResponse]:
    """
    Execute a cached SQL query on the DB and format the response.
    Returns (True, result) if successful, (False, None) if execution fails.
    """
    # ── L4: Run query on database ──
    db_output = None
    persistent_session = _get_persistent_session()

    if persistent_session is not None:
        try:
            db_output = await _execute_sql_with_session(persistent_session, sql)
        except Exception as e:
            log.warning(f"  Persistent MCP session failed during cached query execution ({e}). Restarting…")
            try:
                from api.app_factory import restart_mcp
                fresh_session = await restart_mcp()
                db_output = await _execute_sql_with_session(fresh_session, sql)
            except Exception as e2:
                log.error(f"  MCP restart also failed: {e2}. Falling back to per-request spawn.")

    if db_output is None:
        # ── L4 Fallback: Run temporary database connection ──
        try:
            db_output = await _execute_sql_per_request(sql)
        except Exception as e:
            log.error(f"  Per-request MCP session failed for cached query: {e}")
            return False, {}

    if db_output.startswith("Error") or "Unknown column" in db_output or "Table" in db_output:
        # ── L4: Query execution failed ──
        log.warning(f"  Cached query execution returned database error: {db_output}")
        return False, {}

    # ── L5: Redact sensitive output columns ──
    db_output = redact_db_output_string(db_output)


    # ── RAG: Build schema context for LLM ──
    try:
        # ── RAG: Resolve target tables and schemas ──
        tables = extract_tables_from_sql(sql)
        schema = get_schema(include_tables=set(tables))
    except Exception as e:
        log.warning(f"  Failed to fetch schema for cached query formatting: {e}")
        # ── RAG Fallback: Proceed with empty schema ──
        schema = ""

    agent_cfg = prompt_loader.get_agent_config(intent)
    domain_description = agent_cfg.get("description", "").strip()
    shared_rules = prompt_loader.get_rules()

    formatter_system_prompt = (
        f"{domain_description}\n"
        "Your job is to answer the user's question using the database.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n\n"
        f"Rules:\n{shared_rules}"
    )

    # ── PIPELINE: Format streaming tokens ──
    if stream:
        async def _token_gen():
            async for token in stream_answer_tokens(
                formatter_system_prompt,
                formatter_system_prompt, question, db_output
            ):
                yield token

        # ── PIPELINE: Strip newlines from token headers ──
        sql_flat = " ".join(sql.split())
        safe_sql = sql_flat.encode("ascii", "ignore").decode("ascii")
        headers = {
            "x-agent-name": agent_name,
            "x-sql-query": safe_sql
        }
        return True, StreamingResponse(_token_gen(), media_type="text/event-stream", headers=headers)

    # ── PIPELINE: Format standard JSON response ──
    # ── PIPELINE: Apply LLM answer template ──
    final_answer = format_simple_result(db_output, question)
    formatter_used = "template"

    if final_answer is None:
        formatter_used = "llm"
        try:
            fmt_reply = ollama.chat(
                model=FORMATTER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": formatter_system_prompt},
                    {"role": "user",   "content": question},
                    {
                        "role": "user",
                        "content": (
                            f"The database query returned the following results:\n\n"
                            f"{db_output}\n\n"
                            "Please present this information clearly and naturally to the user."
                        ),
                    },
                ],
                options={"temperature": 0, "num_predict": 1024},
            )
            if isinstance(fmt_reply, dict):
                final_answer = fmt_reply.get("message", {}).get("content", "").strip()
            else:
                final_answer = getattr(fmt_reply.message, "content", "").strip()
        except Exception as e:
            log.error(f"  Formatter LLM failed: {e}")
            return False, {}

    final_answer = re.sub(r'\n{3,}', '\n\n', final_answer or "")

    result = {
        "sql":            sql,
        "columns":        [],
        "rows":           [],
        "natural_answer": final_answer,
        "error":          None,
        "attempts":       1,
        "agent_name":     agent_name,
    }
    return True, result


# ── ROUTER ENDPOINT: Main POST /query Request Pipeline ──
@router.post("/query")
async def handle_query(req: ChatRequest, stream: bool = False):
    """
    Main chat endpoint.

    Accepts a natural-language question and returns:
        sql:            The SQL query that was executed (if any)
        columns:        Column names from the query result
        rows:           Data rows from the query result
        natural_answer: Human-readable answer from the LLM
        error:          Error message if something went wrong
        attempts:       Number of agent loop steps taken
        agent_name:     Display name of the agent that answered

    Query param:
        stream (bool, default False): When True, returns a StreamingResponse
        of newline-delimited token chunks instead of a blocking JSON dict.
    """
    t_request_start = time.perf_counter()
    timer = _Timer("pipeline")

    # ── Guard ────────────────────────────────────────────────────────────────
    is_safe, unsafe_reason = is_safe_prompt(req.user_question)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=unsafe_reason,
        )

    # ── L1: Redact personal data from prompt ──
    req.user_question = redact_pii_from_input(req.user_question)


    log.info(f"━━━ NEW REQUEST: '{req.user_question[:80]}' ━━━")

    # ── Sanitise for cache keys ───────────────────────────────────────────────
    clean_question = re.sub(r'[^\w\s]', '', req.user_question)
    cache_key      = clean_question.strip().lower()

    # ── PIPELINE STEP 1 & 2: CONSOLIDATED CACHE LOOKUPS (EXACT & SEMANTIC) ─────────
    cache_hit = None
    hit_source = None

    if cache_key in api_cache:
        cache_hit = api_cache[cache_key]
        hit_source = "exact"
    else:
        sem_hit = _sem_cache.lookup(req.user_question)
        if sem_hit:
            cache_hit = sem_hit
            hit_source = "semantic"

    if cache_hit:
        if isinstance(cache_hit, dict) and cache_hit.get("sql"):
            # ── CACHE: SQL query matched ──
            log.info(f"  {hit_source.upper()} CACHE HIT (SQL query): '{cache_hit['sql']}'")
            success, result = await _execute_and_format_cached_query(
                sql=cache_hit["sql"],
                intent=cache_hit["intent"],
                question=req.user_question,
                agent_name=cache_hit.get("agent_name", "SQL Agent"),
                stream=stream,
                timer=timer,
            )
            if success:
                timer.lap(f"{hit_source} cache execution and formatting")
                timer.total(f"/query ({hit_source} cache hit)", t_request_start)
                return result
            else:
                log.warning(f"  {hit_source.upper()} CACHE HIT failed execution. Evicting cache entry and falling back to full pipeline.")
                if hit_source == "exact":
                    api_cache.pop(cache_key, None)
                else:
                    _sem_cache.remove(cache_hit)
        else:
            # ── CACHE: Conversational response matched ──
            log.info(f"  {hit_source.upper()} CACHE HIT (non-SQL): '{req.user_question}'")
            timer.total(f"/query ({hit_source} cache hit)", t_request_start)
            return cache_hit

    timer.lap("cache lookup (miss)")

    # ── PIPELINE STEP 3: INTENT ROUTING & REPHRASING (LLM ROUTER CALL) ─────────────
    question, intent = rephrase_and_route(req.user_question, req.chat_history)
    timer.lap(f"rephrase_and_route ({ROUTER_MODEL_NAME})")
    log.info(f"  intent={intent}  question='{question[:80]}'")
    print(f"\n [REPHRASED QUESTION]: '{question}' (Intent: {intent})\n")

    agent_display_name = "Router Agent"

    # ── PIPELINE STEP 4: GENERAL CHAT EXECUTION (NO DATABASE INVOLVEMENT) ──────────
    if intent == "GENERAL":
        reply = handle_general_chat(question)
        timer.lap(f"general_chat ({MODEL_NAME})")
        result = {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": reply,
            "error":          None,
            "attempts":       1,
            "agent_name":     "General Agent",
        }
        cache_entry = {
            "sql":            None,
            "intent":         "GENERAL",
            "natural_answer": reply,
            "agent_name":     "General Agent",
        }
        api_cache[cache_key]  = cache_entry
        _sem_cache.store(req.user_question, cache_entry)
        timer.total("/query (GENERAL, no DB)", t_request_start)
        return result

    # ── Resolve domain table scope ────────────────────────────────────────────
    if intent == "HR_AGENT":
        target_tables = HR_TABLES
    elif intent == "SALES_AGENT":
        target_tables = SALES_TABLES
    else:
        target_tables = None  # CROSS_DOMAIN — all tables

    agent_cfg          = prompt_loader.get_agent_config(intent)
    agent_display_name = agent_cfg.get("display_name", "Cross-Domain Coordinator")
    domain_description = agent_cfg.get("description", "").strip()
    domain_examples    = prompt_loader.get_examples(intent)
    shared_rules       = prompt_loader.get_rules()

    # ── PIPELINE STEP 5: VECTOR-RAG TABLE SCOPE SELECTION & SCHEMA FETCH ───────────
    try:
        if target_tables is not None:
            focused_tables = await asyncio.to_thread(
                _table_selector.select_tables, question, target_tables, engine, None
            )
            timer.lap("VectorRAG table selection")
            log.info(f"  VectorRAG: {set(target_tables)} → {set(focused_tables)}")
            schema = get_schema(include_tables=focused_tables)
            timer.lap("get_schema focused")
        else:
            schema = await asyncio.to_thread(get_schema, None)
            timer.lap("get_schema CROSS_DOMAIN")
    except Exception as e:
        result = {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": None,
            "error":          f"Failed to retrieve database schema: {str(e)}",
            "attempts":       0,
            "agent_name":     agent_display_name,
        }
        return result

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = prompt_loader.get_coder_sql_directive().format(schema_context=schema)
    if domain_examples:
        system_prompt += f"\n\nQuery Examples to Guide Structure:\n{domain_examples}"

    formatter_system_prompt = (
        f"{domain_description}\n"
        "Your job is to answer the user's question using the database.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n\n"
        f"Rules:\n{shared_rules}"
    )

    # ── PIPELINE STEP 6: MCP PERSISTENT SESSION & SQL AGENT REASONING LOOP ───────

    persistent_session = _get_persistent_session()

    if persistent_session is not None:
        try:
            result = await _run_with_session(
                session=persistent_session,
                req=req,
                question=question,
                intent=intent,
                system_prompt=system_prompt,
                formatter_system_prompt=formatter_system_prompt,
                agent_display_name=agent_display_name,
                cache_key=cache_key,
                stream=stream,
                timer=timer,
            )
            timer.total("/query (persistent MCP)", t_request_start)
            return result
        except Exception as e:
            log.warning(f"  Persistent MCP session failed ({e}). Restarting…")
            try:
                from api.app_factory import restart_mcp
                fresh_session = await restart_mcp()
                result = await _run_with_session(
                    session=fresh_session,
                    req=req,
                    question=question,
                    intent=intent,
                    system_prompt=system_prompt,
                    formatter_system_prompt=formatter_system_prompt,
                    agent_display_name=agent_display_name,
                    cache_key=cache_key,
                    stream=stream,
                    timer=timer,
                )
                timer.total("/query (restarted MCP)", t_request_start)
                return result
            except Exception as e2:
                log.error(f"  MCP restart also failed ({e2}). Falling back to per-request spawn.")

    # ── L4 Fallback: Run temporary database connection ──
    result = await _run_per_request_session(
        req=req,
        question=question,
        intent=intent,
        system_prompt=system_prompt,
        formatter_system_prompt=formatter_system_prompt,
        agent_display_name=agent_display_name,
        cache_key=cache_key,
        stream=stream,
        timer=timer,
    )
    timer.total("/query (per-request MCP fallback)", t_request_start)
    return result


def _get_persistent_session() -> ClientSession | None:
    """Return the persistent MCP session, or None if unavailable."""
    try:
        # ── Import app_factory locally to avoid circular import ──
        from api.app_factory import get_mcp_session
        # ── Return persistent session ──
        return get_mcp_session()
    except Exception:
        return None


async def _build_messages(req: ChatRequest, question: str, system_prompt: str) -> list:
    """Construct the Ollama message list for the agent loop."""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.chat_history[-6:]:
        role    = msg.get("role")
        content = msg.get("content")
        if role == "bot":
            role = "assistant"
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


async def _run_with_session(
    session: ClientSession,
    req: ChatRequest,
    question: str,
    intent: str,
    system_prompt: str,
    formatter_system_prompt: str,
    agent_display_name: str,
    cache_key: str,
    stream: bool,
    timer: _Timer,
):
    """Run the agent loop with an already-open ClientSession."""
    messages = await _build_messages(req, question, system_prompt)

    t0 = time.perf_counter()
    result = await run_sql_agent(
        session=session,
        messages=messages,
        system_prompt=system_prompt,
        question=question,
        agent_name=agent_display_name,
        cache_key=cache_key,
        api_cache=api_cache,
        intent=intent,
        formatter_prompt=formatter_system_prompt,
        stream=stream,
    )
    ms = (time.perf_counter() - t0) * 1000
    log.info(f"  ⏱  [pipeline] agent_loop (SQL gen + formatter): {ms:.0f} ms")

    # ── SQL Agent: Extract and cache query ──
    sql_query = None
    if isinstance(result, dict):
        sql_query = result.get("sql")
    elif isinstance(result, StreamingResponse):
        sql_query = result.headers.get("x-sql-query")

    if sql_query:
        cache_entry = {
            "sql":        sql_query,
            "intent":     intent,
            "agent_name": agent_display_name,
            "question":   question
        }
        _sem_cache.store(req.user_question, cache_entry)
        api_cache[cache_key] = cache_entry
    elif isinstance(result, dict):
        cache_entry = {
            "sql":            None,
            "intent":         intent,
            "natural_answer": result.get("natural_answer"),
            "agent_name":     agent_display_name
        }
        _sem_cache.store(req.user_question, cache_entry)
        api_cache[cache_key] = cache_entry

    return result


async def _run_per_request_session(
    req: ChatRequest,
    question: str,
    intent: str,
    system_prompt: str,
    formatter_system_prompt: str,
    agent_display_name: str,
    cache_key: str,
    stream: bool,
    timer: _Timer,
):
    """Open a fresh MCP subprocess for this request (fallback path)."""
    server_params = _make_server_params()
    t0 = time.perf_counter()
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()
                ms = (time.perf_counter() - t0) * 1000
                log.info(f"  ⏱  [pipeline] MCP subprocess spawn + init: {ms:.0f} ms")
                return await _run_with_session(
                    session, req, question, intent,
                    system_prompt, formatter_system_prompt,
                    agent_display_name, cache_key, stream, timer,
                )
    except Exception as e:
        log.error(f"  Per-request MCP session failed: {e}")
        result = {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": None,
            "error":          f"Failed to connect to MySQL MCP Server: {str(e)}",
            "attempts":       0,
            "agent_name":     agent_display_name,
        }
        api_cache[cache_key] = result
        return result


# ── CONSOLIDATED HEALTH & SCHEMA ENDPOINTS ──────────────────────────────────────

# ── ROUTER ENDPOINT: GET /status Health Check ──
@router.get("/status")
async def health_check():
    """Return a simple OK response to confirm the service is running."""
    # ── Return status OK ──
    return {"ok": True, "message": "service running"}


# ── ROUTER ENDPOINT: GET /schema Database Schema ──
@router.get("/schema")
async def schema_endpoint():
    """Return the full database schema as a formatted text string."""
    try:
        # ── Fetch schema metadata and return ──
        return {"schema": get_schema()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

