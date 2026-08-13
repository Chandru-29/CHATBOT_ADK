"""
sql_agent.py — The main SQL reasoning loop.

Sends the user's question to the LLM, runs SQL via the MCP tool,
feeds results back to the LLM, and repeats until a final answer is ready.
"""

# ── MODULE TAG: SQL Generator Agent ──
# ── STITCHGUARD LAYER: L3 (Table Scope) & L4 (Write Blocks) Validation Hooks ──
import re
import json
import asyncio
from collections import defaultdict
from typing import AsyncGenerator
from mcp import ClientSession

from core.llm.llm_client import get_llm_async_client, get_async_client
from core.config.settings import MODEL_NAME, AGENT_MAX_STEPS
from core.config.logger import get_logger
from security.guardrails import (
    validate_sql_before_execution,
    redact_db_output_string,
    extract_tables_from_sql,
    is_db_error,
    READ_ONLY_REFUSAL,
)

log = get_logger(__name__)

# Shared async client — single connection pool, imported from llm_client
_async_client = get_llm_async_client()

# ── Step-count tracker ────────────────────────────────────────────────────────
_step_counts: dict[int, int] = defaultdict(int)


def get_step_counts() -> dict[int, int]:
    """Get statistics on how many reasoning steps (1 to 5) were used to answer questions.

    Returns:
        dict[int, int]: A dictionary mapping step numbers to total question counts.
    """
    return dict(_step_counts)


# ── Shared Execution & Caching Helpers ─────────────────────────────────────────

async def _execute_validated_sql(
    session: ClientSession,
    sql_query: str,
    intent: str | None = None,
    tool_name: str = "execute_read_only_query",
    tool_args: dict | None = None,
) -> tuple[str, bool]:
    """Check SQL query safety, run it on the database, and remove any sensitive data.

    Args:
        session (ClientSession): Connection session to the database tool.
        sql_query (str): The SELECT SQL query string to run.
        intent (str | None, optional): Category of question. Defaults to None.
        tool_name (str, optional): Database tool name. Defaults to "execute_read_only_query".
        tool_args (dict | None, optional): Arguments to pass to the tool. Defaults to None.

    Returns:
        tuple[str, bool]: A tuple containing:
            - str: The clean database result text or error message.
            - bool: True if the query failed safety checks and was skipped, False otherwise.
    """
    is_valid, guard_err = validate_sql_before_execution(sql_query, intent)
    if not is_valid:
        return f"Error: {guard_err}", True

    args = tool_args or {"sql_query": sql_query}
    try:
        result = await session.call_tool(tool_name, args)
        db_output = result.content[0].text if result.content else "No results."
        log.info(f"DB result: {db_output[:200]}")
    except Exception as e:
        db_output = f"Error executing tool '{tool_name}': {str(e)}"
        log.error(db_output)

    return redact_db_output_string(db_output), False


def build_compact_history_summary(
    used_sql: str | None,
    db_output: str | None = None,
    rows: list | None = None,
    columns: list | None = None,
    error: str | None = None,
) -> str:
    """Create a short, compact summary of the SQL query and its results for chat memory.

    Args:
        used_sql (str | None): The executed SQL query string.
        db_output (str | None, optional): Raw database text output. Defaults to None.
        rows (list | None, optional): List of data rows returned. Defaults to None.
        columns (list | None, optional): List of column names returned. Defaults to None.
        error (str | None, optional): Error message if the query failed. Defaults to None.

    Returns:
        str: A short summary tag string (e.g. `[EXECUTED_SQL: ... | RESULT: 3 rows returned (Samples: IT001)]`).
    """
    if not used_sql:
        return ""

    if error or (db_output and db_output.startswith("Error")):
        err_msg = error or db_output
        err_msg = " ".join(str(err_msg).split())
        return f"[EXECUTED_SQL: {used_sql} | ERROR: {err_msg}]"

    if rows is None and db_output:
        parsed_cols, parsed_rows = parse_db_result(db_output)
        rows = parsed_rows
        columns = columns or parsed_cols

    rows = rows or []
    row_count = len(rows)

    if row_count == 0:
        return f"[EXECUTED_SQL: {used_sql} | RESULT: 0 rows returned]"

    samples = [str(r[0]) for r in rows[:3] if r and r[0] is not None]
    samples_str = f" (Samples: {', '.join(samples)})" if samples else ""
    return f"[EXECUTED_SQL: {used_sql} | RESULT: {row_count} rows returned{samples_str}]"


def _build_result(
    used_sql: str | None,
    final_answer: str | None,
    attempts: int,
    agent_name: str,
    error: str | None = None,
    history_summary: str | None = None,
) -> dict:
    """Build the standard answer payload dictionary to return to the user.

    Args:
        used_sql (str | None): The SQL query that was run (if any).
        final_answer (str | None): The natural language answer for display.
        attempts (int): Number of steps taken to get the answer.
        agent_name (str): Name of the agent that answered.
        error (str | None, optional): Error message if something went wrong. Defaults to None.
        history_summary (str | None, optional): Short chat history summary tag. Defaults to None.

    Returns:
        dict: A dictionary containing `sql`, `columns`, `rows`, `natural_answer`, `error`, `attempts`, `agent_name`.
    """
    res = {
        "sql":            used_sql,
        "columns":        [],
        "rows":           [],
        "natural_answer": final_answer,
        "error":          error,
        "attempts":       attempts,
        "agent_name":     agent_name,
    }
    if history_summary:
        res["history_summary"] = history_summary
    return res


def _record_cache_and_steps(
    result: dict,
    used_sql: str | None,
    intent: str | None,
    agent_name: str,
    question: str,
    cache_key: str,
    api_cache: dict,
    step: int,
) -> None:
    """Update execution metrics and cache query results in API and Redis stores.

    Args:
        result (dict): Agent result dictionary payload.
        used_sql (str | None): Executed SQL query statement.
        intent (str | None): Domain intent classification label.
        agent_name (str): Display name of the agent.
        question (str): Original user input question.
        cache_key (str): Unique cache key identifier.
        api_cache (dict): In-memory API response cache mapping dict.
        step (int): Current reasoning loop step index (0-indexed).
    """
    from core.cache.cache_manager import store_cache
    if used_sql:
        entry = {
            "sql":             used_sql,
            "intent":          intent,
            "agent_name":      agent_name,
            "question":        question,
            "history_summary": result.get("history_summary") or build_compact_history_summary(used_sql),
        }
        api_cache[cache_key] = entry
        store_cache(question, entry)
    else:
        api_cache[cache_key] = result
        store_cache(question, result)
    _step_counts[step + 1] += 1


# ── Text Parsing Helpers ───────────────────────────────────────────────────────

def find_json_in_llm_text(text: str) -> dict | None:
    """Scan unformatted text for a JSON payload calling `run_select_query`.

    Args:
        text (str): Raw model completion text output string.

    Returns:
        dict | None: Parsed JSON tool call dictionary if found, or None if invalid.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and obj.get("name") == "run_select_query":
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return None


def extract_sql_from_text(llm_text: str) -> str:
    """Pull a SELECT statement out of raw LLM output."""
    fence_match = re.search(r"```sql\s*(.*?)\s*```", llm_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    select_match = re.search(r"(SELECT\s+.*)", llm_text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).split(';')[0].strip() + ";"

    return ""


def parse_db_result(db_output: str) -> tuple[list[str], list[tuple]]:
    """Parse text output from sql_executor into (columns, rows)."""
    try:
        lines = db_output.strip().splitlines()
        columns: list[str] = []
        rows: list[tuple] = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith("columns:"):
                raw = line[len("columns:"):].strip()
                columns = [c.strip() for c in raw.split(",")]
            elif line.startswith("- "):
                inner = line[2:].strip()
                if inner.startswith("(") and inner.endswith(")"):
                    inner = inner[1:-1].strip()
                parts = [p.strip().strip("'\"") for p in inner.split(",")]
                rows.append(tuple(parts))

        return columns, rows
    except Exception:
        return [], []


STATUS_MAP = {
    "0": "Created",
    "1": "Released",
    "2": "Picking Started",
    "3": "Picked",
    "4": "Putaway Started",
    "5": "Completed",
}


def transform_status_and_completion(columns: list[str], rows: list[tuple]) -> tuple[list[str], list[tuple]]:
    """Transform numeric status codes to human-readable definitions and update completion date column headers."""
    if not columns or not rows:
        return columns, rows

    new_columns = list(columns)
    status_indices = []

    for idx, col in enumerate(columns):
        c_lower = col.lower().strip()
        if "status" in c_lower and "status_name" not in c_lower and "pipeline" not in c_lower:
            status_indices.append(idx)
        if c_lower in ("ud", "updated_date", "updateddate", "completion_date", "completiondate"):
            if c_lower == "ud":
                new_columns[idx] = "Completion / Updated Date"

    new_rows = []
    for row in rows:
        row_list = list(row)
        for s_idx in status_indices:
            val_str = str(row_list[s_idx]).strip()
            if val_str in STATUS_MAP:
                row_list[s_idx] = STATUS_MAP[val_str]
        new_rows.append(tuple(row_list))

    return new_columns, new_rows


def format_db_result_deterministic(db_output: str, question: str = "") -> str:
    """
    Zero-LLM Server-Side Deterministic Response Engine.
    Converts raw database execution output string into clean GitHub-Flavored Markdown.
    
    Handles:
    1. Error & Security Redaction Notices -> Return pass-through message.
    2. Empty Results (0 rows / "No rows returned.") -> Return clear negative response.
    3. Scalar Values (1 row, 1 column) -> Return bold formatted metric/count sentence.
    4. Single-Column Lists (N rows, 1 column) -> Return clean bullet-point list (- Item).
    5. Multi-Column Tables (N rows, M columns) -> Return clean Markdown table (| Col1 | Col2 | ...).
    """
    if not db_output or db_output.strip() == "No rows returned.":
        return "No matching records were found in the database for your query."

    if "omitted for security" in db_output or "sensitive and omitted" in db_output:
        return db_output

    if db_output.startswith("Error"):
        return db_output

    columns, rows = parse_db_result(db_output)
    if not columns or not rows:
        return db_output

    columns, rows = transform_status_and_completion(columns, rows)


    def clean_header(col: str) -> str:
        s = col.replace("_", " ").strip()
        return " ".join(word.capitalize() for word in s.split()) if s else col

    # ── CASE 1: Scalar Metric Result (1 row, 1 column) ──
    if len(rows) == 1 and len(columns) == 1:
        val = str(rows[0][0])
        col_name = columns[0].lower()
        is_aggregate = any(agg in col_name for agg in ["count", "sum", "avg", "min", "max"])
        is_count_question = any(q_word in question.lower() for q_word in ["how many", "total", "count", "number of"])

        if is_aggregate or is_count_question:
            col_label = col_name.replace("_", " ")
            col_label = re.sub(
                r'^(count|sum|avg|max|min)\s*\(.*?\)$', '',
                col_label, flags=re.IGNORECASE
            ).strip()
            stop = {"there", "which", "where", "about", "total", "count",
                    "many", "much", "what", "show", "tell", "list",
                    "gives", "fetch", "find", "query", "give", "please", "provide"}
            nouns = [w for w in question.lower().split() if len(w) > 4 and w not in stop]
            label = nouns[0] if nouns else (col_label or "record")
            log.info(f"[DeterministicFormatter] scalar: value={val!r} label={label!r}")
            return f"There are currently **{val}** {label} in the database."

    # ── CASE 2: Single-Column List (N rows, 1 column) ──
    if len(columns) == 1:
        col_title = clean_header(columns[0])
        items = [str(r[0]) for r in rows]
        bullets = "\n".join(f"- {item}" for item in items)
        log.info(f"[DeterministicFormatter] list: {len(items)} items, col={col_title!r}")
        return f"Here are the {col_title.lower()}:\n\n{bullets}"

    # ── CASE 3: Multi-Column Data Table (N rows, M columns) ──
    headers = [clean_header(c) for c in columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in columns) + " |"
    data_rows = []
    for row in rows:
        row_str = "| " + " | ".join(str(val) for val in row) + " |"
        data_rows.append(row_str)

    table_md = "\n".join([header_row, separator_row] + data_rows)
    if "... (and more rows exist)" in db_output:
        table_md += "\n\n_(and more rows exist)_"

    log.info(f"[DeterministicFormatter] table: {len(rows)} rows x {len(columns)} cols")
    return f"Here are the requested database records:\n\n{table_md}"


format_simple_result = format_db_result_deterministic


async def stream_answer_tokens(
    question: str,
    db_output: str,
) -> AsyncGenerator[str, None]:
    """Yield answer tokens directly from the deterministic markdown formatter."""
    ans = format_db_result_deterministic(db_output, question)
    yield ans


_stream_format_response = stream_answer_tokens


def clean_extracted_sql_pipeline(raw_llm_text) -> str:
    """Cleans raw token inputs to guarantee execution formatting blocks strictly."""
    if not raw_llm_text:
        return ""

    if isinstance(raw_llm_text, dict):
        for v in raw_llm_text.values():
            if isinstance(v, str) and re.search(r"\bSELECT\b", v, re.IGNORECASE):
                v_lower = v.lower()
                if not any(placeholder in v_lower for placeholder in ["query string", "to run", "to execute", "enter your", "placeholder"]):
                    raw_llm_text = v
                    break
        else:
            raw_llm_text = raw_llm_text.get("sql_statement") or raw_llm_text.get("sql_query") or raw_llm_text.get("query") or raw_llm_text.get("sql") or str(raw_llm_text)

    cleaned = str(raw_llm_text).strip()
    cleaned_lower = cleaned.lower()
    if (
        cleaned_lower == "string"
        or len(cleaned) <= 6
        or any(placeholder in cleaned_lower for placeholder in ["query string to run", "query to execute", "placeholder query"])
    ):
        return ""

    markdown_match = re.search(r"```sql\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        cleaned = markdown_match.group(1).strip()
    else:
        select_match = re.search(r"(SELECT\s+.*)", cleaned, re.DOTALL | re.IGNORECASE)
        if select_match:
            cleaned = select_match.group(1).strip()

    cleaned = cleaned.rstrip("} ;").strip()

    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()

    return cleaned + ";"


def get_missing_schema_guidance(sql_query: str, system_prompt: str) -> str:
    """Fetch schema for any tables referenced in the query that are missing from the system prompt."""
    tables = extract_tables_from_sql(sql_query)
    missing_tables = [t for t in tables if f"Table: {t}" not in system_prompt]
    if missing_tables:
        from db.schema import get_schema
        try:
            additional_schema = get_schema(include_tables=set(missing_tables))
            if "Detailed schema for relevant tables below:" in additional_schema:
                additional_schema = additional_schema.split("Detailed schema for relevant tables below:")[-1].strip()
            return f"\n\nHere is the detailed schema for the missing table(s) you queried:\n{additional_schema}"
        except Exception as e:
            log.warning(f"Failed to fetch additional schema for {missing_tables}: {e}")
    return ""


# ── Main Agent Reasoning Loop ─────────────────────────────────────────────────

async def run_sql_agent(
    session:          ClientSession,
    messages:         list,
    system_prompt:    str,
    question:         str,
    agent_name:       str,
    cache_key:        str,
    api_cache:        dict,
    intent:           str = None,
    stream:           bool = False,
) -> dict | AsyncGenerator:
    """
    Run the multi-step SQL agent loop.
    """
    from mcp_service.tools import run_select_query

    used_sql = None

    for step in range(AGENT_MAX_STEPS):
        log.debug(f"Agent Step {step + 1}")

        temp = 0.0 if step == 0 else 0.3
        client = get_llm_async_client()

        # Build messages for LLM SDK
        llm_messages = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "")
            if r == "system":
                continue
            role_name = "assistant" if r in ("bot", "assistant", "model") else "user"
            if r == "tool":
                role_name = "user"
                c = f"Tool Execution Result ({m.get('name', 'tool')}):\n{c}"
            llm_messages.append({"role": role_name, "content": str(c)})

        try:
            llm_reply = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=llm_messages,
                temperature=temp,
            )
            raw_text = llm_reply.choices[0].message.content or ""
            assistant_msg = {"role": "assistant", "content": raw_text}
        except Exception as e:
            log.error(f"Gemini API call error in sql_agent: {e}")
            raw_text = ""
            assistant_msg = {"role": "assistant", "content": ""}

        tool_calls = None

        if tool_calls:
            # ── Normal path: LLM used structured tool_calls ───────────────
            messages.append(assistant_msg)

            for call in tool_calls:
                func = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                tool_name = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
                tool_args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", None)

                log.info(f"Tool call: {tool_name}  args: {tool_args}")
                mcp_tool_name = tool_name.replace("run_select_", "execute_read_only_", 1) if tool_name else tool_name

                if tool_name == "run_select_query" and tool_args:
                    sql_val = tool_args.get("sql_query")
                    sql_str = clean_extracted_sql_pipeline(sql_val)
                    tool_args["sql_query"] = sql_str
                    used_sql = sql_str

                # Execute with pre-execution validation & output redaction helper
                db_output, _ = await _execute_validated_sql(
                    session=session,
                    sql_query=used_sql or "",
                    intent=intent,
                    tool_name=mcp_tool_name,
                    tool_args=tool_args,
                )

                if is_db_error(db_output):
                    if "system schema" in db_output.lower():
                        db_output += (
                            "\n\nCRITICAL: Access to system schemas (like information_schema) is forbidden. "
                            "Do NOT attempt to query system schemas or generate SQL for database structure/counts. "
                            "Instead, read the 'Total tables' and 'All tables' list provided in the DATABASE SCHEMA section "
                            "of your system prompt and answer the user's question directly in natural language."
                        )

                    if "Unknown column" in db_output or "could not be bound" in db_output.lower():
                        db_output += (
                            "\n\nCRITICAL: Missing table JOIN or unknown column error. "
                            "You selected a column from a table that was NOT included in your FROM/JOIN clause. "
                            "Check all column aliases (e.g. l.locationCode requires JOIN LOCATION l ON m.locationId = l.locationId). "
                            "Rewrite the query to explicitly JOIN every table whose columns are referenced."
                        )
                    sql_val = tool_args.get("sql_query") if tool_args else ""
                    if sql_val:
                        guidance = get_missing_schema_guidance(sql_val, system_prompt)
                        if guidance:
                            db_output += f"\n\nCRITICAL: The query failed. {guidance}"

                messages.append({
                    "role":    "tool",
                    "name":    tool_name,
                    "content": db_output,
                })

        else:
            # ── Fallback path: LLM wrote SQL as text instead of tool_calls ─
            raw_text = assistant_msg.get("content", "") if isinstance(assistant_msg, dict) else getattr(assistant_msg, "content", "") or ""

            if "READ_ONLY_REDIRECT" in raw_text and intent == "WMS_AGENT":
                log.warning(f"[HallucinationGuard] Suppressed false-positive READ_ONLY_REDIRECT for domain intent '{intent}'. Forcing SQL generation.")
                raw_text = raw_text.replace("READ_ONLY_REDIRECT", "").strip()

            if "READ_ONLY_REDIRECT" in raw_text:
                result = _build_result(None, READ_ONLY_REFUSAL, step + 1, agent_name)
                _record_cache_and_steps(result, None, intent, agent_name, question, cache_key, api_cache, step)
                log.info(f"SqlAgent: early exit / read-only redirect | steps={step + 1} | agent={agent_name}")
                return result

            # ── STITCHGUARD: Unrelated cross-domain JOIN guard (safety-net token) ──
            if "UNRELATED_DOMAIN_REDIRECT" in raw_text:
                unrelated_msg = (
                    "⚠️ your query contains the two unrelated table data together\n Ask the question separetly\n ."
                )
                result = _build_result(None, unrelated_msg, step + 1, agent_name)
                _record_cache_and_steps(result, None, intent, agent_name, question, cache_key, api_cache, step)
                log.info(f"SqlAgent: early exit / unrelated domain redirect | steps={step + 1} | agent={agent_name}")
                return result

            hidden_call = find_json_in_llm_text(raw_text)
            sql_query = ""
            used_sql_text = False

            if hidden_call:
                args = hidden_call.get("parameters") or hidden_call.get("arguments") or {}
                sql_query = clean_extracted_sql_pipeline(args.get("sql_query", ""))

            if not sql_query:
                extracted = extract_sql_from_text(raw_text)
                if extracted:
                    sql_query = clean_extracted_sql_pipeline(extracted)
                    used_sql_text = True

            is_metadata_query = bool(re.search(r"\b(tables?|schemas?|database structure|database tables?)\b", question, re.IGNORECASE))

            if "METADATA_REDIRECT" in raw_text and not is_metadata_query:
                log.warning(f"[HallucinationGuard] Suppressed false-positive METADATA_REDIRECT for non-metadata query '{question}'. Forcing SQL generation.")
                raw_text = raw_text.replace("METADATA_REDIRECT", "").strip()
                # Force retry — the LLM produced METADATA_REDIRECT by mistake; push a correction message
                if step < AGENT_MAX_STEPS - 1:
                    messages.append(assistant_msg)
                    messages.append({
                        "role":    "user",
                        "content": (
                            "CRITICAL: You incorrectly returned METADATA_REDIRECT for a DATA query. "
                            "The user is asking for actual data rows, NOT database structure information. "
                            "You MUST call run_select_query with a valid SELECT statement immediately. "
                            f"Question: {question}"
                        ),
                    })
                    continue

            if "METADATA_REDIRECT" in raw_text or not sql_query:
                if "METADATA_REDIRECT" not in raw_text and step < 2 and used_sql is None and not is_metadata_query:
                    log.warning(f"[HallucinationGuard] Step {step + 1}: LLM answered without querying the DB. Forcing retry.")
                    messages.append(assistant_msg)
                    messages.append({
                        "role":    "user",
                        "content": (
                            "CRITICAL: You did NOT execute a database query. You MUST call "
                            "the run_select_query tool with a valid SELECT statement to "
                            "retrieve the actual data. Do NOT guess or make up answers. "
                            "Generate the SQL query now."
                        ),
                    })
                    continue

                final_answer = re.sub(r'\n{3,}', '\n\n', raw_text).strip()
                final_answer = final_answer.replace("METADATA_REDIRECT", "").strip()
                
                if not final_answer or is_metadata_query:
                    meta_match = re.search(r"Total tables: (\d+)\nAll tables: ([^\n]+)", system_prompt)
                    if meta_match:
                        count, tables = meta_match.groups()
                        table_list = "\n".join(f"- `{t.strip()}`" for t in tables.split(","))
                        final_answer = f"There are currently **{count}** tables in the database:\n\n{table_list}"
                    elif not final_answer:
                        final_answer = "I can answer questions about the data inside the tables, but I am not permitted to run structure queries directly."

                result = _build_result(used_sql, final_answer, step + 1, agent_name)
                _record_cache_and_steps(result, used_sql, intent, agent_name, question, cache_key, api_cache, step)
                log.info(f"SqlAgent: early exit / no SQL | steps={step + 1} | agent={agent_name}")
                return result

            if sql_query:
                mode = "SQL text" if used_sql_text else "JSON text"
                log.debug(f"[Fallback] LLM returned query as {mode}: {sql_query[:120]}")
                used_sql = sql_query

                # Execute with pre-execution validation & output redaction helper
                db_output, _ = await _execute_validated_sql(
                    session=session,
                    sql_query=sql_query,
                    intent=intent,
                    tool_name="execute_read_only_query",
                )

                if is_db_error(db_output):
                    log.warning(f"[Fallback] Query failed. Feeding error back to LLM to self-correct.")
                    unknown_col_advice = ""
                    if "Unknown column" in db_output:
                        unknown_col_advice = (
                            " Check if the column you are querying belongs to a different table in the schema. "
                            "If the columns span different tables, you MUST perform an explicit JOIN (e.g., JOIN products ON suppliers.product_id = products.product_id) to link them. "
                            "Do not query columns from tables they do not belong to."
                        )
                    guidance = (
                        f"{db_output}\n\n"
                        "CRITICAL: The query failed. Please rewrite the query to fix the error."
                        f"{unknown_col_advice}"
                        " If this is an 'Unknown column' error, you MUST look at the provided SCHEMA CONTEXT and use the exact column names defined there. "
                        "Ensure all quotes and brackets are properly closed."
                    )
                    schema_guidance = get_missing_schema_guidance(sql_query, system_prompt)
                    if schema_guidance:
                        guidance += schema_guidance

                    messages.append(assistant_msg)
                    messages.append({"role": "user", "content": guidance})
                    continue

                if stream:
                    from fastapi.responses import StreamingResponse

                    async def _token_gen():
                        async for token in stream_answer_tokens(question, db_output):
                            yield token

                    _step_counts[step + 1] += 1
                    if used_sql:
                        api_cache[cache_key] = {
                            "sql":        used_sql,
                            "intent":     intent,
                            "agent_name": agent_name,
                            "question":   question
                        }
                    sql_flat = " ".join((used_sql or "").split())
                    safe_sql = sql_flat.encode("ascii", "ignore").decode("ascii")
                    headers = {
                        "x-agent-name": agent_name,
                        "x-sql-query": safe_sql,
                        "x-agent-step": str(step + 1),
                    }
                    return StreamingResponse(_token_gen(), media_type="text/event-stream", headers=headers)

                hist_summary = build_compact_history_summary(used_sql, db_output)
                result = _build_result(used_sql, final_answer, step + 1, agent_name, history_summary=hist_summary)
                _record_cache_and_steps(result, used_sql, intent, agent_name, question, cache_key, api_cache, step)
                log.info(f"SqlAgent: done in {step + 1} step(s) [deterministic] | agent={agent_name}")
                return result

            final_answer = assistant_msg.get("content", "") if isinstance(assistant_msg, dict) else getattr(assistant_msg, "content", "")
            final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)

            result = _build_result(used_sql, final_answer, step + 1, agent_name)
            _record_cache_and_steps(result, used_sql, intent, agent_name, question, cache_key, api_cache, step)
            log.info(f"SqlAgent: done in {step + 1} step(s) | agent={agent_name}")
            return result

    result = _build_result(
        used_sql,
        "I reached the maximum reasoning steps without producing a final answer. Please try rephrasing your question.",
        AGENT_MAX_STEPS,
        agent_name,
        error="Reasoning loop limit exceeded",
    )
    _step_counts[5] += 1
    log.warning(f"SqlAgent: hit 5-step limit | agent={agent_name}")
    return result
