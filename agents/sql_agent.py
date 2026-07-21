"""
sql_agent.py — The main SQL reasoning loop.

Sends the user's question to the LLM, runs SQL via the MCP tool,
feeds results back to the LLM, and repeats until a final answer is ready.

Also handles quirky models (llama3.1, qwen) that output SQL as plain text
instead of using structured tool calls — intercepted and executed anyway.
"""


# ── MODULE TAG: SQL Generator Agent ──
# ── STITCHGUARD LAYER: L3 (Table Scope) & L4 (Write Blocks) Validation Hooks ──
import re
import json
import asyncio
import ollama
from collections import defaultdict
from typing import AsyncGenerator
from mcp import ClientSession

from config.settings import MODEL_NAME, FORMATTER_MODEL_NAME
from config.logger import get_logger
from utils.guardrails import (
    validate_sql_domain_scope,
    is_safe_sql_query,
    redact_db_output_string,
    extract_tables_from_sql,
)

log = get_logger(__name__)

# ── Step-count tracker ────────────────────────────────────────────────────────
# Counts how many LLM steps each request took (1–5).
# Useful for spotting if the model needs better few-shot examples.
_step_counts: dict[int, int] = defaultdict(int)


def get_step_counts() -> dict[int, int]:
    """Return how many requests finished in each step count (1–5)."""
    return dict(_step_counts)


# ── Keep old name as alias so existing code doesn't break ────────────────────
_step_histogram    = _step_counts
get_step_histogram = get_step_counts


# ── Helper: find a hidden JSON tool call inside LLM text ─────────────────────

def find_json_in_llm_text(text: str) -> dict | None:
    """
    Some models write the tool call as raw JSON text instead of using
    the structured tool_calls field. This scans the text for a JSON block
    with `"name": "run_select_query"` and returns it if found.
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


# ── Helper: extract SQL from plain LLM text ───────────────────────────────────

def extract_sql_from_text(llm_text: str) -> str:
    """
    Pull a SELECT statement out of raw LLM output.

    Tries ```sql ... ``` fences first, then falls back to finding
    a bare SELECT keyword. Returns an empty string if nothing found.
    """
    # Try markdown SQL fence first
    fence_match = re.search(r"```sql\s*(.*?)\s*```", llm_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Fallback: bare SELECT
    select_match = re.search(r"(SELECT\s+.*)", llm_text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).split(';')[0].strip() + ";"

    return ""


# ── Helper: parse the plain-text result from sql_executor ────────────────────

# ── SQL AGENT: Database Output Parser Utility ──
def parse_db_result(db_output: str) -> tuple[list[str], list[tuple]]:
    """
    Parse the text that sql_executor returns into (columns, rows).

    sql_executor produces text like:
        Columns: first_name, salary
        Rows (up to 100):
        - (Alice, 50000)
        - (Bob, 60000)

    Returns ([], []) if parsing fails or the format is unrecognised.
    """
    try:
        lines = db_output.strip().splitlines()
        columns: list[str] = []
        rows: list[tuple] = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith("columns:"):
                raw = line[len("columns:"):].strip()
                columns = [c.strip() for c in raw.split(",")]
            elif line.startswith("- ("):
                # e.g.  - (6,)   or   - (Alice, 30)
                inner = line[3:].rstrip(")").strip().rstrip(",")
                if "," in inner:
                    parts = [p.strip().strip("'\"") for p in inner.split(",")]
                else:
                    parts = [inner.strip().strip("'\"")]
                rows.append(tuple(parts))

        return columns, rows
    except Exception:
        return [], []


# ── Helper: fast template formatter (skips the LLM formatter for simple results) ──────

def format_simple_result(db_output: str, question: str) -> str | None:
    """
    Convert simple DB results to a natural sentence — no LLM needed.

    Saves ~500–2000 ms on COUNT queries and short name lists.

    Returns None for multi-column or large results so the caller can
    fall back to the LLM formatter.

    Handled shapes:
      - Single value (COUNT / SUM / scalar):  "There are currently 6 customers."
      - Single column, up to 20 rows:         "Here are the first name: Alice, Bob."
    """
    # Errors and empty results need the LLM to explain them naturally
    if not db_output or db_output.startswith("Error") or db_output == "No rows returned.":
        return None

    columns, rows = parse_db_result(db_output)
    if not columns or not rows:
        return None

    # Single scalar value (COUNT, SUM, etc.)
    if len(rows) == 1 and len(columns) == 1:
        col_name = columns[0].lower()
        is_aggregate = any(agg in col_name for agg in ["count", "sum", "avg", "min", "max"])
        is_count_question = any(q_word in question.lower() for q_word in ["how many", "total", "count", "number of"])

        if is_aggregate or is_count_question:
            value     = rows[0][0]
            col_label = col_name.replace("_", " ")
            # Remove SQL aggregate prefix: count(*) → ""
            col_label = re.sub(
                r'^(count|sum|avg|max|min)\s*\(.*?\)$', '',
                col_label, flags=re.IGNORECASE
            ).strip()
            # Pick a plain noun from the question: "how many customers" → "customers"
            stop = {"there", "which", "where", "about", "total", "count",
                    "many", "much", "what", "show", "tell", "list",
                    "gives", "fetch", "find", "query", "give", "please", "provide"}
            nouns = [w for w in question.lower().split() if len(w) > 4 and w not in stop]
            label = nouns[0] if nouns else (col_label or "record")
            log.info(f"[TemplateFormatter] scalar: value={value!r} label={label!r}")
            return f"There are currently **{value}** {label} in the database."

    # Single-column list (up to 20 rows)
    if len(columns) == 1 and 1 < len(rows) <= 20:
        items = [str(r[0]) for r in rows]
        col_label = columns[0].replace("_", " ")
        bullets = "\n".join(f"- {item}" for item in items)
        log.info(f"[TemplateFormatter] list: {len(items)} items, col={col_label!r}")
        return f"Here are the {col_label}:\n\n{bullets}"

    return None  # complex — let LLM format it


# ── Keep old names as aliases ─────────────────────────────────────────────────
_find_json_tool_call         = find_json_in_llm_text
extract_guaranteed_sql_block = extract_sql_from_text
_parse_tool_output           = parse_db_result
format_response              = format_simple_result


# ── Streaming formatter (opt-in, Change 7) ────────────────────────────────────

async def stream_answer_tokens(
    formatter_prompt: str,
    system_prompt: str,
    question: str,
    db_output: str,
) -> AsyncGenerator[str, None]:
    """
    Yield answer tokens one by one from the LLM formatter.

    Used when the caller requests ?stream=true — lets the UI show text
    as it arrives instead of waiting for the full response.
    """
    try:
        stream = ollama.chat(
            model=FORMATTER_MODEL_NAME,
            messages=[
                {"role": "system", "content": formatter_prompt or system_prompt},
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
            stream=True,
        )
        for chunk in stream:
            if isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = getattr(getattr(chunk, "message", None), "content", "") or ""
            if token:
                yield token
    except Exception as e:
        log.error(f"Streaming formatter error: {e}")
        yield f"\n\n[Streaming error: {e}]"


# ── Keep old name as alias ────────────────────────────────────────────────────
_stream_format_response = stream_answer_tokens


def clean_extracted_sql_pipeline(raw_llm_text) -> str:
    """
    Cleans raw token inputs to guarantee execution formatting blocks strictly.
    Prevents slice KeyError exceptions and invalid query string payload blocks.
    """
    if not raw_llm_text:
        return ""
        
    # If the model returned a structured dict wrapper layout instance, extract fields safely
    if isinstance(raw_llm_text, dict):
        for v in raw_llm_text.values():
            if isinstance(v, str) and re.search(r"\bSELECT\b", v, re.IGNORECASE):
                # Ensure this is not a placeholder description/schema block
                v_lower = v.lower()
                if not any(placeholder in v_lower for placeholder in ["query string", "to run", "to execute", "enter your", "placeholder"]):
                    raw_llm_text = v
                    break
        else:
            raw_llm_text = raw_llm_text.get("sql_statement") or raw_llm_text.get("sql_query") or raw_llm_text.get("query") or raw_llm_text.get("sql") or str(raw_llm_text)
        
    cleaned = str(raw_llm_text).strip()
    
    # Emergency fallback check: if the model returned raw boilerplate fallback string text descriptors
    cleaned_lower = cleaned.lower()
    if (
        cleaned_lower == "string" 
        or len(cleaned) <= 6
        or any(placeholder in cleaned_lower for placeholder in ["query string to run", "query to execute", "placeholder query"])
    ):
        return ""
        
    # Standard markdown SQL wrapper tags removal parsing algorithm regex bounds check
    markdown_match = re.search(r"```sql\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        cleaned = markdown_match.group(1).strip()
    else:
        # Loose regex select statement substring check to remove accidental trailing metadata commentary lines
        select_match = re.search(r"(SELECT\s+.*)", cleaned, re.DOTALL | re.IGNORECASE)
        if select_match:
            cleaned = select_match.group(1).strip()
        
    # Clean up trailing syntax noise from dictionary extractions (like curly braces and semicolons)
    cleaned = cleaned.rstrip("} ;").strip()
    
    # Strip wrapping quotes only if they wrap the entire SELECT query
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
        from database.schema import get_schema
        try:
            additional_schema = get_schema(include_tables=set(missing_tables))
            if "Detailed schema for relevant tables below:" in additional_schema:
                additional_schema = additional_schema.split("Detailed schema for relevant tables below:")[-1].strip()
            return f"\n\nHere is the detailed schema for the missing table(s) you queried:\n{additional_schema}"
        except Exception as e:
            log.warning(f"Failed to fetch additional schema for {missing_tables}: {e}")
    return ""



# ── Main agent loop ──────────────────────────────────────────────────────────

# ── SQL AGENT: Core SQL Generation Agent Orchestrator ──
async def run_sql_agent(
    session:          ClientSession,
    messages:         list,
    system_prompt:    str,
    question:         str,
    agent_name:       str,
    cache_key:        str,
    api_cache:        dict,
    intent:           str = None,
    formatter_prompt: str = None,
    stream:           bool = False,
) -> dict | AsyncGenerator:
    """
    Run the multi-step SQL agent loop.

    What it does each step:
      1. Ask the LLM (qwen2.5-coder) what SQL to run
      2. Execute that SQL on MySQL via the MCP tool
      3. Feed the result back to the LLM and repeat
      4. When the LLM gives a final text answer — format and return it

    Falls back to template formatting for simple results (no LLM needed).
    Falls back to LLM formatting for complex tables.
    """
    from tools.sql_query_tool import run_select_query  # Ollama tool stub

    used_sql = None

    for step in range(5):
        log.info(f"Agent Step {step + 1}")

        # Ask the LLM what SQL to run (use temp 0.3 on retries to break deterministic repetition loops)
        temp = 0.0 if step == 0 else 0.3
        llm_reply = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=[run_select_query],
            options={"temperature": temp},
        )

        if isinstance(llm_reply, dict):
            assistant_msg = llm_reply.get("message", {})
        else:
            assistant_msg = llm_reply.message

        # Did the LLM use structured tool_calls?
        tool_calls = (
            assistant_msg.get("tool_calls")
            if isinstance(assistant_msg, dict)
            else getattr(assistant_msg, "tool_calls", None)
        )

        if tool_calls:
            # ── Normal path: LLM used structured tool_calls ───────────────
            messages.append(assistant_msg)

            for call in tool_calls:
                func = (
                    call.get("function")
                    if isinstance(call, dict)
                    else getattr(call, "function", None)
                )
                tool_name = (
                    func.get("name")
                    if isinstance(func, dict)
                    else getattr(func, "name", None)
                )
                tool_args = (
                    func.get("arguments")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", None)
                )

                log.info(f"Tool call: {tool_name}  args: {tool_args}")

                # sql_executor uses "execute_read_only_query" — map our stub name
                mcp_tool_name = (
                    tool_name.replace("run_select_", "execute_read_only_", 1)
                    if tool_name else tool_name
                )

                db_output = ""
                skip_execution = False

                if tool_name == "run_select_query" and tool_args:
                    sql_val = tool_args.get("sql_query")
                    sql_str = clean_extracted_sql_pipeline(sql_val)
                    tool_args["sql_query"] = sql_str
                    used_sql = sql_str

                    # ── STITCHGUARD LAYER 3: DOMAIN TABLE SCOPE VALIDATION ─────────
                    if intent:
                        is_allowed, scope_err = validate_sql_domain_scope(sql_str, intent)
                        if not is_allowed:
                            db_output = f"Error: {scope_err}"
                            skip_execution = True

                    # ── STITCHGUARD LAYER 4: SQL STRUCTURAL SAFETY CHECK ───────────
                    if not skip_execution:
                        is_safe_sql, sql_err = is_safe_sql_query(sql_str)
                        if not is_safe_sql:
                            db_output = f"Error: {sql_err}"
                            skip_execution = True

                # ── PIPELINE: MCP SUBPROCESS SQL COMMAND EXECUTION ─────────────
                if not skip_execution:
                    try:
                        result     = await session.call_tool(mcp_tool_name, tool_args or {})
                        db_output  = result.content[0].text if result.content else ""
                        log.info(f"DB result: {db_output[:200]}")
                    except Exception as e:
                        db_output = f"Error executing tool '{mcp_tool_name}': {str(e)}"
                        log.error(db_output)

                # ── STITCHGUARD LAYER 5: OUTPUT DATA COLUMN SENSITIVITY REDACTION ─────
                db_output = redact_db_output_string(db_output)



                # Add missing schema guidance if query execution error occurred
                if db_output.startswith("Error") or "Unknown column" in db_output or "Table" in db_output:
                    if "Unknown column" in db_output:
                        db_output += (
                            "\n\nCRITICAL: Unknown column error. Check if the column you are querying belongs to a different table in the schema. "
                            "If the columns span different tables, you MUST perform an explicit JOIN (e.g., JOIN products ON suppliers.product_id = products.product_id) to link them. "
                            "Do not query columns from tables they do not belong to."
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
            raw_text = (
                assistant_msg.get("content", "")
                if isinstance(assistant_msg, dict)
                else getattr(assistant_msg, "content", "")
            ) or ""

            if "READ_ONLY_REDIRECT" in raw_text:
                result = {
                    "sql":            None,
                    "columns":        [],
                    "rows":           [],
                    "natural_answer": "I cannot modify the database. I only have read-only access.",
                    "error":          None,
                    "attempts":       step + 1,
                    "agent_name":     agent_name,
                }
                api_cache[cache_key] = result
                _step_counts[step + 1] += 1
                log.info(
                    f"SqlAgent: early exit / read-only redirect | steps={step + 1} | "
                    f"agent={agent_name}"
                )
                return result

            # Try JSON fallback first, then plain SQL text
            hidden_call = find_json_in_llm_text(raw_text)
            sql_query   = ""
            used_sql_text = False

            if hidden_call:
                args = hidden_call.get("parameters") or hidden_call.get("arguments") or {}
                sql_query = clean_extracted_sql_pipeline(args.get("sql_query", ""))

            if not sql_query:
                extracted = extract_sql_from_text(raw_text)
                if extracted:
                    sql_query = clean_extracted_sql_pipeline(extracted)
                    used_sql_text = True

            # If the user's rule for METADATA_REDIRECT was triggered, or no SQL was found at all:
            if "METADATA_REDIRECT" in raw_text or not sql_query:
                # ── Hallucination guard ───────────────────────────────────────
                # If the LLM produced a text answer WITHOUT running any SQL,
                # and we're still in the early steps, force it to retry with
                # a corrective prompt. Small models sometimes skip the
                # SQL step and hallucinate data answers directly.
                if (
                    "METADATA_REDIRECT" not in raw_text
                    and step < 2
                    and used_sql is None
                ):
                    log.warning(
                        f"[HallucinationGuard] Step {step + 1}: LLM answered without "
                        f"querying the DB. Forcing retry. Text: {raw_text[:120]}"
                    )
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

                # LLM produced a final text answer (no more tool calls)
                final_answer = raw_text
                final_answer = re.sub(r'\n{3,}', '\n\n', final_answer).strip()
                # Remove the raw METADATA_REDIRECT flag from the UI output
                final_answer = final_answer.replace("METADATA_REDIRECT", "").strip()
                
                # If the answer is now empty because it ONLY contained the flag,
                # let's generate a nice fallback answer using the schema context.
                if not final_answer:
                    # Try to extract the metadata header we added earlier from the system prompt
                    meta_match = re.search(r"Total tables: (\d+)\nAll tables: ([^\n]+)", system_prompt)
                    if meta_match:
                        count, tables = meta_match.groups()
                        final_answer = f"There are exactly {count} tables in the database: {tables}."
                    else:
                        final_answer = (
                            "I can answer questions about the data inside the tables, "
                            "but I am not permitted to run structure queries directly."
                        )

                result = {
                    "sql":            used_sql,
                    "columns":        [],
                    "rows":           [],
                    "natural_answer": final_answer,
                    "error":          None,
                    "attempts":       step + 1,
                    "agent_name":     agent_name,
                }
                if used_sql:
                    api_cache[cache_key] = {
                        "sql":        used_sql,
                        "intent":     intent,
                        "agent_name": agent_name,
                        "question":   question
                    }
                else:
                    api_cache[cache_key] = result
                _step_counts[step + 1] += 1
                log.info(
                    f"SqlAgent: early exit / no SQL | steps={step + 1} | "
                    f"sql_len={len(used_sql or '')} | agent={agent_name}"
                )
                return result

            if sql_query:
                mode = "SQL text" if used_sql_text else "JSON text"
                log.warning(f"[Fallback] LLM returned query as {mode}: {sql_query[:120]}")
                used_sql = sql_query

                db_output = ""
                skip_execution = False

                # ── STITCHGUARD LAYER 3: DOMAIN TABLE SCOPE VALIDATION ─────────
                if intent:
                    is_allowed, scope_err = validate_sql_domain_scope(sql_query, intent)
                    if not is_allowed:
                        db_output = f"Error: {scope_err}"
                        skip_execution = True

                # ── STITCHGUARD LAYER 4: SQL STRUCTURAL SAFETY CHECK ───────────
                if not skip_execution:
                    is_safe_sql, sql_err = is_safe_sql_query(sql_query)
                    if not is_safe_sql:
                        db_output = f"Error: {sql_err}"
                        skip_execution = True

                # ── PIPELINE: MCP SUBPROCESS SQL COMMAND EXECUTION ─────────────
                if not skip_execution:
                    try:
                        result    = await session.call_tool(
                            "execute_read_only_query", {"sql_query": sql_query}
                        )
                        db_output = result.content[0].text if result.content else "No results."
                        log.info(f"[Fallback] DB result: {db_output[:200]}")
                    except Exception as e:
                        db_output = f"Error executing query: {str(e)}"
                        log.error(f"[Fallback] {db_output}")

                # ── STITCHGUARD LAYER 5: OUTPUT DATA COLUMN SENSITIVITY REDACTION ─────
                db_output = redact_db_output_string(db_output)



                if db_output.startswith("Error"):
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
                        " If this is an 'Unknown column' error, you MUST look at the provided SCHEMA CONTEXT and use the exact column names defined there (e.g. use 'dept_name' if 'department_name' does not exist). "
                        "Ensure all quotes and brackets are properly closed."
                    )
                    schema_guidance = get_missing_schema_guidance(sql_query, system_prompt)
                    if schema_guidance:
                        guidance += schema_guidance

                    messages.append(assistant_msg)
                    messages.append({
                        "role":    "user",
                        "content": guidance,
                    })
                    continue

                # Streaming path (Change 7)
                if stream:
                    from fastapi.responses import StreamingResponse

                    async def _token_gen():
                        async for token in stream_answer_tokens(
                            formatter_prompt or system_prompt,
                            system_prompt, question, db_output
                        ):
                            yield token

                    _step_counts[step + 1] += 1
                    log.info(
                        f"SqlAgent: streaming | steps={step + 1} | "
                        f"sql_len={len(used_sql or '')} | agent={agent_name}"
                    )
                    if used_sql:
                        api_cache[cache_key] = {
                            "sql":        used_sql,
                            "intent":     intent,
                            "agent_name": agent_name,
                            "question":   question
                        }
                    # Remove all newlines/carriage returns to prevent uvicorn "Invalid HTTP header value" errors
                    sql_flat = " ".join((used_sql or "").split())
                    safe_sql = sql_flat.encode("ascii", "ignore").decode("ascii")
                    headers = {
                        "x-agent-name": agent_name,
                        "x-sql-query": safe_sql
                    }
                    return StreamingResponse(_token_gen(), media_type="text/event-stream", headers=headers)

                # Try fast template formatter first (no LLM, ~0 ms)
                final_answer   = format_simple_result(db_output, question)
                formatter_used = "template"

                if final_answer is None:
                    # Complex result — use formatter LLM
                    formatter_used = "llm"
                    fmt_reply = ollama.chat(
                        model=FORMATTER_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": formatter_prompt or system_prompt},
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

                final_answer = re.sub(r'\n{3,}', '\n\n', final_answer or "")

                result = {
                    "sql":            used_sql,
                    "columns":        [],
                    "rows":           [],
                    "natural_answer": final_answer,
                    "error":          None,
                    "attempts":       step + 1,
                    "agent_name":     agent_name,
                }
                if used_sql:
                    api_cache[cache_key] = {
                        "sql":        used_sql,
                        "intent":     intent,
                        "agent_name": agent_name,
                        "question":   question
                    }
                else:
                    api_cache[cache_key] = result
                _step_counts[step + 1] += 1
                log.info(
                    f"SqlAgent: done in {step + 1} step(s) [{formatter_used}] | "
                    f"sql_len={len(used_sql or '')} | agent={agent_name}"
                )
                return result

            # LLM produced a final text answer (no more tool calls)
            final_answer = (
                assistant_msg.get("content", "")
                if isinstance(assistant_msg, dict)
                else getattr(assistant_msg, "content", "")
            )
            final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)

            result = {
                "sql":            used_sql,
                "columns":        [],
                "rows":           [],
                "natural_answer": final_answer,
                "error":          None,
                "attempts":       step + 1,
                "agent_name":     agent_name,
            }
            if used_sql:
                api_cache[cache_key] = {
                    "sql":        used_sql,
                    "intent":     intent,
                    "agent_name": agent_name,
                    "question":   question
                }
            else:
                api_cache[cache_key] = result
            _step_counts[step + 1] += 1
            log.info(
                f"SqlAgent: done in {step + 1} step(s) | "
                f"sql_len={len(used_sql or '')} | agent={agent_name}"
            )
            return result

    # Hit 5-step limit without a final answer
    result = {
        "sql":            used_sql,
        "columns":        [],
        "rows":           [],
        "natural_answer": (
            "I reached the maximum reasoning steps without producing a final answer. "
            "Please try rephrasing your question."
        ),
        "error":          "Reasoning loop limit exceeded",
        "attempts":       5,
        "agent_name":     agent_name,
    }
    api_cache[cache_key] = result
    _step_counts[5] += 1
    log.warning(
        f"SqlAgent: hit 5-step limit | sql_len={len(used_sql or '')} | agent={agent_name}"
    )
    return result
