import os
import re
import logging
from pathlib import Path
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv(override=True)

app = FastAPI(title="NL to SQL Chatbot API")

# Enable CORS for Streamlit front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_question: str
    db_schema: str = ""        # Optional
    chat_history: list = []    # Optional chat history for context

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_MODEL = "llama3.1"
MAX_RETRIES  = 3
MAX_ROWS     = 500      # cap result rows to avoid flooding UI
DB_TIMEOUT   = 10.0    # seconds before query times out

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from sqlalchemy.engine import URL

# Build connection URI programmatically from environment variables using URL.create
DB_DIALECT = os.getenv("DB_DIALECT", "mysql")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "company_data")

db_url_obj = URL.create(
    drivername=f"{DB_DIALECT}+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)
DB_URL = db_url_obj.render_as_string(hide_password=False)
engine = create_engine(DB_URL)

# ─────────────────────────────────────────────
# BANNED SQL KEYWORDS  (read-only enforcement)
# ─────────────────────────────────────────────

WRITE_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
    "BEGIN", "COMMIT", "ROLLBACK", "VACUUM", "REINDEX",
    "GRANT", "REVOKE", "SAVEPOINT", "RELEASE",
]

DANGEROUS_PATTERNS = [
    r";\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)",   # SQL injection via semicolon
    r"--\s*",                                           # inline SQL comment tricks
    r"/\*.*?\*/",                                       # block comments
    r"UNION\s+ALL\s+SELECT.*FROM\s+sqlite_",            # schema dump via UNION
    r"UNION\s+ALL\s+SELECT.*FROM\s+information_schema", # schema dump via UNION (MySQL schema)
]

# ─────────────────────────────────────────────
# SCHEMA EXTRACTION
# ─────────────────────────────────────────────

def get_db_schema() -> str:
    """
    Extract full schema from MySQL DB.
    Returns a clean human-readable schema string for the LLM prompt.
    """
    try:
        with engine.connect() as conn:
            query_cols = text("""
                SELECT table_name, column_name, data_type, column_key, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = :db_name
                ORDER BY table_name, ordinal_position
            """)
            cols_result = conn.execute(query_cols, {"db_name": DB_NAME}).fetchall()

            if not cols_result:
                return "No tables found in database."

            # Group columns by table
            tables_dict = {}
            for row in cols_result:
                t_name, col_name, dtype, col_key, is_null, col_default = row
                if t_name not in tables_dict:
                    tables_dict[t_name] = []
                tables_dict[t_name].append({
                    "name": col_name,
                    "type": dtype,
                    "pk": col_key == "PRI",
                    "nullable": is_null == "YES",
                    "default": col_default
                })

            schema_parts = []
            for t_name, cols in tables_dict.items():
                col_lines = []
                for col in cols:
                    parts = [f"  {col['name']} {col['type'].upper()}"]
                    if col['pk']:
                        parts.append("PRIMARY KEY")
                    if not col['nullable']:
                        parts.append("NOT NULL")
                    if col['default'] is not None:
                        parts.append(f"DEFAULT {col['default']}")
                    col_lines.append(" ".join(parts))

                try:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM `{t_name}`"))
                    row_count = count_result.scalar()
                    count_hint = f"  -- {row_count} rows"
                except Exception:
                    count_hint = ""

                schema_parts.append(
                    f"Table: {t_name}{count_hint}\n"
                    f"Columns:\n" + "\n".join(col_lines)
                )
        return "\n\n".join(schema_parts)
    except Exception as e:
        log.error(f"Schema extraction failed: {e}")
        raise RuntimeError(f"Cannot read database: {e}")

# ─────────────────────────────────────────────
# SQL SAFETY VALIDATION
# ─────────────────────────────────────────────

def validate_sql_safety(sql: str) -> tuple[bool, str]:
    """
    Multi-layer SQL safety check. Returns (is_safe, reason).
    Blocks anything that isn't a pure SELECT.
    """
    sql_upper = sql.upper().strip()

    # Layer 1: Must start with SELECT
    if not re.match(r"^\s*SELECT\b", sql_upper):
        return False, f"Query must start with SELECT. Got: '{sql[:40]}...'"

    # Layer 2: Banned write keywords
    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_upper):
            return False, f"Forbidden keyword detected: {keyword}"

    # Layer 3: Dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return False, f"Dangerous SQL pattern detected: {pattern}"

    # Layer 4: Multiple statements (semicolon separation)
    stripped = sql.rstrip().rstrip(";")
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"

# ─────────────────────────────────────────────
# SQL EXECUTION
# ─────────────────────────────────────────────

def execute_sql(sql: str) -> tuple[list[str], list[tuple]]:
    """
    Execute a SELECT query using strict read-only execution.
    Returns (column_names, rows).
    """
    sql_clean = sql.strip().rstrip(";")
    is_safe, reason = validate_sql_safety(sql_clean)
    if not is_safe:
        raise ValueError(f"SQL safety check failed: {reason}")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_clean))
            cols = list(result.keys())
            rows = [tuple(row) for row in result.fetchmany(MAX_ROWS)]
            return cols, rows
    except Exception as e:
        raise ValueError(f"SQL execution error: {e}")

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Examples:
Q: How many users are there?
A: SELECT COUNT(*) AS total_users FROM users;

Q: Show me the top 5 products by price
A: SELECT name, price FROM products ORDER BY price DESC LIMIT 5;

Q: What are the orders placed in the last 7 days?
A: SELECT * FROM orders WHERE order_date >= NOW() - INTERVAL 7 DAY;

Q: List customers who have never placed an order
A: SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL;
"""

def build_system_prompt(schema: str) -> str:
    return f"""You are an expert MySQL SQL assistant. Your ONLY job is to convert natural language questions into valid MySQL SELECT queries.


DATABASE SCHEMA:
{schema}

{FEW_SHOT_EXAMPLES}

STRICT RULES — follow every rule, no exceptions:
1. Return ONLY the raw SQL query — no markdown, no backticks, no explanation, no preamble
2. ONLY write SELECT statements — never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, PRAGMA, TRUNCATE, REPLACE, ATTACH, DETACH, SAVEPOINT, RELEASE, or any write operation
3. Use exact table and column names from the schema above
4. Always end the query with a semicolon
5. If the question is ambiguous, ask the user for more information
6. If the question cannot be answered with the available schema, respond with exactly: CANNOT_ANSWER
7. Never use subqueries that modify data
8. Never use ATTACH, DETACH, or access sqlite_master directly
9. The final answer should be in proper English with correct meaning
10. IF the user asks what is in the database, what data is available, or what tables exist, DO NOT write SQL. Read the table names from the schema and respond EXACTLY in this format:
META: The database contains company data. Available tables are: [insert table names]. Please ask detailed questions.
11. IF the user asks a general question or greeting (like 'hi', 'who are you', etc) that does not need a database query, respond EXACTLY in this format:
META: Hello! I am your database assistant. Please ask me questions about the data."""

# ─────────────────────────────────────────────
# LLM INTERACTION
# ─────────────────────────────────────────────

def call_ollama(system_prompt: str, user_message: str) -> str:
    """Call Ollama llama3.1 and return the raw text response."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512,
            }
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.error(f"Ollama error: {e}")
        raise RuntimeError(
            f"Cannot reach Ollama. Make sure it's running: `ollama serve`\nError: {e}"
        )

def clean_llm_sql_output(raw: str) -> str:
    """Strip any markdown or wrapper the LLM accidentally adds."""
    cleaned = re.sub(r"`{3}(?:sql)?\s*([\s\S]*?)`{3}", r"\1", raw, flags=re.IGNORECASE)
    cleaned = cleaned.strip("` \n")
    return cleaned

def generate_natural_answer(question: str, sql: str, columns: list, rows: list) -> str:
    """
    Generate a natural language response in proper English using Ollama.
    """
    if not rows:
        return "No results found matching your query."

    data_summary = f"Columns: {', '.join(columns)}\nRows:\n"
    for row in rows[:100]:
        data_summary += f"- {row}\n"
    if len(rows) > 100:
        data_summary += f"... (and {len(rows) - 100} more rows)\n"

    system_prompt = (
        "You are an expert SQLite assistant. Your job is to answer the user's question "
        "by formatting the database results into a clear, proper English response with correct meaning.\n"
        "RULES:\n"
        "1. Do NOT explain the SQL query, write any code, or show any raw JSON.\n"
        "2. If the results contain multiple records, format them nicely using a numbered list, "
        "bullet points, or a markdown table so it looks highly readable in the chat bubble.\n"
        "3. Make sure to display the full list of names/records found in the database results. "
        "Do NOT truncate the list unless there are more than 100 rows.\n"
        "4. Use single newlines for lists (do NOT leave blank lines or empty lines between items) "
        "to keep the text compact and professional."
    )

    user_message = f"""User Question: {question}
Executed SQL: {sql}
Database Results:
{data_summary}

Please provide the direct English answer below:"""

    try:
        answer = call_ollama(system_prompt, user_message)
        return answer
    except Exception as e:
        log.error(f"Error generating natural answer: {e}")
        return f"Executed query returned {len(rows)} rows."

# ─────────────────────────────────────────────
# ROUTER, REPHRASER & GENERAL CHAT LOGIC
# ─────────────────────────────────────────────

def rephrase_question(user_question: str, chat_history: list) -> str:
    """Uses LLM to rephrase a follow-up question based on conversation history."""
    if not chat_history:
        return user_question

    history_text = "\n".join([
        f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
        for msg in chat_history
    ])
    prompt = f"""You are a question rephraser.
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a complete, standalone question.
If the follow-up question is already standalone and clear, return it exactly as is.
DO NOT answer the question. JUST rephrase it.

History:
{history_text}

Follow-up Question: {user_question}

Standalone Question:"""

    try:
        response = call_ollama(prompt, "")
        return response.strip()
    except Exception as e:
        log.error(f"Rephrase error: {e}")
        return user_question

def classify_intent(user_question: str) -> str:
    """Step 1 Router: Decide if the user wants general chat or database data."""
    router_prompt = """You are an intent classifier.
    Classify the user's input into TWO categories:
    1. 'GENERAL' - Greetings (hi, hello), general chat, or asking what you do.
    2. 'SQL' - Asking for specific data, counts, company details, or what tables exist.
    Reply with EXACTLY ONE WORD: either GENERAL or SQL."""
    try:
        response = call_ollama(router_prompt, user_question)
        if "SQL" in response.upper():
            return "SQL"
        return "GENERAL"
    except Exception:
        return "SQL"   # Default fallback

def handle_general_chat(user_question: str) -> str:
    """General conversation handler without database access."""
    chat_prompt = """You are a helpful Database Assistant.
    Keep your answer polite, short, and friendly (1-2 sentences max).
    Remind them they can ask you to fetch data from the database."""
    return call_ollama(chat_prompt, user_question)

# ─────────────────────────────────────────────
# MCP TOOL STUBS (for Ollama tool-calling serialization)
# FIX: Renamed with "tool_" prefix to avoid shadowing the real
#      get_db_schema() function above. Ollama reads the docstrings
#      to understand what each tool does; the prefix doesn't matter.
# ─────────────────────────────────────────────

def tool_list_tables() -> str:
    """
    List all tables in the SQLite database.
    """
    pass

def tool_get_schema() -> str:
    """
    Retrieve the detailed database schema structure including column names, types, primary keys, and row counts.
    """
    pass

def tool_query_db(sql_query: str) -> str:
    """
    Safely execute a read-only SELECT SQL query on the SQLite database.
    Only SELECT statements are permitted. Banned actions like INSERT, UPDATE, DELETE, etc., will be blocked.

    Args:
        sql_query: The raw SQL string (SELECT query) to execute.
    """
    pass

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/status")
async def status_endpoint():
    """Health-check endpoint."""
    return {"ok": True, "message": "service running"}

@app.get("/schema")
async def schema_endpoint():
    """Return the MySQL schema."""
    try:
        return {"schema": get_db_schema()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Process a natural-language question, generate SQL via Ollama, execute it, and return results."""

    # Input length guard — prevents flooding the LLM prompt
    if len(request.user_question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long. Please keep it under 1000 characters.")

    log.info(f"Received question: {request.user_question}")

    # 0. REPHRASE QUESTION USING HISTORY
    actual_question = request.user_question
    if request.chat_history:
        actual_question = rephrase_question(request.user_question, request.chat_history)
        log.info(f"Rephrased question: {actual_question}")

    # 1. INTENT ROUTING
    intent = classify_intent(actual_question)
    log.info(f"Intent Detected: {intent}")

    # 2. GENERAL CHAT BRANCH
    if intent == "GENERAL":
        chat_response = handle_general_chat(actual_question)
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "natural_answer": chat_response,
            "error": None,
            "attempts": 1
        }

    # 3. SQL BRANCH (using MCP Client + Ollama Tool Calling)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os
    import sys

    # Retrieve DB schema from database
    try:
        db_schema = get_db_schema()
    except Exception as e:
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "natural_answer": None,
            "error": f"Failed to retrieve database schema: {str(e)}",
            "attempts": 0
        }

    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    # Instruct Ollama to use the tool to query the database with schema context and examples
    system_prompt = (
        "You are a helpful MySQL database assistant.\n"
        "Your job is to answer the user's question using the database.\n\n"
        f"DATABASE SCHEMA:\n{db_schema}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "Rules:\n"
        "1. Only query tables, columns, and foreign key relationships defined in the DATABASE SCHEMA above. Do not guess or hallucinate table/column names or relationships.\n"
        "2. Call 'tool_query_db' with a valid, read-only SELECT query to get the data you need. If 'tool_query_db' returns an error, analyze the MySQL error message, correct your query, and try calling 'tool_query_db' again. Do not hallucinate data if the query fails.\n"
        "3. Format the final output into a clear, proper English response. If the query returned database records, you MUST print all of those records/results directly inside your response using a markdown table, numbered list, or bullet points. Do NOT say 'listed above' or 'listed below'; you must write the actual data yourself.\n"
        "4. Do NOT include the generated SQL query or any SQL code in your final response. Only output the natural language response and database results.\n"
        "5. Display all retrieved database records (up to 100 rows). Do not truncate the list unless it exceeds 100 records. Keep the response concise, friendly, and structured.\n"
        "6. If the user asks you to modify, insert, delete, or update any data, do NOT try to run a delete or write query, and do NOT try to verify it. "
        "Instead, directly explain to the user that you only have read-only access and cannot perform database modifications.\n"
        "7. Keep the formatting compact. Do NOT add multiple empty/blank lines between paragraphs or list items. Use single newlines to keep the text compact and professional."
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script, DB_URL]
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the MCP Session
                await session.initialize()

                # Retrieve tools from MCP Server to verify connection
                await session.list_tools()

                messages = [
                    {"role": "system", "content": system_prompt},
                ]

                # Add chat history context (limited to last 6 messages / 3 turns to avoid contamination)
                recent_history = request.chat_history[-6:]
                for msg in recent_history:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "bot":
                        role = "assistant"
                    messages.append({"role": role, "content": content})

                # Add current user question
                messages.append({"role": "user", "content": actual_question})

                executed_sql = None

                # Allow up to 5 steps of tool-calling iterations
                for step in range(5):
                    log.info(f"Agent Loop Step {step + 1}...")
                    response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages,
                        tools=[tool_query_db],
                        options={"temperature": 0.1}
                    )

                    # Inspect Ollama response
                    if isinstance(response, dict):
                        assistant_msg = response.get("message", {})
                    else:
                        assistant_msg = response.message

                    # Check for requested tool calls
                    tool_calls = (
                        assistant_msg.get("tool_calls")
                        if isinstance(assistant_msg, dict)
                        else getattr(assistant_msg, "tool_calls", None)
                    )

                    if tool_calls:
                        # Append Ollama's response (tool call requests) to messages history
                        messages.append(assistant_msg)

                        for tool_call in tool_calls:
                            func = (
                                tool_call.get("function")
                                if isinstance(tool_call, dict)
                                else getattr(tool_call, "function", None)
                            )
                            name = (
                                func.get("name")
                                if isinstance(func, dict)
                                else getattr(func, "name", None)
                            )
                            args = (
                                func.get("arguments")
                                if isinstance(func, dict)
                                else getattr(func, "arguments", None)
                            )

                            log.info(f"LLM requested tool call: {name} with args {args}")

                            # FIX: strip "tool_" prefix to get the real MCP server tool name
                            # e.g. "tool_query_db" → "query_db" (matches mcp_server.py)
                            mcp_tool_name = name.replace("tool_", "") if name else name

                            if name == "tool_query_db":
                                executed_sql = args.get("sql_query") if args else None

                            # Execute on MCP Server using the original tool name
                            try:
                                result = await session.call_tool(mcp_tool_name, args or {})
                                tool_result_text = ""
                                if result.content:
                                    tool_result_text = result.content[0].text

                                log.info(f"Tool '{mcp_tool_name}' returned: {tool_result_text[:200]}...")
                            except Exception as e:
                                tool_result_text = f"Error executing tool '{mcp_tool_name}': {str(e)}"
                                log.error(tool_result_text)

                            # Append tool results back to message history
                            messages.append({
                                "role": "tool",
                                "name": name,
                                "content": tool_result_text
                            })
                    else:
                        # No more tool calls — Ollama has produced the final answer
                        final_answer = (
                            assistant_msg.get("content", "")
                            if isinstance(assistant_msg, dict)
                            else getattr(assistant_msg, "content", "")
                        )

                        # Collapse three or more consecutive newlines into a maximum of two
                        final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)


                        return {
                            "sql": executed_sql,
                            "columns": [],
                            "rows": [],
                            "natural_answer": final_answer,
                            "error": None,
                            "attempts": step + 1
                        }

                # Agent loop exhausted without a final answer
                return {
                    "sql": executed_sql,
                    "columns": [],
                    "rows": [],
                    "natural_answer": (
                        "I reached the maximum reasoning steps without producing a final answer. "
                        "Please try rephrasing your question."
                    ),
                    "error": "Reasoning loop limit exceeded",
                    "attempts": 5
                }

    except Exception as e:
        log.error(f"MCP Session failed: {e}")
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "natural_answer": None,
            "error": f"Failed to connect to SQLite MCP Server: {str(e)}",
            "attempts": 0
        }