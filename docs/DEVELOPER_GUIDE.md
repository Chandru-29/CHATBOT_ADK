# WMS SQL Chatbot — Developer Onboarding & Architecture Guide

---

## Table of Contents

1. [Project Overview & Tech Stack](#1-project-overview--tech-stack)
2. [Directory Structure & Module Breakdown](#2-directory-structure--module-breakdown)
3. [End-to-End Request Lifecycle](#3-end-to-end-request-lifecycle)
4. [How to Run & Test](#4-how-to-run--test)
5. [Guide for Future Contributors](#5-guide-for-future-contributors)
6. [Key Design Decisions](#6-key-design-decisions)

---

## 1. Project Overview & Tech Stack

### What This Project Does

The **WMS SQL Chatbot** is a production-grade, natural-language-to-SQL AI assistant purpose-built for a **Warehouse Management System (WMS)** database. Users type plain English questions ("How many picklists were completed today?") and receive formatted, data-backed answers without ever touching SQL.

The system is built on a **custom local ADK (Agent Development Kit)** — a lightweight, in-process multi-agent orchestration framework modelled after Google's ADK pattern — layered on top of the **Google Gemini 2.5 Flash (`gemini-2.5-flash`)** LLM for SQL generation and the **MCP (Model Context Protocol)** server for safe, isolated SQL execution.

### Core Capabilities

| Capability | Implementation |
|---|---|
| Natural language → SQL | Gemini `gemini-2.5-flash` + multi-step reasoning loop |
| Intent routing (ZERO LLM calls) | Local HuggingFace `all-MiniLM-L6-v2` cosine similarity |
| Schema grounding | VectorRAG via ChromaDB + local `sentence-transformers` |
| Semantic caching | ChromaDB-backed similarity cache (cosine >= 0.99) |
| 6-layer security | StitchGuard pipeline (PII, jailbreak, write, SQL scope, output redaction) |
| Streaming responses | FastAPI `StreamingResponse` + SSE |
| Deterministic formatting | Zero-LLM server-side Markdown formatter |

### Tech Stack

| Layer | Technology | Version / Notes |
|---|---|---|
| **Backend API** | FastAPI | Async, uvicorn ASGI |
| **Frontend UI** | Streamlit | Multi-panel chat interface |
| **LLM** | Google Gemini (`gemini-2.5-flash`) | Via OpenAI-compatible / GenAI client |
| **Embeddings** | Local `sentence-transformers` (`all-MiniLM-L6-v2`) | Fallback: trigram hash vectors |
| **Intent Classifier** | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, no LLM call |
| **Vector Store** | ChromaDB | Persistent on-disk (`chroma_data/`) |
| **SQL Execution** | MCP Server + SQLAlchemy | `mysql+pymysql` dialect |
| **Config / Secrets** | Pydantic + `python-dotenv` | `.env` file |
| **Retry / Resilience** | Tenacity | LLM client retries |
| **Caching** | `cachetools.TTLCache` + ChromaDB SemanticCache | Two-tier |
| **Prompts** | YAML files in `prompts/config/` | Hot-reloadable |
| **Security** | Custom StitchGuard pipeline | 6 layers, regex + rule-based |
| **Audit Logging** | Structured JSON logs | `logs/` directory |

---

## 2. Directory Structure & Module Breakdown

```
E:\CHATBOT_ADK\
|
+-- main.py                         # FastAPI application entry point
+-- app.py                          # Streamlit UI entry point
+-- .env                            # Environment variables (secrets -- never commit)
|
+-- adk/                            # * Local Agent Development Kit (ADK Framework)
|   +-- __init__.py
|   +-- agent.py                    # ADKAgent base class (model binding, tool dispatch, reasoning loop)
|   +-- tool.py                     # ADKTool wrapper (Python fn -> Gemini FunctionDeclaration)
|   +-- middleware.py               # ADKMiddleware (L1-L6 security hooks into agent lifecycle)
|   +-- runner.py                   # ADKRunner (top-level pipeline orchestrator -- main entry)
|
+-- agents/                         # Unified Agents Package
|   +-- __init__.py
|   +-- intent_classifier.py        # Zero-LLM HF embedding intent classifier (WMS_AGENT / GENERAL)
|   +-- router_agent.py             # Intent router + conditional question rephraser (Option A)
|   +-- sql_agent.py                # WMS SQL reasoning loop, deterministic formatter, tool execution
|   +-- general_agent.py           # Conversational fallback (greetings, small talk, write intercepts)
|
+-- api/                            # FastAPI Layer
|   +-- app_factory.py              # create_app() -- FastAPI instance + CORS + router registration
|   +-- models.py                   # Pydantic request/response models (ChatRequest)
|   +-- routes.py                   # REST endpoints: POST /query, GET /status, GET /schema, POST /clear-cache
|
+-- ui/                             # Streamlit Frontend Components
|   +-- chat_window.py              # Main chat message rendering and input box
|   +-- pipeline_panel.py           # Left panel: real-time execution trace viewer
|   +-- right_drawer.py             # Right panel: SQL/table/metadata drawer
|   +-- sidebar.py                  # History, settings, and controls sidebar
|   +-- page_styles.py              # All custom CSS injected via st.markdown
|   +-- session_init.py             # Session state initialization
|
+-- core/                           # Shared Core Infrastructure
|   +-- config/
|   |   +-- settings.py             # ALL environment variables and tunable constants (single source of truth)
|   |   +-- logger.py               # Structured logger factory (get_logger, setup_logging)
|   +-- llm/
|   |   +-- llm_client.py           # Google GenAI SDK singleton client (get_genai_client, ask_llm_async)
|   |   +-- embedder.py             # TextEmbedder: Gemini text-embedding-004 -> trigram fallback + LRU cache
|   +-- cache/
|       +-- cache_manager.py        # Two-tier cache facade: TTLCache (exact) + SemanticCache (fuzzy)
|       +-- semantic_cache.py       # ChromaDB-backed semantic similarity cache (cosine >= 0.99, TTL-aware)
|
+-- db/                             # Database & Vector Storage Layer
|   +-- engine.py                   # SQLAlchemy engine singleton (mysql+pymysql)
|   +-- schema.py                   # get_schema() -- DDL introspection -> formatted text for LLM prompts
|   +-- chromadb.py                 # ChromaDB client + collection factories (table_schemas, semantic_cache)
|   +-- indexer.py                  # index_tables() -- embed table DDL and upsert into ChromaDB
|   +-- table_selector.py           # TableSelector: VectorRAG table narrowing (ChromaDB cosine query)
|   +-- aliases.py                  # Column/table alias resolution helpers
|   +-- similarity.py               # Cosine similarity utility
|
+-- security/                       # StitchGuard 6-Layer Security Engine
|   +-- __init__.py
|   +-- guardrails.py               # GuardrailsPipeline class -- all 6 layers implemented here
|   +-- audit_logger.py             # Structured audit trail (request IDs, pipeline traces, L1 logs)
|   +-- domain_validator.py         # Thin helper for domain scope validation
|
+-- prompts/                        # Prompt Management System
|   +-- loader.py                   # PromptLoader singleton -- loads/caches YAML configs
|   +-- example_selector.py         # Few-shot example selection for SQL prompt construction
|   +-- rule_selector.py            # Domain rule selection from YAML
|   +-- config/                     # YAML prompt/config files (hot-reloadable)
|       +-- agents.yml              # Per-agent system prompts, display names, descriptions
|       +-- examples.yml            # Few-shot SQL Q&A exemplars per intent domain
|       +-- guardrails.yml          # StitchGuard config: PII patterns, jailbreak, banned keywords
|       +-- rules.yml               # Domain-specific SQL generation rules
|
+-- orchestrator/
|   +-- pipeline_orchestrator.py    # Thin delegation shim -> ADKRunner.run_pipeline()
|
+-- mcp_service/                    # MCP (Model Context Protocol) SQL Execution Layer
|   +-- server.py                   # MCP stdio server exposing execute_read_only_query tool
|   +-- session_manager.py          # Manages persistent + per-request MCP client sessions
|   +-- tools.py                    # run_select_query() Python tool declaration
|
+-- chroma_data/                    # ChromaDB on-disk persistence (auto-created, gitignored)
+-- logs/                           # Structured JSON audit logs (auto-created)
```

### Module Responsibility Summary

| Module | Owner Domain | Key Responsibility |
|---|---|---|
| `adk/runner.py` | Orchestration | Top-level pipeline: 6-step sequential workflow, session dispatch, finalization |
| `adk/middleware.py` | Security hooks | Thin adapters bridging ADK lifecycle <-> StitchGuard layers |
| `adk/agent.py` | Agent execution | `ADKAgent.run_step_async()` -- single Gemini API call with tool configuration |
| `adk/tool.py` | Tool abstraction | `ADKTool` -- wraps Python functions into Gemini `FunctionDeclaration` objects |
| `agents/router_agent.py` | Routing | Conditional rephrasing (Option A) + HF embedding classification |
| `agents/intent_classifier.py` | Classification | `EmbeddingIntentClassifier` -- `all-MiniLM-L6-v2` cosine vs. exemplar set |
| `agents/sql_agent.py` | SQL generation | Multi-step Gemini reasoning loop, SQL extraction, L3/L4 validation, L5 redaction, deterministic Markdown |
| `agents/general_agent.py` | Conversational | Write intent intercept, meta explanation guard, LLM fallback reply |
| `security/guardrails.py` | Security | `GuardrailsPipeline` -- all 6 layers, pre-compiled regex, YAML-driven |
| `core/llm/embedder.py` | Embeddings | Gemini `text-embedding-004` + trigram hash fallback + LRU cache |
| `core/cache/semantic_cache.py` | Caching | ChromaDB vector similarity cache (cosine >= 0.99, TTL-aware) |
| `db/table_selector.py` | VectorRAG | ChromaDB cosine query to narrow 14 tables to 2-4 relevant tables |
| `prompts/loader.py` | Config | YAML hot-reload: agent prompts, few-shot examples, domain rules |

---

## 3. End-to-End Request Lifecycle

### High-Level Architecture

```
+--------------------------------------------------------------------------+
|                      STREAMLIT FRONTEND (app.py)                         |
|  chat_window.py | pipeline_panel.py | right_drawer.py | sidebar.py       |
+-------------------------------------+------------------------------------+
                                      |  POST /query  (JSON or stream=True)
                                      v
+--------------------------------------------------------------------------+
|                       FASTAPI BACKEND (main.py)                          |
|  api/routes.py  ->  orchestrator/pipeline_orchestrator.py                |
+-------------------------------------+------------------------------------+
                                      |
                                      v
+--------------------------------------------------------------------------+
|                      ADK RUNNER (adk/runner.py)                          |
|                                                                          |
|  STEP 1 -->  ADKMiddleware.process_l1_input()                            |
|               +-- StitchGuard L1: length, jailbreak, PII redact,         |
|                   write intent block, raw SQL block                      |
|                                                                          |
|  STEP 2 -->  ADKMiddleware.check_l2_cache()                              |
|               +-- Exact TTLCache hit  OR  ChromaDB semantic hit          |
|                   (skip if follow-up question needs_rephrasing=True)     |
|                                                                          |
|  STEP 3 -->  RouterAgent.rephrase_and_route_with_score()                 |
|               +-- Rule fast-path: greeting/farewell -> GENERAL           |
|               +-- Option A: pronoun/ellipsis detected?                   |
|               |    YES -> 1 Gemini call to rephrase question             |
|               |    NO  -> 0 LLM calls                                    |
|               +-- EmbeddingIntentClassifier.predict_with_score()         |
|                   (HuggingFace all-MiniLM-L6-v2, zero LLM calls)        |
|                                                                          |
|  STEP 4 -->  [GENERAL branch]                                            |
|               +-- GeneralAgent.handle_general_chat_async()               |
|                   +-- Write intent intercept -> read-only refusal        |
|                   +-- Gemini conversational reply                        |
|                                                                          |
|  STEP 5 -->  [WMS_AGENT branch] VectorRAG Schema Grounding               |
|               +-- TableSelector.select_tables_with_score()               |
|               |    Embed question -> ChromaDB cosine query               |
|               |    narrow 14 tables to relevant 2-4                     |
|               +-- db/schema.py get_schema(include_tables=focused)        |
|               +-- PromptLoader.get_trimmed_coder_sql_directive()         |
|                   (system prompt = schema + few-shot examples + rules)   |
|                                                                          |
|  STEP 6 -->  WMSSQLAgent Reasoning Loop (sql_agent.run_sql_agent)        |
|               +-- Gemini generate_content (temp=0.0, -> 0.3 on retry)   |
|               +-- Extract SQL (tool_calls -> JSON text -> regex)         |
|               +-- ADKMiddleware.validate_l3_l4_sql()                     |
|               |    L3: Table domain scope check                          |
|               |    L4: SELECT-only, no multi-stmt, banned keywords       |
|               +-- MCP session.call_tool("execute_read_only_query")       |
|               +-- ADKMiddleware.redact_l5_output()                       |
|               |    Drop sensitive columns (passwords, tokens, PII)       |
|               +-- Deterministic Markdown Formatter (zero-LLM)            |
|                   scalar / list / table -> GFM output                   |
|                                                                          |
|  FINALIZE --> ADKMiddleware.sanitize_l6_output()                         |
|               +-- L6: PII re-scan, forbidden phrases, internal scrub     |
|               +-- audit_logger.log_audit_tree()                          |
+--------------------------------------------------------------------------+
```

### Step-by-Step Trace (Annotated with File Locations)

#### STEP 1 — L1 Input Guardrail

```
File: adk/runner.py:67  ->  adk/middleware.py:45  ->  security/guardrails.py:126

is_pass, clean_q, l1_meta = await ADKMiddleware.process_l1_input(original_q)
```

- **Max length check** — default 1000 chars (configurable in `guardrails.yml`)
- **Jailbreak detection** — pre-compiled regex patterns from `guardrails.yml -> input_safety.jailbreak_patterns`
- **PII redaction** — replaces `email`, `phone`, `SSN`, etc. with `[REDACTED_EMAIL]`, etc.
- **Write intent block** — detects "insert", "delete", "update", "CRUD" -> returns `READ_ONLY_REFUSAL`
- **Raw SQL block** — detects direct `SELECT ...`, `DESCRIBE`, `SHOW TABLES` -> asks user to use natural language
- Returns `(True, clean_question, metadata)` or `(False, refusal_message, metadata)`

#### STEP 2 — L2 Cache Interceptor

```
File: adk/runner.py:88  ->  adk/middleware.py:85  ->  core/cache/cache_manager.py

cache_hit, payload, hit_source = ADKMiddleware.check_l2_cache(question, is_follow_up)
```

- **Exact cache** — `cachetools.TTLCache` keyed on sanitized question string (TTL from `settings.py`)
- **Semantic cache** — `SemanticCache.lookup()` -> ChromaDB cosine similarity >= 0.99
- Follow-up questions (`needs_rephrasing=True`) **skip** the cache to avoid stale context
- On hit: re-executes the cached SQL via MCP (fresh DB data), skips LLM entirely

#### STEP 3 — Router Agent (Intent + Rephrasing)

```
File: adk/runner.py:116  ->  agents/router_agent.py:141  ->  agents/intent_classifier.py:78

question, intent, score = await rephrase_and_route_with_score(question, history)
```

Sub-stages:

1. **Rule fast-path** — regex check for greetings/farewells -> instantly returns `GENERAL` (0 LLM calls)
2. **Option A rephrasing** — `needs_rephrasing()` scans for coreference pronouns (`it`, `they`, `this`, etc.) or ellipsis patterns. Only if detected AND history exists -> 1 Gemini call to produce a standalone question
3. **Embedding classification** — `EmbeddingIntentClassifier.predict_with_score()`:
   - Loads `all-MiniLM-L6-v2` locally at startup
   - Pre-computes exemplar embeddings from `prompts/config/examples.yml`
   - Cosine dot-product against exemplar matrix -> best matching `intent in {WMS_AGENT, GENERAL}`
   - Returns `(intent, score)` — no LLM API call

#### STEP 4 — GENERAL Branch

```
File: adk/runner.py:119  ->  agents/general_agent.py:68

reply = await handle_general_chat_async(question, has_write=l1_meta["has_write"])
```

- Write intent double-check -> read-only refusal if detected
- DB-meta keyword guard — if question looks like a mis-routed DB question, prompts user to rephrase
- Otherwise: single Gemini call with the `GENERAL` agent system prompt (from `agents.yml`)
- Result stored in cache (TTL = `API_CACHE_TTL`)

#### STEP 5 — VectorRAG Schema Grounding

```
File: adk/runner.py:145-171  ->  db/table_selector.py  ->  db/schema.py  ->  prompts/loader.py

focused_tables, score = await asyncio.to_thread(table_selector.select_tables_with_score, ...)
schema = get_schema(include_tables=focused_tables)
system_prompt = prompt_loader.get_trimmed_coder_sql_directive(schema, question, focused_tables, ...)
```

- `TableSelector` embeds the user question -> queries ChromaDB `table_schemas` collection
- Returns only tables with cosine similarity >= `VECTOR_RAG_THRESHOLD` (0.70)
- Fallback: all 14 WMS tables if nothing exceeds threshold
- `get_schema()` introspects MySQL DDL -> formatted text: column names, types, PKs, FKs
- `PromptLoader` assembles the final system prompt: schema context + few-shot SQL examples + domain rules

#### STEP 6 — WMSSQLAgent Reasoning Loop

```
File: adk/runner.py:176  ->  mcp_service/session_manager.py  ->  agents/sql_agent.py:367

result = await run_sql_agent(session, messages, system_prompt, question, ...)
```

Inner loop (up to `AGENT_MAX_STEPS=5` iterations):

```
Gemini generate_content (temp=0.0 first step, 0.3 on retry)
|
+-- SQL extracted from response:
|   1. Structured tool_calls  (native Gemini function calling)
|   2. JSON text containing "name": "run_select_query"
|   3. Regex: ```sql ... ``` fence or bare SELECT
|
+-- ADKMiddleware.validate_l3_l4_sql()
|   +-- L3: Verify all FROM/JOIN tables in allowed domain scope
|   +-- L4: Must be SELECT-only, no semicolons, no banned keywords
|
+-- MCP: session.call_tool("execute_read_only_query", {"sql_query": sql})
|
+-- ADKMiddleware.redact_l5_output()
|   +-- Drop columns in guardrails.yml -> output_safety.redact_columns
|
+-- On DB error: feed error + schema guidance back -> continue loop
+-- On success: format_db_result_deterministic(db_output, question)
    +-- 1 row, 1 col  -> "There are currently **42** picklists."
    +-- N rows, 1 col -> bullet list (- item)
    +-- N rows, M col -> GFM table (| Col1 | Col2 | ...)
```

Hallucination guards inside the loop:
- `READ_ONLY_REDIRECT` token in LLM reply -> early exit with read-only message
- `UNRELATED_DOMAIN_REDIRECT` token -> prompt user to split the question
- `METADATA_REDIRECT` on non-metadata query -> force SQL generation retry

#### FINALIZE — L6 Output Sanitizer + Audit

```
File: adk/runner.py:231  ->  adk/middleware.py:111  ->  security/audit_logger.py

result = ADKMiddleware.sanitize_l6_output(result)
log_audit_tree(log, req_id, intent, tables, duration_ms, status)
```

- L6 re-scans natural language answer for PII, forbidden phrases, and internal prompt metadata leakage
- Truncates responses exceeding `output_safety.max_response_characters` (default 5000)
- Writes structured audit log entry: request ID, intent, tables touched, total duration, PASSED/FAILED

### Data Flow Diagram

```
User NL Question
      |
      v
[L1 Guardrail] --UNSAFE--> "Inappropriate query detected."
      |
     SAFE
      |
      v
[L2 Cache Check] --HIT--> Re-execute cached SQL -> Return fresh result
      |
    MISS
      |
      v
[Router] --GENERAL--> GeneralAgent --> Gemini reply --> [Cache] --> [L6] --> Response
      |
   WMS_AGENT
      |
      v
[VectorRAG] -> focused_tables subset of WMS_TABLES (14)
      |
      v
[System Prompt Assembly] <- schema + few-shot examples + rules
      |
      v
[SQL Reasoning Loop] -----------------------------------------------+
      |                                                              |
 Gemini LLM                                                          |
      | SQL                                                          |
      v                                                              |
 [L3 Domain Scope] --BLOCKED--> Error -> retry loop ----------------+
      |
   ALLOWED
      |
      v
 [L4 SQL Safety] --BLOCKED--> Error -> retry loop ------------------+
      |
    SAFE
      |
      v
 MCP: execute_read_only_query(sql)
      |
      v
 [L5 Output Redaction] -> drop sensitive columns
      |
      v
 Deterministic Markdown Formatter (zero-LLM)
      |
      v
 [L6 Answer Sanitizer]
      |
      v
 [Audit Log]
      |
      v
 JSON Response: { sql, columns, rows, natural_answer, attempts, agent_name }
```

---

## 4. How to Run & Test

### Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.11 |
| MySQL / MariaDB | Running locally or remote |
| Google Gemini API Key | `AIza...` |

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd CHATBOT_ADK

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Install sentence-transformers for local HF embeddings
pip install sentence-transformers
```

### Environment Configuration

Fill in your `.env` file:

```bash
# .env -- DO NOT COMMIT THIS FILE

# -- LLM Config --
GEMINI_API_KEY=AIza...your_key_here...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=text-embedding-004

# -- Database Config --
DB_DIALECT=mysql
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=WMS_DB

# -- WMS API (optional external REST endpoint) --
WMS_API_URL=https://your-wms-api/execute-query
WMS_API_TIMEOUT=30
```

All other tunables (cache TTLs, thresholds, history window) are in `core/config/settings.py`.

### Running the FastAPI Backend

```bash
# Development mode (auto-reload on file changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Available endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Health check -> `{"ok": true}` |
| `GET` | `/schema` | Returns full DB schema as formatted text |
| `POST` | `/query` | Main chat endpoint |
| `POST` | `/query?stream=true` | Streaming response (SSE) |
| `POST` | `/clear-cache` | Flush all caches + hot-reload prompts/guardrails |

Request body for `/query`:

```json
{
  "user_question": "How many picklists were completed today?",
  "chat_history": [
    {"role": "user", "content": "Show all warehouses"},
    {"role": "assistant", "content": "Here are the warehouses: ..."}
  ]
}
```

Response body:

```json
{
  "sql": "SELECT COUNT(*) FROM PICKLIST WHERE status = 5 AND DATE(created_at) = CURDATE();",
  "columns": [],
  "rows": [],
  "natural_answer": "There are currently **14** picklists in the database.",
  "error": null,
  "attempts": 2,
  "agent_name": "WMS Assistant"
}
```

### Running the Streamlit Frontend

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The UI communicates with the FastAPI backend at `http://localhost:8000`.

### Quick API Test (curl)

```bash
# Health check
curl http://localhost:8000/status

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_question": "How many active items are in the warehouse?", "chat_history": []}'

# Clear all caches
curl -X POST http://localhost:8000/clear-cache
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_guardrails.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 5. Guide for Future Contributors

### How to Add a New ADK Tool

A **tool** is a Python function that the LLM can call during its reasoning loop. Tools are registered on an `ADKAgent` and automatically exposed to Gemini via `FunctionDeclaration`.

**Step 1 — Define the Python function:**

```python
# agents/tools/my_tool.py
from pydantic import BaseModel

class MyToolParams(BaseModel):
    warehouse_id: str

def get_warehouse_details(warehouse_id: str) -> str:
    """Fetch details for a specific warehouse ID."""
    # ... your implementation
    return f"Warehouse {warehouse_id}: Location A, Capacity 5000"
```

**Step 2 — Wrap it in `ADKTool`:**

```python
# agents/tools/my_tool.py (continued)
from adk.tool import ADKTool

warehouse_tool = ADKTool(
    name="get_warehouse_details",
    description="Fetch details about a specific warehouse given its ID.",
    func=get_warehouse_details,
    parameters_schema=MyToolParams,
)
```

**Step 3 — Attach to an `ADKAgent`:**

```python
from adk.agent import ADKAgent
from agents.tools.my_tool import warehouse_tool

my_agent = ADKAgent(
    name="WMS Extended Agent",
    system_prompt="You are a WMS assistant with access to warehouse detail lookups.",
    tools=[warehouse_tool],
    temperature=0.0,
)

result = await my_agent.run_step_async(question="Get details for warehouse W-001")
```

**Step 4 — Wire the tool into the runner** (if adding to the main SQL pipeline):
- Add the tool's domain scope to `prompts/config/guardrails.yml` under `agent_domain_scopes.WMS_AGENT.allowed_tables`
- Update the system prompt in `prompts/config/agents.yml` to instruct the LLM when to call the tool

---

### How to Modify Guardrails

All guardrail configuration lives in `prompts/config/guardrails.yml` and is loaded at startup by `security/guardrails.py:GuardrailsPipeline`. Changes take effect after calling `POST /clear-cache` (hot-reload) — **no server restart needed**.

**Add a new jailbreak pattern (L1):**

```yaml
# prompts/config/guardrails.yml
input_safety:
  jailbreak_patterns:
    - "ignore (all |previous |above )?(instructions|rules|prompts)"
    - "your new pattern here"          # <- add here
```

**Add a new PII type to redact (L1 & L6):**

```yaml
input_safety:
  pii_patterns:
    email: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
    phone: "\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b"
    new_pii_type: "your_regex_here"    # <- add here
```

**Restrict an agent's table access (L3):**

```yaml
agent_domain_scopes:
  WMS_AGENT:
    allowed_tables:
      - ITEM
      - PICKLIST
      - GRN
      - new_table_name                 # <- add here
```

**Drop a sensitive column from query output (L5):**

```yaml
output_safety:
  redact_columns:
    - password
    - token
    - api_key
    - new_sensitive_column             # <- add here
```

**To add a new guardrail layer in code** (`security/guardrails.py`):

```python
# In GuardrailsPipeline class, add a new method:
def run_layer_7(self, some_input: str) -> Dict[str, Any]:
    """Layer 7: Your custom validation."""
    # ... implementation
    return {"is_safe": True, "reason": None}
```

Then hook it into `adk/middleware.py` as a new static method and call it from `adk/runner.py`.

---

### How to Add a New Intent Domain

By default the system routes to `GENERAL` or `WMS_AGENT`. To add a new domain (e.g., `HR_AGENT`):

**Step 1 — Add exemplars to `prompts/config/examples.yml`:**

```yaml
intents:
  HR_AGENT:
    - "How many employees are in the HR system?"
    - "List all departments"
    - "Show employee attendance for today"
```

**Step 2 — Add domain scope to `prompts/config/guardrails.yml`:**

```yaml
agent_domain_scopes:
  HR_AGENT:
    allowed_tables:
      - EMPLOYEE
      - DEPARTMENT
      - ATTENDANCE
```

**Step 3 — Add agent config to `prompts/config/agents.yml`:**

```yaml
agents:
  HR_AGENT:
    display_name: "HR Assistant"
    description: "You are an HR database assistant."
```

**Step 4 — Update the valid intents set in `agents/router_agent.py:33`:**

```python
_VALID_INTENTS = {"GENERAL", "WMS_AGENT", "HR_AGENT"}  # <- add here
```

**Step 5 — Update `core/config/settings.py`** with the new table pool:

```python
HR_TABLES: frozenset = frozenset({"EMPLOYEE", "DEPARTMENT", "ATTENDANCE"})
```

**Step 6 — Wire into `adk/runner.py`** to dispatch to the new agent when `intent == "HR_AGENT"`.

---

### How to Tweak Agent Instructions (Hot-Reload)

Agent system prompts are loaded from YAML and are **hot-reloadable**. Modify them at:

```
prompts/config/agents.yml    <- system prompt per agent
prompts/config/rules.yml     <- SQL generation rules per domain
prompts/config/examples.yml  <- few-shot Q&A exemplars per domain
```

After editing:
```bash
curl -X POST http://localhost:8000/clear-cache
```

No restart needed. The next request will pick up the new prompts.

---

### How to Change Cache Behavior

All cache TTLs and thresholds are in `core/config/settings.py`:

```python
# Semantic cache: cosine similarity threshold (0.0-1.0)
# Higher = fewer hits, more LLM calls, higher precision
SEMANTIC_CACHE_THRESHOLD: float = 0.99

# How long semantic cache entries survive (seconds)
SEMANTIC_CACHE_TTL: int = 50

# Maximum entries before eviction (LRU)
SEMANTIC_CACHE_MAX: int = 200

# VectorRAG table selection threshold
VECTOR_RAG_THRESHOLD: float = 0.70

# SQL result cache TTL
SQL_CACHE_TTL: int = 30
```

---

### How to Adjust the SQL Agent Reasoning Loop

Key tunables in `core/config/settings.py`:

```python
# Max Gemini API calls per request (increase for complex multi-table queries)
AGENT_MAX_STEPS: int = 5

# How many past messages to include in context (increase for longer conversations)
HISTORY_WINDOW: int = 6
```

Temperature progression in `agents/sql_agent.py`:
- Step 1: `temp=0.0` (deterministic)
- Steps 2+: `temp=0.3` (slight exploration for self-correction)

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero-LLM intent classification** | `all-MiniLM-L6-v2` runs locally — classifies intent in ~5ms with zero API cost. Scales without hitting Gemini rate limits. |
| **Option A Conditional Rephrasing** | Only fires 1 Gemini call when pronouns/ellipsis are detected in multi-turn conversation. 0 LLM calls for all first-turn and self-contained questions. |
| **Deterministic Markdown Formatter** | Replaces LLM-generated natural language summaries with a server-side rule-based formatter. Eliminates hallucination risk on data formatting and reduces token cost. |
| **Two-tier caching (Exact + Semantic)** | Exact cache is O(1) hash lookup. Semantic cache catches rephrased versions of the same question ("count picklists" == "how many picklists"). Reduces LLM calls significantly under repeated workload. |
| **VectorRAG table narrowing** | Sending all 14 tables' DDL to the LLM on every request wastes tokens. ChromaDB cosine search narrows context to only relevant 2-4 tables per query. |
| **MCP for SQL execution** | SQL runs in a sandboxed MCP stdio process, not directly from the FastAPI process. Provides isolation and a clean tool interface matching the Gemini function-calling pattern. |
| **YAML-driven guardrails** | All security rules live in `guardrails.yml` — editable by ops without code changes, hot-reloadable via API. |
| **StitchGuard 6-layer pipeline** | Defense-in-depth: L1 blocks bad input before it reaches the LLM, L3/L4 validate SQL before it touches the DB, L5/L6 redact sensitive data before it reaches the user. |

---

*Generated from source code inspection of `E:\CHATBOT_ADK` — last updated 2026-08-07.*
