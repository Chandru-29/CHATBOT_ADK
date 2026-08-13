# Codebase Architecture Map & File Cheat-Sheet (`MAP.md`)

**Project:** WMS SQL Chatbot (`CHATBOT_ADK`)  
**LLM:** Google Gemini 2.5 Flash (`gemini-2.5-flash`)  
**Framework:** Custom Local ADK (Agent Development Kit) + FastAPI + Streamlit  

---

## 1. Core & API Layer

- [main.py](file:///e:/CHATBOT_ADK/main.py) — FastAPI web application entry point exposing `/query`, `/status`, `/schema`, and `/clear-cache`.
- [api/routes.py](file:///e:/CHATBOT_ADK/api/routes.py) — REST API router defining query, status, schema, and cache clearing endpoints.
- [api/app_factory.py](file:///e:/CHATBOT_ADK/api/app_factory.py) — FastAPI application factory configuring CORS, middleware, and route registrations.
- [api/models.py](file:///e:/CHATBOT_ADK/api/models.py) — Pydantic request and response data models (`ChatRequest`).
- [core/config/settings.py](file:///e:/CHATBOT_ADK/core/config/settings.py) — Single source of truth for environment variables, tunable constants, thresholds, and TTL values.
- [core/config/logger.py](file:///e:/CHATBOT_ADK/core/config/logger.py) — Centralized structured JSON/console logging factory.
- [core/llm/llm_client.py](file:///e:/CHATBOT_ADK/core/llm/llm_client.py) — Google GenAI SDK client singleton with connection pooling and async completion helpers.
- [core/llm/embedder.py](file:///e:/CHATBOT_ADK/core/llm/embedder.py) — Text embedder using Gemini `text-embedding-004` with trigram fallback and LRU cache.
- [core/llm/async_embedder.py](file:///e:/CHATBOT_ADK/core/llm/async_embedder.py) — Non-blocking async embedding generator to prevent thread pool starvation.
- [core/llm/llm_throttle.py](file:///e:/CHATBOT_ADK/core/llm/llm_throttle.py) — Concurrency governor and rate limiter for Gemini API calls.
- [core/request_deduplicator.py](file:///e:/CHATBOT_ADK/core/request_deduplicator.py) — In-flight request deduplicator merging concurrent identical queries into a single execution.

---

## 2. ADK Framework & Multi-Agent Engine

- [adk/runner.py](file:///e:/CHATBOT_ADK/adk/runner.py) — ADK pipeline orchestrator executing the 6-stage request lifecycle, security hooks, and history compaction.
- [adk/agent.py](file:///e:/CHATBOT_ADK/adk/agent.py) — Base `ADKAgent` class managing model binding, tool dispatch, and step execution loops.
- [adk/tool.py](file:///e:/CHATBOT_ADK/adk/tool.py) — `ADKTool` wrapper converting Python functions into Gemini function declarations.
- [adk/middleware.py](file:///e:/CHATBOT_ADK/adk/middleware.py) — ADK middleware lifecycle adapters bridging execution steps to StitchGuard security layers.
- [orchestrator/pipeline_orchestrator.py](file:///e:/CHATBOT_ADK/orchestrator/pipeline_orchestrator.py) — API delegation layer forwarding incoming HTTP requests to `ADKRunner`.
- [agents/router_agent.py](file:///e:/CHATBOT_ADK/agents/router_agent.py) — Option A coreference question rephraser and intent router.
- [agents/intent_classifier.py](file:///e:/CHATBOT_ADK/agents/intent_classifier.py) — Zero-LLM HuggingFace `all-MiniLM-L6-v2` embedding intent classifier (`WMS_AGENT` vs `GENERAL`).
- [agents/sql_agent.py](file:///e:/CHATBOT_ADK/agents/sql_agent.py) — WMS SQL reasoning loop, SQL extraction, MCP tool execution, compact history tag builder, and deterministic Markdown formatter.
- [agents/general_agent.py](file:///e:/CHATBOT_ADK/agents/general_agent.py) — Conversational agent for greetings, meta-questions, and read-only write-intent refusals.

---

## 3. Database & VectorRAG Grounding

- [db/schema.py](file:///e:/CHATBOT_ADK/db/schema.py) — Database DDL introspection engine formatting schema into Compact Schema Notation (CSN) (`Table: NAME(col:TYPE)`).
- [db/table_selector.py](file:///e:/CHATBOT_ADK/db/table_selector.py) — VectorRAG search narrowing 14 WMS tables down to 2-4 query-relevant tables via ChromaDB cosine similarity.
- [db/indexer.py](file:///e:/CHATBOT_ADK/db/indexer.py) — ChromaDB vector table indexer embedding CSN table metadata into the `table_schemas` collection.
- [db/engine.py](file:///e:/CHATBOT_ADK/db/engine.py) — SQLAlchemy MySQL database connection engine singleton (`mysql+pymysql`).
- [db/chromadb.py](file:///e:/CHATBOT_ADK/db/chromadb.py) — ChromaDB client factory managing vector collections for table schemas and semantic cache.
- [db/aliases.py](file:///e:/CHATBOT_ADK/db/aliases.py) — Table and column synonym dictionary mappings for schema grounding.
- [db/similarity.py](file:///e:/CHATBOT_ADK/db/similarity.py) — Vector cosine similarity mathematical utility function.

---

## 4. Memory, Caching & Security Engine

- [redis_store/session_store.py](file:///e:/CHATBOT_ADK/redis_store/session_store.py) — Redis-backed chat session thread manager storing multi-turn history with 24-hour TTL expiration.
- [redis_store/client.py](file:///e:/CHATBOT_ADK/redis_store/client.py) — Centralized async Redis connection manager with automatic local RAM fallback mode.
- [redis_store/exact_cache.py](file:///e:/CHATBOT_ADK/redis_store/exact_cache.py) — Redis exact query response string cache with local TTLCache fallback.
- [redis_store/semantic_cache.py](file:///e:/CHATBOT_ADK/redis_store/semantic_cache.py) — Redis/ChromaDB vector similarity cache (cosine >= 0.99) bypassing LLM execution on hits.
- [redis_store/rate_limiter.py](file:///e:/CHATBOT_ADK/redis_store/rate_limiter.py) — Redis sliding window API rate limiter per user/IP.
- [core/cache/cache_manager.py](file:///e:/CHATBOT_ADK/core/cache/cache_manager.py) — Unified two-tier caching facade coordinating local TTLCache and ChromaDB/Redis caches.
- [security/guardrails.py](file:///e:/CHATBOT_ADK/security/guardrails.py) — StitchGuard 6-layer security engine enforcing input safety, PII redaction, write blocking, SQL domain scope, and output sanitization.
- [security/audit_logger.py](file:///e:/CHATBOT_ADK/security/audit_logger.py) — Structured audit trail logger generating request IDs and pipeline execution log trees.
- [security/domain_validator.py](file:///e:/CHATBOT_ADK/security/domain_validator.py) — SQL table domain access scope validation helper.

---

## 5. Prompt System & Token Optimization

- [prompts/loader.py](file:///e:/CHATBOT_ADK/prompts/loader.py) — YAML prompt loader managing Gemini context caching static prefix invariance and adaptive prompt assembly.
- [prompts/example_selector.py](file:///e:/CHATBOT_ADK/prompts/example_selector.py) — Vector RAG few-shot exemplar selector with adaptive dynamic Top-K complexity classification (K=1, K=3, K=5).
- [prompts/rule_selector.py](file:///e:/CHATBOT_ADK/prompts/rule_selector.py) — Dynamic rule selector filtering mandatory core rules vs active table scenario rules.
- [prompts/config/instructions.yml](file:///e:/CHATBOT_ADK/prompts/config/instructions.yml) — System prompt directives for WMS SQL agent, router, rephraser, and shared SQL rules.
- [prompts/config/examples.yml](file:///e:/CHATBOT_ADK/prompts/config/examples.yml) — Few-shot SQL Q&A exemplars organized by intent domain.
- [prompts/config/guardrails.yml](file:///e:/CHATBOT_ADK/prompts/config/guardrails.yml) — StitchGuard security configuration storing PII regex patterns, banned keywords, and domain access scopes.
- [prompts/config/rules.yml](file:///e:/CHATBOT_ADK/prompts/config/rules.yml) — Domain-specific SQL generation rules.

---

## 6. MCP Execution Service

- [mcp_service/server.py](file:///e:/CHATBOT_ADK/mcp_service/server.py) — MCP stdio server exposing sandboxed `execute_read_only_query` tool to LLM sessions.
- [mcp_service/session_manager.py](file:///e:/CHATBOT_ADK/mcp_service/session_manager.py) — Client session manager executing queries via MCP sessions or cached SQL execution.
- [mcp_service/mcp_session_pool.py](file:///e:/CHATBOT_ADK/mcp_service/mcp_session_pool.py) — Persistent MCP process session pool eliminating process startup overhead on concurrent queries.
- [mcp_service/tools.py](file:///e:/CHATBOT_ADK/mcp_service/tools.py) — MCP tool declarations defining `run_select_query` parameter schemas.

---

## 7. UI Layer (Streamlit)

- [app.py](file:///e:/CHATBOT_ADK/app.py) — Streamlit frontend user interface entry point.
- [ui/chat_window.py](file:///e:/CHATBOT_ADK/ui/chat_window.py) — Main chat window rendering message history, user input, and markdown response tables.
- [ui/pipeline_panel.py](file:///e:/CHATBOT_ADK/ui/pipeline_panel.py) — Real-time left drawer displaying live step-by-step pipeline execution traces.
- [ui/right_drawer.py](file:///e:/CHATBOT_ADK/ui/right_drawer.py) — Right side drawer displaying generated SQL query, raw execution table rows, and column metadata.
- [ui/sidebar.py](file:///e:/CHATBOT_ADK/ui/sidebar.py) — Left control sidebar for chat history threads, session clearing, and system status indicators.
- [ui/page_styles.py](file:///e:/CHATBOT_ADK/ui/page_styles.py) — Custom CSS stylesheet injected into Streamlit UI.
- [ui/session_init.py](file:///e:/CHATBOT_ADK/ui/session_init.py) — Streamlit session state initialization helper.

---

## 8. Testing & Automation Suite

- [tests/test_compact_schema.py](file:///e:/CHATBOT_ADK/tests/test_compact_schema.py) — Unit tests for Proposal 1 Compact Schema Notation (CSN) schema generation and token reduction.
- [tests/test_context_caching_prefix.py](file:///e:/CHATBOT_ADK/tests/test_context_caching_prefix.py) — Unit tests for Proposal 2 Gemini 2.5 Flash static prefix prompt invariance.
- [tests/test_adaptive_example_selector.py](file:///e:/CHATBOT_ADK/tests/test_adaptive_example_selector.py) — Unit tests for Proposal 3 Adaptive Dynamic Top-K exemplar selection (K=1, K=3, K=5).
- [tests/test_history_compaction.py](file:///e:/CHATBOT_ADK/tests/test_history_compaction.py) — Unit tests for Proposal 4 Enhanced Chat History Compaction (`[EXECUTED_SQL: ... | RESULT: ...]` tags).
- [tests/test_redis_integration.py](file:///e:/CHATBOT_ADK/tests/test_redis_integration.py) — Integration tests for Redis session store, exact cache, semantic cache, and rate limiter.
- [tests/locustfile.py](file:///e:/CHATBOT_ADK/tests/locustfile.py) — Locust performance load testing script simulating concurrent chat users.
- [pytest.ini](file:///e:/CHATBOT_ADK/pytest.ini) — Pytest configuration file enabling automatic local package import resolution (`pythonpath = .`).
- [run_server.ps1](file:///e:/CHATBOT_ADK/run_server.ps1) — PowerShell launcher script spinning up FastAPI backend and Streamlit UI servers concurrently.

---

## 🚀 Quick Developer Commands

```powershell
# 1. Run FastAPI Backend API (port 8000)
uvicorn main:app --reload --port 8000

# 2. Run Streamlit UI (port 8501)
streamlit run app.py

# 3. Run Both Servers Concurrently
.\run_server.ps1

# 4. Run Token Optimization & Feature Unit Tests
pytest tests/test_compact_schema.py tests/test_context_caching_prefix.py tests/test_adaptive_example_selector.py tests/test_history_compaction.py -v

# 5. Clear Caches & Hot-Reload Prompts/Guardrails via API
curl -X POST http://localhost:8000/clear-cache
```
