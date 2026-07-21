"""
utils/guardrails.py — The StitchGuard validation and enforcement engine.

Loads rules, patterns, and list configs dynamically from prompts/guardrails.yml
and exposes validation functions for input queries, intent routing, SQL compile-time
scoping, SQL execution, and output data redaction.
"""


# ── MODULE TAG: StitchGuard Security Engine ──
# ── STITCHGUARD LAYER: Core Safety Logic (L1, L3, L4, L5 Checkers) ──
import os
import re
import yaml
from typing import Tuple, List, Any, Optional

from config.settings import PROMPTS_DIR
from config.logger import get_logger

log = get_logger(__name__)

# ── CONFIG: Config cache ──
_config: Optional[dict] = None


# ── UTILS: Load Guardrails Configuration ──
def get_guardrails_config() -> dict:
    """Return the loaded guardrails config. Loads and caches it on first use."""
    global _config
    # ── Return cached config if available ──
    if _config is not None:
        return _config

    # ── Load configuration from disk ──
    path = os.path.join(PROMPTS_DIR, "config", "guardrails.yml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
        log.info("Guardrails: Loaded configuration from guardrails.yml")
    except Exception as e:
        log.error(f"Guardrails: Failed to load guardrails.yml from {path}: {e}")
        _config = {}
    return _config


# ── UTILS: Reload Guardrails Configuration ──
def reload_guardrails_config() -> dict:
    """Evict cache and reload the guardrails configuration from file."""
    global _config
    # ── Clear cache and reload ──
    _config = None
    return get_guardrails_config()


# ── Layer 1: Input Safety Filters ─────────────────────────────────────────────

# ── STITCHGUARD FUNCTION: Layer 1 Injection Check ──
def is_safe_prompt(user_question: str) -> tuple[bool, Optional[str]]:
    """
    Check if the user prompt is safe from injection/jailbreak attempts and length limits.

    Returns:
        (True, None) if safe, or (False, reason_message) if unsafe.
    """
    cfg = get_guardrails_config()
    safety_cfg = cfg.get("input_safety", {})

    # ── L1: Validate character length ──
    max_chars = safety_cfg.get("max_characters", 1000)
    if len(user_question) > max_chars:
        return False, f"Question too long. Please keep it under {max_chars} characters."

    # ── L1: Scan for injection patterns ──
    patterns = safety_cfg.get("jailbreak_patterns", [])
    for pattern in patterns:
        try:
            if re.search(pattern, user_question, re.IGNORECASE):
                log.warning(f"Guardrails: Jailbreak pattern matched: {pattern!r}")
                return False, "Inappropriate query detected. Please stick to database queries."
        except re.error as e:
            log.error(f"Guardrails: Invalid jailbreak regex pattern {pattern!r}: {e}")

    return True, None


# ── STITCHGUARD FUNCTION: Layer 1 PII Masking ──
def redact_pii_from_input(user_question: str) -> str:
    """
    Scan user prompt for PII (SSN, Email, Phone, Credit Cards) and replace with placeholders.
    """
    cfg = get_guardrails_config()
    pii_patterns = cfg.get("input_safety", {}).get("pii_patterns", {})
    
    redacted = user_question
    for label, pattern in pii_patterns.items():
        try:
            placeholder = f"[REDACTED_{label.upper()}]"
            # Perform regex substitution
            redacted = re.sub(pattern, placeholder, redacted)
        except re.error as e:
            log.error(f"Guardrails: Invalid PII regex pattern for {label} ({pattern!r}): {e}")
            
    if redacted != user_question:
        log.info("Guardrails: PII redacted from user question.")
        
    return redacted


# ── Layer 3: SQL Domain Scope Checker ─────────────────────────────────────────

# ── UTILS: SQL Table Extractor Heuristic ──
def extract_tables_from_sql(sql: str) -> list[str]:
    """
    Extract table names from SQL query string, ignoring string literals and comments.
    """
    if not sql:
        return []
    
    # ── Strip SQL comments ──
    sql_clean = re.sub(r"--.*?\n|/\*.*?\*/", "", sql, flags=re.DOTALL)
    
    # ── Strip string literals ──
    sql_clean = re.sub(r"'[^']*'|\"[^\"]*\"", "", sql_clean)
    
    # ── Find FROM/JOIN keywords ──
    matches = re.findall(r"\b(?:FROM|JOIN)\s+[`'\"#]?([a-zA-Z0-9_]+)[`'\"#]?", sql_clean, re.IGNORECASE)
    return list(set(matches))


# ── STITCHGUARD FUNCTION: Layer 3 Table Scope Validation ──
def validate_sql_domain_scope(sql_query: str, intent: str) -> tuple[bool, Optional[str]]:
    """
    Ensure the SQL query only targets tables within the authorized domain of the active intent.
    If intent is CROSS_DOMAIN, it can access the union of all agent tables.

    Returns:
        (True, None) if scope is valid, or (False, error_description) if out of scope.
    """
    if not sql_query:
        return True, None

    cfg = get_guardrails_config()
    scopes = cfg.get("agent_domain_scopes", {})
    
    # ── L3: Resolve allowed tables ──
    allowed_tables = set()
    if intent == "CROSS_DOMAIN":
        # CROSS_DOMAIN can query tables across all agent scopes
        for scope_cfg in scopes.values():
            allowed_tables.update(scope_cfg.get("allowed_tables", []))
    else:
        # Standard agent scope
        agent_scope = scopes.get(intent, {})
        allowed_tables.update(agent_scope.get("allowed_tables", []))

    # ── L3: Block SQL if no tables allowed ──
    if not allowed_tables:
        return False, f"SQL generation is blocked for {intent} intent."

    queried_tables = extract_tables_from_sql(sql_query)
    for table in queried_tables:
        if table.lower() not in [t.lower() for t in allowed_tables]:
            log.warning(f"Guardrails: Unauthorized table access blocked: table={table} intent={intent}")
            return False, f"Table '{table}' is outside your authorized scope for the {intent} domain."

    return True, None


# ── Layer 4: SQL Execution Security Filter ────────────────────────────────────

# ── STITCHGUARD FUNCTION: Layer 4 DML/DDL Write Blockers ──
def is_safe_sql_query(sql_query: str) -> tuple[bool, Optional[str]]:
    """
    Examine SQL query structure to ensure it's read-only, single statement,
    and does not access forbidden keywords or schemas.

    Returns:
        (True, None) if query passes all filters, or (False, error_description) if blocked.
    """
    if not sql_query:
        return False, "Query is empty."

    cfg = get_guardrails_config()
    safety_cfg = cfg.get("database_safety", {})

    # ── L4: Clean SQL string ──
    sql_clean = re.sub(r"--.*?\n|/\*.*?\*/", "", sql_query, flags=re.DOTALL).strip()
    sql_upper = sql_clean.upper()

    # ── L4: Validate SQL prefix ──
    allowed_prefixes = safety_cfg.get("allowed_prefixes", ["SELECT"])
    starts_valid = any(sql_upper.startswith(prefix) for prefix in allowed_prefixes)
    if not starts_valid:
        return False, "Only read-only queries (e.g., SELECT) are permitted."

    # ── L4: Block multiple statements ──
    # Remove trailing semicolons first
    stripped = sql_clean.rstrip().rstrip(";")
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed."

    # ── L4: Block forbidden keywords ──
    banned_keywords = safety_cfg.get("banned_keywords", [])
    for keyword in banned_keywords:
        # Use word bounds to prevent false positives (e.g., matching "UPDATE" in "update_timestamp" column)
        if re.search(rf"\b{keyword}\b", sql_upper):
            log.warning(f"Guardrails: Forbidden keyword detected: {keyword}")
            return False, f"Forbidden keyword detected in query: {keyword}"

    # ── L4: Block system schemas ──
    banned_schemas = safety_cfg.get("banned_schemas", [])
    for schema in banned_schemas:
        if re.search(rf"\b{schema}\b", sql_upper) or f"{schema}." in sql_clean.lower():
            log.warning(f"Guardrails: Access to system schema blocked: {schema}")
            return False, f"Access to system schema '{schema}' is forbidden."

    return True, None


# ── Layer 5: Output Data Redaction ────────────────────────────────────────────

# ── STITCHGUARD FUNCTION: Layer 5 Output Column Filtering ──
def redact_sensitive_columns(columns: list[str], rows: list[tuple]) -> tuple[list[str], list[tuple]]:
    """
    Scan the column headers. If any column matches output redaction rules,
    replace its values in all rows with '[REDACTED]'.
    """
    if not columns or not rows:
        return columns, rows

    cfg = get_guardrails_config()
    redact_cols = cfg.get("output_safety", {}).get("redact_columns", [])
    redact_cols_lower = [c.lower() for c in redact_cols]

    # ── L5: Match column indexes for redaction ──
    redact_indices = []
    for i, col in enumerate(columns):
        if col.lower() in redact_cols_lower:
            redact_indices.append(i)

    if not redact_indices:
        return columns, rows

    # ── L5: Redact values in matched columns ──
    new_rows = []
    for row in rows:
        row_list = list(row)
        for idx in redact_indices:
            row_list[idx] = "[REDACTED]"
        new_rows.append(tuple(row_list))

    log.info(f"Guardrails: Redacted sensitive column(s) at indices: {redact_indices}")
    return columns, new_rows


# ── STITCHGUARD FUNCTION: Layer 5 Output Parsing and Redaction ──
def redact_db_output_string(db_output: str) -> str:
    """
    Parse a db_output string, redact any sensitive columns,
    and reconstruct it back to the original string representation.
    """
    if not db_output or db_output.startswith("Error") or db_output == "No rows returned.":
        return db_output

    # Import locally to avoid circular dependencies
    from agents.sql_agent import parse_db_result
    columns, rows = parse_db_result(db_output)
    if not columns or not rows:
        return db_output

    # Run redaction
    redacted_columns, redacted_rows = redact_sensitive_columns(columns, rows)
    
    # Rebuild the string
    result_str = f"Columns: {', '.join(redacted_columns)}\nRows (up to 100):\n"
    for row in redacted_rows:
        result_str += f"- {tuple(row)}\n"
    
    # ── L5: Keep rows count indicator ──
    if "... (and more rows exist)" in db_output:
        result_str += "... (and more rows exist)"
        
    return result_str

 