"""
security/guardrails.py — The StitchGuard validation and enforcement engine.

Contains the centralized `GuardrailsPipeline` class and exposes top-level functions
for input queries, intent routing, SQL compile-time scoping, SQL execution, and output data redaction.

Pre-compiles all regex patterns into module-level compiled objects for maximum throughput.
"""

# ── MODULE TAG: StitchGuard Security Engine ──
import os
import re
import yaml
import asyncio
from typing import Tuple, List, Any, Optional, Set, Dict

from core.config.settings import PROMPTS_DIR
from core.config.logger import get_logger

log = get_logger(__name__)

# ── Refusal & Error Constants ────────────────────────────────────────────────
READ_ONLY_REFUSAL = "I cannot modify the database. I only have read-only access."
RAW_SQL_REFUSAL = (
    "Please ask your question in natural language rather than entering raw SQL queries directly."
)

_DEFAULT_RAW_SQL_PATTERNS = [
    r"^\s*select\s+[\*\w\`\"\,\s]+\s+from\s+[\w\.\`\"]+",
    r"^\s*show\s+(?:tables|columns|databases|schemas|create\s+table|full\s+tables|processlist|variables|status|grants|indexes|index|keys|triggers|events)\b",
    r"^\s*with\s+[\w]+\s+as\s*\(",
    r"^\s*describe\s+[\w\.\`\"]+",
    r"^\s*explain\s+(?:select|insert|update|delete)\b",
    r"^\s*pragma\s+\w+",
]

_DEFAULT_WRITE_PATTERNS = [
    r"\bcrud\b",
    r"\b(?:add|insert|update|delete|remove|change|modify|edit|alter|drop|create|truncate|purge|erase|wipe)\s+(?:the\s+|a\s+|an\s+|any\s+|all\s+|some\s+|our\s+|their\s+|this\s+|that\s+)?(?:new\s+|old\s+)?(?:picklist|inventory|grn|location|warehouse|sku|suid|item|employee|customer|department|project|order|product|supplier|row|record|user|admin|data|table|column|database|db|schema|entry|entries|field|value|operation|action|query|queries|command|task|statement)s?\b",
    r"\bcan\s+you\s+(?:perform|do|execute|run|write|support|help\s+with)?\s*(?:a\s+|an\s+|any\s+|the\s+)?(?:delete|insert|update|remove|modify|create|drop|alter|truncate|write|crud|mutation|editing)\b",
    r"\bcan\s+(?:you|i)\s+(?:delete|insert|update|remove|modify|create|drop|truncate|alter|edit|wipe|purge|erase)\b",
    r"\b(?:how\s+to|how\s+can\s+i|how\s+do\s+i|how\s+can\s+you)\s+(?:delete|insert|update|remove|modify|create|drop|truncate|alter|edit|wipe|purge|erase)\b",
    r"\b(?:perform|do|execute|run|write)\s+(?:a\s+|an\s+|the\s+|any\s+)?(?:delete|insert|update|crud|write|mutation|modification|ddl|dml)\s+(?:operation|action|query|queries|command|task|statement)s?\b",
    r"\b(?:create|drop|truncate|alter)\s+(?:[\w\s]+)?(?:table|database|db|schema|index|view)\b",
    r"\b(?:delete|insert\s+into|update\s+[\w`\"]+\s+set)\b",
]


class GuardrailsPipeline:
    """
    StitchGuard 6-Layer Security & Data Governance Pipeline.

    Enforces:
      Layer 1: Input Gate (Length limit, injection scanning, input PII masking, write intent refusal, raw SQL refusal)
      Layer 2: Intent Scope & Policy Validation
      Layer 3: SQL Compile-Time Domain Scope
      Layer 4: SQL Structural Execution Security (SELECT only, no multi-statements, banned keywords/schemas)
      Layer 5: Database Output Data Redaction (Column dropping & value masking)
      Layer 6: LLM Output Content Safety (Output PII redaction, forbidden phrase filtering, internal term scrubbing)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(PROMPTS_DIR, "config", "guardrails.yml")
        self.config: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        """Reload guardrails.yml and re-compile all regex patterns in memory."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            log.info(f"GuardrailsPipeline: Loaded config from {self.config_path}")
        except Exception as e:
            log.error(f"GuardrailsPipeline: Failed to load config from {self.config_path}: {e}")
            self.config = {}

        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns from config for performance."""
        # 1. Jailbreak Patterns
        jb_patterns = self.config.get("input_safety", {}).get("jailbreak_patterns", [])
        self._jailbreak_patterns = [re.compile(pat, re.IGNORECASE) for pat in jb_patterns]

        # 2. Input PII Patterns
        pii_patterns = self.config.get("input_safety", {}).get("pii_patterns", {})
        self._pii_patterns = {label: re.compile(pat, re.IGNORECASE) for label, pat in pii_patterns.items()}

        # 3. Write Intent Patterns
        self._write_patterns = [re.compile(pat, re.IGNORECASE) for pat in _DEFAULT_WRITE_PATTERNS]

        # 4. Raw SQL Patterns
        self._raw_sql_patterns = [re.compile(pat, re.IGNORECASE) for pat in _DEFAULT_RAW_SQL_PATTERNS]

        # 5. Agent Domain Scopes
        scopes = self.config.get("agent_domain_scopes", {})
        self._normalized_agent_domain_scopes = {
            agent: {tbl.lower() for tbl in config.get("allowed_tables", [])}
            for agent, config in scopes.items()
        }

        # 6. Database Safety - Banned Keywords & Schemas
        banned_kw = self.config.get("database_safety", {}).get("banned_keywords", [])
        self._banned_kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in banned_kw]

        banned_schemas = self.config.get("database_safety", {}).get("banned_schemas", [])
        self._banned_schema_patterns = [(sch, re.compile(rf"\b{re.escape(sch)}\b", re.IGNORECASE)) for sch in banned_schemas]

        # 7. Redact Columns
        redact_cols = self.config.get("output_safety", {}).get("redact_columns", [])
        self._normalized_redact_columns = {re.sub(r'[^a-z0-9]', '', col.lower()) for col in redact_cols}

        # 8. Output Forbidden Patterns
        out_forbidden = self.config.get("output_safety", {}).get("forbidden_output_patterns", [])
        self._output_forbidden_patterns = [re.compile(pat) for pat in out_forbidden]

        # 9. Output Scrub Internal Terms
        scrub_terms = self.config.get("output_safety", {}).get("scrub_internal_terms", [])
        self._output_scrub_terms = [
            (re.compile(entry["pattern"], re.IGNORECASE), entry.get("replacement", ""))
            for entry in scrub_terms
            if "pattern" in entry
        ]

    # ── LAYER 1: Input Gate ──────────────────────────────────────────────────
    def run_layer_1(self, user_question: str) -> Dict[str, Any]:
        """Layer 1 Input Validation (Length, jailbreak, PII, write intent, raw SQL)."""
        max_chars = self.config.get("input_safety", {}).get("max_characters", 1000)

        if len(user_question) > max_chars:
            return {
                "is_safe": False,
                "redacted_question": user_question,
                "detected_pii": [],
                "has_write": False,
                "write_reason": None,
                "has_raw_sql": False,
                "raw_sql_reason": None,
                "unsafe_reason": f"Question too long. Please keep it under {max_chars} characters.",
            }

        for pat in self._jailbreak_patterns:
            if pat.search(user_question):
                log.warning(f"Guardrails L1: Jailbreak pattern matched: {pat.pattern!r}")
                return {
                    "is_safe": False,
                    "redacted_question": user_question,
                    "detected_pii": [],
                    "has_write": False,
                    "write_reason": None,
                    "has_raw_sql": False,
                    "raw_sql_reason": None,
                    "unsafe_reason": "Inappropriate query detected. Please stick to database queries.",
                }

        redacted_q = user_question
        detected_pii: List[str] = []
        for label, pat in self._pii_patterns.items():
            placeholder = f"[REDACTED_{label.upper()}]"
            new_text, count = pat.subn(placeholder, redacted_q)
            if count > 0:
                detected_pii.append(label)
                redacted_q = new_text

        has_write = False
        write_reason = None
        for pat in self._write_patterns:
            if pat.search(user_question):
                log.warning(f"Guardrails L1: Write intent detected: {pat.pattern!r}")
                has_write = True
                write_reason = READ_ONLY_REFUSAL
                break

        has_raw_sql = False
        raw_sql_reason = None
        q_trim = user_question.strip()
        for pat in self._raw_sql_patterns:
            if pat.search(q_trim):
                log.warning(f"Guardrails L1: Raw SQL pattern matched: {pat.pattern!r}")
                has_raw_sql = True
                raw_sql_reason = RAW_SQL_REFUSAL
                break

        return {
            "is_safe": True,
            "redacted_question": redacted_q,
            "detected_pii": detected_pii,
            "has_write": has_write,
            "write_reason": write_reason,
            "has_raw_sql": has_raw_sql,
            "raw_sql_reason": raw_sql_reason,
            "unsafe_reason": None,
        }

    async def run_layer_1_async(self, user_question: str) -> Dict[str, Any]:
        """Async runner for Layer 1 checks."""
        return self.run_layer_1(user_question)

    # ── LAYER 2: Intent Policy ───────────────────────────────────────────────
    def run_layer_2(self, intent: str, question: str) -> Dict[str, Any]:
        valid_intents = {"GENERAL", "WMS_AGENT"}
        if intent not in valid_intents:
            return {"is_valid": False, "intent": intent, "reason": f"Unknown intent '{intent}'"}
        return {"is_valid": True, "intent": intent, "reason": None}

    # ── LAYER 3: SQL Compile-Time Domain Scope ───────────────────────────────
    def run_layer_3(self, sql_query: str, intent: str) -> Dict[str, Any]:
        if not sql_query:
            return {"is_allowed": True, "queried_tables": [], "reason": None}

        allowed_tables = self._normalized_agent_domain_scopes.get(intent, set())
        if not allowed_tables:
            return {"is_allowed": False, "queried_tables": [], "reason": f"SQL generation blocked for {intent}"}

        sql_clean = re.sub(r"--.*?\n|/\*.*?\*/", "", sql_query, flags=re.DOTALL)
        sql_clean = re.sub(r"'[^']*'|\"[^\"]*\"", "", sql_clean)
        queried_tables = list(set(re.findall(r"\b(?:FROM|JOIN)\s+[`'\"#]?([a-zA-Z0-9_]+)[`'\"#]?", sql_clean, re.IGNORECASE)))

        for table in queried_tables:
            if table.lower() not in allowed_tables:
                log.warning(f"Guardrails L3: Unauthorized table access blocked: table={table} intent={intent}")
                return {
                    "is_allowed": False,
                    "queried_tables": queried_tables,
                    "reason": f"Table '{table}' is outside authorized scope for {intent} domain."
                }

        return {"is_allowed": True, "queried_tables": queried_tables, "reason": None}

    # ── LAYER 4: SQL Execution Security Filter ───────────────────────────────
    def run_layer_4(self, sql_query: str) -> Dict[str, Any]:
        if not sql_query:
            return {"is_safe": False, "reason": "Query is empty."}

        safety_cfg = self.config.get("database_safety", {})
        sql_clean = re.sub(r"--.*?\n|/\*.*?\*/", "", sql_query, flags=re.DOTALL).strip()
        sql_upper = sql_clean.upper()

        allowed_prefixes = safety_cfg.get("allowed_prefixes", ["SELECT"])
        if not any(sql_upper.startswith(prefix) for prefix in allowed_prefixes):
            return {"is_safe": False, "reason": "Only read-only queries (e.g., SELECT) are permitted."}

        stripped = sql_clean.rstrip().rstrip(";")
        if ";" in stripped:
            return {"is_safe": False, "reason": "Multiple SQL statements are not allowed."}

        for kw, pat in self._banned_kw_patterns:
            if pat.search(sql_upper):
                log.warning(f"Guardrails L4: Banned keyword detected: {kw}")
                return {"is_safe": False, "reason": f"Forbidden keyword detected in query: {kw}"}

        for schema, pat in self._banned_schema_patterns:
            if pat.search(sql_upper) or f"{schema}." in sql_clean.lower():
                log.warning(f"Guardrails L4: Access to system schema blocked: {schema}")
                return {"is_safe": False, "reason": f"Access to system schema '{schema}' is forbidden."}

        return {"is_safe": True, "reason": None}

    # ── LAYER 5: Output Data Redaction ───────────────────────────────────────
    def run_layer_5(self, db_output: str) -> Dict[str, Any]:
        if not db_output or db_output.startswith("Error") or db_output == "No rows returned.":
            return {"redacted_output": db_output, "redacted_columns": []}

        try:
            from sql_agent.sql_agent import parse_db_result
            columns, rows = parse_db_result(db_output)
        except Exception:
            return {"redacted_output": db_output, "redacted_columns": []}

        if not columns or not rows:
            return {"redacted_output": db_output, "redacted_columns": []}

        redact_indices = set()
        redacted_col_names = []
        for i, col in enumerate(columns):
            col_norm = re.sub(r'[^a-z0-9]', '', col.lower())
            if col_norm in self._normalized_redact_columns or any(rc == col_norm or (len(rc) >= 5 and rc in col_norm) for rc in self._normalized_redact_columns):
                redact_indices.add(i)
                redacted_col_names.append(col)

        if not redact_indices:
            return {"redacted_output": db_output, "redacted_columns": []}

        kept_columns = [col for i, col in enumerate(columns) if i not in redact_indices]
        if not kept_columns:
            log.info(f"Guardrails L5: Dropped all sensitive column(s): {redacted_col_names}")
            return {
                "redacted_output": "The requested columns contain sensitive personal data and were omitted for security.",
                "redacted_columns": redacted_col_names,
            }

        new_rows = []
        for row in rows:
            kept_row = tuple(val for i, val in enumerate(row) if i not in redact_indices)
            new_rows.append(kept_row)

        result_str = f"Columns: {', '.join(kept_columns)}\nRows (up to 100):\n"
        for row in new_rows:
            result_str += f"- {tuple(row)}\n"

        if "... (and more rows exist)" in db_output:
            result_str += "... (and more rows exist)"

        log.info(f"Guardrails L5: Dropped sensitive column(s): {redacted_col_names}")
        return {"redacted_output": result_str, "redacted_columns": redacted_col_names}

    # ── LAYER 6: LLM Output Content Safety ───────────────────────────────────
    def run_layer_6(self, natural_answer: str, sql_used: Optional[str] = None) -> Dict[str, Any]:
        if not natural_answer:
            return {"sanitized_answer": natural_answer, "modifications": []}

        max_chars = self.config.get("output_safety", {}).get("max_response_characters", 5000)
        answer = natural_answer
        modifications: List[str] = []

        for label, pat in self._pii_patterns.items():
            before = answer
            answer = pat.sub("-", answer)
            if answer != before:
                modifications.append(f"Redacted {label} PII")

        for pat in self._output_forbidden_patterns:
            before = answer
            answer = pat.sub("-", answer)
            if answer != before:
                modifications.append("Filtered forbidden output phrase")

        for pat, replacement in self._output_scrub_terms:
            before = answer
            answer = pat.sub(replacement, answer)
            if answer != before:
                modifications.append("Scrubbed internal prompt metadata")

        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(" ", 1)[0] + "\n\n_(Response truncated for brevity.)_"
            modifications.append(f"Truncated response length to {max_chars} chars")

        return {"sanitized_answer": answer, "modifications": modifications}


# Shared singleton instance
guardrails = GuardrailsPipeline()


# Global helper functions
def reload_guardrails_config() -> None:
    guardrails.reload()


def is_safe_prompt(user_question: str) -> bool:
    l1 = guardrails.run_layer_1(user_question)
    return l1["is_safe"] and not l1["has_write"] and not l1["has_raw_sql"]


def redact_pii_from_input(user_question: str) -> tuple[str, list[str]]:
    l1 = guardrails.run_layer_1(user_question)
    return l1["redacted_question"], l1["detected_pii"]


def get_pii_reassurance_message(detected_labels: list[str] = None) -> str:
    labels_str = ", ".join(detected_labels) if detected_labels else "sensitive data"
    return (
        f"Please do not share your sensitive personal data ({labels_str}). "
        f"I have redacted your data for security. It's safe now!"
    )


def has_raw_sql_intent(user_question: str) -> tuple[bool, Optional[str]]:
    l1 = guardrails.run_layer_1(user_question)
    return l1["has_raw_sql"], l1["raw_sql_reason"]


def has_write_intent(user_question: str) -> tuple[bool, Optional[str]]:
    l1 = guardrails.run_layer_1(user_question)
    return l1["has_write"], l1["write_reason"]


def is_db_error(db_output: str) -> bool:
    if not db_output:
        return False
    return (
        db_output.startswith("Error")
        or "Unknown column" in db_output
        or bool(re.search(r"Table '.+' doesn't exist", db_output, re.IGNORECASE))
    )


def extract_tables_from_sql(sql: str) -> list[str]:
    l3 = guardrails.run_layer_3(sql, "WMS_AGENT")
    return l3.get("queried_tables", [])


def validate_sql_domain_scope(sql_query: str, intent: str) -> tuple[bool, Optional[str]]:
    l3 = guardrails.run_layer_3(sql_query, intent)
    return l3["is_allowed"], l3["reason"]


def is_safe_sql_query(sql_query: str) -> tuple[bool, Optional[str]]:
    l4 = guardrails.run_layer_4(sql_query)
    return l4["is_safe"], l4["reason"]


def validate_sql_before_execution(
    sql_query: str,
    intent: str | None = None,
) -> tuple[bool, str | None]:
    if intent:
        is_allowed, scope_err = validate_sql_domain_scope(sql_query, intent)
        if not is_allowed:
            return False, scope_err
    return is_safe_sql_query(sql_query)


def redact_db_output_string(db_output: str) -> str:
    l5 = guardrails.run_layer_5(db_output)
    return l5["redacted_output"]


def sanitize_output(natural_answer: str, sql_used: Optional[str] = None) -> str:
    l6 = guardrails.run_layer_6(natural_answer, sql_used)
    return l6["sanitized_answer"]
