"""
router_agent.py — Combines question rephrasing and intent routing into a single step.
It handles greetings/polite filler without LLM using regex pre-filtering.
"""


# ── MODULE TAG: Intent Router Agent ──
# ── STITCHGUARD LAYER: L2 (Router Validation & Fallback) ──
import re
import json

from config.logger import get_logger

log = get_logger(__name__)

# Rule-based greeting/farewell/thanks filter (saves LLM call)
_GENERAL_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|day|night))[!.,\s]*$",
    r"^\s*(bye|goodbye|see\s+you|take\s+care|farewell)[!.,\s]*$",
    r"^\s*(thanks?|thank\s+you|ty|thx|cheers)[!.,\s]*$",
    r"^\s*(ok|okay|got\s+it|noted|sounds?\s+good|great|nice|cool|sure|perfect)[!.,\s]*$",
    r"^\s*(yes|no|yep|nope|yeah|nah)[!.,\s]*$",
]]

_VALID_INTENTS = {"GENERAL", "HR_AGENT", "SALES_AGENT", "CROSS_DOMAIN"}
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ── INTENT ROUTER: Simple Conversation Classifier Heuristics ──
def is_simple_conversational_turn(question: str) -> bool:
    """Return True if the question is a basic greeting, farewell, thanks or agreement."""
    for pattern in _GENERAL_PATTERNS:
        if pattern.match(question.strip()):
            return True
    return False


# ── INTENT ROUTER: LLM Response JSON Parser ──
def parse_router_json(raw_text: str) -> tuple[str, str] | None:
    """
    Extract the standalone question and intent classification from the LLM JSON response.
    Returns (rephrased_question, intent) or None.
    """
    md_match = _JSON_BLOCK_RE.search(raw_text)
    candidate = md_match.group(1) if md_match else raw_text.strip()

    if not md_match:
        # Fallback: Find the first balanced { } block
        depth, start = 0, -1
        for i, ch in enumerate(raw_text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = raw_text[start: i + 1]
                    break

    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    rephrased = str(obj.get("rephrased_question", "")).strip()
    intent    = str(obj.get("intent", "")).strip().upper()

    if intent not in _VALID_INTENTS:
        return None

    return rephrased or None, intent


# ── INTENT ROUTER: LLM Intent Routing & Question Rephraser ──
def rephrase_and_route(question: str, chat_history: list) -> tuple[str, str]:
    """ 
    Decide intent and rephrase follow-up questions in one step.
    
    Returns:
        (rephrased_question, intent_label)
    """
    # ── STEP 1: CONVERSATIONAL PRE-FILTERING (GREETINGS, THANKS, FAREWELLS) ────────
    if is_simple_conversational_turn(question):
        log.info(f"Router: simple greeting detected: '{question[:60]}'")
        return question, "GENERAL"

    # ── STEP 2: CONTEXT FORMATTING & PROMPT LOADING ────────────────────────────────
    from prompts.prompt_loader import prompt_loader
    from llm.llm_client import ask_llm
    from config.settings import ROUTER_MODEL_NAME

    combined_prompt = prompt_loader.get_rephrase_and_route_prompt()

    # Format history context
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in chat_history[-6:]
        )

    user_msg = (
        f"Conversation history:\n{history_text}\n\n" if history_text else ""
    ) + f"User question: {question}"

    # ── STEP 3: ROUTER LLM CALL WITH LAYER 2 JSON SAFETY RETRIES ──────────────────
    for attempt in range(2):
        try:
            raw = ask_llm(combined_prompt, user_msg, model_name=ROUTER_MODEL_NAME, max_tokens=128)
            parsed = parse_router_json(raw)
            if parsed:
                rephrased, intent = parsed
                final_q = rephrased if rephrased else question
                log.info(f"Router: resolved intent={intent}, stand-alone='{final_q[:80]}' (attempt {attempt + 1})")
                return final_q, intent
            else:
                log.warning(f"Router: JSON parse failed on attempt {attempt + 1} for: {raw[:120]!r}")
        except Exception as e:
            log.error(f"Router: LLM call failed on attempt {attempt + 1} ({e})")

    # ── STEP 4: SAFE FALLBACK ROUTING (ZERO DATABASE PRIVILEGES) ──────────────────
    log.warning("Router: Falling back to GENERAL intent for safety after failed classification.")
    return question, "GENERAL"




# Aliases
route_question = rephrase_and_route
_is_general_by_rule = is_simple_conversational_turn
_parse_llm_json = parse_router_json
