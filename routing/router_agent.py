"""
router_agent.py — High-performance Intent Router & Option A Conditional Question Rephraser.

Uses local Hugging Face all-MiniLM-L6-v2 embeddings for ZERO-LLM intent classification:
  - GENERAL
  - WMS_AGENT

Uses Option A Conditional Smart Rephrasing:
  - 0 LLM calls for first turns or self-contained queries (no pronouns/ellipsis).
  - 1 LLM call ONLY when multi-turn follow-up pronouns/ellipsis are detected.
"""

# ── MODULE TAG: Intent Router Agent ──
# ── STITCHGUARD LAYER: L2 (Router Validation & Fallback) ──
import re
import json

from core.config.logger import get_logger
from core.config.settings import HISTORY_WINDOW, REPHRASER_MODEL_NAME, ROUTER_MODEL_NAME
from routing.intent_classifier import classify_intent

log = get_logger(__name__)

# Rule-based greeting/farewell/thanks filter (saves LLM call)
_GENERAL_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|day|night))[!.,\s]*$",
    r"^\s*(bye|goodbye|see\s+you|take\s+care|farewell)[!.,\s]*$",
    r"^\s*(thanks?|thank\s+you|ty|thx|cheers)[!.,\s]*$",
    r"^\s*(ok|okay|got\s+it|noted|sounds?\s+good|great|nice|cool|sure|perfect)[!.,\s]*$",
    r"^\s*(yes|no|yep|nope|yeah|nah)[!.,\s]*$",
]]

_VALID_INTENTS = {"GENERAL", "WMS_AGENT"}

# Option A Coreference Pronouns & Ellipsis Patterns
_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|he|him|his|she|her|hers|they|them|their|theirs|this|that|these|those|former|latter)\b",
    re.IGNORECASE,
)
_ELLIPSIS_PATTERN = re.compile(
    r"^\s*(how\s+many\??|what\s+about\s+.*|and\s+.*|who\s+else\??|where\s+else\??|which\s+ones\??|how\s+much\??)\s*$",
    re.IGNORECASE,
)


# ── STITCHGUARD GUARDRAIL: Layer 2 - Fast-Path Conversational Pre-Filter ──
def is_simple_conversational_turn(question: str) -> bool:
    """Return True if the question is a basic greeting, farewell, thanks or agreement."""
    for pattern in _GENERAL_PATTERNS:
        if pattern.match(question.strip()):
            return True
    return False


# ── OPTION A: CONDITIONAL REPHRASING CHECK ─────────────────────────────────────
def needs_rephrasing(question: str, chat_history: list) -> bool:
    """
    Option A: Return True only if chat_history exists AND the question contains
    coreference pronouns or short elliptical follow-up expressions.
    """
    if not chat_history:
        return False

    q_lower = question.lower().strip()

    # Check if pronoun exists
    pronoun_match = _PRONOUN_PATTERN.search(q_lower)
    if not pronoun_match:
        if _ELLIPSIS_PATTERN.match(q_lower):
            return True
        return False

    # Ignore intra-sentential possessive pronouns if the subject entity noun is in the prompt itself
    found_pronoun = pronoun_match.group(1)
    if found_pronoun in ("their", "theirs", "its", "his", "her", "hers"):
        subject_nouns = {
            "picklist", "picklists", "item", "items", "grn",
            "warehouse", "warehouses", "sku", "skuitem",
            "customer", "customers", "user", "users"
        }
        prompt_words = set(re.findall(r'\w+', q_lower))
        if prompt_words & subject_nouns:
            log.info(f"Router: Intra-sentential pronoun '{found_pronoun}' detected with prompt subject -> Skipping rephrasing")
            return False

    return True


# ── STITCHGUARD GUARDRAIL: Layer 2 - Intent Classification & Routing Engine ──
async def rephrase_and_route(question: str, chat_history: list) -> tuple[str, str]:
    """ 
    Route intent via zero-LLM local embedding model and conditionally rephrase follow-up queries.
    
    Returns:
        (rephrased_question, intent_label)
    """
    # ── STEP 1: CONVERSATIONAL PRE-FILTERING (GREETINGS, THANKS, FAREWELLS) ────────
    if is_simple_conversational_turn(question):
        log.info(f"Router: [Rule Fast-Path] simple greeting detected: '{question[:60]}'")
        return question, "GENERAL"

    # ── STEP 2: OPTION A CONDITIONAL REPHRASING CHECK ──────────────────────────────
    final_question = question
    if needs_rephrasing(question, chat_history):
        log.info("Router: [Option A Triggered] Pronoun/Ellipsis detected in multi-turn question. Rephrasing...")
        from core.llm.llm_client import ask_llm_async

        rephrase_system_prompt = (
            "You are an expert query contextualizer. "
            "Given conversation history and a follow-up question, rewrite the question "
            "to be a clear, standalone question. Do NOT answer the question, output ONLY the rephrased question text."
        )

        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in chat_history[-HISTORY_WINDOW:]
        )
        user_msg = f"Conversation History:\n{history_text}\n\nFollow-up question: {question}"

        try:
            rephrased = await ask_llm_async(
                rephrase_system_prompt,
                user_msg,
                model_name=REPHRASER_MODEL_NAME or ROUTER_MODEL_NAME,
                max_tokens=128
            )
            rephrased_clean = rephrased.strip().strip('"\'')
            if rephrased_clean:
                log.info(f"Router: Rephrased '{question}' -> '{rephrased_clean}'")
                final_question = rephrased_clean
        except Exception as e:
            log.warning(f"Router: Rephrasing failed ({e}), using raw question.")

    # ── STEP 3: ZERO-LLM INTENT CLASSIFICATION ON CONTEXTUALIZED QUESTION ──────────
    intent = classify_intent(final_question)
    log.info(f"Router: [HF Embedding Classifier] intent={intent} for query='{final_question[:60]}'")

    return final_question, intent


async def rephrase_and_route_with_score(question: str, chat_history: list) -> tuple[str, str, float]:
    """Route intent via local embedding model and return score."""
    if is_simple_conversational_turn(question):
        return question, "GENERAL", 1.0

    final_question = question
    if needs_rephrasing(question, chat_history):
        from core.llm.llm_client import ask_llm_async
        rephrase_system_prompt = (
            "You are an expert query contextualizer. "
            "Given conversation history and a follow-up question, rewrite the question "
            "to be a clear, standalone question. Do NOT answer the question, output ONLY the rephrased question text."
        )
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in chat_history[-HISTORY_WINDOW:]
        )
        user_msg = f"Conversation History:\n{history_text}\n\nFollow-up question: {question}"
        try:
            rephrased = await ask_llm_async(
                rephrase_system_prompt,
                user_msg,
                model_name=REPHRASER_MODEL_NAME or ROUTER_MODEL_NAME,
                max_tokens=128
            )
            rephrased_clean = rephrased.strip().strip('"\'')
            if rephrased_clean:
                final_question = rephrased_clean
        except Exception:
            pass

    from routing.intent_classifier import get_intent_classifier
    clf = get_intent_classifier()
    intent, score = clf.predict_with_score(final_question)
    return final_question, intent, score



# Synchronous fallback wrapper if called synchronously
def rephrase_and_route_sync(question: str, chat_history: list) -> tuple[str, str]:
    import asyncio
    return asyncio.run(rephrase_and_route(question, chat_history))


# Aliases
route_question = rephrase_and_route
_is_general_by_rule = is_simple_conversational_turn
