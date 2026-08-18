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
from agents.intent_classifier import classify_intent

log = get_logger(__name__)

# Rule-based greeting/farewell/thanks/praise filter (saves LLM call)
_GENERAL_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|day|night))[!.,\s]*$",
    r"^\s*(bye|goodbye|see\s+you|take\s+care|farewell)[!.,\s]*$",
    r"^\s*(thanks?|thank\s+you|ty|thx|cheers)[!.,\s]*$",
    r"^\s*(ok|okay|got\s+it|noted|sounds?\s+good|great|nice|cool|sure|perfect|awesome|brilliant|wonderful)[!.,\s]*$",
    r"^\s*(yes|no|yep|nope|yeah|nah)[!.,\s]*$",
    r".*\b(you\s+are|you're|who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|who\s+made\s+you|who\s+created\s+you)\b.*",
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
    """Check if the user is saying a simple greeting, thank you, or goodbye.

    Args:
        question (str): The question or message typed by the user.

    Returns:
        bool: True if it is a simple greeting or farewell, False if it is a data question.
    """
    for pattern in _GENERAL_PATTERNS:
        if pattern.match(question.strip()):
            return True
    return False


# ── OPTION A: CONDITIONAL REPHRASING CHECK ─────────────────────────────────────
def needs_rephrasing(question: str, chat_history: list) -> bool:
    """Check if a follow-up question contains words like "it", "them", or "their" that need context from history.

    Args:
        question (str): The current user question.
        chat_history (list): Previous chat messages list.

    Returns:
        bool: True if the question needs to be rewritten with past context, False if it stands on its own.
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


def _is_valid_rephrased_question(rephrased: str, original: str) -> bool:
    """Validate that the rephrased question is a complete, non-dangling question sentence.

    Args:
        rephrased (str): The candidate rephrased string from LLM.
        original (str): The original user question.

    Returns:
        bool: True if valid, False if incomplete or cut off.
    """
    if not rephrased:
        return False
    words = rephrased.strip().split()
    if len(words) < 3:
        return False
    # Check if cut off prematurely (ends with dangling preposition, determiner, or verb)
    dangling_words = {
        "the", "of", "a", "an", "in", "to", "for", "and", "or", "with",
        "at", "by", "from", "on", "about", "into", "over", "after", "is", "are", "were", "was"
    }
    last_word = words[-1].strip("?.,!\"'").lower()
    if last_word in dangling_words:
        return False
    return True


async def _rephrase_question_async(question: str, chat_history: list) -> str:
    """Rewrite a follow-up question into a clear, standalone question using Gemini LLM.

    Args:
        question (str): The raw follow-up user question.
        chat_history (list): Conversation history turns.

    Returns:
        str: Rephrased standalone question string, or original question if rephrasing fails/invalid.
    """
    if not needs_rephrasing(question, chat_history):
        return question

    log.info("Router: [Option A Triggered] Pronoun/Ellipsis detected in multi-turn question. Rephrasing...")
    from core.llm.llm_client import ask_llm_async

    rephrase_system_prompt = (
        "You are an expert query contextualizer.\n"
        "Given a conversation history and a follow-up question containing ambiguous pronouns or short expressions "
        "(e.g., 'it', 'them', 'these', 'how many of them'), rewrite the follow-up question into a complete, clear, standalone question "
        "by replacing pronouns with the explicit subjects/entities mentioned in the history.\n\n"
        "Rules:\n"
        "1. Output ONLY the rephrased standalone question text. Do not add explanations, intros, markdown fences, or quotes.\n"
        "2. Maintain original intent and fix minor typos (e.g., 'deleiveried' -> 'delivered').\n"
        "3. Ensure the rephrased question is a complete sentence ending with a question mark.\n\n"
        "Examples:\n"
        "History:\n"
        "User: How many open picklists do we have?\n"
        "Assistant: There are currently 121 open picklists.\n"
        "Follow-up question: how many of them were deleiveried\n"
        "Rephrased question: How many of the open picklists were delivered?\n\n"
        "History:\n"
        "User: Show me all items in warehouse WH-01\n"
        "Assistant: Found 5 items in WH-01.\n"
        "Follow-up question: list their quantities\n"
        "Rephrased question: List the quantities for items in warehouse WH-01?\n"
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
        if _is_valid_rephrased_question(rephrased_clean, question):
            log.info(f"Router: Rephrased '{question}' -> '{rephrased_clean}'")
            return rephrased_clean
        else:
            log.warning(f"Router: Rephrased output '{rephrased_clean}' failed validation (dangling/incomplete). Falling back to raw question.")
    except Exception as e:
        log.warning(f"Router: Rephrasing failed ({e}), using raw question.")

    return question


# ── STITCHGUARD GUARDRAIL: Layer 2 - Intent Classification & Routing Engine ──
async def rephrase_and_route(question: str, chat_history: list) -> tuple[str, str]:
    """Rewrite follow-up questions to be self-contained and categorize the user's intent.

    Args:
        question (str): User question text.
        chat_history (list): Conversation history list.

    Returns:
        tuple[str, str]: A tuple containing:
            - str: The clean, standalone question text.
            - str: The intent category (`"WMS_AGENT"` for database queries, `"GENERAL"` for chat).
    """
    final_question, intent, _ = await rephrase_and_route_with_score(question, chat_history)
    return final_question, intent


async def rephrase_and_route_with_score(question: str, chat_history: list) -> tuple[str, str, float]:
    """Rewrite follow-up questions and return the intent category along with a confidence score.

    Args:
        question (str): User question text.
        chat_history (list): Conversation history list.

    Returns:
        tuple[str, str, float]: A tuple containing:
            - str: The clean, standalone question text.
            - str: The intent category.
            - float: The confidence score for the classification (0.0 to 1.0).
    """
    if is_simple_conversational_turn(question):
        return question, "GENERAL", 1.0

    final_question = await _rephrase_question_async(question, chat_history)

    from agents.intent_classifier import get_intent_classifier
    clf = get_intent_classifier()
    intent, score = clf.predict_with_score(final_question)
    if score < 0.30 and intent == "WMS_AGENT":
        log.info(f"Router: Low similarity score ({score:.4f} < 0.30) -> Defaulting to GENERAL intent")
        intent = "GENERAL"
    return final_question, intent, score


# Synchronous fallback wrapper if called synchronously
def rephrase_and_route_sync(question: str, chat_history: list) -> tuple[str, str]:
    """Synchronous helper wrapper for rephrase_and_route.

    Args:
        question (str): User question text.
        chat_history (list): Conversation history list.

    Returns:
        tuple[str, str]: Standalone question text and intent category tuple.
    """
    import asyncio
    return asyncio.run(rephrase_and_route(question, chat_history))


# Aliases
route_question = rephrase_and_route
_is_general_by_rule = is_simple_conversational_turn
