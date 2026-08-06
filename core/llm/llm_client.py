"""
llm_client.py — Wrapper around ollama.chat() with Tenacity exponential retries.

Every module that needs to ask the LLM a question calls ask_llm() from here.
Includes automatic resilience retries for transient Ollama service disruptions.
"""

# ── MODULE TAG: Ollama LLM Client ──
import logging
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from core.config.settings import MODEL_NAME
from core.config.logger import get_logger

log = get_logger(__name__)

# Persistent async client singleton
_async_client = ollama.AsyncClient()


def get_async_client() -> ollama.AsyncClient:
    """Return the shared persistent Ollama AsyncClient instance."""
    return _async_client


def _extract_reply_content(reply) -> str:
    """Extract clean message content string from dict or object responses."""
    if isinstance(reply, dict):
        return reply.get("message", {}).get("content", "").strip()
    return getattr(getattr(reply, "message", None), "content", "").strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.2, max=2.0),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _chat_with_retry(model: str, messages: list, options: dict):
    """Execute raw ollama.chat with exponential backoff retries via Tenacity."""
    return ollama.chat(model=model, messages=messages, options=options)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.2, max=2.0),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
async def _async_chat_with_retry(model: str, messages: list, options: dict):
    """Execute async _async_client.chat with exponential backoff retries via Tenacity."""
    return await _async_client.chat(model=model, messages=messages, options=options)


def _build_chat_payload(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 512) -> tuple[str, list[dict], dict]:
    """Construct standard model, messages, and options parameters."""
    model = model_name or MODEL_NAME
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]
    options = {
        "temperature": 0,
        "top_p": 0.9,
        "num_predict": max_tokens,
    }
    return model, messages, options


def ask_llm(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 512) -> str:
    """Send a message to Ollama with automatic resilience retries and return reply text."""
    model, messages, options = _build_chat_payload(system_prompt, user_msg, model_name, max_tokens)

    try:
        reply = _chat_with_retry(model, messages, options)
        return _extract_reply_content(reply)
    except Exception as e:
        log.error(f"Ollama sync API call failed after 3 retries: {e}")
        raise RuntimeError(
            f"Cannot reach Ollama. Make sure it's running: `ollama serve`\nError: {e}"
        )


async def ask_llm_async(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 512) -> str:
    """Asynchronously send a message to Ollama with automatic resilience retries and return reply text."""
    model, messages, options = _build_chat_payload(system_prompt, user_msg, model_name, max_tokens)

    try:
        reply = await _async_chat_with_retry(model, messages, options)
        return _extract_reply_content(reply)
    except Exception as e:
        log.error(f"Ollama async API call failed after 3 retries: {e}")
        raise RuntimeError(
            f"Cannot reach Ollama. Make sure it's running: `ollama serve`\nError: {e}"
        )
