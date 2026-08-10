"""
llm_client.py — Google Gemini API Client Wrapper using google-genai SDK with Tenacity retries.

Every module that needs to ask Gemini a question calls ask_llm() or ask_llm_async() from here.
Includes resilience retries for network/API rate-limit disruptions.
"""

# ── MODULE TAG: Google Gemini LLM Client ──
import os
import logging
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from core.config.settings import GEMINI_API_KEY, GEMINI_MODEL
from core.config.logger import get_logger

log = get_logger(__name__)

# Single shared persistent Gemini client
_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Return the shared persistent Google GenAI Client instance."""
    global _client
    if _client is None:
        api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_KEY")
        if not api_key:
            log.warning("GEMINI_API_KEY not found in environment. Using placeholder key 'MOCK_GEMINI_API_KEY'. Please set GEMINI_API_KEY in .env")
            api_key = "MOCK_GEMINI_API_KEY"
        _client = genai.Client(api_key=api_key)
    return _client


# Backwards compatibility helper
def get_async_client() -> genai.Client:
    """Return the shared persistent GenAI client (supports .aio for async calls)."""
    return get_genai_client()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.2, max=2.0),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _generate_with_retry(model: str, system_prompt: str, user_msg: str, max_tokens: int, temperature: float = 0.0):
    """Execute synchronous client.models.generate_content with Tenacity exponential retries."""
    client = get_genai_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return client.models.generate_content(
        model=model,
        contents=user_msg,
        config=config,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.2, max=2.0),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
async def _async_generate_with_retry(model: str, system_prompt: str, user_msg: str, max_tokens: int, temperature: float = 0.0):
    """Execute async client.aio.models.generate_content with Tenacity exponential retries."""
    client = get_genai_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return await client.aio.models.generate_content(
        model=model,
        contents=user_msg,
        config=config,
    )


def ask_llm(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Send a message to Google Gemini API with automatic retries and return reply text."""
    model = model_name or GEMINI_MODEL
    try:
        resp = _generate_with_retry(model, system_prompt, user_msg, max_tokens, temperature)
        return (resp.text or "").strip()
    except Exception as e:
        log.error(f"Gemini API sync call failed after retries: {e}")
        raise RuntimeError(f"Error calling Google Gemini API ({model}): {e}")


async def ask_llm_async(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Asynchronously send a message to Google Gemini API with automatic retries and return reply text."""
    model = model_name or GEMINI_MODEL
    try:
        resp = await _async_generate_with_retry(model, system_prompt, user_msg, max_tokens, temperature)
        return (resp.text or "").strip()
    except Exception as e:
        log.error(f"Gemini API async call failed after retries: {e}")
        raise RuntimeError(f"Error calling Google Gemini API async ({model}): {e}")
