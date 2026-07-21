"""
llm_client.py — Thin wrapper around ollama.chat().

Every module that needs to ask the LLM a question calls ask_llm() from here.
This keeps the Ollama API call in one place so it's easy to swap models
or add retry logic without touching the calling code.
"""

import ollama
from config.settings import MODEL_NAME
from config.logger import get_logger

log = get_logger(__name__)


def ask_llm(system_prompt: str, user_msg: str, model_name: str = None, max_tokens: int = 512) -> str:
    """
    Send a message to Ollama and return the reply text.

    Args:
        system_prompt: The system role instruction for the LLM.
        user_msg:      The user's message to process.
        model_name:    Optional custom model name. Defaults to config.settings.MODEL_NAME.
        max_tokens:    Optional maximum tokens to generate. Defaults to 512.

    Returns:
        The LLM's text response, stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    try:
        model = model_name or MODEL_NAME
        reply = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            options={
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
        )
        if isinstance(reply, dict):
            return reply.get("message", {}).get("content", "").strip()
        else:
            return getattr(reply.message, "content", "").strip()
    except Exception as e:
        log.error(f"Ollama error: {e}")
        raise RuntimeError(
            f"Cannot reach Ollama. Make sure it's running: `ollama serve`\nError: {e}"
        )
