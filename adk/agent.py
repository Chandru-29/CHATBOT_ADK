"""
agent.py — ADK Agent Core Class using Gemini API.

Defines declarative Agents with system instructions, model binding, attached ADK Tools,
and reasoning step execution loops.
"""

# ── MODULE TAG: ADK Agent Engine ──
import re
import json
from typing import List, Dict, Any, Optional

from core.config.settings import MODEL_NAME, AGENT_MAX_STEPS
from core.config.logger import get_logger
from core.llm.llm_client import get_llm_async_client
from adk.tool import ADKTool

log = get_logger(__name__)


class ADKAgent:
    """Standardized ADK Agent entity encapsulating model parameters and instructions.

    Attributes:
        name (str): Agent display identifier.
        system_prompt (str): Base system prompt instructions.
        tools (List[ADKTool]): List of attached tool declarations.
        tool_map (Dict[str, ADKTool]): Map of tool names to tool objects.
        model_name (str): Gemini model identifier string.
        temperature (float): Sampling temperature value.
        max_steps (int): Maximum reasoning loop step limit.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[ADKTool]] = None,
        model_name: str = MODEL_NAME,
        temperature: float = 0.0,
        max_steps: int = AGENT_MAX_STEPS,
    ) -> None:
        """Initialize ADKAgent instance.

        Args:
            name (str): Agent display name identifier.
            system_prompt (str): System prompt directives string.
            tools (Optional[List[ADKTool]], optional): List of attached tools. Defaults to None.
            model_name (str, optional): Target model identifier string. Defaults to MODEL_NAME.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            max_steps (int, optional): Maximum step execution limit. Defaults to AGENT_MAX_STEPS.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_map = {t.name: t for t in self.tools}
        self.model_name = model_name or MODEL_NAME
        self.temperature = temperature
        self.max_steps = max_steps

    async def run_step_async(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single agent reasoning step via Gemini LLM completion.

        Args:
            question (str): User question string.
            chat_history (Optional[List[Dict[str, str]]], optional): Conversation history turns. Defaults to None.
            system_prompt_override (Optional[str], optional): Override system prompt. Defaults to None.

        Returns:
            Dict[str, Any]: Result payload containing `sql`, `columns`, `rows`, `natural_answer`, `error`, `attempts`, `agent_name`.
        """
        client = get_llm_async_client()
        sys_prompt = system_prompt_override or self.system_prompt

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                role_name = "assistant" if role in ("bot", "model", "assistant") else "user"
                messages.append({"role": role_name, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": question})

        try:
            resp = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            text_output = resp.choices[0].message.content or ""
            return {
                "sql": None,
                "columns": [],
                "rows": [],
                "natural_answer": text_output,
                "error": None,
                "attempts": 1,
                "agent_name": self.name,
                "raw_response": resp,
            }
        except Exception as e:
            log.error(f"ADKAgent '{self.name}' execution error: {e}")
            from core.llm.llm_client import format_llm_api_error
            return {
                "sql": None,
                "columns": [],
                "rows": [],
                "natural_answer": format_llm_api_error(e),
                "error": str(e),
                "attempts": 1,
                "agent_name": self.name,
            }
