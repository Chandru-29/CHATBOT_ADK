"""
agent.py — ADK Agent Core Class using Google Gemini API.

Defines declarative Agents with system instructions, model binding, attached ADK Tools,
and reasoning step execution loops.
"""

# ── MODULE TAG: ADK Agent Engine ──
import re
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from core.config.settings import GEMINI_MODEL, AGENT_MAX_STEPS
from core.config.logger import get_logger
from core.llm.llm_client import get_genai_client
from adk.tool import ADKTool

log = get_logger(__name__)


class ADKAgent:
    """
    Standardized ADK Agent entity encapsulating model parameters, system prompt,
    attached tools, and execution capabilities.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[ADKTool]] = None,
        model_name: str = GEMINI_MODEL,
        temperature: float = 0.0,
        max_steps: int = AGENT_MAX_STEPS,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_map = {t.name: t for t in self.tools}
        self.model_name = model_name
        self.temperature = temperature
        self.max_steps = max_steps

    def _get_gemini_tools(self) -> Optional[List[Any]]:
        """Return attached Python tool functions directly for GenAI SDK automatic tool dispatching."""
        if not self.tools:
            return None
        return [t.func for t in self.tools]

    async def run_step_async(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute agent reasoning loop using Google Gemini API.
        Returns standard result dict: {sql, columns, rows, natural_answer, error, attempts, agent_name}.
        """
        client = get_genai_client()
        sys_prompt = system_prompt_override or self.system_prompt
        raw_tools = self._get_gemini_tools()

        # Build contents from history and current question
        contents = []
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                if role == "bot":
                    role = "model"
                elif role == "assistant":
                    role = "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.get("content", ""))]))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        config = types.GenerateContentConfig(
            system_instruction=sys_prompt if sys_prompt else None,
            temperature=self.temperature,
            tools=raw_tools if raw_tools else None,
        )

        try:
            resp = await client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            text_output = resp.text or ""
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
            return {
                "sql": None,
                "columns": [],
                "rows": [],
                "natural_answer": None,
                "error": str(e),
                "attempts": 1,
                "agent_name": self.name,
            }
