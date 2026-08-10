"""
tool.py — ADK Tool Abstraction for Google Gemini API Function Calling.

Wraps Python functions into Gemini-compatible tool objects (types.Tool)
and manages execution with error handling.
"""

# ── MODULE TAG: ADK Tool Management ──
import inspect
from typing import Callable, Any, Dict
from pydantic import BaseModel
from google.genai import types

from core.config.logger import get_logger

log = get_logger(__name__)


class ADKTool:
    """
    Encapsulates a Python tool function with name, description, parameters,
    and Gemini API FunctionDeclaration representation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters_schema: type[BaseModel] | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters_schema = parameters_schema

    def to_gemini_tool(self) -> types.Tool:
        """Convert this ADK Tool into a Google GenAI types.Tool instance."""
        if hasattr(types, "FunctionDeclaration"):
            fn_decl = types.FunctionDeclaration(
                name=self.name,
                description=self.description,
            )
            return types.Tool(function_declarations=[fn_decl])
        return types.Tool(function_declarations=[self.func])

    async def execute_async(self, **kwargs) -> Any:
        """Execute the encapsulated tool function asynchronously."""
        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return self.func(**kwargs)
        except Exception as e:
            log.error(f"Error executing ADK Tool '{self.name}': {e}")
            return f"Error executing tool '{self.name}': {str(e)}"
