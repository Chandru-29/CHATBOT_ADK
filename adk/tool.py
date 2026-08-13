"""
tool.py — ADK Tool Abstraction.

Wraps Python functions into reusable ADK tool objects and manages execution with error handling.
"""

# ── MODULE TAG: ADK Tool Management ──
import inspect
from typing import Callable, Any, Dict
from pydantic import BaseModel

from core.config.logger import get_logger

log = get_logger(__name__)


class ADKTool:
    """Encapsulates a Python tool function with name, description, and execution capabilities.

    Attributes:
        name (str): Unique tool identifier.
        description (str): Functional tool description.
        func (Callable): Encapsulated Python function or coroutine.
        parameters_schema (type[BaseModel] | None): Optional Pydantic schema model.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters_schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize ADKTool instance.

        Args:
            name (str): Tool identifier string.
            description (str): Tool description string.
            func (Callable): Target function or coroutine.
            parameters_schema (type[BaseModel] | None, optional): Pydantic parameter schema class. Defaults to None.
        """
        self.name = name
        self.description = description
        self.func = func
        self.parameters_schema = parameters_schema

    async def execute_async(self, **kwargs) -> Any:
        """Execute the encapsulated tool function asynchronously with error handling.

        Args:
            **kwargs: Keyword arguments passed directly to the target function.

        Returns:
            Any: Tool execution result object or error string.
        """
        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return self.func(**kwargs)
        except Exception as e:
            log.error(f"Error executing ADK Tool '{self.name}': {e}")
            return f"Error executing tool '{self.name}': {str(e)}"
