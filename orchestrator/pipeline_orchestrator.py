"""
pipeline_orchestrator.py — Request Pipeline Orchestrator.

Delegates execution lifecycle to the local ADK Framework Workflow Orchestrator (ADKRunner).
"""

# ── MODULE TAG: Pipeline Execution Orchestrator Service ──
import time
from typing import Union
from fastapi.responses import StreamingResponse

from core.config.logger import get_logger
from api.models import ChatRequest
from adk.runner import ADKRunner

log = get_logger(__name__)


class Timer:
    """A stopwatch to measure how long each step takes in milliseconds.

    Attributes:
        label (str): Stopwatch section name.
    """

    def __init__(self, label: str) -> None:
        """Set up the Timer stopwatch.

        Args:
            label (str): Name for the stopwatch measurement.
        """
        self.label = label
        self._t0 = time.perf_counter()

    def lap(self, name: str) -> float:
        """Record and return the milliseconds passed since the last checkpoint.

        Args:
            name (str): Milestone name.

        Returns:
            float: Milliseconds passed since the last step.
        """
        now = time.perf_counter()
        ms = (now - self._t0) * 1000
        log.debug(f"  ⏱  [{self.label}] {name}: {ms:.0f} ms")
        self._t0 = now
        return ms

    def total(self, name: str, t_start: float) -> None:
        """Log the total time taken from the start of the request.

        Args:
            name (str): Operation description string.
            t_start (float): Start time marker.
        """
        ms = (time.perf_counter() - t_start) * 1000
        log.debug(f"  ⏱  [{self.label}] ── TOTAL {name}: {ms:.0f} ms ──")


async def process_query_pipeline(req: ChatRequest, stream: bool = False) -> Union[dict, StreamingResponse]:
    """Run the main process for handling a user's chat question.

    Args:
        req (ChatRequest): Incoming chat question request object.
        stream (bool, optional): True to stream answer piece by piece, False for a full response. Defaults to False.

    Returns:
        Union[dict, StreamingResponse]: Answer result dictionary or StreamingResponse object.
    """
    return await ADKRunner.run_pipeline(req, stream=stream)
