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
    """Lightweight wall-clock stopwatch for pipeline step timing."""

    def __init__(self, label: str):
        self.label = label
        self._t0 = time.perf_counter()

    def lap(self, name: str) -> float:
        """Log and return elapsed ms since the last lap (or start)."""
        now = time.perf_counter()
        ms = (now - self._t0) * 1000
        log.debug(f"  ⏱  [{self.label}] {name}: {ms:.0f} ms")
        self._t0 = now
        return ms

    def total(self, name: str, t_start: float) -> None:
        """Log total elapsed time from t_start."""
        ms = (time.perf_counter() - t_start) * 1000
        log.debug(f"  ⏱  [{self.label}] ── TOTAL {name}: {ms:.0f} ms ──")


async def process_query_pipeline(req: ChatRequest, stream: bool = False) -> Union[dict, StreamingResponse]:
    """
    Main chat execution pipeline.
    Delegates directly to local ADK Framework Workflow Orchestrator (ADKRunner).
    """
    return await ADKRunner.run_pipeline(req, stream=stream)
