"""
test_router_agent.py — Unit tests for agents.router_agent module:
- _is_valid_rephrased_question
"""

import pytest
from agents.router_agent import _is_valid_rephrased_question


def test_is_valid_rephrased_question():
    """Verify _is_valid_rephrased_question detects complete vs dangling truncated questions."""
    assert _is_valid_rephrased_question("How many of the open picklists were delivered?", "how many of them were deleiveried") is True
    assert _is_valid_rephrased_question("How many of the", "how many of them were deleiveried") is False
    assert _is_valid_rephrased_question("What is the status of", "what is it") is False
    assert _is_valid_rephrased_question("List all open items in warehouse WH-01?", "list them") is True
