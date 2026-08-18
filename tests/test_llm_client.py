"""
test_llm_client.py — Unit tests for rate limit error detection and formatting in core.llm.llm_client.
"""

from unittest.mock import MagicMock
import pytest
from openai import RateLimitError
from core.llm.llm_client import is_rate_limit_error, format_llm_api_error


def test_is_rate_limit_error_with_openai_exception():
    """Verify is_rate_limit_error returns True for OpenAI RateLimitError."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.request = mock_request
    err = RateLimitError(
        message="Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests",
        response=mock_response,
        body=None,
    )
    assert is_rate_limit_error(err) is True


def test_is_rate_limit_error_with_status_code_429():
    """Verify is_rate_limit_error returns True for status code 429 or 429 string."""
    class CustomAPIError(Exception):
        def __init__(self, message, status_code=429):
            super().__init__(message)
            self.status_code = status_code

    err = CustomAPIError("Error code: 429 - RESOURCE_EXHAUSTED")
    assert is_rate_limit_error(err) is True

    err_str_only = Exception("You exceeded your current quota, please check your plan. Error 429")
    assert is_rate_limit_error(err_str_only) is True


def test_is_rate_limit_error_non_429():
    """Verify is_rate_limit_error returns False for non-rate-limit errors."""
    err = ValueError("Invalid syntax in SQL query")
    assert is_rate_limit_error(err) is False


def test_format_llm_api_error_rate_limit():
    """Verify format_llm_api_error formats 429 rate limit messages clearly."""
    err = Exception("Error code: 429 - RESOURCE_EXHAUSTED Quota exceeded")
    msg = format_llm_api_error(err)
    assert "API Quota Exceeded (429 Rate Limit)" in msg
    assert "Gemini API request quota" in msg


def test_format_llm_api_error_general():
    """Verify format_llm_api_error formats general errors cleanly."""
    err = Exception("Connection timed out to Gemini endpoint")
    msg = format_llm_api_error(err)
    assert "LLM Service Error" in msg
    assert "Connection timed out" in msg
