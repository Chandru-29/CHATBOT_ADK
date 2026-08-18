"""
test_sql_agent.py — Unit tests for agents.sql_agent module:
- build_compact_history_summary
- format_db_result_deterministic
"""

import pytest
from agents.sql_agent import build_compact_history_summary, format_db_result_deterministic


def test_build_compact_history_summary_standard_result():
    """Verify build_compact_history_summary formats standard query results with samples."""
    used_sql = "SELECT item_code, item_name FROM ITEM WHERE status = 1;"
    columns = ["item_code", "item_name"]
    rows = [("IT001", "Widget A"), ("IT002", "Widget B"), ("IT003", "Widget C"), ("IT004", "Widget D")]
    
    summary = build_compact_history_summary(used_sql=used_sql, rows=rows, columns=columns)
    assert summary == "[EXECUTED_SQL: SELECT item_code, item_name FROM ITEM WHERE status = 1; | RESULT: 4 rows returned (Samples: IT001, IT002, IT003)]"


def test_build_compact_history_summary_zero_rows():
    """Verify build_compact_history_summary formats 0-row results without Samples tag."""
    used_sql = "SELECT item_code FROM ITEM WHERE status = 99;"
    columns = ["item_code"]
    rows = []
    
    summary = build_compact_history_summary(used_sql=used_sql, rows=rows, columns=columns)
    assert summary == "[EXECUTED_SQL: SELECT item_code FROM ITEM WHERE status = 99; | RESULT: 0 rows returned]"
    assert "Samples" not in summary


def test_build_compact_history_summary_db_error():
    """Verify build_compact_history_summary formats DB errors correctly."""
    used_sql = "SELECT location_id FROM ITEM;"
    error_msg = "Error: Unknown column 'location_id' in 'ITEM'"
    
    summary = build_compact_history_summary(used_sql=used_sql, error=error_msg)
    assert summary == "[EXECUTED_SQL: SELECT location_id FROM ITEM; | ERROR: Error: Unknown column 'location_id' in 'ITEM']"


def test_format_db_result_deterministic_scalar_pluralization_and_camelcase():
    """Verify format_db_result_deterministic correctly handles camelCase column labels and pluralization."""
    # Test camelCase column openPicklists with > 1 result
    db_out = "Columns: openPicklists\nRows (up to 100):\n- 121\n"
    formatted = format_db_result_deterministic(db_out, "how many open picklist we have ?")
    assert formatted == "There are currently **121** open picklists in the database."

    # Test camelCase column openPicklists with 1 result
    db_out_single = "Columns: openPicklists\nRows (up to 100):\n- 1\n"
    formatted_single = format_db_result_deterministic(db_out_single, "how many open picklist we have ?")
    assert formatted_single == "There is currently **1** open picklist in the database."

    # Test fallback record with no noun
    db_out_fallback = "Columns: total\nRows (up to 100):\n- 42\n"
    formatted_fallback = format_db_result_deterministic(db_out_fallback, "how many?")
    assert formatted_fallback == "There are currently **42** records in the database."


def test_run_sql_agent_rate_limit_error():
    """Verify run_sql_agent exits immediately on 429 quota error without looping or dumping table list."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    from agents.sql_agent import run_sql_agent

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception(
            "Error code: 429 - [{'error': {'code': 429, 'message': 'Quota exceeded for metric: generativelanguage.googleapis.com', 'status': 'RESOURCE_EXHAUSTED'}}]"
        )
    )

    system_prompt = "Total tables: 14\nAll tables: ITEM, SKUITEM, PICKLIST, LOCATION"
    
    with patch("agents.sql_agent.get_llm_async_client", return_value=mock_client):
        res = asyncio.run(
            run_sql_agent(
                session=None,
                messages=[],
                system_prompt=system_prompt,
                question="list the top two open picklist",
                agent_name="WMS Assistant",
                cache_key="test_key",
                api_cache={},
                intent="WMS_AGENT",
                stream=False,
            )
        )

    assert isinstance(res, dict)
    assert res["attempts"] == 1
    assert res["error"] == "Quota/Rate Limit Exceeded (HTTP 429)"
    assert "API Quota Exceeded (429 Rate Limit)" in res["natural_answer"]
    # Crucial assertion: Must NOT dump the database tables metadata fallback message!
    assert "There are currently 14 tables in the database" not in res["natural_answer"]
    # Assert chat completions create was called exactly 1 time (no multi-step retries on 429 error)
    assert mock_client.chat.completions.create.call_count == 1