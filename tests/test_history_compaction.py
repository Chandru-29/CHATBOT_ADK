"""
test_history_compaction.py — Unit tests for Enhanced Chat History Compaction.
"""

import pytest
from agents.sql_agent import build_compact_history_summary
from adk.runner import ADKRunner
from api.models import ChatRequest


def test_compact_summary_standard_result():
    """Verify build_compact_history_summary formats standard query results with samples."""
    used_sql = "SELECT item_code, item_name FROM ITEM WHERE status = 1;"
    columns = ["item_code", "item_name"]
    rows = [("IT001", "Widget A"), ("IT002", "Widget B"), ("IT003", "Widget C"), ("IT004", "Widget D")]
    
    summary = build_compact_history_summary(used_sql=used_sql, rows=rows, columns=columns)
    assert summary == "[EXECUTED_SQL: SELECT item_code, item_name FROM ITEM WHERE status = 1; | RESULT: 4 rows returned (Samples: IT001, IT002, IT003)]"


def test_compact_summary_zero_rows():
    """Verify build_compact_history_summary formats 0-row results without Samples tag."""
    used_sql = "SELECT item_code FROM ITEM WHERE status = 99;"
    columns = ["item_code"]
    rows = []
    
    summary = build_compact_history_summary(used_sql=used_sql, rows=rows, columns=columns)
    assert summary == "[EXECUTED_SQL: SELECT item_code FROM ITEM WHERE status = 99; | RESULT: 0 rows returned]"
    assert "Samples" not in summary


def test_compact_summary_db_error():
    """Verify build_compact_history_summary formats DB errors correctly."""
    used_sql = "SELECT location_id FROM ITEM;"
    error_msg = "Error: Unknown column 'location_id' in 'ITEM'"
    
    summary = build_compact_history_summary(used_sql=used_sql, error=error_msg)
    assert summary == "[EXECUTED_SQL: SELECT location_id FROM ITEM; | ERROR: Error: Unknown column 'location_id' in 'ITEM']"


def test_build_messages_replaces_markdown_tables():
    """Verify ADKRunner._build_messages compacts past assistant history."""
    markdown_table_response = (
        "Here are the requested database records:\n\n"
        "| Item ID | Item Code | Item Name |\n"
        "| --- | --- | --- |\n" +
        "\n".join(f"| {i} | ITEM-{i} | Product {i} |" for i in range(50))
    )
    
    req = ChatRequest(
        user_question="what about their locations?",
        chat_history=[
            {"role": "user", "content": "Show active items"},
            {
                "role": "assistant",
                "content": markdown_table_response,
                "history_summary": "[EXECUTED_SQL: SELECT itemCode FROM ITEM WHERE status = 1; | RESULT: 50 rows returned (Samples: ITEM-0, ITEM-1, ITEM-2)]"
            }
        ]
    )
    
    messages = ADKRunner._build_messages(req, req.user_question, "system prompt")
    
    # Assert assistant message is compacted to history_summary
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    assert assistant_msg["content"] == "[EXECUTED_SQL: SELECT itemCode FROM ITEM WHERE status = 1; | RESULT: 50 rows returned (Samples: ITEM-0, ITEM-1, ITEM-2)]"
    assert "| --- |" not in assistant_msg["content"]
    assert len(assistant_msg["content"]) < 150
