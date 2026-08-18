"""
test_runner.py — Unit tests for adk.runner module:
- _build_messages (history compaction)
"""

import pytest
from adk.runner import ADKRunner
from api.models import ChatRequest


def test_build_messages_replaces_markdown_tables():
    """Verify ADKRunner._build_messages compacts past assistant history into history_summary tags."""
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
