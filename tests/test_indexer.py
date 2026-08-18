"""
test_indexer.py — Unit tests for db.indexer module:
- build_rich_table_doc
"""

import pytest
from unittest.mock import patch

from db.schema import get_schema, clear_schema_cache
from db.indexer import build_rich_table_doc


MOCK_COLUMNS = [
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemId", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemCode", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemName", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"},
    {"TABLE_NAME": "user", "COLUMN_NAME": "userId", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
]


@pytest.fixture(autouse=True)
def reset_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


def test_build_rich_table_doc_csn():
    """Verify that db.indexer build_rich_table_doc extracts CSN table lines correctly."""
    with patch("db.schema._execute_api", return_value=MOCK_COLUMNS):
        schema_str = get_schema(include_tables={"ITEM", "user"})
        
        doc_item = build_rich_table_doc("ITEM", schema_str)
        assert "Table: ITEM(itemId:INT NOT NULL, itemCode:VARCHAR NOT NULL, itemName:VARCHAR)" in doc_item
        
        doc_user = build_rich_table_doc("user", schema_str)
        assert "Table: [user](userId:INT NOT NULL)" in doc_user
