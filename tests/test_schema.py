"""
test_schema.py — Unit tests for db.schema module:
- get_schema
- clear_schema_cache
"""

import re
import pytest
from unittest.mock import patch

from db.schema import get_schema, clear_schema_cache


MOCK_COLUMNS = [
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemId", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemCode", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "ITEM", "COLUMN_NAME": "itemName", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"},
    {"TABLE_NAME": "LOCATION", "COLUMN_NAME": "locationId", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "LOCATION", "COLUMN_NAME": "locationCode", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "NO"},
    {"TABLE_NAME": "user", "COLUMN_NAME": "userId", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
]


def mock_execute_api(sql_query: str) -> list[dict]:
    """Dynamically filter MOCK_COLUMNS based on TABLE_NAME IN clause."""
    in_match = re.search(r"WHERE TABLE_NAME IN \((.*?)\)", sql_query, re.IGNORECASE)
    if in_match:
        raw_tables = [t.strip(" '\"[]") for t in in_match.group(1).split(",")]
        return [row for row in MOCK_COLUMNS if row["TABLE_NAME"] in raw_tables]
    return MOCK_COLUMNS


@pytest.fixture(autouse=True)
def reset_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


def test_get_schema_csn_format():
    """Verify that get_schema produces dense single-line CSN for each table."""
    clear_schema_cache()
    with patch("db.schema._execute_api", side_effect=mock_execute_api):
        schema_str = get_schema(include_tables=frozenset({"ITEM", "LOCATION", "user"}))
        
        # Verify table single-line CSN notation
        assert "Table: ITEM(itemId:INT NOT NULL, itemCode:VARCHAR NOT NULL, itemName:VARCHAR)" in schema_str
        assert "Table: LOCATION(locationId:INT NOT NULL, locationCode:VARCHAR NOT NULL)" in schema_str
        assert "Table: [user](userId:INT NOT NULL)" in schema_str


def test_get_schema_csn_token_reduction():
    """Verify CSN uses significantly fewer characters/tokens than verbose multi-line format."""
    clear_schema_cache()
    with patch("db.schema._execute_api", side_effect=mock_execute_api):
        csn_schema = get_schema(include_tables=frozenset({"ITEM", "LOCATION"}))
        
        # Calculate character size of table definitions (excluding header)
        csn_lines = [line for line in csn_schema.split("\n") if line.startswith("Table:")]
        csn_char_count = sum(len(line) for line in csn_lines)
        
        # Verbose representation for exact same mock columns:
        verbose_char_count = len(
            "Table: ITEM\nColumns:\n  itemId INT NOT NULL\n  itemCode VARCHAR NOT NULL\n  itemName VARCHAR\n"
            "Table: LOCATION\nColumns:\n  locationId INT NOT NULL\n  locationCode VARCHAR NOT NULL"
        )
        
        # Assert CSN achieves character/token reduction on table definitions
        assert csn_char_count < verbose_char_count


def test_get_schema_sql_agent_compatibility():
    """Verify that f'Table: {t}' substring check in sql_agent passes with CSN format."""
    clear_schema_cache()
    with patch("db.schema._execute_api", side_effect=mock_execute_api):
        schema_str = get_schema(include_tables=frozenset({"ITEM"}))
        
        # SQL Agent check pattern: f"Table: {t}" in system_prompt
        assert "Table: ITEM" in schema_str
        assert "Table: [user]" not in schema_str
