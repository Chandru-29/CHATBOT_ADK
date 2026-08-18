"""
test_loader.py — Unit tests for prompts.loader module:
- get_trimmed_coder_sql_directive
- Static prompt prefix context caching invariance
"""

import pytest
from prompts.loader import prompt_loader


def extract_prompt_static_prefix(full_prompt: str) -> str:
    """Extract the invariant static prefix (everything up to <database_schema>)."""
    if "<database_schema>" in full_prompt:
        return full_prompt.split("<database_schema>")[0]
    return full_prompt


def test_get_trimmed_coder_sql_directive_prefix_invariance():
    """
    Verify that the static prompt prefix (System Role + Core Mandatory Rules + Output Format)
    is 100% byte-for-byte identical across different queries and table selections.
    """
    schema_a = "Table: PICKLIST(picklistId:INT NOT NULL, status:INT NOT NULL)"
    tables_a = {"PICKLIST"}
    question_a = "How many picklists were completed today?"

    schema_b = "Table: ITEMLOCACNMAP(itemId:INT NOT NULL, locationId:INT NOT NULL)\nTable: WAREHOUSE(warehouseId:INT NOT NULL)"
    tables_b = {"ITEMLOCACNMAP", "WAREHOUSE"}
    question_b = "Show warehouse location mapping for items"

    prompt_a = prompt_loader.get_trimmed_coder_sql_directive(schema_a, question_a, tables_a)
    prompt_b = prompt_loader.get_trimmed_coder_sql_directive(schema_b, question_b, tables_b)

    prefix_a = extract_prompt_static_prefix(prompt_a)
    prefix_b = extract_prompt_static_prefix(prompt_b)

    # Core requirement for Gemini Context Caching:
    # Token 0 to <database_schema> must be 100% identical byte-for-byte across queries!
    assert prefix_a == prefix_b, "Static prompt prefix differs between queries! Prefix caching broken."
    assert len(prefix_a) > 500, "Static prefix too short, expected system role + mandatory core rules."


def test_get_trimmed_coder_sql_directive_dynamic_context_placement():
    """Verify that schema_context, scenario_rules, and exemplars are placed in the dynamic suffix."""
    schema = "Table: GRN(grnId:INT NOT NULL, vendorCode:VARCHAR NOT NULL)"
    tables = {"GRN"}
    question = "Show GRN details by vendor"

    prompt = prompt_loader.get_trimmed_coder_sql_directive(schema, question, tables)

    # Check structure
    assert "<system_role>" in prompt
    assert "<critical_rules>" in prompt
    assert "<output_format>" in prompt
    assert "<database_schema>" in prompt

    # Verify order: <database_schema> comes AFTER <output_format>
    output_fmt_index = prompt.index("<output_format>")
    db_schema_index = prompt.index("<database_schema>")
    assert db_schema_index > output_fmt_index, "<database_schema> must be placed after static prefix for caching!"

    # Verify dynamic schema context is included
    assert schema in prompt
