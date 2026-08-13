"""
test_adaptive_example_selector.py — Unit tests for Adaptive Few-Shot Exemplar Selector (Dynamic Top-K).
"""

import pytest
from prompts.example_selector import example_selector


RAW_MOCK_EXAMPLES = [
    {"q": "How many picklists are completed?", "a": "SELECT COUNT(*) FROM PICKLIST WHERE status = 5 AND isDeleted = 0;"},
    {"q": "List all warehouses", "a": "SELECT warehouseId, warehouseName FROM WAREHOUSE WHERE isDeleted = 0;"},
    {"q": "Show items and location mapping", "a": "SELECT i.itemCode, l.locationCode FROM ITEM i JOIN ITEMLOCACNMAP m ON i.itemId = m.itemId JOIN LOCATION l ON m.locationId = l.locationId WHERE i.isDeleted = 0;"},
    {"q": "Show stock by vendor", "a": "SELECT g.vendorCode, s.qty FROM GRN g JOIN ITEM i ON g.itemCode = i.itemCode JOIN SKUITEM sk ON i.itemId = sk.itemId JOIN SULOCATION s ON sk.skuId = s.skuId WHERE g.isDeleted = 0;"},
    {"q": "Which picklist has the most items and what is its status?", "a": "SELECT TOP 1 p.picklistCode, p.status, COUNT(pi.itemId) AS cnt FROM PICKLIST p JOIN PICKLISTITEM pi ON p.picklistId = pi.picklistId WHERE p.isDeleted = 0 GROUP BY p.picklistCode, p.status ORDER BY cnt DESC;"},
]


def test_determine_adaptive_k_simple():
    """Verify simple count and listing queries select K=1."""
    k_count = example_selector.determine_adaptive_k("How many picklists are completed?", {"PICKLIST"})
    assert k_count == 1

    k_list = example_selector.determine_adaptive_k("list all warehouses", {"WAREHOUSE"})
    assert k_list == 1


def test_determine_adaptive_k_intermediate():
    """Verify 2-table queries or group-by aggregations select K=3."""
    k_join = example_selector.determine_adaptive_k("show items and their location mapping", {"ITEM", "LOCATION"})
    assert k_join == 3

    k_vendor = example_selector.determine_adaptive_k("show stock by vendor", {"GRN", "SKUITEM"})
    assert k_vendor == 3


def test_determine_adaptive_k_complex():
    """Verify 3+ table queries or superlative queries select K=5."""
    k_superlative = example_selector.determine_adaptive_k("which picklist has the most items and what is its putaway location?", {"PICKLIST", "PICKLISTITEM", "LOCATION"})
    assert k_superlative == 5

    k_3tables = example_selector.determine_adaptive_k("show activity log for picklist PL001", {"PICKLIST", "SUIDACTIVITYLOG", "SULOCATION"})
    assert k_3tables == 5


def test_select_top_k_examples_adaptive():
    """Verify select_top_k_examples produces adaptive top-K exemplars when top_k is None."""
    # Simple query -> 1 exemplar
    res_simple = example_selector.select_top_k_examples(
        question="How many picklists are completed?",
        intent="WMS_AGENT",
        raw_examples=RAW_MOCK_EXAMPLES,
        top_k=None,
        focused_tables={"PICKLIST"},
    )
    assert res_simple.count("Q:") == 1

    # Intermediate query -> 3 exemplars
    res_inter = example_selector.select_top_k_examples(
        question="show items and location mapping",
        intent="WMS_AGENT",
        raw_examples=RAW_MOCK_EXAMPLES,
        top_k=None,
        focused_tables={"ITEM", "LOCATION"},
    )
    assert res_inter.count("Q:") == 3

    # Complex query -> 5 exemplars
    res_complex = example_selector.select_top_k_examples(
        question="which picklist has the most items and what is its status?",
        intent="WMS_AGENT",
        raw_examples=RAW_MOCK_EXAMPLES,
        top_k=None,
        focused_tables={"PICKLIST", "PICKLISTITEM", "LOCATION"},
    )
    assert res_complex.count("Q:") == 5
