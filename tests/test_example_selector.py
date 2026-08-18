"""
test_example_selector.py — Unit tests for prompts.example_selector module:
- determine_adaptive_k
- select_top_k_examples
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


def test_select_top_k_examples_default():
    """Verify select_top_k_examples selects DEFAULT_TOP_K=5 exemplars when top_k is None."""
    res_default = example_selector.select_top_k_examples(
        question="How many picklists are completed?",
        intent="WMS_AGENT",
        raw_examples=RAW_MOCK_EXAMPLES,
        top_k=None,
        focused_tables={"PICKLIST"},
    )
    assert res_default.count("Q:") == 5


def test_select_top_k_examples_explicit_top_k():
    """Verify select_top_k_examples respects an explicit top_k parameter."""
    res_custom = example_selector.select_top_k_examples(
        question="show items and location mapping",
        intent="WMS_AGENT",
        raw_examples=RAW_MOCK_EXAMPLES,
        top_k=2,
        focused_tables={"ITEM", "LOCATION"},
    )
    assert res_custom.count("Q:") == 2


def test_precompute_embeddings_and_chroma_cache():
    """Verify precompute_embeddings populates ChromaDB 'fewshot_exemplars' collection."""
    from db.chromadb import get_fewshot_exemplars_collection
    raw_dict = {"WMS_AGENT": RAW_MOCK_EXAMPLES}
    example_selector.clear_cache()
    
    # Precompute embeddings into ChromaDB
    example_selector.precompute_embeddings(raw_dict, force_recompute=True)
    collection = get_fewshot_exemplars_collection()
    
    assert collection.count() == 5
    curr_hash = example_selector._compute_examples_hash(raw_dict)
    assert collection.metadata.get("hash") == curr_hash


def test_examples_hash_invalidation():
    """Verify SHA-256 hash invalidation and collection re-indexing when raw exemplar content changes."""
    from db.chromadb import get_fewshot_exemplars_collection
    raw_dict_1 = {"WMS_AGENT": [{"q": "How many completed picklists?", "a": "SELECT COUNT(*) FROM PICKLIST WHERE status = 5;"}]}
    raw_dict_2 = {"WMS_AGENT": [{"q": "Show warehouse list", "a": "SELECT * FROM WAREHOUSE;"}]}

    # Index first dictionary
    example_selector.precompute_embeddings(raw_dict_1, force_recompute=True)
    col1 = get_fewshot_exemplars_collection()
    hash1 = col1.metadata.get("hash")
    assert col1.count() == 1

    # Index second dictionary (hash mismatch triggers re-index)
    example_selector.precompute_embeddings(raw_dict_2)
    col2 = get_fewshot_exemplars_collection()
    hash2 = col2.metadata.get("hash")
    assert col2.count() == 1
    assert hash1 != hash2



