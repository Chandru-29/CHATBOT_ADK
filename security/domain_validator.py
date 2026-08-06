"""
domain_validator.py — Validates whether a SQL query accesses tables allowed
by the active agent domain scope.
"""

from typing import Set

def is_query_allowed_for_domain(queried_tables: list[str], allowed_tables: Set[str]) -> bool:
    """Return True if all queried_tables are within allowed_tables for the domain."""
    if not queried_tables:
        return True
    allowed_lower = {t.lower() for t in allowed_tables}
    return all(tbl.lower() in allowed_lower for tbl in queried_tables)
