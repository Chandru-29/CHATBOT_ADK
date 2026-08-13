"""
domain_validator.py — Validates whether a SQL query accesses tables allowed
by the active agent domain scope.
"""

from typing import Set

def is_query_allowed_for_domain(queried_tables: list[str], allowed_tables: Set[str]) -> bool:
    """Check if all database tables in a SQL query are included in the allowed table list.

    Args:
        queried_tables (list[str]): List of table names found in the SQL query.
        allowed_tables (Set[str]): Set of allowed table names for this topic.

    Returns:
        bool: True if every table in the query is allowed, False if any table is forbidden.
    """
    if not queried_tables:
        return True
    allowed_lower = {t.lower() for t in allowed_tables}
    return all(tbl.lower() in allowed_lower for tbl in queried_tables)
