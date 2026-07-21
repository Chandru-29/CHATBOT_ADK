"""
aliases.py — Predefined and dynamically generated semantic synonyms for database tables.

This module provides TABLE_ALIASES and STATIC_DESCRIPTIONS. If new tables are
added to the database, it automatically generates synonyms and descriptions on
the fly using heuristic word splitting, avoiding slow runtime LLM calls.
"""


# ── MODULE TAG: RAG Table Synonym Mapping ──
# ── STITCHGUARD LAYER: L3 (Allowed Tables Synonyms) ──
import re
from config.logger import get_logger

log = get_logger(__name__)

# Predefined manual semantic synonyms for core company database tables
_STATIC_ALIASES = {
    "employees": [
        "staff", "worker", "team", "people", "headcount", "personnel",
        "payroll", "workforce", "member", "employee", "person", "hire", "salary"
    ],
    "departments": [
        "dept", "division", "unit", "group", "branch", "section", "org",
        "department", "team"
    ],
    "projects": [
        "task", "assignment", "benchmark", "initiative", "deliverable",
        "milestone", "sprint", "project", "goal", "target", "objective"
    ],
    "customers": [
        "client", "buyer", "consumer", "account", "user", "subscriber",
        "patron", "customer"
    ],
    "orders": [
        "purchase", "transaction", "sale", "invoice", "billing", "order",
        "receipt", "deal", "payment"
    ],
    "suppliers": [
        "vendor", "supplier", "provider", "manufacturer", "distributor", "merchant"
    ],
    "leave_requests": [
        "vacation", "time off", "holiday", "absence", "leave", "request"
    ],
    "performance_reviews": [
        "review", "evaluation", "rating", "feedback", "grade", "score", "performance"
    ],
    "products": [
        "item", "inventory", "stock", "goods", "merchandise", "product"
    ],
    "attendance": [
        "check-in", "clock-in", "hours", "timesheet", "presence", "attendance"
    ]
}

# Predefined manual database table descriptions
_STATIC_DESCRIPTIONS = {
    "attendance": "Employee data tracking table.",
    "customers": "Customer data table with ID, name, and email.",
    "departments": "Department information including ID, name, and head.",
    "employees": "Employee data with ID, name, department, salary, and role.",
    "leave_requests": "Employee data for leave requests.",
    "orders": "Sales and product data for orders.",
    "performance_reviews": "Employee data with performance grades and ratings.",
    "products": "Sales and product data for inventory management.",
    "projects": "Employee data linked to projects.",
    "suppliers": "Supplier details and defect counts for products."
}


class DynamicTableAliases(dict):
    """
    A dictionary-like object that returns predefined synonyms for known tables,
    and automatically falls back to clean, word-split tokens for any new tables.
    """

    def __init__(self):
        super().__init__(_STATIC_ALIASES)

    def get(self, key, default=None):
        # ── Return static alias if defined ──
        if key in self:
            return super().__getitem__(key)
        
        # ── Return regex split fallback synonyms ──
        words = re.findall(r"[a-zA-Z0-9]+", key.lower())
        if words:
            if key.lower() not in words:
                words.append(key.lower())
            log.info(f"Aliases: Generated fallback synonyms for table '{key}': {words}")
            return words
        return default or [key]

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # ── Fallback to get() on KeyError ──
            return self.get(key)


class DynamicTableDescriptions(dict):
    """
    A dictionary-like object that returns predefined descriptions for known tables,
    and dynamically constructs generic descriptions for any new tables.
    """

    def __init__(self):
        super().__init__(_STATIC_DESCRIPTIONS)

    def get(self, key, default=None):
        # ── Return static description if defined ──
        if key in self:
            return super().__getitem__(key)
        
        # ── Return generic fallback description ──
        words = re.findall(r"[a-zA-Z0-9]+", key.lower())
        name_str = " ".join(words) if words else key
        desc = f"Table containing {name_str} data."
        log.info(f"Aliases: Generated fallback description for table '{key}': '{desc}'")
        return desc

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # ── Fallback to get() on KeyError ──
            return self.get(key)


# Global instances exported to schema retriever / indexer
TABLE_ALIASES = DynamicTableAliases()
STATIC_DESCRIPTIONS = DynamicTableDescriptions()
