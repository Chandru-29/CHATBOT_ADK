"""
rule_selector.py — Table-Aware Dynamic Rule Selector for SQL Coder Directives.

Filters 30 SQL directive rules into mandatory core rules vs. table/scenario-specific rules,
injecting scenario rules only when relevant to the user query and VectorRAG focused tables.
"""

# ── MODULE TAG: Dynamic Rule Selector ──
import re
from typing import List, Set

from core.config.logger import get_logger

log = get_logger(__name__)

_MANDATORY_CORE_RULE_NUMBERS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 29, 30}


class RuleSelector:
    """Selects relevant SQL rules based on mandatory core rules + active table/scenario triggers."""

    @staticmethod
    def select_relevant_rules(
        all_rules: List[str],
        question: str,
        focused_tables: Set[str],
    ) -> List[str]:
        """Filter *all_rules* to return only core rules + relevant scenario rules."""
        if not all_rules:
            return []

        q_lower = question.lower()
        tables_upper = {t.upper() for t in focused_tables}

        active_rule_numbers = set(_MANDATORY_CORE_RULE_NUMBERS)

        if "PICKLIST" in tables_upper or any(kw in q_lower for kw in ["status", "picking", "putaway", "completed"]):
            active_rule_numbers.add(18)

        if "GRN" in tables_upper or "SKUITEM" in tables_upper or "vendor" in q_lower:
            active_rule_numbers.add(20)

        if "WAREHOUSE" in tables_upper or "ITEMLOCACNMAP" in tables_upper:
            active_rule_numbers.add(21)

        if "PICKLISTITEM" in tables_upper or "PICKLIST" in tables_upper:
            active_rule_numbers.add(22)

        if "PICKLIST" in tables_upper or any(kw in q_lower for kw in ["grn picklist", "work order", "transfer order", "document"]):
            active_rule_numbers.add(23)

        if "FGTRANSACTION" in tables_upper or any(kw in q_lower for kw in ["vin", "fgcode", "putaway", "delivered", "hold"]):
            active_rule_numbers.add(24)

        if "SUIDACTIVITYLOG" in tables_upper or "activity log" in q_lower or "remark" in q_lower:
            active_rule_numbers.add(25)

        if "FGMODEL" in tables_upper or "fg model" in q_lower:
            active_rule_numbers.add(26)

        if "ITEMLOCACNMAP" in tables_upper or "mapping" in q_lower:
            active_rule_numbers.add(27)

        if "GRN" in tables_upper and any(kw in q_lower for kw in ["stock", "inventory", "store", "where"]):
            active_rule_numbers.add(28)

        selected_rules = []
        for r in all_rules:
            r_str = r.strip()
            if not r_str:
                continue
            m = re.match(r"^(\d+)\.", r_str)
            if m:
                rule_num = int(m.group(1))
                if rule_num in active_rule_numbers:
                    selected_rules.append(r_str)
            else:
                selected_rules.append(r_str)

        log.info(
            f"RuleSelector: Selected {len(selected_rules)}/{len(all_rules)} rules "
            f"for tables={tables_upper}"
        )
        return selected_rules


# Singleton instance
rule_selector = RuleSelector()
