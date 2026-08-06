"""
loader.py — Loads instructions.yml and examples.yml from the prompts/config
directory at startup and caches them in memory.

The PromptLoader class exposes clean get_*() methods so other modules
never have to read YAML files directly or know where the files live.
Includes dynamic prompt trimming via RAG example selection and table-aware rule filtering.
"""

# ── MODULE TAG: Prompt Engineering Loader ──
import os
import re
import yaml
from typing import Optional, Set
from core.config.settings import PROMPTS_DIR
from core.config.logger import get_logger

log = get_logger(__name__)


class PromptLoader:
    """
    Reads instructions.yml and examples.yml once at startup and provides
    helper methods to extract the relevant section for each agent or intent.
    """

    def __init__(self):
        self._instructions = self._load_yaml("instructions.yml")
        self._examples     = self._load_yaml("examples.yml")
        log.info("PromptLoader: YAML prompt files loaded successfully.")

    def reload(self) -> None:
        """Reload instructions.yml and examples.yml into memory."""
        self._instructions = self._load_yaml("instructions.yml")
        self._examples     = self._load_yaml("examples.yml")

        try:
            from prompts.example_selector import example_selector
            example_selector.clear_cache()
        except Exception as e:
            log.warning(f"PromptLoader: could not clear example_selector cache: {e}")

        log.info("PromptLoader: YAML prompt files reloaded successfully.")

    @staticmethod
    def _load_yaml(filename: str) -> dict:
        path = os.path.join(PROMPTS_DIR, "config", filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            log.error(f"PromptLoader: {filename} not found at {path}")
            return {}
        except yaml.YAMLError as e:
            log.error(f"PromptLoader: Failed to parse {filename}: {e}")
            return {}

    # ── Instructions ──────────────────────────────────────────────────────────

    def get_router_prompt(self) -> str:
        """Return the system prompt used to classify the user's intent."""
        return self._instructions.get("router", {}).get("prompt", "").strip()

    def get_rephraser_prompt(self) -> str:
        """Return the system prompt used to rewrite follow-up questions."""
        return self._instructions.get("rephraser", {}).get("prompt", "").strip()

    def get_rephrase_and_route_prompt(self) -> str:
        """Return the combined rephrase+intent prompt (used by router_agent.py)."""
        return self._instructions.get("rephrase_and_route", {}).get("prompt", "").strip()

    def get_agent_config(self, intent: str) -> dict:
        """Return {'display_name': ..., 'description': ...} for the given intent."""
        return self._instructions.get("agents", {}).get(intent, {
            "display_name": "General Agent",
            "description":  "You are a helpful database assistant.",
        })

    def get_rules(self) -> str:
        """Return the numbered shared rules as a single formatted string."""
        rules = self._instructions.get("rules", [])
        return "\n".join(f"{i + 1}. {r.strip()}" for i, r in enumerate(rules))

    def get_coder_sql_directive(self) -> str:
        """Return the complete coder SQL directive template."""
        return self._instructions.get("coder_sql_directive", "").strip()

    def get_trimmed_coder_sql_directive(
        self,
        schema_context: str,
        question: str,
        focused_tables: Set[str],
        top_k_examples: int = 5,
        intent: str = "WMS_AGENT",
    ) -> str:
        """
        Dynamically construct a compressed system prompt by selecting core rules
        and retrieving top-K semantically matching Q/A exemplars.
        """
        raw_directive = self.get_coder_sql_directive()

        raw_rules = []
        rules_match = re.search(r"<critical_rules>(.*?)</critical_rules>", raw_directive, re.DOTALL)
        if rules_match:
            rules_block = rules_match.group(1).strip()
            raw_rules = [r.strip() for r in rules_block.split("\n") if r.strip()]

        try:
            from prompts.rule_selector import rule_selector
            selected_rules = rule_selector.select_relevant_rules(raw_rules, question, focused_tables)
        except Exception as e:
            log.warning(f"PromptLoader: Dynamic rule selection failed ({e}). Using full rule set.")
            selected_rules = raw_rules

        selected_rules_str = "\n  ".join(selected_rules)
        dynamic_directive = re.sub(
            r"<critical_rules>.*?</critical_rules>",
            f"<critical_rules>\n  {selected_rules_str}\n  </critical_rules>",
            raw_directive,
            flags=re.DOTALL,
        )

        formatted_directive = dynamic_directive.format(schema_context=schema_context)

        examples_str = self.get_examples(intent, question=question, top_k=top_k_examples)
        if examples_str:
            formatted_directive += f"\n\n{examples_str}"

        return formatted_directive

    # ── Examples ──────────────────────────────────────────────────────────────

    def get_examples(self, intent: str, question: Optional[str] = None, top_k: int = 5) -> str:
        """Return SQL examples for the given intent domain."""
        domain_examples = self._examples.get(intent, [])
        if not domain_examples:
            return ""

        if question:
            try:
                from prompts.example_selector import example_selector
                return example_selector.select_top_k_examples(
                    question=question,
                    intent=intent,
                    raw_examples=domain_examples,
                    top_k=top_k,
                )
            except Exception as e:
                log.warning(f"PromptLoader: Dynamic example selection failed ({e}). Falling back to full examples.")

        lines = ["Examples:"]
        for ex in domain_examples:
            q = ex.get("q", "").strip()
            a = ex.get("a", "").strip()
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_exemplar_questions_by_intent(self) -> dict[str, list[str]]:
        """Extract question strings grouped by domain intent from examples.yml."""
        result: dict[str, list[str]] = {}
        for intent, items in self._examples.items():
            if isinstance(items, list):
                qs = [
                    item.get("q", "").strip()
                    for item in items
                    if isinstance(item, dict) and item.get("q")
                ]
                if qs:
                    result[intent] = qs
        return result


# Instantiate once at module load
prompt_loader = PromptLoader()
