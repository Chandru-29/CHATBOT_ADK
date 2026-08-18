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
    """Reads instructions.yml and examples.yml configuration files at startup.

    Provides helper methods to retrieve prompts, rules, directives, and exemplars for agents.
    """

    def __init__(self) -> None:
        """Initialize PromptLoader instance and load configuration YAML files."""
        self._instructions = self._load_yaml("instructions.yml")
        self._examples     = self._load_yaml("examples.yml")
        log.info("PromptLoader: YAML prompt files loaded successfully.")

        try:
            from prompts.example_selector import example_selector
            example_selector.precompute_embeddings(self._examples)
        except Exception as e:
            log.warning(f"PromptLoader: could not pre-compute exemplar embeddings: {e}")

    def reload(self) -> None:
        """Reload instructions.yml and examples.yml files into memory and clear caches."""
        self._instructions = self._load_yaml("instructions.yml")
        self._examples     = self._load_yaml("examples.yml")

        try:
            from prompts.example_selector import example_selector
            example_selector.clear_cache()
            example_selector.precompute_embeddings(self._examples, force_recompute=True)
        except Exception as e:
            log.warning(f"PromptLoader: could not clear or pre-compute example_selector cache: {e}")

        log.info("PromptLoader: YAML prompt files reloaded successfully.")

    @staticmethod
    def _load_yaml(filename: str) -> dict:
        """Read and parse a YAML file from the prompts config directory.

        Args:
            filename (str): Name of YAML file to load.

        Returns:
            dict: Parsed dictionary content object.
        """
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
        """Retrieve system prompt string used to classify user intent.

        Returns:
            str: Router system prompt string.
        """
        return self._instructions.get("router", {}).get("prompt", "").strip()

    def get_rephraser_prompt(self) -> str:
        """Retrieve system prompt string used to rewrite follow-up questions.

        Returns:
            str: Rephraser system prompt string.
        """
        return self._instructions.get("rephraser", {}).get("prompt", "").strip()

    def get_rephrase_and_route_prompt(self) -> str:
        """Retrieve combined rephrase and routing classification system prompt.

        Returns:
            str: Combined rephrase and router system prompt string.
        """
        return self._instructions.get("rephrase_and_route", {}).get("prompt", "").strip()

    def get_agent_config(self, intent: str) -> dict:
        """Retrieve display name and description config dictionary for an intent domain.

        Args:
            intent (str): Domain intent string.

        Returns:
            dict: Agent configuration dict `{"display_name": ..., "description": ...}`.
        """
        return self._instructions.get("agents", {}).get(intent, {
            "display_name": "General Agent",
            "description":  "You are a helpful database assistant.",
        })

    def get_rules(self) -> str:
        """Retrieve shared numbered system rules as a single formatted string.

        Returns:
            str: Formatted rules block string.
        """
        rules = self._instructions.get("rules", [])
        return "\n".join(f"{i + 1}. {r.strip()}" for i, r in enumerate(rules))

    def get_coder_sql_directive(self) -> str:
        """Retrieve complete raw SQL generation directive template string.

        Returns:
            str: Raw directive template string.
        """
        return self._instructions.get("coder_sql_directive", "").strip()

    def get_trimmed_coder_sql_directive(
        self,
        schema_context: str,
        question: str,
        focused_tables: Set[str],
        top_k_examples: Optional[int] = None,
        intent: str = "WMS_AGENT",
    ) -> str:
        """Construct a compressed system prompt with core static rules and dynamic exemplars.

        Args:
            schema_context (str): Formatted schema text string.
            question (str): User question string.
            focused_tables (Set[str]): Grounded table names set.
            top_k_examples (Optional[int], optional): Exemplar count limit. Defaults to None.
            intent (str, optional): Domain intent. Defaults to "WMS_AGENT".

        Returns:
            str: Assembled system prompt string.
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

        mandatory_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 29, 30}
        core_rules = []
        scenario_rules = []

        for r in selected_rules:
            m = re.match(r"^(\d+)\.", r)
            if m and int(m.group(1)) in mandatory_numbers:
                core_rules.append(r)
            else:
                scenario_rules.append(r)

        core_rules_str = "\n  ".join(core_rules if core_rules else raw_rules)
        dynamic_directive = re.sub(
            r"<critical_rules>.*?</critical_rules>",
            f"<critical_rules>\n  {core_rules_str}\n  </critical_rules>",
            raw_directive,
            flags=re.DOTALL,
        )

        formatted_directive = dynamic_directive.format(schema_context=schema_context)

        if scenario_rules:
            scenario_str = "\n  ".join(scenario_rules)
            formatted_directive += f"\n\n  <scenario_rules>\n  {scenario_str}\n  </scenario_rules>"

        examples_str = self.get_examples(intent, question=question, top_k=top_k_examples, focused_tables=focused_tables)
        if examples_str:
            formatted_directive += f"\n\n{examples_str}"

        return formatted_directive

    # ── Examples ──────────────────────────────────────────────────────────────

    def get_examples(self, intent: str, question: Optional[str] = None, top_k: Optional[int] = None, focused_tables: Optional[Set[str]] = None) -> str:
        """Retrieve formatted SQL Q/A exemplar strings for an intent domain.

        Args:
            intent (str): Intent domain string.
            question (Optional[str], optional): User question. Defaults to None.
            top_k (Optional[int], optional): Top-K match limit. Defaults to None.
            focused_tables (Optional[Set[str]], optional): Table scope set. Defaults to None.

        Returns:
            str: Formatted Markdown examples block string.
        """
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
                    focused_tables=focused_tables,
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
        """Extract question strings grouped by domain intent from examples.yml.

        Returns:
            dict[str, list[str]]: Dictionary mapping domain intent labels to question string lists.
        """
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
