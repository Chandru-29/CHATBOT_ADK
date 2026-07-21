"""
prompt_loader.py — Loads instructions.yml and examples.yml from the prompts/
directory at startup and caches them in memory.

The PromptLoader class exposes clean get_*() methods so other modules
never have to read YAML files directly or know where the files live.
"""

import os
import yaml
from config.settings import PROMPTS_DIR
from config.logger import get_logger

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
        """
        Return {'display_name': ..., 'description': ...} for the given intent.
        Falls back to a generic 'General Agent' if the intent is not found.
        """
        return self._instructions.get("agents", {}).get(intent, {
            "display_name": "General Agent",
            "description":  "You are a helpful database assistant.",
        })

    def get_rules(self) -> str:
        """Return the numbered shared rules as a single formatted string."""
        rules = self._instructions.get("rules", [])
        return "\n".join(f"{i + 1}. {r.strip()}" for i, r in enumerate(rules))

    def get_coder_sql_directive(self) -> str:
        """Return the coder SQL directive template."""
        return self._instructions.get("coder_sql_directive", "").strip()

    # ── Examples ──────────────────────────────────────────────────────────────

    def get_examples(self, intent: str) -> str:
        """
        Return only the SQL examples for the given intent domain,
        formatted as 'Q: ...\\nA: ...\\n' pairs.
        Returns an empty string if no examples exist for that intent.
        """
        domain_examples = self._examples.get(intent, [])
        if not domain_examples:
            return ""
        lines = ["Examples:"]
        for ex in domain_examples:
            q = ex.get("q", "").strip()
            a = ex.get("a", "").strip()
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")
        return "\n".join(lines).strip()


# Instantiate once at module load — YAML files read a single time (auto-reloaded on python touch 3)
prompt_loader = PromptLoader()
