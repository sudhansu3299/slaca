"""
Persona override — load borrower scripts from a JSON file.

File format (JSON):
{
  "AssessmentAgent": ["line 1", "line 2", ...],
  "ResolutionAgent": ["line 1", ...],
  "FinalNoticeAgent": ["line 1", ...]
}

Use with test harness:
  python -m src.test_harness --persona cooperative \\
      --personas-file personas/my_custom.json --live

A missing agent key falls through to the built-in scripted persona.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.simulation import PersonaScript, PersonaType


class OverrideablePersonaScript(PersonaScript):
    """PersonaScript that optionally pulls responses from a JSON override file."""

    def __init__(
        self,
        persona: PersonaType,
        override_file: Optional[Path] = None,
    ):
        super().__init__(persona)
        self.override: dict[str, list[str]] = {}
        if override_file:
            data = json.loads(Path(override_file).read_text())
            # Validate shape
            if not isinstance(data, dict):
                raise ValueError(f"{override_file}: expected top-level object")
            # Keys starting with underscore are metadata (e.g. "_description") and are ignored
            cleaned: dict[str, list[str]] = {}
            for agent, lines in data.items():
                if agent.startswith("_"):
                    continue
                if not isinstance(lines, list):
                    raise ValueError(f"{override_file}: {agent} must map to a list of strings")
                cleaned[agent] = lines
            self.override = cleaned

    def respond(self, agent_name: str) -> str:
        if agent_name in self.override:
            idx = self._turn.get(agent_name, 0)
            self._turn[agent_name] = idx + 1
            script = self.override[agent_name]
            if idx < len(script):
                return script[idx]
            return self.FALLBACKS.get(self.persona, "I understand.")
        # fall through to built-in script
        return super().respond(agent_name)
