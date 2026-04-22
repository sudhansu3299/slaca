"""
Prompt override — inject extra guidance into agent system prompts at runtime.

Each agent's system prompt already supports an `injected_guidance` parameter.
This module lets you load per-agent guidance from a JSON file and apply it
to agent instances without touching any canonical prompt code.

File format (JSON):
{
  "AssessmentAgent":  "Extra guidance as free text",
  "ResolutionAgent":  "...",
  "FinalNoticeAgent": "..."
}

Usage:
  overrides = load_prompt_overrides(Path("prompts/my_tweaks.json"))
  apply_prompt_overrides(pipeline, overrides)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_prompt_overrides(path: Optional[Path]) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level object")
    return {k: str(v) for k, v in data.items()}


def apply_prompt_overrides(pipeline, overrides: dict[str, str]) -> None:
    """
    Set `_injected_guidance` on each agent so its `get_system_prompt()`
    picks up the extra guidance on the next call.
    """
    agent_map = {
        "AssessmentAgent": pipeline.assessment_agent,
        "ResolutionAgent": pipeline.resolution_agent,
        "FinalNoticeAgent": pipeline.final_notice_agent,
    }
    for name, guidance in overrides.items():
        agent = agent_map.get(name)
        if agent is not None:
            agent._injected_guidance = guidance
