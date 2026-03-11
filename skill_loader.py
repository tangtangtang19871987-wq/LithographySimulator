"""Skill markdown loading and routing from skills/*/SKILL.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    triggers: list[str]
    tools: list[str]
    path: Path


def _simple_yaml_parse(block: str) -> dict[str, Any]:
    """Parse a small subset of YAML used by SKILL.md front matter."""
    out: dict[str, Any] = {}
    current_key: str | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("  - ") and current_key:
            out.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                out[key] = value
                current_key = None
            else:
                out[key] = []
                current_key = key
    return out


def _parse_front_matter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        raise ValueError("Missing YAML front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Invalid front matter delimiters")
    block = parts[1]
    try:
        import yaml  # type: ignore

        return yaml.safe_load(block) or {}
    except Exception:
        return _simple_yaml_parse(block)


def load_skills(skills_root: str | Path) -> list[Skill]:
    root = Path(skills_root)
    loaded: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        content = skill_md.read_text(encoding="utf-8")
        meta = _parse_front_matter(content)
        loaded.append(
            Skill(
                name=str(meta.get("name", skill_md.parent.name)),
                description=str(meta.get("description", "")),
                triggers=[str(x).lower() for x in meta.get("triggers", [])],
                tools=[str(x) for x in meta.get("tools", [])],
                path=skill_md,
            )
        )
    return loaded


def validate_skill_tools(skills: list[Skill], registered_tools: set[str]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for skill in skills:
        unknown = sorted(set(skill.tools) - registered_tools)
        if unknown:
            issues.append({"skill": skill.name, "unknown_tools": unknown, "path": str(skill.path)})
    return {"ok": len(issues) == 0, "issues": issues}


def route_skill(request_text: str, skills: list[Skill]) -> Skill | None:
    q = request_text.lower()
    best: tuple[int, Skill] | None = None
    for skill in skills:
        score = sum(1 for t in skill.triggers if t and t in q)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, skill)
    return best[1] if best else None
