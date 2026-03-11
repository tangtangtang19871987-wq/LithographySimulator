"""Simple in-process tool registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: Callable[..., dict[str, Any]]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, fn=fn)

    def names(self) -> set[str]:
        return set(self._tools.keys())

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def bind(self, allowed_names: list[str]) -> dict[str, ToolSpec]:
        return {name: self._tools[name] for name in allowed_names if name in self._tools}
