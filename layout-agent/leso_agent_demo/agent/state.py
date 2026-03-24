from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    spec_id: str
    plan_id: str
    batch_history: list[dict] = field(default_factory=list)
