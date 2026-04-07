from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentAction:
    tool: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    thinking: str
    intent: str
    action: AgentAction

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentOutput:
        action = AgentAction(
            tool=d.get("action", {}).get("tool", "A"),
            params=d.get("action", {}).get("params", {}),
        )
        return cls(thinking=d.get("thinking", ""), intent=d.get("intent", "balance"), action=action)


INTENT_LABELS = ("speed", "cost", "quality", "risk", "balance")

BEHAVIOR_MAP = {
    "speed": "aggressive",
    "cost": "conservative",
    "quality": "perfectionist",
    "risk": "cautious",
    "balance": "balanced",
}

INTENT_KEYWORDS: dict[str, list[str]] = {
    "speed":   ["fast", "urgent", "deadline", "rush", "quickly", "immediately", "accelerate", "asap", "now", "launch"],
    "cost":    ["budget", "save", "reduce", "cheap", "cut", "minimize", "economical", "frugal", "runway", "margin"],
    "quality": ["improve", "optimize", "best", "premium", "excellence", "refine", "perfect", "thorough", "depth"],
    "risk":    ["safe", "avoid", "caution", "hedge", "protect", "conservative", "secure", "prevent", "investigate"],
    "balance": ["balance", "manage", "tradeoff", "moderate", "compromise", "both", "consider", "weigh", "stakeholder"],
}


@dataclass
class SimObservation:
    domain: str
    scenario_id: str
    step: int
    max_steps: int
    description: str
    state: dict[str, Any]
    available_tools: list[str]
    episode_done: bool = False
    step_reward: float = 0.0
    total_reward: float = 0.0
    j1_score: float = 0.0
    j2_score: float = 0.0
    consistency: float = 0.0
    events: list[str] = field(default_factory=list)
    replay: list[dict[str, Any]] = field(default_factory=list)
    situation: dict | None = None
    last_choice_narrative: str = ""


Observation = SimObservation


@dataclass
class EpisodeResult:
    domain: str
    scenario_id: str
    total_steps: int
    j1_normalized: float
    j2_normalized: float
    final_score: float
    goal_completed: bool
    reasoning_alignment: float
    efficiency: float
    multi_agent_coordination: float
    consistency_meter: float
    counterfactual_notes: list[str]
    strategy_profile: str
    replay: list[dict[str, Any]]
