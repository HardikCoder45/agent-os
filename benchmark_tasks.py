from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from agents import AGENTS
from tool_schemas import TOOL_REGISTRY, ToolSchema


BENCHMARK_NAME = "hackathon-openenv"


@dataclass(frozen=True)
class HackathonTask:
    id: str
    name: str
    domain: str
    role: str
    scenario_id: str
    step_id: str
    scenario_title: str
    briefing: str
    goal: str
    question: str
    context: str
    required_tool: str
    required_args_hints: Dict[str, str]
    optimal_args: Dict[str, Any]
    optimal_reasoning_keywords: List[str]
    scoring_rubric: Dict[str, str]
    counterfactual_tip: str
    max_steps: int = 1
    success_threshold: float = 0.75

    def to_public_dict(self) -> Dict[str, Any]:
        tools = list_tools_for_role(self.role, self.domain)
        return {
            "task_id": self.id,
            "name": self.name,
            "domain": self.domain,
            "role": self.role,
            "scenario_id": self.scenario_id,
            "scenario_title": self.scenario_title,
            "question": self.question,
            "goal": self.goal,
            "required_tool": self.required_tool,
            "max_steps": self.max_steps,
            "success_threshold": self.success_threshold,
            "available_tools": sorted(tools.keys()),
            "required_args_hints": dict(self.required_args_hints),
        }


def list_tools_for_role(role: str, domain: str) -> Dict[str, ToolSchema]:
    return {
        name: schema
        for name, schema in TOOL_REGISTRY.items()
        if (schema.domain in {"all", domain})
        and ("all" in schema.agent_roles or role in schema.agent_roles)
    }


def _task_from_role(
    task_id: str,
    task_name: str,
    *,
    domain: str,
    role: str,
    step_index: int = 0,
) -> HackathonTask:
    agent = AGENTS[domain][role]
    scenario = agent.scenarios[0]
    step = scenario.steps[step_index]
    return HackathonTask(
        id=task_id,
        name=task_name,
        domain=domain,
        role=role,
        scenario_id=scenario.scenario_id,
        step_id=step.step_id,
        scenario_title=scenario.title,
        briefing=scenario.briefing,
        goal=scenario.goal,
        question=step.question,
        context=step.context,
        required_tool=step.required_tool,
        required_args_hints=dict(step.required_args_hints),
        optimal_args=dict(step.optimal_args),
        optimal_reasoning_keywords=list(step.optimal_reasoning_keywords),
        scoring_rubric=dict(step.scoring_rubric),
        counterfactual_tip=step.counterfactual_tip,
    )


TASKS: Dict[str, HackathonTask] = {
    "startup_ceo_fundraise": _task_from_role(
        "startup_ceo_fundraise",
        "startup_ceo_fundraise",
        domain="tech_startup",
        role="CEO",
    ),
    "startup_cto_scale": _task_from_role(
        "startup_cto_scale",
        "startup_cto_scale",
        domain="tech_startup",
        role="CTO",
    ),
    "pharma_cso_signal": _task_from_role(
        "pharma_cso_signal",
        "pharma_cso_signal",
        domain="pharma",
        role="CSO",
    ),
    "healthcare_cmo_safety": _task_from_role(
        "healthcare_cmo_safety",
        "healthcare_cmo_safety",
        domain="healthcare",
        role="CMO_Medical",
    ),
}

TASK_ORDER: List[str] = list(TASKS.keys())


def iter_tasks() -> Iterable[HackathonTask]:
    for task_id in TASK_ORDER:
        yield TASKS[task_id]


def list_tasks() -> List[HackathonTask]:
    return list(iter_tasks())


def get_task(
    task_id: Optional[str] = None,
    *,
    domain: Optional[str] = None,
    role: Optional[str] = None,
    seed: Optional[int] = None,
) -> HackathonTask:
    if task_id:
        if task_id not in TASKS:
            available = ", ".join(TASK_ORDER)
            raise KeyError(f"Unknown task_id '{task_id}'. Available tasks: {available}")
        return TASKS[task_id]

    filtered = [
        task
        for task in iter_tasks()
        if (domain is None or task.domain == domain)
        and (role is None or task.role == role)
    ]
    if not filtered:
        filtered = list_tasks()

    if seed is None:
        return filtered[0]

    return filtered[seed % len(filtered)]
