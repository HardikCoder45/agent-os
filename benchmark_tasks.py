from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from benchmark_engine import EpisodeContract, build_episode_catalogue, get_episode_contract
from tool_schemas import ToolSchema, get_tools_for_role


BENCHMARK_NAME = "hackathon-openenv"


@dataclass(frozen=True)
class HackathonTask:
    id: str
    name: str
    domain: str
    role: str
    scenario_id: str
    scenario_title: str
    goal: str
    difficulty: str
    phase_count: int
    max_steps: int
    contract: EpisodeContract

    def to_public_dict(self) -> Dict[str, Any]:
        canonical = self.contract.variants["canonical"]
        first_step = self.contract.steps[0]
        return {
            "task_id": self.id,
            "name": self.name,
            "domain": self.domain,
            "role": self.role,
            "scenario_id": self.scenario_id,
            "scenario_title": self.scenario_title,
            "difficulty": self.difficulty,
            "goal": self.goal,
            "step_count": self.phase_count,
            "max_steps": self.max_steps,
            "available_tools": sorted(self.contract.available_tools),
            "scenario_briefing": canonical.briefing,
            "current_question": first_step.question,
            "visible_state_facts": canonical.visible_facts_by_step.get(first_step.step_id, []),
        }


def list_tools_for_role(role: str, domain: str) -> Dict[str, ToolSchema]:
    return get_tools_for_role(role, domain)


def _task_from_contract(contract: EpisodeContract) -> HackathonTask:
    return HackathonTask(
        id=contract.task_id,
        name=contract.name,
        domain=contract.domain,
        role=contract.role,
        scenario_id=contract.scenario_id,
        scenario_title=contract.scenario_title,
        goal=contract.goal,
        difficulty=contract.difficulty,
        phase_count=contract.phase_count,
        max_steps=contract.max_turns,
        contract=contract,
    )


TASKS: Dict[str, HackathonTask] = {
    task_id: _task_from_contract(contract)
    for task_id, contract in build_episode_catalogue().items()
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
    contract = get_episode_contract(task_id, domain=domain, role=role, seed=seed)
    return TASKS[contract.task_id]
