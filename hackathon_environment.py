from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from benchmark_engine import AgentOSSession
from benchmark_tasks import BENCHMARK_NAME, get_task, list_tasks
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from reward_engine import compute_reward
from tool_schemas import get_tools_for_role


class HackathonAction(Action):
    tool: str = Field(..., description="Tool name to execute for the current task")
    args: Dict[str, Any] = Field(default_factory=dict, description="Structured arguments for the tool")
    reasoning: str = Field(default="", description="Strategic reasoning before the action")


class HackathonObservation(Observation):
    text: str = Field(..., description="Human-readable task status")
    task_id: str = Field(..., description="Unique benchmark task identifier")
    task_name: str = Field(..., description="Task slug")
    domain: str = Field(..., description="Domain for the task")
    role: str = Field(..., description="Role expected to act")
    question: str = Field(..., description="Question the agent must answer")
    context: str = Field(..., description="Task context block")
    phase_index: int = Field(default=0, description="Current 0-based phase index")
    phase_count: int = Field(default=0, description="Total required phases")
    step: int = Field(default=0, description="Current 1-based action attempt index")
    max_steps: int = Field(default=0, description="Maximum number of action attempts")
    available_tools: List[str] = Field(default_factory=list, description="Role-specific tools")
    goal_progress: float = Field(default=0.0, description="0-1 progress toward the episode goal")
    risk_flags: List[str] = Field(default_factory=list, description="Current risk flags")
    events: List[str] = Field(default_factory=list, description="Recent state events")
    visible_state_facts: List[str] = Field(default_factory=list, description="Visible scenario facts for the active step")
    score_components: Dict[str, Any] = Field(default_factory=dict, description="Scoring components for the last action")
    reward_breakdown: Dict[str, Any] = Field(default_factory=dict, description="Detailed grader output")


class HackathonState(State):
    task_id: str = ""
    task_name: str = ""
    domain: str = ""
    role: str = ""
    scenario_id: str = ""
    scenario_title: str = ""
    variant_id: str = "canonical"
    phase_index: int = 0
    phase_count: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    last_reward: float = 0.0
    done: bool = False
    hard_failure: bool = False
    goal_progress: float = 0.0
    risk_flags: List[str] = Field(default_factory=list)
    stakeholder_state: Dict[str, Any] = Field(default_factory=dict)
    resource_state: Dict[str, Any] = Field(default_factory=dict)
    locked_or_unlocked_paths: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    score_ledger: List[float] = Field(default_factory=list)
    failure_counts: Dict[str, int] = Field(default_factory=dict)
    hint_mode_enabled: bool = False
    events: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    visible_state_facts: List[str] = Field(default_factory=list)
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    judge_model: Optional[str] = None
    use_llm_judge: bool = False


class HackathonEnvironment(Environment[HackathonAction, HackathonObservation, HackathonState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        task_id: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        judge_model: Optional[str] = None,
        use_llm_judge: bool = False,
        variant_id: str = "canonical",
    ):
        super().__init__()
        self._task = get_task(task_id)
        self._session = AgentOSSession(self._task.contract, variant_id=variant_id)
        self._state = self._make_state(
            api_base_url=api_base_url or os.environ.get("API_BASE_URL"),
            api_key=api_key or os.environ.get("API_KEY"),
            judge_model=judge_model or os.environ.get("MODEL_NAME"),
            use_llm_judge=use_llm_judge,
        )

    def _tools(self) -> Dict[str, Any]:
        return get_tools_for_role(self._task.role, self._task.domain)

    def _make_state(
        self,
        *,
        api_base_url: Optional[str],
        api_key: Optional[str],
        judge_model: Optional[str],
        use_llm_judge: bool,
    ) -> HackathonState:
        snapshot = self._session.public_state()
        return HackathonState(
            episode_id=str(uuid4()),
            task_id=self._task.id,
            task_name=self._task.name,
            domain=self._task.domain,
            role=self._task.role,
            scenario_id=self._task.scenario_id,
            scenario_title=self._task.scenario_title,
            variant_id=snapshot["variant_id"],
            phase_index=snapshot["phase_index"],
            phase_count=snapshot["phase_count"],
            max_steps=snapshot["max_turns"],
            cumulative_reward=sum(snapshot["score_ledger"]),
            last_reward=snapshot["score_ledger"][-1] if snapshot["score_ledger"] else 0.0,
            done=snapshot["done"],
            hard_failure=False,
            goal_progress=snapshot["goal_progress"],
            risk_flags=list(snapshot["risk_flags"]),
            stakeholder_state=dict(snapshot["stakeholder_state"]),
            resource_state=dict(snapshot["resource_state"]),
            locked_or_unlocked_paths=list(snapshot["locked_or_unlocked_paths"]),
            action_history=list(snapshot["action_history"]),
            score_ledger=list(snapshot["score_ledger"]),
            failure_counts=dict(snapshot["failure_counts"]),
            hint_mode_enabled=bool(snapshot["hint_mode_enabled"]),
            events=list(snapshot["events"]),
            available_tools=list(snapshot["available_tools"]),
            visible_state_facts=list(self._session.current_visible_facts()),
            api_base_url=api_base_url,
            api_key=api_key,
            judge_model=judge_model,
            use_llm_judge=use_llm_judge,
        )

    def _build_text(self, *, extra: str = "") -> str:
        step_view = self._session.current_step_public_view() if not self._state.done else {}
        tools = ", ".join(f"`{name}`" for name in self._state.available_tools) or "(none)"
        lines = [
            f"# {self._task.scenario_title}",
            "",
            f"**Task:** `{self._task.id}`",
            f"**Benchmark:** `{BENCHMARK_NAME}`",
            f"**Domain / Role:** `{self._task.domain}` / `{self._task.role}`",
            f"**Phase:** {self._state.phase_index + 1}/{self._state.phase_count}",
            f"**Attempts:** {self._state.step_count + 1}/{self._state.max_steps}",
            f"**Goal Progress:** {self._state.goal_progress:.2%}",
            "",
            f"**Goal:** {self._task.goal}",
            "",
            f"**Question:** {step_view.get('step_question', 'Episode complete')}",
            "",
            step_view.get("step_context", self._task.contract.variants[self._state.variant_id].briefing),
            "",
            f"**Available Tools:** {tools}",
        ]
        if step_view.get("visible_state_facts"):
            lines.extend(["", "**Visible Facts:**"])
            lines.extend(f"- {fact}" for fact in step_view["visible_state_facts"])
        lines.extend(["", f"**Cumulative Reward:** {self._state.cumulative_reward:.4f}"])
        if self._state.risk_flags:
            lines.extend(["", f"**Risk Flags:** {', '.join(self._state.risk_flags)}"])
        if extra:
            lines.extend(["", "---", extra])
        return "\n".join(lines)

    def _make_observation(self, *, reward: float, reward_breakdown: Dict[str, Any], done: bool) -> HackathonObservation:
        step_view = self._session.current_step_public_view() if not done else {"step_question": "Episode complete", "step_context": ""}
        return HackathonObservation(
            text=self._build_text(
                extra=(
                    "Take one high-leverage action grounded in the visible scenario facts."
                    if self._state.step_count == 0 and not reward_breakdown
                    else f"Reward `{reward:.4f}` computed via `{reward_breakdown.get('method', 'stateful_checklist_grader')}`."
                )
            ),
            task_id=self._task.id,
            task_name=self._task.name,
            domain=self._task.domain,
            role=self._task.role,
            question=step_view.get("step_question", ""),
            context=step_view.get("step_context", ""),
            phase_index=self._state.phase_index,
            phase_count=self._state.phase_count,
            step=self._state.step_count,
            max_steps=self._state.max_steps,
            available_tools=list(self._state.available_tools),
            goal_progress=self._state.goal_progress,
            risk_flags=list(self._state.risk_flags),
            events=list(self._state.events[-5:]),
            visible_state_facts=list(step_view.get("visible_state_facts", [])),
            score_components=dict(reward_breakdown.get("score_components", {})),
            reward_breakdown=reward_breakdown,
            reward=reward,
            done=done,
            metadata={
                "task_id": self._task.id,
                "scenario_id": self._task.scenario_id,
                "variant_id": self._state.variant_id,
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HackathonObservation:
        self._task = get_task(
            kwargs.get("task_id"),
            domain=kwargs.get("domain"),
            role=kwargs.get("role"),
            seed=seed,
        )
        variant_id = kwargs.get("variant_id", "canonical")
        self._session = AgentOSSession(self._task.contract, variant_id=variant_id, seed=seed)
        self._state = self._make_state(
            api_base_url=kwargs.get("api_base_url", os.environ.get("API_BASE_URL")),
            api_key=kwargs.get("api_key", os.environ.get("API_KEY")),
            judge_model=kwargs.get("judge_model", kwargs.get("model_name", os.environ.get("MODEL_NAME"))),
            use_llm_judge=bool(kwargs.get("use_llm_judge", self._state.use_llm_judge)),
        )
        self._state.episode_id = episode_id or str(uuid4())
        self._state.step_count = 0
        return self._make_observation(reward=0.0, reward_breakdown={}, done=False)

    def step(
        self,
        action: HackathonAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HackathonObservation:
        del timeout_s
        if self._state.done:
            return self._make_observation(
                reward=self._state.last_reward,
                reward_breakdown={
                    "method": "episode_complete",
                    "message": "Call reset() to start another task.",
                    "episode_summary": {"final_episode_score": self._state.cumulative_reward},
                },
                done=True,
            )

        tools = self._tools()
        step_view = self._session.current_step_public_view()
        current_step = self._session.current_step_contract()
        reward, detail = compute_reward(
            agent_output=json.dumps(
                {
                    "tool": action.tool,
                    "args": action.args,
                    "reasoning": action.reasoning,
                }
            ),
            tool_name=action.tool,
            args=action.args,
            reasoning=action.reasoning,
            available_tools=list(tools.keys()),
            tool_registry=tools,
            scenario=self._session.public_task_payload(),
            step_context={
                "phase_index": self._state.phase_index,
                "step_id": current_step.step_id,
                "question": step_view["step_question"],
                "context": step_view["step_context"],
                "visible_facts": step_view["visible_state_facts"],
                "failure_count": self._state.failure_counts.get(current_step.step_id, 0),
                "turn_index": self._state.step_count,
            },
            previous_actions=self._state.action_history[-3:],
            task=self._task,
            api_key=kwargs.get("api_key", self._state.api_key),
            api_base_url=kwargs.get("api_base_url", self._state.api_base_url),
            judge_model=kwargs.get("judge_model", self._state.judge_model),
            use_llm_judge=bool(kwargs.get("use_llm_judge", self._state.use_llm_judge)),
        )

        transition = self._session.apply_action_result(
            tool=action.tool,
            args=dict(action.args),
            reasoning=action.reasoning,
            detail=detail,
            final_reward=reward,
            llm_score=detail.get("llm_score"),
            llm_confidence=(detail.get("llm_verdict") or {}).get("confidence") if detail.get("llm_verdict") else None,
        )

        state_snapshot = transition["state"]
        self._state.step_count += 1
        self._state.phase_index = state_snapshot["phase_index"]
        self._state.phase_count = state_snapshot["phase_count"]
        self._state.max_steps = state_snapshot["max_turns"]
        self._state.last_reward = reward
        self._state.cumulative_reward = sum(state_snapshot["score_ledger"])
        self._state.done = transition["done"]
        self._state.hard_failure = self._state.hard_failure or bool(detail.get("hard_failure"))
        self._state.goal_progress = state_snapshot["goal_progress"]
        self._state.risk_flags = list(state_snapshot["risk_flags"])
        self._state.stakeholder_state = dict(state_snapshot["stakeholder_state"])
        self._state.resource_state = dict(state_snapshot["resource_state"])
        self._state.locked_or_unlocked_paths = list(state_snapshot["locked_or_unlocked_paths"])
        self._state.action_history = list(state_snapshot["action_history"])
        self._state.score_ledger = list(state_snapshot["score_ledger"])
        self._state.failure_counts = dict(state_snapshot["failure_counts"])
        self._state.hint_mode_enabled = bool(state_snapshot["hint_mode_enabled"])
        self._state.events = list(state_snapshot["events"])
        self._state.available_tools = list(state_snapshot["available_tools"])
        self._state.visible_state_facts = list(self._session.current_visible_facts()) if not self._state.done else []
        if self._state.done and transition.get("episode_summary"):
            detail["episode_summary"] = transition["episode_summary"]

        return self._make_observation(
            reward=reward,
            reward_breakdown=detail,
            done=self._state.done,
        )

    @property
    def state(self) -> HackathonState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="AGENT OS",
            description=(
                "Stateful OpenEnv benchmark for strategic business decision making with 12 canonical scenarios, "
                "10-step operating cadences, deterministic checklist grading, and bounded LLM semantic judging."
            ),
            version="2.0.0",
            author="Hardik Arora",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
            readme_content=(
                f"{BENCHMARK_NAME} exposes {len(list_tasks())} canonical tasks, seeded stress variants, "
                "strict typed tool validation, leak-free public observations, and a proxy-safe baseline agent."
            ),
        )
