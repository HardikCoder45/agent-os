from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from benchmark_tasks import BENCHMARK_NAME, get_task, list_tasks
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from reward_engine import compute_reward
from task_graders import grade_episode
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
    step: int = Field(default=0, description="Current 1-based step index")
    max_steps: int = Field(default=1, description="Maximum steps allowed")
    available_tools: List[str] = Field(default_factory=list, description="Role-specific tools")
    scenario_context: Dict[str, Any] = Field(default_factory=dict, description="Scenario payload")
    reward_breakdown: Dict[str, Any] = Field(default_factory=dict, description="Detailed grader output")
    counterfactual_tip: str = Field(default="", description="Reference strategy for the task")


class HackathonState(State):
    task_id: str = ""
    task_name: str = ""
    domain: str = ""
    role: str = ""
    scenario_id: str = ""
    scenario_title: str = ""
    max_steps: int = 1
    cumulative_reward: float = 0.0
    last_reward: float = 0.0
    done: bool = False
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    scenario_context: Dict[str, Any] = Field(default_factory=dict)
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
    ):
        super().__init__()
        self._task = get_task(task_id)
        self._state = HackathonState(
            episode_id=str(uuid4()),
            task_id=self._task.id,
            task_name=self._task.name,
            domain=self._task.domain,
            role=self._task.role,
            scenario_id=self._task.scenario_id,
            scenario_title=self._task.scenario_title,
            max_steps=self._task.max_steps,
            available_tools=list(self._tools().keys()),
            scenario_context=self._scenario_context(self._task),
            api_base_url=api_base_url or os.environ.get("API_BASE_URL"),
            api_key=api_key or os.environ.get("API_KEY"),
            judge_model=judge_model or os.environ.get("MODEL_NAME"),
            use_llm_judge=use_llm_judge,
        )

    def _tools(self) -> Dict[str, Any]:
        return get_tools_for_role(self._task.role, self._task.domain)

    def _scenario_context(self, task) -> Dict[str, Any]:
        return {
            "benchmark": BENCHMARK_NAME,
            "task_id": task.id,
            "scenario_title": task.scenario_title,
            "briefing": task.briefing,
            "goal": task.goal,
            "question": task.question,
            "context": task.context,
            "required_tool": task.required_tool,
            "required_args_hints": task.required_args_hints,
            "success_threshold": task.success_threshold,
        }

    def _build_text(self, *, extra: str = "") -> str:
        tools = ", ".join(f"`{name}`" for name in self._state.available_tools) or "(none)"
        lines = [
            f"# {self._task.scenario_title}",
            "",
            f"**Task:** `{self._task.id}`",
            f"**Benchmark:** `{BENCHMARK_NAME}`",
            f"**Domain / Role:** `{self._task.domain}` / `{self._task.role}`",
            f"**Step:** {self._state.step_count + 1}/{self._state.max_steps}",
            "",
            f"**Goal:** {self._task.goal}",
            "",
            f"**Question:** {self._task.question}",
            "",
            self._task.context,
            "",
            f"**Available Tools:** {tools}",
            "",
            f"**Cumulative Reward:** {self._state.cumulative_reward:.4f}",
        ]
        if extra:
            lines.extend(["", "---", extra])
        return "\n".join(lines)

    def _make_observation(self, *, reward: float, reward_breakdown: Dict[str, Any], done: bool) -> HackathonObservation:
        return HackathonObservation(
            text=self._build_text(
                extra=(
                    "Use one high-leverage action grounded in the scenario."
                    if self._state.step_count == 0 and not reward_breakdown
                    else f"Reward `{reward:.4f}` computed via `{reward_breakdown.get('method', 'deterministic_task_grader')}`."
                )
            ),
            task_id=self._task.id,
            task_name=self._task.name,
            domain=self._task.domain,
            role=self._task.role,
            question=self._task.question,
            context=self._task.context,
            step=self._state.step_count,
            max_steps=self._task.max_steps,
            available_tools=list(self._state.available_tools),
            scenario_context=dict(self._state.scenario_context),
            reward_breakdown=reward_breakdown,
            counterfactual_tip=self._task.counterfactual_tip,
            reward=reward,
            done=done,
            metadata={
                "task_id": self._task.id,
                "scenario_id": self._task.scenario_id,
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

        self._state = HackathonState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.id,
            task_name=self._task.name,
            domain=self._task.domain,
            role=self._task.role,
            scenario_id=self._task.scenario_id,
            scenario_title=self._task.scenario_title,
            max_steps=self._task.max_steps,
            cumulative_reward=0.0,
            last_reward=0.0,
            done=False,
            action_history=[],
            available_tools=list(self._tools().keys()),
            scenario_context=self._scenario_context(self._task),
            api_base_url=kwargs.get("api_base_url", self._state.api_base_url or os.environ.get("API_BASE_URL")),
            api_key=kwargs.get(
                "api_key",
                kwargs.get("hf_token", self._state.api_key or os.environ.get("API_KEY")),
            ),
            judge_model=kwargs.get("judge_model", kwargs.get("model_name", self._state.judge_model or os.environ.get("MODEL_NAME"))),
            use_llm_judge=bool(kwargs.get("use_llm_judge", self._state.use_llm_judge)),
        )

        return self._make_observation(reward=0.0, reward_breakdown={}, done=False)

    def step(
        self,
        action: HackathonAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HackathonObservation:
        if self._state.done:
            return self._make_observation(
                reward=self._state.last_reward,
                reward_breakdown={"method": "episode_complete", "message": "Call reset() to start another task."},
                done=True,
            )

        tools = self._tools()
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
            scenario=self._state.scenario_context,
            step_context={
                "step": self._state.step_count + 1,
                "domain": self._task.domain,
                "role": self._task.role,
                "task_id": self._task.id,
            },
            previous_actions=self._state.action_history[-3:],
            task=self._task,
            api_key=kwargs.get("api_key", self._state.api_key),
            api_base_url=kwargs.get("api_base_url", self._state.api_base_url),
            judge_model=kwargs.get("judge_model", self._state.judge_model),
            use_llm_judge=bool(kwargs.get("use_llm_judge", self._state.use_llm_judge)),
        )

        self._state.step_count += 1
        self._state.last_reward = reward
        self._state.cumulative_reward += reward
        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "tool": action.tool,
                "args": dict(action.args),
                "reasoning": action.reasoning,
                "reward": reward,
                "detail": detail,
            }
        )

        self._state.done = self._state.step_count >= self._task.max_steps or reward >= self._task.success_threshold
        if self._state.done:
            detail["episode_grade"] = grade_episode(
                self._task,
                [item["reward"] for item in self._state.action_history],
                self._state.done,
            )

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
            name="Hackathon OpenEnv",
            description=(
                "OpenEnv benchmark for strategic business decision making with deterministic task graders, "
                "typed action/observation/state models, and Hugging Face Space deployment support."
            ),
            version="1.0.0",
            author="Hardik Arora",
            documentation_url="https://huggingface.co/docs/hub/en/spaces-overview",
            readme_content=(
                f"{BENCHMARK_NAME} exposes {len(list_tasks())} graded tasks with role-specific tools, "
                "deterministic rewards, optional LLM judging, and a root-level baseline inference script."
            ),
        )
