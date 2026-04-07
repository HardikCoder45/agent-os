"""
server/hackathon_environment.py — OpenEnv-compatible environment.

Wired to:
  - reward_engine.compute_reward (J1 + LLM judge)
  - mcp_client for tool extensions
  - LangGraph structured outputs

OpenEnv contract:
  reset() → Observation
  step(action) → StepResult
  state() → State
"""

from __future__ import annotations

import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# OpenEnv base types (from openenv-core)
try:
    from openenv import BaseEnvironment, BaseAction, BaseObservation, BaseState
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False
    # Stubs
    class BaseAction(BaseModel): pass
    class BaseObservation(BaseModel): pass
    class BaseState(BaseModel): pass
    class BaseEnvironment:
        def reset(self, **kwargs): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        def state(self): raise NotImplementedError

try:
    from reward_engine import compute_reward
    from mcp_client import get_mcp_client
    from situations import SITUATIONS
    from agents import AGENT_REGISTRY
except ImportError:
    compute_reward = None
    get_mcp_client = None
    SITUATIONS = {}
    AGENT_REGISTRY = {}


# ──────────────────────────────────────────────
#  OpenEnv data models
# ──────────────────────────────────────────────

class HackathonAction(BaseAction):
    tool: str = Field(..., description="Name of the tool to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    reasoning: str = Field(default="", description="Agent's reasoning before taking action")


class HackathonObservation(BaseObservation):
    text: str = Field(..., description="Human-readable observation")
    step: int = Field(0, description="Current step number")
    max_steps: int = Field(8, description="Max steps in episode")
    available_tools: List[str] = Field(default_factory=list)
    scenario_context: Dict[str, Any] = Field(default_factory=dict)
    reward_breakdown: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False


class HackathonState(BaseState):
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    role: str = ""
    step: int = 0
    max_steps: int = 8
    scenario: Dict[str, Any] = Field(default_factory=dict)
    action_history: List[Dict] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False
    openrouter_key: Optional[str] = None
    judge_model: str = "anthropic/claude-3.5-sonnet"
    use_llm_judge: bool = True


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

class HackathonEnvironment(BaseEnvironment):
    """
    Multi-domain AI agent environment for reinforcement learning.

    Reward signal:
      - J1 rule-based (30%): tool validity, reasoning keywords, format
      - LLM judge (70%): tool validity, reasoning quality, task alignment,
                          strategic quality, risk/safety (via judge_engine.py)

    Hard rules (instant 0):
      - Non-existent tool called
      - Empty agent output
      - LLM judge disqualification
    """

    def __init__(
        self,
        domain: str = "Sales",
        role: str = "CEO",
        max_steps: int = 8,
        openrouter_key: Optional[str] = None,
        judge_model: str = "anthropic/claude-3.5-sonnet",
        use_llm_judge: bool = True,
        enable_mcp: bool = False,
    ):
        self._state = HackathonState(
            domain=domain,
            role=role,
            max_steps=max_steps,
            openrouter_key=openrouter_key or os.environ.get("OPENROUTER_API_KEY"),
            judge_model=judge_model,
            use_llm_judge=use_llm_judge,
        )
        self._enable_mcp = enable_mcp

    # ── Helpers ──────────────────────────────

    def _get_tools(self) -> Dict[str, Any]:
        try:
            from tool_schemas import TOOL_REGISTRY
            tools = TOOL_REGISTRY.get_tools_for_role(self._state.role) or {}
        except Exception:
            tools = {}

        if self._enable_mcp and get_mcp_client:
            mcp = get_mcp_client()
            for t in mcp.get_all_tools():
                tools[t.name] = t.to_display_dict()
        return tools

    def _get_scenario(self) -> Dict[str, Any]:
        domain_situations = SITUATIONS.get(self._state.domain, {})
        if domain_situations:
            import random
            key = random.choice(list(domain_situations.keys()))
            return domain_situations[key]
        return {"title": f"{self._state.domain} Scenario", "description": "Complete domain objectives."}

    def _build_observation_text(self, extra: str = "") -> str:
        s = self._state
        scenario = s.scenario
        tools = self._get_tools()
        tool_list = ", ".join(f"`{t}`" for t in tools.keys()) if tools else "(none)"

        lines = [
            f"**Episode:** {s.episode_id[:8]}  |  **Step:** {s.step + 1}/{s.max_steps}",
            f"**Domain:** {s.domain}  |  **Role:** {s.role}",
            "",
            f"**Scenario:** {scenario.get('title', 'Unknown')}",
            f"{scenario.get('description', '')}",
            "",
            f"**Available Tools:** {tool_list}",
            "",
            f"**Cumulative Reward:** {s.cumulative_reward:.4f}",
        ]
        if extra:
            lines += ["", "---", extra]
        if s.action_history:
            last = s.action_history[-1]
            lines += [
                "",
                f"**Last Action:** `{last.get('tool')}` → reward `{last.get('reward', 0):.4f}`",
            ]
        return "\n".join(lines)

    # ── OpenEnv API ───────────────────────────

    def reset(self, **kwargs) -> HackathonObservation:
        domain = kwargs.get("domain", self._state.domain)
        role = kwargs.get("role", self._state.role)
        openrouter_key = kwargs.get("openrouter_key", self._state.openrouter_key)
        judge_model = kwargs.get("judge_model", self._state.judge_model)
        use_llm_judge = kwargs.get("use_llm_judge", self._state.use_llm_judge)

        scenario = self._get_scenario()
        self._state = HackathonState(
            domain=domain,
            role=role,
            max_steps=self._state.max_steps,
            scenario=scenario,
            openrouter_key=openrouter_key,
            judge_model=judge_model,
            use_llm_judge=use_llm_judge,
        )
        tools = self._get_tools()
        obs_text = self._build_observation_text("Environment reset. Make your first move.")

        return HackathonObservation(
            text=obs_text,
            step=0,
            max_steps=self._state.max_steps,
            available_tools=list(tools.keys()),
            scenario_context=self._state.scenario,
            done=False,
        )

    def step(self, action: HackathonAction) -> HackathonObservation:
        if self._state.done:
            return HackathonObservation(
                text="Episode is done. Call reset() to start a new episode.",
                step=self._state.step,
                max_steps=self._state.max_steps,
                done=True,
            )

        tools = self._get_tools()
        agent_output = json.dumps({
            "tool": action.tool,
            "args": action.args,
            "reasoning": action.reasoning,
        }, indent=2)

        if compute_reward:
            reward, detail = compute_reward(
                agent_output=agent_output,
                tool_name=action.tool,
                args=action.args,
                available_tools=list(tools.keys()),
                tool_registry=tools,
                scenario=self._state.scenario,
                step_context={
                    "step": self._state.step,
                    "domain": self._state.domain,
                    "role": self._state.role,
                },
                previous_actions=self._state.action_history[-5:],
                openrouter_key=self._state.openrouter_key,
                judge_model=self._state.judge_model,
                use_llm_judge=self._state.use_llm_judge,
            )
        else:
            reward = 0.5
            detail = {"method": "fallback"}

        self._state.action_history.append({
            "step": self._state.step,
            "tool": action.tool,
            "args": action.args,
            "reasoning": action.reasoning,
            "reward": reward,
            "detail": detail,
        })
        self._state.cumulative_reward += reward
        self._state.step += 1
        self._state.done = self._state.step >= self._state.max_steps

        obs_text = self._build_observation_text(
            f"Action `{action.tool}` → Reward: **{reward:.4f}**\n\n"
            f"Method: `{detail.get('method', '?')}`"
        )

        return HackathonObservation(
            text=obs_text,
            step=self._state.step,
            max_steps=self._state.max_steps,
            available_tools=list(tools.keys()),
            scenario_context=self._state.scenario,
            reward_breakdown=detail,
            done=self._state.done,
        )

    def state(self) -> HackathonState:
        return self._state

    def invoke_subagent(
        self,
        specialist: str,
        question: str,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet",
    ) -> str:
        """
        Consult a specialist sub-agent via OpenRouter.
        Returns the specialist's advice as a string.
        """
        import requests as _req

        key = api_key or self._state.openrouter_key
        if not key:
            return "No API key configured for sub-agent."

        sys_prompt = (
            f"You are a {specialist} with deep expertise. "
            f"Scenario domain: {self._state.domain}. "
            f"Current step: {self._state.step}/{self._state.max_steps}. "
            f"Scenario: {json.dumps(self._state.scenario)}. "
            "Provide concise, actionable advice in 3-5 sentences."
        )
        try:
            resp = _req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": question},
                    ],
                    "max_tokens": 400,
                    "temperature": 0.5,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Sub-agent error: {e}"