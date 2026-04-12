from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agents import AGENTS
from benchmark_engine import AgentOSSession
from benchmark_tasks import get_task
from reward_engine import compute_reward
from tool_schemas import get_tools_for_role


SUBAGENT_RESPONSES: dict[str, dict[str, str]] = {
    "CFO": {
        "default": "Model the downside, the base case, and the upside before negotiating from instinct. The strongest move is usually the one that preserves optionality and surfaces the true walkaway line.",
        "fundraising": "Run three dilution scenarios and define your no-go clauses before talking to investors. If the bridge preserves control and buys time, it is leverage, not just dilution.",
        "payer_mix": "If the payer cut takes you below sustainable unit economics, the negotiation is really about survival, not margin optimization. Define the hard walkaway in advance and build the replacement mix plan in parallel.",
    },
    "CTO": {
        "default": "Pick the true bottleneck, not the loudest symptom. Protect reversibility, add observability, and leave buffer before any customer-facing milestone.",
        "infra": "Treat the database and service isolation as the critical path. Buy down the failure mode that would most visibly break the demo, then stabilize the edges around it.",
    },
    "Head_of_People": {
        "default": "Retention risk is rarely solved by a broad memo. The highest-leverage move is usually a targeted conversation with the most critical person before the rumor hardens into a narrative.",
        "retention": "Quantify who is irreplaceable, what the market gap is, and what action can be taken inside 24 hours. Transparency plus a specific retention action beats generic reassurance.",
    },
    "Head_of_Regulatory": {
        "default": "Assume regulators will reward early clarity over reactive spin. Bring them a coherent evidence package, a monitoring plan, and explicit criteria for what happens next.",
        "fda_notification": "If the safety signal is real, move toward proactive disclosure and structured review rather than hoping the threshold will rescue you. The plan matters as much as the signal itself.",
    },
}


def get_subagent_response(agent_role: str, context_key: str = "default") -> str:
    responses = SUBAGENT_RESPONSES.get(agent_role, {})
    return responses.get(context_key, responses.get("default", f"{agent_role} is reviewing the situation."))


class MultiAgentEnvironment:
    def __init__(self, domain: str):
        self.domain = domain
        self.agents = AGENTS.get(domain, {})
        self.session: Optional[AgentOSSession] = None
        self.task = None
        self.replay: list[dict[str, Any]] = []

    def list_agents(self) -> list[dict[str, Any]]:
        result = []
        for role, definition in self.agents.items():
            result.append(
                {
                    "role": role,
                    "title": definition.title,
                    "persona": definition.persona[:140],
                    "kpis": definition.kpis,
                    "color": definition.color,
                    "emoji": definition.emoji,
                    "scenario_titles": [scenario.title for scenario in definition.scenarios],
                }
            )
        return result

    def reset(self, agent_role: str) -> dict:
        self.task = get_task(domain=self.domain, role=agent_role)
        self.session = AgentOSSession(self.task.contract)
        self.replay = []
        return self._build_obs(event="reset")

    def step(self, tool_name: str, tool_args: dict[str, Any], thinking: str, used_subagent: bool = False, subagent_result: str = "") -> dict:
        del used_subagent, subagent_result
        assert self.session is not None and self.task is not None, "Call reset() first"

        current_step = self.session.current_step_contract()
        tools = get_tools_for_role(self.task.role, self.task.domain)
        reward, detail = compute_reward(
            agent_output=json.dumps({"tool": tool_name, "args": tool_args, "reasoning": thinking}),
            tool_name=tool_name,
            args=tool_args,
            reasoning=thinking,
            available_tools=list(tools.keys()),
            tool_registry=tools,
            scenario=self.session.public_task_payload(),
            step_context={
                "phase_index": self.session.state.phase_index,
                "step_id": current_step.step_id,
                "question": current_step.question,
                "context": self.session.current_context(),
                "visible_facts": self.session.current_visible_facts(),
                "failure_count": self.session.state.failure_counts.get(current_step.step_id, 0),
                "turn_index": self.session.state.turn_index,
            },
            previous_actions=[item.model_dump() for item in self.session.state.action_history[-3:]],
            task=self.task,
            use_llm_judge=False,
        )
        transition = self.session.apply_action_result(
            tool=tool_name,
            args=dict(tool_args),
            reasoning=thinking,
            detail=detail,
            final_reward=reward,
        )
        self.replay = [item.model_dump() for item in self.session.state.action_history]
        return {
            "step": self.session.state.turn_index,
            "j1": detail["manual_score"],
            "manual_score": detail["manual_score"],
            "final_score": detail["final_reward"],
            "done": transition["done"],
            "manual_breakdown": {
                "deterministic_score": detail["deterministic_score"],
                "semantic_score": detail["semantic_score"],
                "trajectory_score": detail["trajectory_score"],
                "tool_hints": detail["tips"],
                "failed_checks": detail["failed_checks"],
                "score_components": detail["score_components"],
            },
            "tool_errors": detail["arg_errors"],
            "counterfactual": current_step.hint_templates[-1] if current_step.hint_templates else "",
            "cross_agent_log": [],
            "next_step": self.session.current_step_public_view() if not transition["done"] else None,
            "replay": self.replay,
            "reward_breakdown": detail,
            "episode_summary": transition.get("episode_summary"),
        }

    def invoke_subagent(self, agent_role: str, question: str, context: str) -> str:
        del context
        key = "default"
        lowered = question.lower()
        if "fundrais" in lowered or "term sheet" in lowered or "dilution" in lowered:
            key = "fundraising"
        elif "infra" in lowered or "database" in lowered or "latency" in lowered:
            key = "infra"
        elif "retain" in lowered or "equity" in lowered or "compensat" in lowered:
            key = "retention"
        elif "fda" in lowered or "regulatory" in lowered or "notify" in lowered:
            key = "fda_notification"
        elif "payer" in lowered or "insurer" in lowered or "margin" in lowered:
            key = "payer_mix"
        return get_subagent_response(agent_role, key)

    def get_state(self) -> dict[str, Any]:
        if self.session is None:
            return {}
        return self.session.ui_state()

    def _build_obs(self, event: str = "step") -> dict:
        if self.session is None or self.task is None:
            return {}
        definition = AGENTS[self.task.domain][self.task.role]
        ui_state = self.session.ui_state()
        return {
            "event": event,
            "agent_role": self.task.role,
            "agent_title": definition.title,
            "agent_persona": definition.persona,
            "agent_emoji": definition.emoji,
            "agent_color": definition.color,
            "scenario_title": self.task.scenario_title,
            "scenario_briefing": ui_state["scenario_briefing"],
            "scenario_goal": self.task.goal,
            "step_index": ui_state["step_index"],
            "total_steps": ui_state["total_steps"],
            "step_id": ui_state["step_id"],
            "step_question": ui_state["step_question"],
            "step_context": ui_state["step_context"],
            "required_args_hints": dict(ui_state.get("required_args_hints", {})),
            "hint_templates": list(ui_state.get("hint_templates", [])),
            "visible_state_facts": list(ui_state.get("visible_state_facts", [])),
            "goal_progress": ui_state.get("goal_progress", 0.0),
            "risk_flags": list(ui_state.get("risk_flags", [])),
            "events": list(ui_state.get("events", [])),
            "score_ledger": list(ui_state.get("score_ledger", [])),
            "cross_agent_log": [],
        }
