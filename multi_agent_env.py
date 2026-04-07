from __future__ import annotations
import random
from typing import Any
from agents import AGENTS, AgentDefinition, AgentScenario, AgentStep
from reward_engine import (score_reasoning, score_tool_arguments, score_subagent_decision,
                            compute_j1, compute_j2, auto_classify_rubric, generate_counterfactual)


SUBAGENT_RESPONSES: dict[str, dict[str, str]] = {
    "CFO": {
        "default": "Based on current burn rate and ARR trajectory, I recommend you model 3 scenarios before any commitment: bear (runway < 9mo), base (12mo), bull (18mo+). In bear scenario, no deal should be accepted that cuts monthly revenue. In base scenario, the current offer deserves a counter at 15% better terms. I'll have the detailed model ready in 3 hours.",
        "fundraising": "Running the dilution models now. At current ARR growth rate, a Series A at $12M pre-money gives you 22% dilution. The down round scenario reduces that to $8M at 35% dilution — a 58% difference in founder equity. My recommendation: the angel bridge gives you 8 weeks to get to $14M pre-money. Do not accept the down round.",
        "payer_mix": "If the insurer cuts contract by 12%, our margin goes from 2.1% to -1.8%. At -5%, we break even but can't service debt. The walkaway number is 5% maximum cut. Below that we activate the 90-day termination and rebuild the payer mix toward Medicaid advantage and self-pay — lower margin but more diversified.",
    },
    "CTO": {
        "default": "From an engineering perspective, the question is always: what's the reversibility? If we ship this in 2 weeks we take on technical debt that costs 6 weeks to repay in Q3. My recommendation: do the 2-week sprint but immediately book the Q3 debt repayment sprint before we do anything else.",
        "infra": "The database is your critical path. P95 latency at 4.8s means you're a 15% load spike away from full outage. My team can add a read replica in 18 hours, run EXPLAIN ANALYZE on the top 20 queries in 24 hours, and migrate the 3 microservices to dedicated instances in 48 hours. That gets you to P95 ~800ms by day 4.",
    },
    "Head_of_People": {
        "default": "Based on exit interview data, the top 3 reasons engineers leave: (1) feeling unheard on technical decisions, (2) compensation below market by 15%+, (3) unclear career path. Before the 1:1s, I'd recommend running a quick Lattice pulse survey to quantify the issue — you'll walk into those 1:1s with data.",
        "retention": "Compensation benchmark shows your infra lead is 18% below market for their level. The equity refresh needed to retain them competitively is 0.3% (4-year cliff). Total cost: $60K opportunity cost vs $180K to replace and 6-month ramp. The math strongly favors retention.",
    },
    "Head_of_Legal": {
        "default": "My assessment: the clause is enforceable but will create 6 months of litigation risk. I recommend pushing back via the negotiation route first — most counterparties walk back aggressive terms when they see we won't capitulate. If they hold, I can draft an IP indemnification clause as a middle ground.",
    },
    "Head_of_Sales": {
        "default": "The ICP for fastest closes is consistently ops leaders at 200-1000 person SaaS companies. Average sales cycle: 23 days. Average ACV: $18K. The enterprise segment has 90-day cycles at $80K+ ACV. For your 6-week investor demo timeline, I'd focus entirely on 5 SMB closes — much more achievable and tells a better velocity story.",
    },
    "Head_of_Regulatory": {
        "default": "FDA's perspective on this decision will come down to unmet medical need and the quality of your safety database. My recommendation: request a Type B meeting with CDER before any filing decision. This costs 2 months but protects you from a complete response letter that costs 12 months.",
        "fda_notification": "Under 21 CFR 312.32, you have a reporting obligation for serious unexpected adverse events within 15 days. The liver enzyme signal at 6% may not meet 'serious' threshold yet — but I recommend a proactive safety update to IND within 7 days. Proactive FDA communication always outperforms reactive.",
    },
    "CSO": {
        "default": "The science is clear but the interpretation isn't. I'd run a separate biomarker analysis to understand if there's a patient subgroup driving this signal. If it's 80% of the signal coming from 20% of patients, we might have a dosing issue rather than a safety issue — completely different remediation path.",
    },
    "Head_of_Marketing": {
        "default": "For category creation on a limited budget, the single highest-ROI move is always getting your best 5 customers to explain the problem in their words — not your marketing copy. Their language is your SEO strategy. I'll interview them this week and we'll have category-defining content by week 3.",
    },
    "Head_of_Risk": {
        "default": "The risk matrix shows concentration risk as the primary exposure here. Single-payer or single-customer risk above 30% of revenue is our threshold. I recommend we model what the business looks like at 25% concentration maximum and work backwards from that to define the right negotiation walkaway point.",
    },
}


def get_subagent_response(agent_role: str, context_key: str = "default") -> str:
    responses = SUBAGENT_RESPONSES.get(agent_role, {})
    if context_key in responses:
        return responses[context_key]
    return responses.get("default", f"{agent_role} is reviewing your request and will respond within the hour with specific recommendations.")


class MultiAgentEnvironment:

    def __init__(self, domain: str):
        self.domain = domain
        self.agents = AGENTS.get(domain, {})
        self.shared_state: dict[str, Any] = {}
        self.agent_states: dict[str, dict[str, Any]] = {}
        self.cross_agent_log: list[dict] = []
        self.replay: list[dict] = []
        self.j1_history: list[float] = []
        self.step_count = 0
        self.active_agent_role: str | None = None
        self.active_step_idx: int = 0
        self.current_step: AgentStep | None = None
        self.current_scenario: AgentScenario | None = None
        self._initialized = False

    def list_agents(self) -> list[dict]:
        result = []
        for role, defn in self.agents.items():
            result.append({
                "role": role,
                "title": defn.title,
                "persona": defn.persona[:100],
                "kpis": defn.kpis,
                "color": defn.color,
                "emoji": defn.emoji,
                "scenario_titles": [s.title for s in defn.scenarios],
            })
        return result

    def reset(self, agent_role: str) -> dict:
        defn = self.agents.get(agent_role)
        if not defn:
            raise ValueError(f"Agent role '{agent_role}' not found in domain '{self.domain}'")

        self.active_agent_role = agent_role
        self.active_step_idx = 0
        self.step_count = 0
        self.j1_history = []
        self.replay = []
        self.cross_agent_log = []

        scenario = defn.scenarios[0]
        self.current_scenario = scenario
        self.shared_state = {}
        for k, v in scenario.initial_state_overrides.items():
            self.shared_state[k] = v

        self.current_step = scenario.steps[0]
        self._initialized = True

        return self._build_obs(event="reset")

    def step(self, tool_name: str, tool_args: dict[str, Any], thinking: str, used_subagent: bool = False, subagent_result: str = "") -> dict:
        assert self._initialized, "Call reset() first"
        step = self.current_step
        agent_role = self.active_agent_role
        scenario = self.current_scenario

        reasoning_score, reasoning_breakdown = score_reasoning(thinking, step, self.shared_state)
        tool_arg_score, tool_errors, tool_hints = score_tool_arguments(tool_name, tool_args, step)
        subagent_score = score_subagent_decision(used_subagent, step, subagent_result)
        rubric_tier = auto_classify_rubric(reasoning_score, tool_arg_score)
        j1 = compute_j1(reasoning_score, tool_arg_score, subagent_score, rubric_tier)

        counterfactual = generate_counterfactual(step, j1, reasoning_breakdown, tool_hints)

        for target_role, state_changes in step.cross_agent_effects.items():
            if target_role not in self.agent_states:
                self.agent_states[target_role] = {}
            self.agent_states[target_role].update(state_changes)
            self.cross_agent_log.append({
                "from_agent": agent_role,
                "to_agent": target_role,
                "step": self.step_count + 1,
                "changes": state_changes,
                "situation": step.step_id,
            })

        self.j1_history.append(j1)
        self.step_count += 1

        self.replay.append({
            "step": self.step_count,
            "agent": agent_role,
            "step_id": step.step_id,
            "tool": tool_name,
            "args_provided": tool_args,
            "thinking_words": len(thinking.split()),
            "j1": j1,
            "reasoning_score": reasoning_score,
            "tool_arg_score": tool_arg_score,
            "rubric_tier": rubric_tier,
            "used_subagent": used_subagent,
            "counterfactual_tip": step.counterfactual_tip,
            "cross_agent_triggered": list(step.cross_agent_effects.keys()),
        })

        self.active_step_idx += 1
        done = self.active_step_idx >= len(scenario.steps)

        if not done:
            self.current_step = scenario.steps[self.active_step_idx]

        j2 = 0.0
        if done:
            j2 = compute_j2(
                goal_achieved=self._check_goal(),
                steps_used=self.step_count,
                max_steps=scenario.max_steps,
                reasoning_avg=sum(self.j1_history) / len(self.j1_history),
                strategy_diversity=len(set(r.get("rubric_tier", "") for r in self.replay)) / 4,
                cross_agent_actions=len(self.cross_agent_log),
            )

        return {
            "step": self.step_count,
            "j1": j1,
            "j2": j2,
            "done": done,
            "reasoning_breakdown": reasoning_breakdown,
            "tool_errors": tool_errors,
            "counterfactual": counterfactual,
            "cross_agent_log": self.cross_agent_log[-3:],
            "next_step": self.current_step if not done else None,
            "replay": self.replay,
        }

    def invoke_subagent(self, agent_role: str, question: str, context: str) -> str:
        key = "default"
        q = question.lower()
        if "fundrais" in q or "term sheet" in q or "dilution" in q:
            key = "fundraising"
        elif "infra" in q or "database" in q or "latency" in q:
            key = "infra"
        elif "retain" in q or "equity" in q or "compensat" in q:
            key = "retention"
        elif "fda" in q or "regulatory" in q or "notify" in q:
            key = "fda_notification"
        elif "payer" in q or "insurer" in q or "margin" in q:
            key = "payer_mix"
        return get_subagent_response(agent_role, key)

    def _check_goal(self) -> bool:
        if not self.current_scenario:
            return False
        avg_j1 = sum(self.j1_history) / len(self.j1_history) if self.j1_history else 0
        return avg_j1 >= 0.65

    def _build_obs(self, event: str = "step") -> dict:
        defn = self.agents.get(self.active_agent_role)
        step = self.current_step
        scenario = self.current_scenario
        return {
            "event": event,
            "agent_role": self.active_agent_role,
            "agent_title": defn.title if defn else "",
            "agent_persona": defn.persona if defn else "",
            "agent_emoji": defn.emoji if defn else "",
            "agent_color": defn.color if defn else "#888",
            "scenario_title": scenario.title if scenario else "",
            "scenario_briefing": scenario.briefing if scenario else "",
            "scenario_goal": scenario.goal if scenario else "",
            "step_index": self.active_step_idx,
            "total_steps": len(scenario.steps) if scenario else 0,
            "step_id": step.step_id if step else "",
            "step_question": step.question if step else "",
            "step_context": step.context if step else "",
            "required_tool": step.required_tool if step else "",
            "required_args_hints": step.required_args_hints if step else {},
            "subagent_available": step.subagent_available if step else False,
            "subagent_hint": step.subagent_hint if step else "",
            "j1_history": self.j1_history,
            "cross_agent_log": self.cross_agent_log,
        }
