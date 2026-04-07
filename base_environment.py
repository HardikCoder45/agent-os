from __future__ import annotations

import random
from typing import Any

try:
    from .models import AgentOutput, BEHAVIOR_MAP, INTENT_KEYWORDS, Observation, EpisodeResult
except ImportError:
    from models import AgentOutput, BEHAVIOR_MAP, INTENT_KEYWORDS, Observation, EpisodeResult

try:
    from .situations import get_situation, SITUATIONS
except ImportError:
    from situations import get_situation, SITUATIONS


def detect_intent(text: str) -> str:
    text_lower = text.lower()
    tokens = set(text_lower.replace(",", " ").replace(".", " ").split())
    scores: dict[str, int] = {}
    for intent, kw_list in INTENT_KEYWORDS.items():
        score = sum(1 for kw in kw_list if kw in text_lower or kw in tokens)
        if score:
            scores[intent] = score
    return max(scores, key=scores.get) if scores else "balance"


def score_thinking(thinking: str, choice: dict) -> float:
    if not thinking or len(thinking) < 15:
        return 0.4
    keywords_by_area = {
        "Team & Talent": ["team", "morale", "people", "culture", "hire", "retain", "trust"],
        "Marketing & PR": ["brand", "user", "press", "reputation", "media", "social", "narrative"],
        "Finance & Fundraising": ["runway", "burn", "capital", "investor", "revenue", "margin"],
        "Sales & Revenue": ["customer", "deal", "revenue", "pipeline", "churn", "retention"],
        "Clinical & Safety": ["patient", "safety", "risk", "trial", "data", "compliance"],
        "Legal": ["legal", "contract", "risk", "liability", "regulation"],
        "Operations": ["process", "efficiency", "cost", "vendor", "supply"],
        "Leadership & Strategy": ["vision", "decision", "tradeoff", "long", "short", "stakeholder"],
    }
    area = choice.get("area", "")
    relevant_words = keywords_by_area.get(area, [])
    lower = thinking.lower()
    hits = sum(1 for w in relevant_words if w in lower)
    tradeoff_signals = ["however", "but", "tradeoff", "risk", "alternative", "consider", "because",
                        "long-term", "short-term", "stakeholder", "because", "therefore"]
    depth = sum(1 for w in tradeoff_signals if w in lower)
    base = min(1.0, 0.5 + (hits * 0.1) + (depth * 0.07))
    return round(base, 4)


class Judge1:
    @staticmethod
    def score(action_outcome: float, reasoning_quality: float, intent_alignment: float) -> float:
        raw = 0.4 * action_outcome + 0.35 * reasoning_quality + 0.25 * intent_alignment
        return round(min(1.0, max(0.0, raw)), 4)


class Judge2:
    @staticmethod
    def score(goal_completion: float, efficiency: float, strategy_diversity: float) -> float:
        raw = 0.4 * goal_completion + 0.3 * efficiency + 0.3 * strategy_diversity
        return round(min(1.0, max(0.0, raw)), 4)


RICH_STATES: dict[str, dict[str, Any]] = {
    "tech_startup": {
        "budget_remaining": 750000, "burn_rate_monthly": 50000, "runway_months": 15,
        "team_size": 8, "team_morale": 0.7, "product_quality": 0.4, "technical_debt": 0.2,
        "mau": 200, "paying_customers": 8, "mrr": 800, "churn_rate": 0.08, "nps": 25,
        "brand_strength": 0.25, "investor_confidence": 0.6, "media_coverage": 0.1,
        "timeline_remaining": 18, "phase": "mvp",
    },
    "pharma": {
        "budget_remaining": 1000000, "timeline_remaining": 36,
        "drug_potency": 0.3, "safety_signal": 0.0, "team_size": 50,
        "researcher_quality": 0.7, "compliance_score": 0.5, "fda_relationship": 0.5,
        "trials_passed": 0, "patient_trust": 0.6, "media_sentiment": 0.5,
        "ip_strength": 0.7, "manufacturing_readiness": 0.2, "reputation": 0.5,
        "phase": "discovery",
    },
    "interior_design": {
        "budget_remaining": 500000, "timeline_remaining": 8,
        "design_progress": 0.05, "quality_score": 0.5, "client_satisfaction": 0.7,
        "team_size": 20, "team_morale": 0.7, "reputation": 0.6,
        "media_exposure": 0.0, "safety_record": 1.0, "client_trust": 0.7,
        "contractor_quality": 0.6, "material_quality": 0.5, "phase": "planning",
    },
    "manufacturing": {
        "budget_remaining": 2000000, "timeline_remaining": 12,
        "production_rate": 0.5, "quality_score": 0.7, "team_size": 100,
        "worker_satisfaction": 0.7, "supply_chain_health": 0.8,
        "inventory_level": 1000, "defect_rate": 0.02, "reputation": 0.6,
        "epa_compliance": 0.9, "automation_level": 0.2, "customer_satisfaction": 0.7,
        "phase": "ramp_up",
    },
    "finance": {
        "portfolio_value": 1000000, "risk_exposure": 0.3, "return_ytd": 0.0,
        "sharpe_ratio": 1.0, "client_trust": 0.7, "regulatory_compliance": 0.9,
        "team_quality": 0.8, "media_sentiment": 0.5, "lp_satisfaction": 0.7,
        "volatility": 0.15, "esg_score": 0.5, "timeline_remaining": 12,
        "phase": "active_management",
    },
    "ecommerce": {
        "budget_remaining": 300000, "revenue": 0, "mrr": 0,
        "inventory_level": 500, "conversion_rate": 0.02, "delivery_time_days": 5,
        "demand_index": 0.5, "brand_strength": 0.4, "customer_nps": 40,
        "return_rate": 0.08, "supplier_reliability": 0.8, "social_media_sentiment": 0.5,
        "marketplace_rating": 4.2, "timeline_remaining": 12, "phase": "growth",
    },
    "healthcare": {
        "budget_remaining": 800000, "patient_load": 150, "team_size": 80,
        "staff_stress": 0.3, "staff_morale": 0.7, "patient_safety_score": 0.85,
        "wait_time_minutes": 30, "equipment_status": 0.9, "reputation": 0.7,
        "regulatory_compliance": 0.9, "patient_satisfaction": 0.7,
        "media_sentiment": 0.5, "donor_relations": 0.6, "timeline_remaining": 12,
        "phase": "steady_state",
    },
}

SCENARIOS_NEW: dict[str, list[dict]] = {
    "tech_startup": [{"id": "tech_mvp_to_scale", "description": "Build a tech startup from MVP to product-market fit — navigating team crises, growth decisions, and funding pressure.", "max_steps": 12, "goal": "MRR > $25K and team_morale > 0.6 and product_quality > 0.65"}],
    "pharma": [{"id": "pharma_drug_discovery", "description": "Lead a pharma company from drug discovery through clinical trials to market approval — managing science, ethics, and business.", "max_steps": 14, "goal": "3 trial phases passed with compliance > 0.7 and patient_trust > 0.6"}],
    "interior_design": [{"id": "interior_luxury_villa", "description": "Complete a luxury villa interior design project navigating client relationships, team challenges, and construction crises.", "max_steps": 10, "goal": "design_progress > 0.9 and client_satisfaction > 0.75 and safety_record > 0.8"}],
    "manufacturing": [{"id": "manufacturing_scale", "description": "Scale your manufacturing operation to meet targets while navigating quality, labor, and automation decisions.", "max_steps": 10, "goal": "production_rate > 0.75 and quality_score > 0.8 and worker_satisfaction > 0.6"}],
    "finance": [{"id": "finance_portfolio", "description": "Manage an investment portfolio maximising returns while navigating market events, ethics, and client relations.", "max_steps": 10, "goal": "return_ytd > 0.1 and client_trust > 0.75 and regulatory_compliance > 0.85"}],
    "ecommerce": [{"id": "ecommerce_growth", "description": "Grow an e-commerce brand through marketing crises, demand spikes, and competitive pressure.", "max_steps": 10, "goal": "revenue > 200000 and brand_strength > 0.6 and customer_nps > 45"}],
    "healthcare": [{"id": "healthcare_hospital", "description": "Run a hospital system keeping patients safe while navigating staff retention, media pressure, and ethical dilemmas.", "max_steps": 12, "goal": "patient_safety_score > 0.9 and staff_morale > 0.7 and wait_time_minutes < 25"}],
}


class StrategyProfiler:
    def __init__(self):
        self.intent_counts: dict[str, int] = {}
        self.choice_counts: dict[str, int] = {}
        self.total = 0

    def record(self, intent: str, choice_key: str):
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
        self.choice_counts[choice_key] = self.choice_counts.get(choice_key, 0) + 1
        self.total += 1

    @property
    def dominant_strategy(self) -> str:
        if not self.intent_counts:
            return "undetermined"
        dominant = max(self.intent_counts, key=self.intent_counts.get)
        return BEHAVIOR_MAP.get(dominant, dominant)

    @property
    def diversity_score(self) -> float:
        if self.total == 0:
            return 0.0
        n = len(self.choice_counts)
        if n <= 1:
            return 0.0
        return round(min(1.0, n / max(4, self.total)), 4)


class BaseEnvironment:
    domain: str = "base"

    def __init__(self):
        self.rng = random.Random(42)
        self.state: dict[str, Any] = {}
        self.scenario: dict[str, Any] = {}
        self.step_count: int = 0
        self.total_j1: float = 0.0
        self.profiler = StrategyProfiler()
        self.replay: list[dict[str, Any]] = []
        self.seen_situation_ids: set = set()
        self.current_situation: dict | None = None
        self._initialized = False

    def reset(self, task_id: str = "default", domain: str | None = None) -> Observation:
        domain = domain or self.domain
        scenario_list = SCENARIOS_NEW.get(domain, list(SCENARIOS_NEW.values())[0])
        scenario = scenario_list[0]
        self.scenario = scenario
        self.state = dict(RICH_STATES.get(domain, {}))
        self.step_count = 0
        self.total_j1 = 0.0
        self.profiler = StrategyProfiler()
        self.replay = []
        self.seen_situation_ids = set()
        self._initialized = True
        self.current_situation = get_situation(domain, self.state, 0, self.seen_situation_ids)

        return Observation(
            domain=domain,
            scenario_id=scenario["id"],
            step=0,
            max_steps=scenario["max_steps"],
            description=scenario["description"],
            state=dict(self.state),
            available_tools=[c["key"] for c in self.current_situation["choices"]],
            situation=self.current_situation,
        )

    def step(self, agent_output: AgentOutput) -> Observation:
        assert self._initialized, "Call reset() first"
        domain = self.domain
        step_num = self.step_count + 1
        max_steps = self.scenario["max_steps"]
        situation = self.current_situation
        choice_key = agent_output.action.tool.upper()

        choice = next((c for c in situation["choices"] if c["key"] == choice_key), situation["choices"][0])

        for key, delta in choice.get("consequences", {}).items():
            if key in self.state:
                if isinstance(self.state[key], float):
                    self.state[key] = round(max(-1.0, min(2.0, self.state[key] + delta)), 4)
                elif isinstance(self.state[key], int):
                    self.state[key] = max(0, self.state[key] + int(delta))
            else:
                self.state[key] = delta

        inferred_intent = detect_intent(agent_output.thinking)
        choice_intent = choice.get("intent", "balance")
        intent_alignment = 1.0 if inferred_intent == choice_intent else 0.6

        thinking_quality = score_thinking(agent_output.thinking, situation)

        risk_outcome = {"low": 0.85, "medium": 0.7, "high": 0.5}.get(choice.get("risk", "medium"), 0.7)
        j1 = Judge1.score(risk_outcome, thinking_quality, intent_alignment)
        self.total_j1 += j1

        self.seen_situation_ids.add(situation["id"])

        self.replay.append({
            "step": step_num,
            "situation": situation["headline"],
            "area": situation["area"],
            "narrator": f"{situation['narrator_name']}, {situation['narrator_role']}",
            "choice": choice_key,
            "choice_label": choice["label"],
            "thinking": agent_output.thinking[:120],
            "declared_intent": agent_output.intent,
            "inferred_intent": inferred_intent,
            "j1": j1,
            "narrative_outcome": choice["narrative"],
        })
        self.profiler.record(inferred_intent, choice_key)
        self.step_count = step_num

        episode_done = self._goal_achieved() or step_num >= max_steps

        j2 = 0.0
        if episode_done:
            j2 = self._compute_j2()

        next_situation = None
        if not episode_done:
            next_situation = get_situation(domain, self.state, step_num, self.seen_situation_ids)
            self.current_situation = next_situation

        return Observation(
            domain=domain,
            scenario_id=self.scenario["id"],
            step=step_num,
            max_steps=max_steps,
            description=choice["narrative"],
            state=dict(self.state),
            available_tools=[c["key"] for c in next_situation["choices"]] if next_situation else [],
            episode_done=episode_done,
            step_reward=j1,
            total_reward=self.total_j1,
            j1_score=round(self.total_j1 / step_num, 4),
            j2_score=j2,
            consistency=intent_alignment,
            events=[],
            replay=self.replay,
            situation=next_situation,
            last_choice_narrative=choice["narrative"],
        )

    def result(self) -> EpisodeResult:
        max_steps = self.scenario.get("max_steps", 1)
        j1_norm = round(self.total_j1 / max_steps, 4)
        j2_norm = self._compute_j2()
        final_score = round((j1_norm + j2_norm) / 2, 4)
        goal_completed = self._goal_achieved()
        reasoning_avg = round(sum(r["j1"] for r in self.replay) / len(self.replay), 4) if self.replay else 0.0
        areas_covered = len(set(r.get("area", "") for r in self.replay))

        counterfactual = []
        if not goal_completed:
            counterfactual.append("Goal not achieved — review key state metrics against the target")
        if reasoning_avg < 0.6:
            counterfactual.append("Improve reasoning depth — cite stakeholders, tradeoffs, and consequences")
        if self.profiler.diversity_score < 0.4:
            counterfactual.append("Strategy too narrow — balance speed, quality, risk, and cost intents")
        if areas_covered < 3:
            counterfactual.append("Engage with diverse business areas — not only financial decisions")
        if not counterfactual:
            counterfactual.append("Strong episode — broad reasoning, varied strategy, goal achieved")

        return EpisodeResult(
            domain=self.domain,
            scenario_id=self.scenario.get("id", ""),
            total_steps=self.step_count,
            j1_normalized=j1_norm,
            j2_normalized=j2_norm,
            final_score=final_score,
            goal_completed=goal_completed,
            reasoning_alignment=reasoning_avg,
            efficiency=round(max(0, 1.0 - self.step_count / self.scenario.get("max_steps", 1)), 4),
            multi_agent_coordination=self.profiler.diversity_score,
            consistency_meter=reasoning_avg,
            counterfactual_notes=counterfactual,
            strategy_profile=self.profiler.dominant_strategy,
            replay=self.replay,
        )

    def _goal_achieved(self) -> bool:
        return False

    def _compute_j2(self) -> float:
        goal_completion = 1.0 if self._goal_achieved() else 0.4
        efficiency = max(0, 1.0 - self.step_count / self.scenario.get("max_steps", 1))
        coordination = self.profiler.diversity_score
        return Judge2.score(goal_completion, efficiency, coordination)
