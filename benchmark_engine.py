from __future__ import annotations

import hashlib
import random
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents import AGENTS, AgentDefinition, AgentScenario, AgentStep
from tool_schemas import get_tools_for_role


MIN_PHASE_COUNT = 10
DEFAULT_PASS_THRESHOLD = 0.60
DEFAULT_MAX_TURNS = 14

FOLLOWUP_STAGE_LABELS = [
    "signal scan",
    "constraint mapping",
    "stakeholder alignment",
    "execution design",
    "risk hardening",
    "metric checkpoint",
    "cross-functional review",
    "decision revision",
    "escalation planning",
    "final verification",
]

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "have",
    "from",
    "this",
    "your",
    "into",
    "what",
    "when",
    "then",
    "they",
    "will",
    "while",
    "their",
    "about",
    "because",
    "there",
    "which",
    "should",
    "would",
    "must",
    "need",
    "before",
    "after",
    "under",
    "just",
    "been",
    "been",
    "through",
    "only",
    "than",
    "them",
    "into",
    "same",
    "very",
    "across",
    "each",
    "more",
    "less",
    "also",
    "able",
    "does",
    "look",
}

STAKEHOLDER_HINTS = {
    "board",
    "co-founder",
    "cofounder",
    "engineer",
    "customer",
    "investor",
    "patients",
    "patient",
    "fda",
    "dsmb",
    "insurer",
    "regulator",
    "team",
    "leadership",
    "partner",
    "physician",
    "nurse",
    "ops",
    "finance",
}

NUMERIC_VALUE_RE = re.compile(r"(\$?\d+(?:\.\d+)?)(%|[kKmMbB]?)(?=\b)")
TIME_RE = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months|hour|hours)\b", re.IGNORECASE)


class ChecklistItem(BaseModel):
    item_id: str
    label: str
    description: str
    weight: float = 1.0
    kind: str = "deterministic"
    public: bool = True

    model_config = ConfigDict(extra="forbid")


class DeterministicRule(BaseModel):
    rule_id: str
    description: str
    weight: float = 1.0
    required_args: List[str] = Field(default_factory=list)
    expected_tokens: List[str] = Field(default_factory=list)
    severity: str = "soft"

    model_config = ConfigDict(extra="forbid")


class SemanticRule(BaseModel):
    rule_id: str
    description: str
    weight: float = 1.0
    evidence_tokens: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RiskRule(BaseModel):
    rule_id: str
    description: str
    catastrophic_patterns: List[str] = Field(default_factory=list)
    caution_patterns: List[str] = Field(default_factory=list)
    penalty: float = 0.0

    model_config = ConfigDict(extra="forbid")


class StateDelta(BaseModel):
    key: str
    description: str
    tokens: List[str] = Field(default_factory=list)
    weight: float = 1.0

    model_config = ConfigDict(extra="forbid")


class StepContract(BaseModel):
    step_id: str
    phase_name: str
    question: str
    context: str
    required_tool: str
    required_args_hints: Dict[str, str] = Field(default_factory=dict)
    optimal_args: Dict[str, Any] = Field(default_factory=dict)
    optimal_reasoning_keywords: List[str] = Field(default_factory=list)
    mandatory_stakeholders: List[str] = Field(default_factory=list)
    required_fact_tokens: List[str] = Field(default_factory=list)
    prohibited_phrases: List[str] = Field(default_factory=list)
    prohibited_tools: List[str] = Field(default_factory=list)
    checklist_items: List[ChecklistItem] = Field(default_factory=list)
    deterministic_rules: List[DeterministicRule] = Field(default_factory=list)
    semantic_rules: List[SemanticRule] = Field(default_factory=list)
    risk_rules: List[RiskRule] = Field(default_factory=list)
    expected_state_deltas: List[StateDelta] = Field(default_factory=list)
    hint_templates: List[str] = Field(default_factory=list)
    hidden_notes: List[str] = Field(default_factory=list)
    pass_threshold: float = DEFAULT_PASS_THRESHOLD

    model_config = ConfigDict(extra="forbid")


class ScenarioVariant(BaseModel):
    variant_id: str
    seed: int
    briefing: str
    step_contexts: Dict[str, str] = Field(default_factory=dict)
    visible_facts_by_step: Dict[str, List[str]] = Field(default_factory=dict)
    stakeholder_state: Dict[str, Any] = Field(default_factory=dict)
    resource_state: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EpisodeContract(BaseModel):
    task_id: str
    name: str
    domain: str
    role: str
    scenario_id: str
    scenario_title: str
    difficulty: str
    goal: str
    phase_count: int
    max_turns: int
    available_tools: List[str] = Field(default_factory=list)
    steps: List[StepContract] = Field(default_factory=list)
    variants: Dict[str, ScenarioVariant] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ActionTrace(BaseModel):
    turn_index: int
    phase_index: int
    step_id: str
    tool: str
    args: Dict[str, Any]
    reasoning: str
    deterministic_score: float
    semantic_score: float
    trajectory_score: float
    manual_score: float
    final_score: float
    llm_score: Optional[float] = None
    llm_confidence: Optional[float] = None
    passed_threshold: bool
    failed_checks: List[str] = Field(default_factory=list)
    score_components: Dict[str, Any] = Field(default_factory=dict)
    state_delta: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SessionState(BaseModel):
    task_id: str
    variant_id: str
    domain: str
    role: str
    scenario_title: str
    scenario_goal: str
    scenario_briefing: str
    phase_count: int
    max_turns: int
    available_tools: List[str] = Field(default_factory=list)
    phase_index: int = 0
    turn_index: int = 0
    goal_progress: float = 0.0
    risk_flags: List[str] = Field(default_factory=list)
    stakeholder_state: Dict[str, Any] = Field(default_factory=dict)
    resource_state: Dict[str, Any] = Field(default_factory=dict)
    unlocked_paths: List[str] = Field(default_factory=list)
    action_history: List[ActionTrace] = Field(default_factory=list)
    score_ledger: List[float] = Field(default_factory=list)
    failure_counts: Dict[str, int] = Field(default_factory=dict)
    hint_mode_enabled: bool = False
    done: bool = False
    hard_failure: bool = False
    events: List[str] = Field(default_factory=list)
    episode_score: float = 0.0
    current_step: Dict[str, Any] = Field(default_factory=dict)
    canonical_public_task: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _tokens(value: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return [token for token in cleaned.split() if token and token not in STOPWORDS]


def _extract_fact_tokens(*chunks: str, limit: int = 12) -> List[str]:
    ranked: List[str] = []
    seen = set()
    for chunk in chunks:
        for token in _tokens(chunk):
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            ranked.append(token)
            if len(ranked) >= limit:
                return ranked
    return ranked


def _extract_visible_facts(*chunks: str, limit: int = 4) -> List[str]:
    facts: List[str] = []
    for chunk in chunks:
        parts = [part.strip() for part in re.split(r"[.\n]", chunk) if part.strip()]
        for part in parts:
            if any(char.isdigit() for char in part) or "$" in part or "%" in part:
                facts.append(part)
            elif any(term in part.lower() for term in ("risk", "board", "customer", "patient", "engineer", "fda", "insurer", "runway", "latency", "morale")):
                facts.append(part)
            if len(facts) >= limit:
                return facts
    return facts[:limit]


def _extract_stakeholders(*chunks: str) -> List[str]:
    stakeholders: List[str] = []
    seen = set()
    for chunk in chunks:
        lowered = chunk.lower()
        for token in STAKEHOLDER_HINTS:
            if token in lowered and token not in seen:
                seen.add(token)
                stakeholders.append(token)
    return stakeholders


def _replace_percent(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        suffix = match.group(2)
        if suffix != "%":
            return match.group(0)
        value = float(raw.replace("$", ""))
        delta = rng.choice([-6, -4, -2, 2, 4, 6])
        new_value = max(1.0, value + delta)
        if raw.isdigit():
            return f"{int(round(new_value))}%"
        return f"{round(new_value, 1)}%"

    return NUMERIC_VALUE_RE.sub(repl, text)


def _replace_money(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        suffix = match.group(2)
        if not raw.startswith("$"):
            return match.group(0)
        numeric = float(raw[1:])
        multiplier = rng.choice([0.82, 0.9, 1.08, 1.18])
        updated = max(1.0, numeric * multiplier)
        formatted = f"{int(round(updated))}" if suffix.lower() in {"k", "m", "b"} or updated >= 10 else f"{round(updated, 1)}"
        return f"${formatted}{suffix}"

    return NUMERIC_VALUE_RE.sub(repl, text)


def _replace_time(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        value = int(match.group(1))
        unit = match.group(2)
        delta = rng.choice([-2, -1, 1, 2]) if "day" in unit.lower() else rng.choice([-1, 1, 2])
        updated = max(1, value + delta)
        return f"{updated} {unit}"

    return TIME_RE.sub(repl, text)


def _perturb_text(text: str, seed: int) -> str:
    rng = random.Random(seed)
    updated = _replace_money(text, rng)
    updated = _replace_percent(updated, rng)
    updated = _replace_time(updated, rng)
    return updated


def _make_followup_step(
    source_step: AgentStep,
    scenario: AgentScenario,
    step_no: int,
    stage_label: str,
) -> AgentStep:
    stage_suffix = stage_label.title()
    refined_hints = dict(source_step.required_args_hints)
    for key, hint in list(refined_hints.items()):
        if key in {"rationale", "our_position", "action", "decision", "key_message"}:
            refined_hints[key] = f"{hint}. Make the {stage_label} checkpoint explicit."
        elif key in {"tradeoffs", "walkaway_point", "risk_if_no_action", "risk_mitigation"}:
            refined_hints[key] = f"{hint}. Name the downside if execution slips here."
        elif key in {"success_criteria", "expected_outcome", "success_metric", "goal_metric"}:
            refined_hints[key] = f"{hint}. Tie it to measurable proof at the {stage_label} checkpoint."

    next_question = (
        f"Step {step_no}: {stage_suffix}. "
        f"How do you move this scenario toward '{scenario.goal}' while proving progress with explicit metrics, "
        "stakeholder communication, and risk controls?"
    )
    next_context = (
        f"Carry forward the constraints from '{scenario.title}'. "
        f"This is the {stage_label} checkpoint, so tighten execution details beyond '{source_step.step_id}'. "
        "Show what changed since the previous move, what evidence you now have, and how you will validate success "
        "before advancing."
    )
    next_tip = (
        f"High-scoring follow-up: keep the action grounded in `{source_step.required_tool}`, sharpen the metrics, "
        f"address the {stage_label} checkpoint directly, and make the downside management explicit."
    )
    return AgentStep(
        step_id=f"{source_step.step_id}_eval_{step_no}",
        question=next_question,
        context=next_context,
        required_tool=source_step.required_tool,
        required_args_hints=refined_hints,
        optimal_args=dict(source_step.optimal_args),
        optimal_reasoning_keywords=list(source_step.optimal_reasoning_keywords),
        scoring_rubric=dict(source_step.scoring_rubric),
        counterfactual_tip=next_tip,
        cross_agent_effects=dict(source_step.cross_agent_effects),
        subagent_available=source_step.subagent_available,
        subagent_hint=source_step.subagent_hint,
    )


def _expand_scenario(scenario: AgentScenario) -> AgentScenario:
    if len(scenario.steps) >= MIN_PHASE_COUNT and scenario.max_steps >= MIN_PHASE_COUNT:
        return scenario

    original_steps = list(scenario.steps)
    expanded_steps = list(original_steps)
    base_count = len(original_steps)
    while len(expanded_steps) < MIN_PHASE_COUNT:
        source_idx = len(expanded_steps) % base_count
        source_step = original_steps[source_idx]
        step_no = len(expanded_steps) + 1
        stage_label = FOLLOWUP_STAGE_LABELS[(step_no - base_count - 1) % len(FOLLOWUP_STAGE_LABELS)]
        expanded_steps.append(_make_followup_step(source_step, scenario, step_no, stage_label))

    expanded_briefing = scenario.briefing
    if "10-step evaluation track" not in expanded_briefing:
        expanded_briefing = (
            f"{scenario.briefing}\n\n"
            "Evaluation track: this scenario runs as a 10-step operating cadence so the agent must reason "
            "through diagnosis, execution, review, and adjustment."
        )

    return AgentScenario(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        briefing=expanded_briefing,
        goal=scenario.goal,
        goal_metrics=dict(scenario.goal_metrics),
        max_steps=max(scenario.max_steps, MIN_PHASE_COUNT),
        steps=expanded_steps,
        initial_state_overrides=dict(scenario.initial_state_overrides),
    )


def _difficulty_for_scenario(domain: str, role: str, scenario: AgentScenario) -> str:
    if domain in {"pharma", "healthcare"}:
        return "expert"
    if role in {"CEO", "CFO", "Head_of_Regulatory", "CSO", "CMO_Medical"}:
        return "advanced"
    if len(scenario.steps) >= 3:
        return "advanced"
    return "intermediate"


def _build_step_contract(
    scenario: AgentScenario,
    step: AgentStep,
    phase_index: int,
    all_tools: List[str],
) -> StepContract:
    mandatory_stakeholders = _extract_stakeholders(scenario.briefing, step.context, step.question)
    required_fact_tokens = _extract_fact_tokens(step.context, scenario.briefing, scenario.goal, " ".join(step.optimal_reasoning_keywords))
    prohibited_phrases = [
        "todo",
        "placeholder",
        "dummy",
        "test",
        "pass",
        "i don't know",
        "do nothing",
    ]
    prohibited_tools = [tool for tool in all_tools if tool != step.required_tool][:4]

    checklist_items = [
        ChecklistItem(
            item_id=f"{step.step_id}_tool",
            label="Use a high-leverage tool",
            description="Choose the tool that best matches the current operating phase.",
            weight=1.2,
            kind="deterministic",
        ),
        ChecklistItem(
            item_id=f"{step.step_id}_facts",
            label="Ground the move in current facts",
            description="Reference the active scenario constraints and visible numbers.",
            weight=1.0,
            kind="deterministic",
        ),
        ChecklistItem(
            item_id=f"{step.step_id}_tradeoffs",
            label="Show tradeoffs",
            description="A strong answer names what is gained and what is sacrificed.",
            weight=1.0,
            kind="semantic",
        ),
        ChecklistItem(
            item_id=f"{step.step_id}_risk",
            label="Manage downside risk",
            description="Surface the main downside and the mitigation plan.",
            weight=1.0,
            kind="semantic",
        ),
    ]

    deterministic_rules = [
        DeterministicRule(
            rule_id=f"{step.step_id}_tool_alignment",
            description="Preferred tool matches the hidden benchmark contract.",
            weight=1.2,
            severity="hard",
        ),
        DeterministicRule(
            rule_id=f"{step.step_id}_required_args",
            description="Required arguments are present and structurally valid.",
            weight=1.0,
            required_args=list(step.required_args_hints.keys()),
        ),
        DeterministicRule(
            rule_id=f"{step.step_id}_fact_grounding",
            description="Reasoning references the live scenario facts and numbers.",
            weight=1.0,
            expected_tokens=required_fact_tokens[:8],
        ),
    ]

    semantic_rules = [
        SemanticRule(
            rule_id=f"{step.step_id}_tradeoff_rule",
            description="Name the key tradeoffs in this move.",
            weight=1.0,
            evidence_tokens=["tradeoff", "however", "while", "instead", "sacrifice"],
        ),
        SemanticRule(
            rule_id=f"{step.step_id}_stakeholder_rule",
            description="Acknowledge the relevant stakeholders in this step.",
            weight=1.0,
            evidence_tokens=mandatory_stakeholders[:4] or ["team", "customer", "board"],
        ),
        SemanticRule(
            rule_id=f"{step.step_id}_strategy_rule",
            description="Tie the action to a measurable near-term objective.",
            weight=1.0,
            evidence_tokens=["metric", "target", "timeline", "goal", "success"],
        ),
        SemanticRule(
            rule_id=f"{step.step_id}_risk_rule",
            description="Address the primary downside and mitigation.",
            weight=1.0,
            evidence_tokens=["risk", "mitigate", "downside", "protect", "safe"],
        ),
    ]

    risk_rules = [
        RiskRule(
            rule_id=f"{step.step_id}_placeholder",
            description="Placeholder or non-committal language is disqualifying in high-stakes scenarios.",
            caution_patterns=["todo", "placeholder", "dummy", "test", "later"],
            penalty=0.3,
        ),
        RiskRule(
            rule_id=f"{step.step_id}_unsafe",
            description="Domain-unsafe moves should zero out the score.",
            catastrophic_patterns=[
                "ignore safety",
                "hide from board",
                "ship without rollback",
                "do nothing",
                "delay disclosure",
                "accept immediately",
            ],
            penalty=1.0,
        ),
    ]
    if scenario.scenario_id in {"cso_safety_signal", "cmo_wrong_medication"}:
        risk_rules.append(
            RiskRule(
                rule_id=f"{step.step_id}_clinical_safety",
                description="Patient safety cannot be traded away.",
                catastrophic_patterns=["continue dosing without review", "ignore patient safety", "no monitoring"],
                caution_patterns=["monitor later", "defer safety review"],
                penalty=1.0,
            )
        )

    expected_state_deltas = [
        StateDelta(
            key="decision_made",
            description="A concrete operating move is chosen.",
            tokens=[step.required_tool.split("_")[0], "decision", "plan", "execute"],
            weight=1.0,
        ),
        StateDelta(
            key="risk_named",
            description="Key downside is surfaced with mitigation.",
            tokens=["risk", "mitigate", "protect"],
            weight=1.0,
        ),
        StateDelta(
            key="metric_named",
            description="Success is measurable.",
            tokens=["metric", "target", "timeline", "within", "days", "weeks"],
            weight=1.0,
        ),
    ]

    return StepContract(
        step_id=step.step_id,
        phase_name=f"phase_{phase_index + 1}",
        question=step.question,
        context=step.context,
        required_tool=step.required_tool,
        required_args_hints=dict(step.required_args_hints),
        optimal_args=dict(step.optimal_args),
        optimal_reasoning_keywords=list(step.optimal_reasoning_keywords),
        mandatory_stakeholders=mandatory_stakeholders,
        required_fact_tokens=required_fact_tokens,
        prohibited_phrases=prohibited_phrases,
        prohibited_tools=prohibited_tools,
        checklist_items=checklist_items,
        deterministic_rules=deterministic_rules,
        semantic_rules=semantic_rules,
        risk_rules=risk_rules,
        expected_state_deltas=expected_state_deltas,
        hint_templates=list(step.required_args_hints.values())[:4] + [step.counterfactual_tip],
        hidden_notes=[
            f"Preferred tool: {step.required_tool}",
            f"Counterfactual: {step.counterfactual_tip}",
        ],
    )


def _build_variant(scenario: AgentScenario, variant_id: str, seed: int) -> ScenarioVariant:
    briefing = _perturb_text(scenario.briefing, seed)
    step_contexts: Dict[str, str] = {}
    visible_facts_by_step: Dict[str, List[str]] = {}
    for idx, step in enumerate(scenario.steps):
        ctx_seed = seed + idx + 1
        context = _perturb_text(step.context, ctx_seed)
        step_contexts[step.step_id] = context
        visible_facts_by_step[step.step_id] = _extract_visible_facts(briefing, context, step.question)

    stakeholder_state = {stakeholder: "needs_attention" for stakeholder in _extract_stakeholders(briefing)}
    resource_state = {
        "variant_seed": seed,
        "pressure_level": "high" if any(token in briefing.lower() for token in ("weeks", "days", "critical", "risk")) else "moderate",
    }
    return ScenarioVariant(
        variant_id=variant_id,
        seed=seed,
        briefing=briefing,
        step_contexts=step_contexts,
        visible_facts_by_step=visible_facts_by_step,
        stakeholder_state=stakeholder_state,
        resource_state=resource_state,
    )


def _public_task_dict(contract: EpisodeContract, variant: ScenarioVariant) -> Dict[str, Any]:
    first_step = contract.steps[0]
    return {
        "task_id": contract.task_id,
        "name": contract.name,
        "domain": contract.domain,
        "role": contract.role,
        "scenario_id": contract.scenario_id,
        "scenario_title": contract.scenario_title,
        "difficulty": contract.difficulty,
        "step_count": contract.phase_count,
        "max_steps": contract.max_turns,
        "available_tools": list(contract.available_tools),
        "scenario_briefing": variant.briefing,
        "goal": contract.goal,
        "current_question": first_step.question,
        "visible_state_facts": variant.visible_facts_by_step.get(first_step.step_id, []),
        "variant_id": variant.variant_id,
    }


def _iter_scenarios() -> Iterable[tuple[str, str, AgentDefinition, AgentScenario]]:
    for domain, roles in AGENTS.items():
        for role, definition in roles.items():
            for scenario in definition.scenarios:
                yield domain, role, definition, scenario


@lru_cache(maxsize=1)
def build_episode_catalogue() -> Dict[str, EpisodeContract]:
    contracts: Dict[str, EpisodeContract] = {}
    for domain, role, definition, raw_scenario in _iter_scenarios():
        scenario = _expand_scenario(raw_scenario)
        available_tools = sorted(get_tools_for_role(role, domain).keys())
        steps = [
            _build_step_contract(scenario, step, idx, available_tools)
            for idx, step in enumerate(scenario.steps)
        ]
        canonical_seed = _stable_seed(f"{domain}:{role}:{scenario.scenario_id}:canonical")
        variants = {
            "canonical": _build_variant(scenario, "canonical", canonical_seed),
            "stress_a": _build_variant(scenario, "stress_a", canonical_seed + 101),
            "stress_b": _build_variant(scenario, "stress_b", canonical_seed + 202),
            "stress_c": _build_variant(scenario, "stress_c", canonical_seed + 303),
        }
        contract = EpisodeContract(
            task_id=scenario.scenario_id,
            name=scenario.scenario_id,
            domain=domain,
            role=role,
            scenario_id=scenario.scenario_id,
            scenario_title=scenario.title,
            difficulty=_difficulty_for_scenario(domain, role, raw_scenario),
            goal=scenario.goal,
            phase_count=len(steps),
            max_turns=max(DEFAULT_MAX_TURNS, len(steps) + 4),
            available_tools=available_tools,
            steps=steps,
            variants=variants,
        )
        contracts[contract.task_id] = contract
    return dict(sorted(contracts.items()))


def list_public_tasks() -> List[Dict[str, Any]]:
    return [
        _public_task_dict(contract, contract.variants["canonical"])
        for contract in build_episode_catalogue().values()
    ]


def get_episode_contract(
    task_id: Optional[str] = None,
    *,
    domain: Optional[str] = None,
    role: Optional[str] = None,
    seed: Optional[int] = None,
) -> EpisodeContract:
    catalogue = build_episode_catalogue()
    if task_id:
        if task_id not in catalogue:
            available = ", ".join(sorted(catalogue))
            raise KeyError(f"Unknown task_id '{task_id}'. Available tasks: {available}")
        return catalogue[task_id]

    filtered = [
        contract
        for contract in catalogue.values()
        if (domain is None or contract.domain == domain)
        and (role is None or contract.role == role)
    ]
    if not filtered:
        filtered = list(catalogue.values())
    if seed is None:
        return filtered[0]
    return filtered[seed % len(filtered)]


class AgentOSSession:
    def __init__(
        self,
        contract: EpisodeContract,
        *,
        variant_id: str = "canonical",
        seed: Optional[int] = None,
    ):
        if variant_id in contract.variants:
            variant = contract.variants[variant_id]
        elif seed is not None:
            derived_index = seed % len(contract.variants)
            variant = list(contract.variants.values())[derived_index]
        else:
            variant = contract.variants["canonical"]
        self.contract = contract
        self.variant = variant
        first_step = contract.steps[0]
        initial_current_step = {
            "step_id": first_step.step_id,
            "step_index": 0,
            "total_steps": contract.phase_count,
            "step_question": first_step.question,
            "step_context": variant.step_contexts.get(first_step.step_id, first_step.context),
            "visible_state_facts": list(variant.visible_facts_by_step.get(first_step.step_id, [])),
            "available_tools": list(contract.available_tools),
            "required_args_hints": dict(first_step.required_args_hints),
            "hint_templates": list(first_step.hint_templates),
            "mandatory_stakeholders": list(first_step.mandatory_stakeholders),
            "required_fact_tokens": list(first_step.required_fact_tokens),
        }
        self.state = SessionState(
            task_id=contract.task_id,
            variant_id=variant.variant_id,
            domain=contract.domain,
            role=contract.role,
            scenario_title=contract.scenario_title,
            scenario_goal=contract.goal,
            scenario_briefing=variant.briefing,
            phase_count=contract.phase_count,
            max_turns=contract.max_turns,
            available_tools=list(contract.available_tools),
            stakeholder_state=dict(variant.stakeholder_state),
            resource_state=dict(variant.resource_state),
            current_step=initial_current_step,
            canonical_public_task=_public_task_dict(contract, contract.variants["canonical"]),
            events=["Episode initialized."],
        )

    def current_step_contract(self) -> StepContract:
        return self.contract.steps[min(self.state.phase_index, self.contract.phase_count - 1)]

    def current_context(self) -> str:
        step = self.current_step_contract()
        return self.variant.step_contexts.get(step.step_id, step.context)

    def current_visible_facts(self) -> List[str]:
        step = self.current_step_contract()
        return list(self.variant.visible_facts_by_step.get(step.step_id, []))

    def current_step_public_view(self, *, include_hidden: bool = False) -> Dict[str, Any]:
        step = self.current_step_contract()
        payload = {
            "step_id": step.step_id,
            "step_index": self.state.phase_index,
            "total_steps": self.contract.phase_count,
            "step_question": step.question,
            "step_context": self.current_context(),
            "visible_state_facts": self.current_visible_facts(),
            "available_tools": list(self.contract.available_tools),
        }
        if include_hidden:
            payload["required_args_hints"] = dict(step.required_args_hints)
            payload["hint_templates"] = list(step.hint_templates)
            payload["mandatory_stakeholders"] = list(step.mandatory_stakeholders)
            payload["required_fact_tokens"] = list(step.required_fact_tokens)
        return payload

    def public_task_payload(self) -> Dict[str, Any]:
        return _public_task_dict(self.contract, self.variant)

    def public_state(self) -> Dict[str, Any]:
        return {
            "task_id": self.state.task_id,
            "variant_id": self.state.variant_id,
            "domain": self.state.domain,
            "role": self.state.role,
            "scenario_title": self.state.scenario_title,
            "scenario_goal": self.state.scenario_goal,
            "scenario_briefing": self.state.scenario_briefing,
            "phase_index": self.state.phase_index,
            "phase_count": self.state.phase_count,
            "turn_index": self.state.turn_index,
            "max_turns": self.state.max_turns,
            "available_tools": list(self.state.available_tools),
            "goal_progress": round(self.state.goal_progress, 4),
            "risk_flags": list(self.state.risk_flags),
            "stakeholder_state": dict(self.state.stakeholder_state),
            "resource_state": dict(self.state.resource_state),
            "locked_or_unlocked_paths": list(self.state.unlocked_paths),
            "action_history": [item.model_dump() for item in self.state.action_history],
            "score_ledger": list(self.state.score_ledger),
            "failure_counts": dict(self.state.failure_counts),
            "hint_mode_enabled": self.state.hint_mode_enabled,
            "events": list(self.state.events),
            "done": self.state.done,
        }

    def ui_state(self) -> Dict[str, Any]:
        public = self.public_state()
        public.update(
            {
                "agent_role": self.contract.role,
                "agent_title": AGENTS[self.contract.domain][self.contract.role].title,
                "agent_persona": AGENTS[self.contract.domain][self.contract.role].persona,
                "agent_emoji": AGENTS[self.contract.domain][self.contract.role].emoji,
                "agent_color": AGENTS[self.contract.domain][self.contract.role].color,
                "scenario_title": self.contract.scenario_title,
                "scenario_briefing": self.variant.briefing,
                "scenario_goal": self.contract.goal,
                "step_index": self.state.phase_index,
                "total_steps": self.contract.phase_count,
                "step_id": self.current_step_contract().step_id,
                "step_question": self.current_step_contract().question,
                "step_context": self.current_context(),
                "required_args_hints": dict(self.current_step_contract().required_args_hints),
                "hint_templates": list(self.current_step_contract().hint_templates),
                "subagent_available": self.current_step_contract().required_tool == "summon_subagent" or bool(self.current_step_contract().mandatory_stakeholders),
                "score_ledger": list(self.state.score_ledger),
                "progress": round(self.state.goal_progress, 4),
            }
        )
        return public

    def _update_goal_progress(self) -> None:
        completed_phases = self.state.phase_index
        recent_bonus = 0.0
        if self.state.score_ledger:
            recent_bonus = min(0.1, sum(self.state.score_ledger[-3:]) / max(len(self.state.score_ledger[-3:]), 1) * 0.1)
        self.state.goal_progress = min(1.0, completed_phases / max(self.contract.phase_count, 1) + recent_bonus)

    def finalize_episode(self) -> Dict[str, Any]:
        mean_step_score = sum(self.state.score_ledger) / max(len(self.state.score_ledger), 1)
        final_outcome_score = min(1.0, self.state.goal_progress + (0.08 if self.state.phase_index >= self.contract.phase_count else 0.0))
        risk_penalty = min(0.5, 0.08 * len(self.state.risk_flags))
        risk_management_score = max(0.0, 1.0 - risk_penalty)
        efficiency = self.state.phase_index / max(self.state.turn_index, 1)
        efficiency_and_consistency = min(1.0, max(0.0, efficiency * 0.7 + (mean_step_score * 0.3)))
        episode_score = max(
            0.0,
            min(
                1.0,
                0.55 * mean_step_score
                + 0.20 * final_outcome_score
                + 0.15 * risk_management_score
                + 0.10 * efficiency_and_consistency,
            ),
        )
        self.state.episode_score = round(episode_score, 4)
        return {
            "mean_step_score": round(mean_step_score, 4),
            "final_outcome_score": round(final_outcome_score, 4),
            "risk_management_score": round(risk_management_score, 4),
            "efficiency_and_consistency_score": round(efficiency_and_consistency, 4),
            "final_episode_score": round(episode_score, 4),
        }

    def apply_action_result(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        reasoning: str,
        detail: Dict[str, Any],
        final_reward: float,
        llm_score: Optional[float] = None,
        llm_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        step = self.current_step_contract()
        passed = bool(detail.get("passed_threshold", final_reward >= step.pass_threshold))
        failed_checks = list(detail.get("failed_checks", []))
        self.state.turn_index += 1
        self.state.score_ledger.append(round(final_reward, 4))

        trace = ActionTrace(
            turn_index=self.state.turn_index,
            phase_index=self.state.phase_index,
            step_id=step.step_id,
            tool=tool,
            args=dict(args),
            reasoning=reasoning,
            deterministic_score=float(detail.get("deterministic_score", 0.0)),
            semantic_score=float(detail.get("semantic_score", 0.0)),
            trajectory_score=float(detail.get("trajectory_score", 0.0)),
            manual_score=float(detail.get("manual_score", final_reward)),
            final_score=round(final_reward, 4),
            llm_score=llm_score,
            llm_confidence=llm_confidence,
            passed_threshold=passed,
            failed_checks=failed_checks,
            score_components=dict(detail.get("score_components", {})),
            state_delta=dict(detail.get("state_delta", {})),
        )
        self.state.action_history.append(trace)

        if passed:
            self.state.events.append(f"Phase {self.state.phase_index + 1} cleared.")
            self.state.unlocked_paths.append(step.phase_name)
            self.state.failure_counts.pop(step.step_id, None)
            self.state.phase_index += 1
            if detail.get("state_delta"):
                self.state.resource_state.update(detail["state_delta"])
        else:
            count = self.state.failure_counts.get(step.step_id, 0) + 1
            self.state.failure_counts[step.step_id] = count
            if count >= 3:
                self.state.hint_mode_enabled = True
            if count >= 4:
                self.state.risk_flags.append(f"stalled_{step.step_id}")
            self.state.events.append(f"Phase {self.state.phase_index + 1} blocked on attempt {count}.")

        for flag in detail.get("risk_flags", []):
            if flag not in self.state.risk_flags:
                self.state.risk_flags.append(flag)

        self._update_goal_progress()
        if self.state.phase_index >= self.contract.phase_count:
            self.state.done = True
            self.state.events.append("All required phases completed.")
        elif self.state.turn_index >= self.contract.max_turns:
            self.state.done = True
            self.state.events.append("Max turns reached.")

        if detail.get("hard_failure"):
            self.state.done = True
            self.state.hard_failure = True
            self.state.events.append("Episode terminated by hard failure.")

        if self.state.done:
            detail["episode_summary"] = self.finalize_episode()

        self.state.current_step = (
            self.current_step_public_view(include_hidden=True)
            if not self.state.done
            else {}
        )
        return {
            "done": self.state.done,
            "phase_index": self.state.phase_index,
            "turn_index": self.state.turn_index,
            "goal_progress": self.state.goal_progress,
            "state": self.public_state(),
            "ui_state": self.ui_state(),
            "episode_summary": detail.get("episode_summary"),
        }
