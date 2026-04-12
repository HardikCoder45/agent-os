from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from benchmark_engine import StepContract
from judge_engine import ChecklistJudgeVerdict, run_llm_judge
from tool_schemas import ToolSchema, validate_args


PLACEHOLDER_PATTERN = re.compile(r"\b(test|dummy|placeholder|todo|fixme|pass|null|tbd)\b", re.IGNORECASE)
TRADEOFF_TOKENS = ("tradeoff", "however", "while", "instead", "but", "sacrifice", "not prioritizing")
RISK_TOKENS = ("risk", "downside", "mitigate", "protect", "safe", "rollback", "monitor")
STRATEGY_TOKENS = ("goal", "metric", "target", "timeline", "success", "within", "weeks", "days")
HISTORY_TOKENS = ("next", "now", "since", "after", "previous", "earlier", "update", "checkpoint")


def _clip(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def _tokens(value: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return {token for token in cleaned.split() if token}


def _semantic_match(actual: Any, expected: Any) -> float:
    if actual is None:
        return 0.0
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            return 0.0
        if actual == expected:
            return 1.0
        if expected == 0:
            return 0.0
        return _clip(min(actual, expected) / max(actual, expected))
    if isinstance(expected, str):
        actual_text = str(actual).strip()
        expected_text = expected.strip()
        if not actual_text:
            return 0.0
        if actual_text.lower() == expected_text.lower():
            return 1.0
        actual_tokens = _tokens(actual_text)
        expected_tokens = _tokens(expected_text)
        if not expected_tokens:
            return 0.5
        overlap = len(actual_tokens & expected_tokens) / len(expected_tokens)
        contains_bonus = 0.2 if expected_text.lower() in actual_text.lower() else 0.0
        return _clip(overlap + contains_bonus)
    return 1.0 if actual == expected else 0.0


def _contains_any(text: str, tokens: tuple[str, ...] | list[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for token in tokens if token in lowered)
    if hits <= 0:
        return 0.0
    if hits == 1:
        return 0.55
    if hits == 2:
        return 0.8
    return 1.0


def _arg_coverage(args: Dict[str, Any], step_contract: StepContract) -> float:
    required = list(step_contract.required_args_hints.keys())
    if not required:
        return 1.0
    present = sum(1 for key in required if key in args and args.get(key) not in (None, "", []))
    return _clip(present / len(required))


def _fact_grounding_score(reasoning: str, args: Dict[str, Any], step_contract: StepContract, visible_facts: List[str]) -> float:
    combined = f"{reasoning} {json.dumps(args, sort_keys=True)}".lower()
    target_tokens = list(dict.fromkeys(step_contract.required_fact_tokens + [token for fact in visible_facts for token in _tokens(fact)]))
    target_tokens = [token for token in target_tokens if len(token) >= 3][:12]
    if not target_tokens:
        return 0.5
    hits = sum(1 for token in target_tokens if token in combined)
    return _clip(hits / len(target_tokens))


def _state_delta_score(reasoning: str, args: Dict[str, Any], step_contract: StepContract) -> Tuple[float, Dict[str, Any]]:
    combined = f"{reasoning} {json.dumps(args, sort_keys=True)}".lower()
    scored = {}
    total = 0.0
    weight_total = 0.0
    for delta in step_contract.expected_state_deltas:
        token_hits = sum(1 for token in delta.tokens if token in combined)
        score = 1.0 if token_hits >= 2 else 0.5 if token_hits == 1 else 0.0
        scored[delta.key] = score
        total += score * delta.weight
        weight_total += delta.weight
    return (_clip(total / weight_total) if weight_total else 0.0), scored


def _stakeholder_score(reasoning: str, step_contract: StepContract) -> float:
    if not step_contract.mandatory_stakeholders:
        return 0.8
    lowered = reasoning.lower()
    hits = sum(1 for stakeholder in step_contract.mandatory_stakeholders if stakeholder in lowered)
    return _clip(hits / len(step_contract.mandatory_stakeholders))


def _trajectory_score(reasoning: str, tool_name: str, args: Dict[str, Any], previous_actions: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    if not previous_actions:
        return 0.85, []
    notes: List[str] = []
    last = previous_actions[-1]
    repeated = last.get("tool") == tool_name and last.get("args") == args
    repetition_score = 0.0 if repeated else 1.0
    if repeated:
        notes.append("Repeated the same tool and arguments without new evidence.")
    history_awareness = _contains_any(reasoning, HISTORY_TOKENS)
    novelty = 0.4 if repeated else 0.9
    score = _clip(0.5 * repetition_score + 0.25 * history_awareness + 0.25 * novelty)
    return score, notes


def _catastrophic_risk_flags(reasoning: str, args: Dict[str, Any], step_contract: StepContract) -> tuple[bool, List[str], float]:
    combined = f"{reasoning} {json.dumps(args, sort_keys=True)}".lower()
    risk_flags: List[str] = []
    gate = 1.0
    for risk_rule in step_contract.risk_rules:
        for pattern in risk_rule.catastrophic_patterns:
            if pattern and pattern in combined:
                risk_flags.append(risk_rule.rule_id)
                return True, risk_flags, 0.0
        caution_hits = [pattern for pattern in risk_rule.caution_patterns if pattern and pattern in combined]
        if caution_hits:
            gate = min(gate, max(0.0, 1.0 - risk_rule.penalty))
            risk_flags.append(risk_rule.rule_id)
    return False, risk_flags, gate


def _tips_from_failures(
    *,
    arg_errors: List[str],
    failed_checks: List[str],
    hallucinated_args: List[str],
    step_contract: StepContract,
    fact_grounding: float,
    semantic_score: float,
) -> List[str]:
    tips: List[str] = []
    tips.extend(arg_errors[:2])
    if hallucinated_args:
        tips.append(f"Remove unsupported arguments: {', '.join(hallucinated_args[:3])}.")
    if fact_grounding < 0.45:
        tips.append("Reference the live scenario numbers, constraints, and visible facts more explicitly.")
    if semantic_score < 0.55:
        tips.append("Name the tradeoff, the downside risk, and the measurable success target in the reasoning.")
    if not tips:
        tips.extend(step_contract.hint_templates[:3])
    else:
        tips.extend(step_contract.hint_templates[: max(0, 3 - len(tips))])
    tips.extend(failed_checks[:2])
    return list(dict.fromkeys(tips))[:4]


def deterministic_program_score(
    *,
    tool_name: str,
    args: Dict[str, Any],
    reasoning: str,
    step_contract: StepContract,
    available_tools: List[str],
    tool_schema: Optional[ToolSchema],
    visible_facts: List[str],
    previous_actions: List[Dict[str, Any]],
    failure_count: int = 0,
) -> Dict[str, Any]:
    gate_multiplier = 1.0
    failed_checks: List[str] = []
    risk_flags: List[str] = []
    hard_failure = False

    if not tool_name or tool_name not in available_tools:
        return {
            "deterministic_score": 0.0,
            "semantic_score": 0.0,
            "trajectory_score": 0.0,
            "manual_score": 0.0,
            "gate_multiplier": 0.0,
            "arg_errors": [f"Tool '{tool_name}' is not available for this role."],
            "hallucinated_args": [],
            "failed_checks": ["Choose a valid tool from the visible tool list."],
            "risk_flags": ["invalid_tool"],
            "hard_failure": False,
            "tips": step_contract.hint_templates[:3],
            "passed_threshold": False,
            "state_delta": {"blocked_phase": step_contract.phase_name},
            "score_components": {
                "tool_alignment": 0.0,
                "arg_coverage": 0.0,
                "arg_quality": 0.0,
                "fact_grounding": 0.0,
                "state_delta": 0.0,
                "stakeholder_awareness": 0.0,
            },
        }

    tool_alignment = 1.0 if tool_name == step_contract.required_tool else 0.05
    if tool_name != step_contract.required_tool:
        gate_multiplier = min(gate_multiplier, 0.25)
        failed_checks.append("Selected a lower-leverage tool than the benchmark expects for this phase.")

    if tool_schema is None:
        arg_valid, arg_errors, arg_quality, normalized_args, hallucinated_args = False, [f"No schema found for `{tool_name}`."], 0.0, {}, []
    else:
        arg_valid, arg_errors, arg_quality, normalized_args, hallucinated_args = validate_args(tool_schema, args)

    if hallucinated_args:
        gate_multiplier = min(gate_multiplier, 0.5)
        failed_checks.append("Passed unsupported arguments.")

    arg_coverage = _arg_coverage(normalized_args if normalized_args else args, step_contract)
    if arg_coverage < 1.0:
        gate_multiplier = min(gate_multiplier, 0.5)
        failed_checks.append("Did not cover the required argument set.")

    reasoning_text = reasoning.strip()
    if len(reasoning_text.split()) < 8:
        gate_multiplier = min(gate_multiplier, 0.5)
        failed_checks.append("Reasoning is too short for a high-stakes step.")

    placeholder_language = bool(PLACEHOLDER_PATTERN.search(reasoning_text)) or bool(
        PLACEHOLDER_PATTERN.search(json.dumps(args, sort_keys=True))
    )
    if placeholder_language:
        gate_multiplier = min(gate_multiplier, 0.35)
        failed_checks.append("Placeholder language is not acceptable in the benchmark.")

    catastrophic, catastrophic_flags, risk_gate = _catastrophic_risk_flags(reasoning_text, args, step_contract)
    if catastrophic:
        gate_multiplier = 0.0
        hard_failure = True
        risk_flags.extend(catastrophic_flags)
        failed_checks.append("Triggered a catastrophic safety or governance pattern.")
    else:
        gate_multiplier = min(gate_multiplier, risk_gate)
        risk_flags.extend(catastrophic_flags)

    fact_grounding = _fact_grounding_score(reasoning_text, normalized_args if normalized_args else args, step_contract, visible_facts)
    if fact_grounding < 0.35:
        failed_checks.append("The reasoning is not grounded enough in the current scenario facts.")

    state_delta_score, state_delta = _state_delta_score(reasoning_text, normalized_args if normalized_args else args, step_contract)
    stakeholder_awareness = _stakeholder_score(reasoning_text, step_contract)

    optimal_alignment_components = [
        _semantic_match((normalized_args if normalized_args else args).get(key), expected)
        for key, expected in step_contract.optimal_args.items()
    ]
    optimal_alignment = _clip(
        sum(optimal_alignment_components) / max(len(optimal_alignment_components), 1)
    )

    deterministic_score = _clip(
        tool_alignment * 0.32
        + ((arg_quality * 0.65) + (arg_coverage * 0.35)) * 0.26
        + fact_grounding * 0.16
        + state_delta_score * 0.14
        + stakeholder_awareness * 0.07
        + optimal_alignment * 0.05
    )

    tradeoff_score = _contains_any(reasoning_text, TRADEOFF_TOKENS)
    risk_score = _contains_any(reasoning_text, RISK_TOKENS)
    strategy_score = _contains_any(reasoning_text, STRATEGY_TOKENS)
    semantic_score = _clip(
        tradeoff_score * 0.3
        + stakeholder_awareness * 0.25
        + strategy_score * 0.25
        + risk_score * 0.2
    )
    if semantic_score < 0.5:
        failed_checks.append("The answer needs clearer tradeoffs, stakeholders, and risk framing.")

    trajectory_score, trajectory_notes = _trajectory_score(reasoning_text, tool_name, normalized_args if normalized_args else args, previous_actions)
    failed_checks.extend(trajectory_notes)

    manual_score = _clip(
        gate_multiplier
        * (
            0.70 * deterministic_score
            + 0.20 * semantic_score
            + 0.10 * trajectory_score
        )
    )
    if failure_count >= 2 and manual_score < step_contract.pass_threshold:
        manual_score = min(manual_score, 0.4)
        failed_checks.append("Repeated low-quality retries are capped until the strategy materially changes.")

    tips = _tips_from_failures(
        arg_errors=arg_errors,
        failed_checks=failed_checks,
        hallucinated_args=hallucinated_args,
        step_contract=step_contract,
        fact_grounding=fact_grounding,
        semantic_score=semantic_score,
    )

    passed_threshold = manual_score >= step_contract.pass_threshold and not hard_failure
    visible_state_delta = {
        "last_tool": tool_name,
        "phase_name": step_contract.phase_name,
        "decision_quality": round(manual_score, 4),
        "stakeholder_awareness": round(stakeholder_awareness, 4),
    }
    if passed_threshold:
        visible_state_delta["phase_cleared"] = step_contract.phase_name
    else:
        visible_state_delta["blocked_phase"] = step_contract.phase_name

    return {
        "deterministic_score": deterministic_score,
        "semantic_score": semantic_score,
        "trajectory_score": trajectory_score,
        "manual_score": manual_score,
        "gate_multiplier": gate_multiplier,
        "arg_errors": arg_errors,
        "hallucinated_args": hallucinated_args,
        "failed_checks": list(dict.fromkeys(failed_checks)),
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "hard_failure": hard_failure,
        "tips": tips,
        "passed_threshold": passed_threshold,
        "state_delta": visible_state_delta,
        "score_components": {
            "tool_alignment": round(tool_alignment, 4),
            "arg_coverage": round(arg_coverage, 4),
            "arg_quality": round(arg_quality, 4),
            "fact_grounding": round(fact_grounding, 4),
            "state_delta": round(state_delta_score, 4),
            "stakeholder_awareness": round(stakeholder_awareness, 4),
            "optimal_alignment": round(optimal_alignment, 4),
            "tradeoff_awareness": round(tradeoff_score, 4),
            "risk_awareness": round(risk_score, 4),
            "strategy_awareness": round(strategy_score, 4),
        },
        "state_delta_components": state_delta,
    }


def blend_scores(
    deterministic_score: float,
    llm_score: Optional[float],
    *,
    llm_confidence: Optional[float] = None,
    gate_multiplier: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    deterministic_score = _clip(deterministic_score)
    if gate_multiplier == 0.0:
        return 0.0, {
            "manual_weight": 1.0,
            "llm_weight": 0.0,
            "agreement_gap": None,
            "agreement_adjustment": 0.0,
            "method": "hard_gate_zero",
            "llm_confidence": llm_confidence,
        }

    if llm_score is None:
        return deterministic_score, {
            "manual_weight": 1.0,
            "llm_weight": 0.0,
            "agreement_gap": None,
            "agreement_adjustment": 0.0,
            "method": "manual_only",
            "llm_confidence": llm_confidence,
        }

    llm_score = _clip(llm_score)
    confidence = llm_confidence if llm_confidence is not None else 0.75
    manual_weight = 0.8
    llm_weight = 0.2
    if confidence < 0.60:
        manual_weight, llm_weight = 0.95, 0.05

    if deterministic_score < 0.20:
        blended = min(0.30, deterministic_score * manual_weight + llm_score * llm_weight)
        return _clip(blended), {
            "manual_weight": manual_weight,
            "llm_weight": llm_weight,
            "agreement_gap": _clip(abs(deterministic_score - llm_score)),
            "agreement_adjustment": 0.0,
            "method": "low_manual_cap",
            "llm_confidence": confidence,
        }

    agreement_gap = abs(deterministic_score - llm_score)
    agreement_adjustment = 0.0
    if agreement_gap <= 0.10:
        agreement_adjustment = 0.02
    elif agreement_gap >= 0.35:
        llm_weight *= 0.5
        manual_weight = 1.0 - llm_weight
        agreement_adjustment = -0.02

    blended = deterministic_score * manual_weight + llm_score * llm_weight + agreement_adjustment
    return _clip(blended), {
        "manual_weight": round(manual_weight, 4),
        "llm_weight": round(llm_weight, 4),
        "agreement_gap": _clip(agreement_gap),
        "agreement_adjustment": agreement_adjustment,
        "method": "manual_llm_blend",
        "llm_confidence": confidence,
    }


def compute_reward(
    agent_output: str,
    tool_name: str,
    args: Dict[str, Any],
    reasoning: str,
    available_tools: List[str],
    tool_registry: Dict[str, ToolSchema],
    scenario: Dict[str, Any],
    step_context: Dict[str, Any],
    previous_actions: List[Dict],
    task: Any = None,
    rubric: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    judge_model: str = "openai/gpt-4.1-mini",
    use_llm_judge: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    del agent_output, rubric
    if task is None or not hasattr(task, "contract"):
        raise ValueError("compute_reward now requires a benchmark task with an episode contract.")

    phase_index = int(step_context.get("phase_index", step_context.get("step", 1) - 1))
    phase_index = max(0, min(phase_index, len(task.contract.steps) - 1))
    step_contract: StepContract = task.contract.steps[phase_index]
    visible_facts = list(step_context.get("visible_facts", []) or [])
    failure_count = int(step_context.get("failure_count", 0))
    tool_schema = tool_registry.get(tool_name)

    deterministic_detail = deterministic_program_score(
        tool_name=tool_name,
        args=args,
        reasoning=reasoning,
        step_contract=step_contract,
        available_tools=available_tools,
        tool_schema=tool_schema,
        visible_facts=visible_facts,
        previous_actions=previous_actions,
        failure_count=failure_count,
    )

    manual_score = float(deterministic_detail["manual_score"])
    llm_score: Optional[float] = None
    llm_verdict: Optional[ChecklistJudgeVerdict] = None
    if use_llm_judge and api_key and api_base_url and deterministic_detail["gate_multiplier"] > 0.0:
        llm_verdict = run_llm_judge(
            reasoning=reasoning,
            tool_name=tool_name,
            args=args,
            step_contract=step_contract,
            scenario=scenario,
            step_context=step_context,
            previous_actions=previous_actions,
            api_key=api_key,
            api_base_url=api_base_url,
            model=judge_model,
        )
        if llm_verdict is not None:
            llm_score = llm_verdict.semantic_score

    final_score, blend_detail = blend_scores(
        manual_score,
        llm_score,
        llm_confidence=(llm_verdict.confidence if llm_verdict is not None else None),
        gate_multiplier=float(deterministic_detail["gate_multiplier"]),
    )

    detail = {
        "method": "stateful_checklist_grader",
        "deterministic_score": deterministic_detail["deterministic_score"],
        "semantic_score": deterministic_detail["semantic_score"],
        "trajectory_score": deterministic_detail["trajectory_score"],
        "manual_score": manual_score,
        "llm_score": llm_score,
        "llm_verdict": llm_verdict.model_dump() if llm_verdict is not None else None,
        "blend_detail": blend_detail,
        "gate_multiplier": deterministic_detail["gate_multiplier"],
        "failed_checks": deterministic_detail["failed_checks"],
        "arg_errors": deterministic_detail["arg_errors"],
        "hallucinated_args": deterministic_detail["hallucinated_args"],
        "risk_flags": deterministic_detail["risk_flags"],
        "hard_failure": deterministic_detail["hard_failure"],
        "tips": deterministic_detail["tips"],
        "passed_threshold": final_score >= step_contract.pass_threshold and not deterministic_detail["hard_failure"],
        "pass_threshold": step_contract.pass_threshold,
        "score_components": deterministic_detail["score_components"],
        "state_delta": deterministic_detail["state_delta"],
        "state_delta_components": deterministic_detail["state_delta_components"],
        "final_reward": final_score,
    }
    if llm_verdict is not None:
        detail["judge_feedback"] = llm_verdict.overall_feedback
        detail["judge_tips"] = llm_verdict.improvement_tips
    return final_score, detail
