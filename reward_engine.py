"""
reward_engine.py — Unified reward engine integrating:
  1. Rule-based J1 scoring (fast, deterministic)
  2. LLM multi-dimensional judge via judge_engine.py (deep, semantic)
  3. Final combined reward ∈ [0, 1]

Design principle: J1 catches objective violations instantly;
LLM judge provides rich semantic evaluation that J1 cannot.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from judge_engine import run_llm_judge, FinalVerdict
from task_graders import grade_task_action
from tool_schemas import TOOL_REGISTRY, validate_args

# ──────────────────────────────────────────────
#  J1: Fast rule-based scorer
# ──────────────────────────────────────────────

REASONING_KEYWORDS = [
    "because", "therefore", "first", "then", "next", "since",
    "given", "as a result", "consequently", "this means", "in order to",
    "the goal is", "i need to", "i should", "this will", "so that",
]

FORMAT_KEYWORDS = ["tool", "args", "reasoning"]

PENALTY_PATTERNS = [
    (r"\b(test|dummy|placeholder|todo|fixme|pass|null)\b", -0.10, "Placeholder/stub language"),
    (r"(.)\1{6,}", -0.08, "Repetitive characters"),
]

BONUS_PATTERNS = [
    (r"\b(icp|ideal customer|segment|persona|revenue|conversion|churn|ltv|cac)\b", 0.03, "Business domain awareness"),
    (r"\b(risk|compliance|legal|regulatory|gdpr|soc2)\b", 0.03, "Risk/compliance awareness"),
    (r"\b(kpi|metric|benchmark|baseline|measurement)\b", 0.02, "Metrics awareness"),
]

PLACEHOLDER_PATTERN = re.compile(r"\b(test|dummy|placeholder|todo|fixme|pass|null)\b", re.IGNORECASE)
STRATEGIC_SIGNAL_WORDS = (
    "tradeoff",
    "risk",
    "metric",
    "timeline",
    "success",
    "goal",
    "stakeholder",
    "runway",
    "dilution",
    "retention",
    "governance",
    "budget",
)


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


def blend_scores(manual_score: float, llm_score: Optional[float]) -> Tuple[float, Dict[str, Any]]:
    manual_score = _clip(manual_score)
    if llm_score is None:
        return manual_score, {
            "manual_weight": 1.0,
            "llm_weight": 0.0,
            "agreement_adjustment": 0.0,
            "agreement_gap": None,
            "method": "manual_only",
        }

    llm_score = _clip(llm_score)
    gap = abs(manual_score - llm_score)
    agreement_adjustment = 0.0
    if gap <= 0.10:
        agreement_adjustment = 0.03
    elif gap >= 0.35:
        agreement_adjustment = -0.04
    elif gap >= 0.22:
        agreement_adjustment = -0.02

    final = _clip((manual_score * 0.55) + (llm_score * 0.45) + agreement_adjustment)
    return final, {
        "manual_weight": 0.55,
        "llm_weight": 0.45,
        "agreement_adjustment": agreement_adjustment,
        "agreement_gap": _clip(gap),
        "method": "manual_llm_blend",
    }


def j1_score(
    agent_output: str,
    tool_name: str,
    args: Dict,
    available_tools: List[str],
    rubric: Optional[Dict] = None,
) -> Tuple[float, List[str]]:
    """
    Fast J1 rule-based scoring.
    Returns (score ∈ [0, 1], list of reasons).
    """
    score = 0.5  # baseline
    reasons = []

    # ─ Tool validity (hard gate) ─
    if not tool_name:
        return 0.0, ["No tool called"]
    if available_tools and tool_name not in available_tools:
        return 0.0, [f"Tool '{tool_name}' not in available tools"]

    # ─ Args completeness ─
    if not args:
        score -= 0.15
        reasons.append("Empty args")
    elif len(args) >= 3:
        score += 0.05
        reasons.append("Rich arg set")

    # ─ Reasoning quality ─
    if not agent_output or len(agent_output.strip()) < 20:
        score -= 0.20
        reasons.append("No/minimal reasoning")
    else:
        kw_count = sum(1 for kw in REASONING_KEYWORDS if kw in agent_output.lower())
        if kw_count == 0:
            score -= 0.10
            reasons.append("No reasoning connectors")
        elif kw_count >= 3:
            score += 0.10
            reasons.append(f"Strong reasoning ({kw_count} connectors)")
        elif kw_count >= 1:
            score += 0.04
            reasons.append(f"Basic reasoning ({kw_count} connectors)")

    # ─ Format compliance ─
    fmt_hits = sum(1 for kw in FORMAT_KEYWORDS if kw in agent_output.lower())
    if fmt_hits < 2:
        score -= 0.05
        reasons.append("Missing format keywords")

    # ─ Penalty patterns ─
    for pattern, penalty, label in PENALTY_PATTERNS:
        if re.search(pattern, agent_output, re.IGNORECASE):
            score += penalty
            reasons.append(f"Penalty: {label}")

    # ─ Bonus patterns ─
    for pattern, bonus, label in BONUS_PATTERNS:
        if re.search(pattern, agent_output, re.IGNORECASE):
            score += bonus
            reasons.append(f"Bonus: {label}")

    # ─ Rubric check ─
    if rubric:
        for criterion, weight in rubric.items():
            if criterion.lower() in agent_output.lower():
                score += weight * 0.1
                reasons.append(f"Rubric match: {criterion}")

    return round(max(0.0, min(1.0, score)), 4), reasons


# ──────────────────────────────────────────────
#  Combined reward
# ──────────────────────────────────────────────

def compute_reward(
    agent_output: str,
    tool_name: str,
    args: Dict,
    reasoning: str,
    available_tools: List[str],
    tool_registry: Dict[str, Any],
    scenario: Dict[str, Any],
    step_context: Dict[str, Any],
    previous_actions: List[Dict],
    task: Any = None,
    rubric: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    judge_model: str = "anthropic/claude-3.5-sonnet",
    use_llm_judge: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute unified reward combining J1 and optional LLM judge.

    Returns:
      (final_reward ∈ [0, 1], detail_dict)
    """
    task_grade: Optional[Dict[str, Any]] = None
    if task is not None:
        task_grade = grade_task_action(
            task,
            tool_name=tool_name,
            args=args,
            reasoning=reasoning,
            available_tools=available_tools,
            tool_schema=tool_registry.get(tool_name),
        )
        base_score = task_grade["final_reward"]
        j1 = base_score
        j1_reasons = task_grade.get("notes", []) + task_grade.get("errors", [])
    else:
        j1, j1_reasons = j1_score(agent_output, tool_name, args, available_tools, rubric)
        base_score = j1

    # Hard gate: J1 zero means instant zero overall
    if base_score == 0.0:
        return 0.0, {
            "manual_score": 0.0,
            "manual_reasons": j1_reasons,
            "deterministic_grade": task_grade,
            "llm_verdict": None,
            "llm_score": None,
            "blend_detail": {
                "manual_weight": 1.0,
                "llm_weight": 0.0,
                "agreement_adjustment": 0.0,
                "agreement_gap": None,
                "method": "instant_zero",
            },
            "final_reward": 0.0,
            "method": "task_instant_zero" if task_grade else "j1_instant_zero",
        }

    llm_verdict: Optional[FinalVerdict] = None
    if use_llm_judge and api_key and api_base_url:
        try:
            llm_verdict = run_llm_judge(
                agent_output=agent_output,
                tool_registry=tool_registry,
                scenario=scenario,
                step_context=step_context,
                available_tools=available_tools,
                previous_actions=previous_actions,
                api_key=api_key,
                api_base_url=api_base_url,
                model=judge_model,
            )
        except Exception:
            llm_verdict = None

    llm_score: Optional[float] = None
    blend_detail: Dict[str, Any]
    if llm_verdict is not None:
        if llm_verdict.instant_zero:
            return 0.0, {
                "manual_score": base_score,
                "manual_reasons": j1_reasons,
                "deterministic_grade": task_grade,
                "llm_verdict": llm_verdict.model_dump(),
                "llm_score": 0.0,
                "final_reward": 0.0,
                "method": "llm_instant_zero",
            }
        llm_score = llm_verdict.total_score
        final, blend_detail = blend_scores(base_score, llm_score)
        method = "combined_task_llm" if task_grade else "combined_manual_llm"
    else:
        final, blend_detail = blend_scores(base_score, None)
        method = "task_only" if task_grade else "manual_only"

    return final, {
        "manual_score": base_score,
        "manual_reasons": j1_reasons,
        "deterministic_grade": task_grade,
        "llm_verdict": llm_verdict.model_dump() if llm_verdict else None,
        "llm_score": llm_score,
        "blend_detail": blend_detail,
        "final_reward": final,
        "method": method,
    }



# ──────────────────────────────────────────────
#  Legacy stub functions for backward compatibility
# ──────────────────────────────────────────────

def score_reasoning(thinking: str, step: Any, shared_state: Dict) -> Tuple[float, Dict]:
    """
    Legacy function - scores reasoning quality.
    Returns (score, breakdown_dict).
    """
    cleaned = (thinking or "").strip()
    if not cleaned:
        return 0.0, {
            "word_count": 0,
            "connector_hits": 0,
            "keyword_hit_ratio": 0.0,
            "context_hit_ratio": 0.0,
            "strategic_signal_ratio": 0.0,
            "placeholder_language": False,
        }

    words = cleaned.split()
    lower = cleaned.lower()
    connectors = sum(1 for kw in REASONING_KEYWORDS if kw in lower)
    keyword_ratio = 0.0
    context_ratio = 0.0
    strategic_ratio = 0.0

    optimal_keywords = getattr(step, "optimal_reasoning_keywords", []) or []
    if optimal_keywords:
        keyword_ratio = len([kw for kw in optimal_keywords if kw.lower() in lower]) / len(optimal_keywords)

    context_tokens = _tokens(
        f"{getattr(step, 'question', '')} {getattr(step, 'context', '')} "
        f"{getattr(step, 'counterfactual_tip', '')}"
    )
    if context_tokens:
        context_ratio = len(context_tokens & _tokens(cleaned)) / len(context_tokens)

    strategic_hits = sum(1 for word in STRATEGIC_SIGNAL_WORDS if word in lower)
    strategic_ratio = min(1.0, strategic_hits / 5)
    placeholder_language = bool(PLACEHOLDER_PATTERN.search(cleaned))

    score = 0.08
    score += min(0.18, len(words) / 55 * 0.18)
    score += min(0.16, connectors * 0.04)
    score += min(0.28, keyword_ratio * 0.28)
    score += min(0.14, context_ratio * 0.14)
    score += min(0.16, strategic_ratio * 0.16)

    if len(words) < 18:
        score -= 0.10
    if connectors == 0:
        score -= 0.06
    if placeholder_language:
        score -= 0.25

    return _clip(score), {
        "word_count": len(words),
        "connector_hits": connectors,
        "keyword_hit_ratio": _clip(keyword_ratio),
        "context_hit_ratio": _clip(context_ratio),
        "strategic_signal_ratio": _clip(strategic_ratio),
        "placeholder_language": placeholder_language,
    }


def score_tool_arguments(tool_name: str, tool_args: Dict, step: Any) -> Tuple[float, List[str], List[str]]:
    """
    Legacy function - scores tool arguments.
    Returns (score, errors_list, hints_list).
    """
    errors = []
    hints = []

    if not tool_name:
        errors.append("No tool specified")
        return 0.0, errors, hints

    required_tool = getattr(step, "required_tool", "")
    if required_tool and tool_name != required_tool:
        errors.append(f"Expected tool `{required_tool}`, got `{tool_name}`")

    schema = TOOL_REGISTRY.get(tool_name)
    arg_quality = 0.0
    validation_ok = False
    if schema is not None:
        validation_ok, arg_errors, arg_quality = validate_args(schema, tool_args)
        errors.extend(arg_errors)
    else:
        errors.append(f"No schema found for tool `{tool_name}`")

    expected_args = getattr(step, "required_args_hints", {}) or {}
    if expected_args:
        provided_required = sum(1 for name in expected_args if name in tool_args and tool_args.get(name) not in (None, "", []))
        coverage_ratio = provided_required / len(expected_args)
    else:
        coverage_ratio = 0.5 if tool_args else 0.0

    optimal_args = getattr(step, "optimal_args", {}) or {}
    if optimal_args:
        semantic_scores = [_semantic_match(tool_args.get(key), expected) for key, expected in optimal_args.items()]
        optimal_alignment = sum(semantic_scores) / len(semantic_scores)
    else:
        optimal_alignment = 0.5 if tool_args else 0.0

    richness_scores = []
    for value in tool_args.values():
        if isinstance(value, str):
            richness_scores.append(min(1.0, len(value.split()) / 10))
        elif isinstance(value, (int, float)):
            richness_scores.append(0.9)
        else:
            richness_scores.append(0.7)
    arg_richness = sum(richness_scores) / len(richness_scores) if richness_scores else 0.0

    tool_match_score = 1.0 if not required_tool or tool_name == required_tool else 0.12
    score = (
        tool_match_score * 0.40
        + arg_quality * 0.20
        + coverage_ratio * 0.20
        + optimal_alignment * 0.15
        + arg_richness * 0.05
    )

    if coverage_ratio < 0.75:
        missing = [name for name in expected_args if name not in tool_args or tool_args.get(name) in (None, "", [])]
        if missing:
            hints.append(f"Add the missing core args: {', '.join(missing[:4])}")
    if optimal_alignment < 0.65 and optimal_args:
        hints.append("Match the step's expected direction more closely with scenario-specific values.")
    if validation_ok and coverage_ratio >= 0.8:
        hints.append("Argument structure is solid.")

    return _clip(score), errors, hints


def score_subagent_decision(used_subagent: bool, step: Any, subagent_result: Optional[str]) -> float:
    """
    Legacy function - scores subagent usage.
    Returns score.
    """
    if not used_subagent:
        return 0.5  # neutral
    
    if subagent_result and len(subagent_result) > 50:
        return 0.8  # good usage
    
    return 0.6


def compute_j1(reasoning_score: float, tool_arg_score: float, subagent_score: float, rubric_tier: str) -> float:
    """
    Legacy function - computes J1 score.
    Returns combined score.
    """
    base = (reasoning_score * 0.45 + tool_arg_score * 0.45 + subagent_score * 0.10)
    
    tier_bonus = {
        "excellent": 0.06,
        "good": 0.03,
        "acceptable": 0.0,
        "poor": -0.06,
    }.get(rubric_tier, 0.0)

    return _clip(base + tier_bonus)


def compute_j2(goal_achieved: bool, steps_used: int, max_steps: int, 
               reasoning_avg: float = 0.5, strategy_diversity: float = 0.5,
               cross_agent_actions: int = 0, efficiency: float = 0.5, 
               shared_state: Dict = None) -> float:
    """
    Legacy function - computes J2 (episode-level) score.
    Returns score.
    """
    if not goal_achieved:
        return 0.0
    
    step_efficiency = 1.0 - (steps_used / max_steps) if max_steps > 0 else 0.5
    
    return (step_efficiency * 0.3 + reasoning_avg * 0.4 + strategy_diversity * 0.2 + min(cross_agent_actions / 5, 0.1))


def auto_classify_rubric(reasoning_score: float, tool_arg_score: float) -> str:
    """
    Legacy function - classifies performance into rubric tier.
    Returns tier string.
    """
    avg = (reasoning_score + tool_arg_score) / 2
    
    if avg >= 0.8:
        return "excellent"
    elif avg >= 0.6:
        return "good"
    elif avg >= 0.4:
        return "acceptable"
    else:
        return "poor"


def generate_counterfactual(step: Any, j1: float, reasoning_breakdown: Dict, tool_hints: List[str]) -> str:
    """
    Legacy function - generates counterfactual feedback.
    Returns feedback string.
    """
    required_tool = getattr(step, "required_tool", "the expected tool")
    connector_hits = reasoning_breakdown.get("connector_hits", 0)
    keyword_ratio = reasoning_breakdown.get("keyword_hit_ratio", 0.0)

    if j1 >= 0.8:
        return (
            f"Strong performance. Keep the same direction and sharpen the next move with explicit metrics, "
            f"stakeholder impact, and why `{required_tool}` is still the right tool."
        )
    if j1 >= 0.6:
        improvement = ", ".join(tool_hints[:2]) if tool_hints else "tighten the argument details and make the tradeoffs explicit"
        return f"Solid attempt. To score higher, keep `{required_tool}` but {improvement}."
    if j1 >= 0.4:
        return (
            f"Needs improvement. Your reasoning only showed {connector_hits} structured connectors and "
            f"{keyword_ratio:.2f} keyword alignment. Use `{required_tool}` with more scenario-specific numbers, risks, and success criteria."
        )
    return (
        f"Low-scoring move. Re-anchor on the step objective, use `{required_tool}` if appropriate, and answer with "
        "complete args, explicit tradeoffs, concrete metrics, and direct scenario grounding."
    )
