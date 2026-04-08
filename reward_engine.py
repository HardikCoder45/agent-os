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

RULE_WEIGHT = 0.75
LLM_WEIGHT = 0.25


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
            "j1_score": 0.0,
            "j1_reasons": j1_reasons,
            "deterministic_grade": task_grade,
            "llm_verdict": None,
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

    if llm_verdict is not None:
        if llm_verdict.instant_zero:
            return 0.0, {
                "j1_score": base_score,
                "j1_reasons": j1_reasons,
                "deterministic_grade": task_grade,
                "llm_verdict": llm_verdict.model_dump(),
                "final_reward": 0.0,
                "method": "llm_instant_zero",
            }
        llm_score = llm_verdict.total_score
        final = round(RULE_WEIGHT * base_score + LLM_WEIGHT * llm_score, 4)
        method = "combined_task_llm" if task_grade else "combined_j1_llm"
    else:
        final = base_score
        method = "task_only" if task_grade else "j1_only"

    return final, {
        "j1_score": base_score,
        "j1_reasons": j1_reasons,
        "deterministic_grade": task_grade,
        "llm_verdict": llm_verdict.model_dump() if llm_verdict else None,
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
    if not thinking or len(thinking.strip()) < 20:
        return 0.2, {"length": "too_short", "connectors": 0}
    
    connectors = sum(1 for kw in REASONING_KEYWORDS if kw in thinking.lower())
    score = min(1.0, 0.4 + (connectors * 0.1))
    
    return score, {"length": len(thinking), "connectors": connectors}


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
    
    if not tool_args:
        errors.append("Empty arguments")
        return 0.3, errors, hints
    
    score = 0.7 if len(tool_args) >= 2 else 0.5
    
    if len(tool_args) >= 3:
        hints.append("Good argument coverage")
    
    return score, errors, hints


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
    base = (reasoning_score * 0.4 + tool_arg_score * 0.4 + subagent_score * 0.2)
    
    tier_bonus = {
        "excellent": 0.1,
        "good": 0.05,
        "acceptable": 0.0,
        "poor": -0.1,
    }.get(rubric_tier, 0.0)
    
    return max(0.0, min(1.0, base + tier_bonus))


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
    if j1 >= 0.8:
        return "Strong performance. Consider exploring alternative approaches for even better results."
    elif j1 >= 0.6:
        return f"Good attempt. Improvements: {', '.join(tool_hints) if tool_hints else 'add more detailed reasoning'}"
    elif j1 >= 0.4:
        return f"Needs improvement. Issues: {reasoning_breakdown.get('connectors', 0)} reasoning connectors found. Add more structured thinking."
    else:
        return "Poor performance. Review the task requirements and provide detailed reasoning with proper tool arguments."
