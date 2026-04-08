from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from benchmark_tasks import HackathonTask
from tool_schemas import ToolSchema, validate_args


CONNECTOR_WORDS = (
    "because",
    "therefore",
    "however",
    "first",
    "next",
    "tradeoff",
    "risk",
    "impact",
    "goal",
)

PLACEHOLDER_PATTERN = re.compile(r"\b(test|dummy|placeholder|todo|fixme|pass|null)\b", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return {token for token in cleaned.split() if token}


def _clip(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def _semantic_match(actual: Any, expected: Any) -> float:
    if actual is None:
        return 0.0
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            return 0.0
        if expected == actual:
            return 1.0
        if expected == 0:
            return 0.0
        ratio = min(expected, actual) / max(expected, actual)
        return _clip(ratio)
    if isinstance(expected, str):
        actual_text = str(actual).strip()
        if not actual_text:
            return 0.0
        if actual_text.lower() == expected.strip().lower():
            return 1.0
        expected_tokens = _tokens(expected)
        actual_tokens = _tokens(actual_text)
        if not expected_tokens:
            return 0.5
        overlap = len(expected_tokens & actual_tokens) / len(expected_tokens)
        substring_bonus = 0.2 if expected.strip().lower() in actual_text.lower() else 0.0
        return _clip(overlap + substring_bonus)
    return 1.0 if actual == expected else 0.0


def _reasoning_score(reasoning: str, keywords: Iterable[str]) -> Tuple[float, Dict[str, Any]]:
    cleaned = reasoning.strip()
    if not cleaned:
        return 0.0, {
            "length_words": 0,
            "connector_hits": 0,
            "keyword_hits": 0,
            "placeholder_language": False,
        }

    words = cleaned.split()
    connector_hits = sum(1 for connector in CONNECTOR_WORDS if connector in cleaned.lower())
    keyword_hits = sum(1 for keyword in keywords if keyword.lower() in cleaned.lower())
    placeholder_language = bool(PLACEHOLDER_PATTERN.search(cleaned))

    score = 0.2
    score += min(0.25, len(words) / 80)
    score += min(0.2, connector_hits * 0.05)
    if keywords:
        score += min(0.35, (keyword_hits / max(len(list(keywords)), 1)) * 0.35)
    if placeholder_language:
        score -= 0.25

    return _clip(score), {
        "length_words": len(words),
        "connector_hits": connector_hits,
        "keyword_hits": keyword_hits,
        "placeholder_language": placeholder_language,
    }


def grade_task_action(
    task: HackathonTask,
    *,
    tool_name: str,
    args: Dict[str, Any],
    reasoning: str,
    available_tools: List[str],
    tool_schema: ToolSchema | None,
) -> Dict[str, Any]:
    if tool_name not in available_tools:
        return {
            "method": "deterministic_task_grader",
            "task_id": task.id,
            "task_name": task.name,
            "final_reward": 0.0,
            "tool_score": 0.0,
            "arg_validation_score": 0.0,
            "optimal_alignment_score": 0.0,
            "reasoning_score": 0.0,
            "completion_score": 0.0,
            "errors": [f"Tool '{tool_name}' is not available for role {task.role}."],
            "notes": ["Pick a tool from the role-specific registry."],
        }

    tool_score = 1.0 if tool_name == task.required_tool else 0.15

    if tool_schema is None:
        arg_validation_ok = False
        arg_errors = [f"No schema found for tool '{tool_name}'."]
        arg_quality = 0.0
    else:
        arg_validation_ok, arg_errors, arg_quality = validate_args(tool_schema, args)

    optimal_alignment_components = [
        _semantic_match(args.get(key), expected)
        for key, expected in task.optimal_args.items()
    ]
    optimal_alignment_score = _clip(
        sum(optimal_alignment_components) / max(len(optimal_alignment_components), 1)
    )

    reasoning_score, reasoning_detail = _reasoning_score(
        reasoning,
        task.optimal_reasoning_keywords,
    )

    completion_score = 0.0
    if tool_name == task.required_tool:
        completion_score += 0.4
    if arg_validation_ok:
        completion_score += 0.3
    if optimal_alignment_score >= 0.7:
        completion_score += 0.2
    if reasoning_score >= 0.65:
        completion_score += 0.1
    completion_score = _clip(completion_score)

    final_reward = _clip(
        (tool_score * 0.35)
        + (arg_quality * 0.2)
        + (optimal_alignment_score * 0.25)
        + (reasoning_score * 0.15)
        + (completion_score * 0.05)
    )

    notes = []
    if tool_name != task.required_tool:
        notes.append(
            f"Task expects `{task.required_tool}` first; `{tool_name}` is lower leverage for this benchmark."
        )
    if not arg_validation_ok:
        notes.append("Some required arguments are missing or malformed.")
    if optimal_alignment_score < 0.7:
        notes.append("Arguments only partially align with the benchmark reference solution.")
    if reasoning_score < 0.65:
        notes.append("Reasoning needs clearer tradeoffs, risks, or scenario grounding.")

    return {
        "method": "deterministic_task_grader",
        "task_id": task.id,
        "task_name": task.name,
        "tool_score": _clip(tool_score),
        "arg_validation_score": _clip(arg_quality),
        "optimal_alignment_score": optimal_alignment_score,
        "reasoning_score": reasoning_score,
        "completion_score": completion_score,
        "reasoning_detail": reasoning_detail,
        "errors": arg_errors,
        "notes": notes,
        "final_reward": final_reward,
    }


def grade_episode(task: HackathonTask, rewards: List[float], done: bool) -> Dict[str, Any]:
    if not rewards:
        return {
            "task_id": task.id,
            "task_name": task.name,
            "score": 0.0,
            "success": False,
        }

    score = _clip(sum(rewards) / len(rewards))
    return {
        "task_id": task.id,
        "task_name": task.name,
        "score": score,
        "success": bool(done and score >= task.success_threshold),
    }
