from __future__ import annotations

import re
from typing import Any, Dict, List

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
        return _clip(len(actual_tokens & expected_tokens) / len(expected_tokens))
    return 1.0 if actual == expected else 0.0


def _reasoning_score(reasoning: str, keywords: List[str]) -> float:
    cleaned = reasoning.strip()
    if not cleaned:
        return 0.0
    words = cleaned.split()
    connector_hits = sum(1 for connector in CONNECTOR_WORDS if connector in cleaned.lower())
    keyword_hits = sum(1 for keyword in keywords if keyword.lower() in cleaned.lower())
    placeholder_language = bool(PLACEHOLDER_PATTERN.search(cleaned))
    score = 0.15
    score += min(0.25, len(words) / 100)
    score += min(0.2, connector_hits * 0.05)
    if keywords:
        score += min(0.4, (keyword_hits / len(keywords)) * 0.4)
    if placeholder_language:
        score -= 0.35
    return _clip(score)


def grade_task_action(
    task: HackathonTask,
    *,
    tool_name: str,
    args: Dict[str, Any],
    reasoning: str,
    available_tools: List[str],
    tool_schema: ToolSchema | None,
) -> Dict[str, Any]:
    step = task.contract.steps[0]
    if tool_name not in available_tools:
        return {
            "method": "legacy_task_grader",
            "task_id": task.id,
            "task_name": task.name,
            "final_reward": 0.0,
            "errors": [f"Tool '{tool_name}' is not available for role {task.role}."],
            "notes": ["Pick a tool from the role-specific registry."],
        }

    tool_score = 1.0 if tool_name == step.required_tool else 0.05
    if tool_schema is None:
        arg_validation_ok, arg_errors, arg_quality, normalized_args, hallucinated_args = False, [f"No schema found for `{tool_name}`."], 0.0, {}, []
    else:
        arg_validation_ok, arg_errors, arg_quality, normalized_args, hallucinated_args = validate_args(tool_schema, args)

    optimal_alignment_components = [
        _semantic_match((normalized_args if normalized_args else args).get(key), expected)
        for key, expected in step.optimal_args.items()
    ]
    optimal_alignment_score = _clip(sum(optimal_alignment_components) / max(len(optimal_alignment_components), 1))
    reasoning_score = _reasoning_score(reasoning, step.optimal_reasoning_keywords)

    final_reward = _clip(
        tool_score * 0.35
        + arg_quality * 0.25
        + optimal_alignment_score * 0.20
        + reasoning_score * 0.20
    )
    if hallucinated_args:
        final_reward = min(final_reward, 0.4)
    if PLACEHOLDER_PATTERN.search(reasoning):
        final_reward = min(final_reward, 0.25)

    notes = []
    if tool_name != step.required_tool:
        notes.append("The selected tool is lower leverage than the preferred phase action.")
    if not arg_validation_ok:
        notes.append("Some required arguments are missing or malformed.")
    if optimal_alignment_score < 0.7:
        notes.append("Arguments only partially align with the benchmark reference strategy.")
    if reasoning_score < 0.65:
        notes.append("Reasoning needs clearer tradeoffs, risks, or scenario grounding.")

    return {
        "method": "legacy_task_grader",
        "task_id": task.id,
        "task_name": task.name,
        "tool_score": _clip(tool_score),
        "arg_validation_score": _clip(arg_quality),
        "optimal_alignment_score": optimal_alignment_score,
        "reasoning_score": reasoning_score,
        "errors": arg_errors,
        "notes": notes,
        "final_reward": final_reward,
    }


def grade_episode(task: HackathonTask, rewards: List[float], done: bool, final_episode_score: float | None = None) -> Dict[str, Any]:
    if final_episode_score is not None:
        score = _clip(final_episode_score)
    elif rewards:
        score = _clip(sum(rewards) / len(rewards))
    else:
        score = 0.0
    return {
        "task_id": task.id,
        "task_name": task.name,
        "score": score,
        "success": bool(done and score >= 0.70),
    }
