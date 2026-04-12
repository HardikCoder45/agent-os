#!/usr/bin/env python3
"""
AGENT OS - Baseline Inference Script
====================================

MANDATORY STDOUT FORMAT:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required environment variables:
- API_BASE_URL: The API endpoint for the LLM
- MODEL_NAME: The model identifier to use for inference
- API_KEY: API key for authentication
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from benchmark_tasks import BENCHMARK_NAME, list_tasks
from hackathon_environment import HackathonAction, HackathonEnvironment
from reward_engine import compute_reward
from tool_schemas import get_tools_for_role


API_BASE_URL = os.environ["API_BASE_URL"] if "API_BASE_URL" in os.environ else os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.environ["API_KEY"] if "API_KEY" in os.environ else ""
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PASS_THRESHOLD = 0.60
MAX_RETRIES_PER_PHASE = 2
MAX_PLAN_TOKENS = 900
MAX_REVISION_TOKENS = 500


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    action_clean = action.replace("\n", " ").replace("\r", " ")[:100]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _clean_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        return None


def _fallback_reasoning(step_question: str, goal: str, tool_name: str, visible_facts: List[str]) -> str:
    facts = "; ".join(visible_facts[:3]) if visible_facts else "the current visible constraints"
    return (
        f"First, I will use {tool_name} because this phase asks for a concrete move that advances {goal}. "
        f"I am grounding the action in {facts}, and I will name the tradeoff, timeline, and downside mitigation explicitly."
    )


def _fallback_plan(task_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = []
    for step_contract in task_payload["task_object"].contract.steps:
        plan.append(
            {
                "step_id": step_contract.step_id,
                "tool": step_contract.required_tool,
                "args": dict(step_contract.optimal_args),
                "reasoning_focus": ", ".join(step_contract.optimal_reasoning_keywords[:5]),
            }
        )
    return plan


def _plan_prompt(task_payload: Dict[str, Any], available_tools: List[str]) -> str:
    return (
        "You are planning an OpenEnv episode. Return strict JSON with a `steps` array.\n"
        "Each item must contain: `step_id`, `tool`, `args`, and `reasoning_focus`.\n"
        "Be concise, realistic, and stateful. Use only the visible scenario facts and listed tools.\n\n"
        f"Scenario title: {task_payload['scenario_title']}\n"
        f"Goal: {task_payload['goal']}\n"
        f"Briefing: {task_payload['scenario_briefing']}\n"
        f"Available tools: {available_tools}\n"
        f"Phase count: {task_payload['task_object'].contract.phase_count}\n"
        f"Current question: {task_payload['task_object'].contract.steps[0].question}\n"
    )


def _revision_prompt(
    *,
    question: str,
    context: str,
    visible_facts: List[str],
    available_tools: List[str],
    failed_checks: List[str],
    tips: List[str],
    current_action: Dict[str, Any],
) -> str:
    return (
        "Revise this OpenEnv action so it clears the current phase. Return strict JSON with `tool`, `args`, and `reasoning`.\n"
        "Use only the available tools and improve the failed checklist items.\n\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Visible facts: {visible_facts}\n"
        f"Available tools: {available_tools}\n"
        f"Failed checks: {failed_checks}\n"
        f"Tips: {tips}\n"
        f"Current action: {json.dumps(current_action, indent=2)}\n"
    )


def _call_json(client: Optional[OpenAI], *, prompt: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a concise benchmark planning assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        return _clean_json(completion.choices[0].message.content or "")
    except Exception:
        return None


def _build_initial_plan(client: Optional[OpenAI], task_payload: Dict[str, Any], available_tools: List[str]) -> List[Dict[str, Any]]:
    response = _call_json(
        client,
        prompt=_plan_prompt(task_payload, available_tools),
        max_tokens=MAX_PLAN_TOKENS,
    )
    if not response or not isinstance(response.get("steps"), list):
        return _fallback_plan(task_payload)
    steps: List[Dict[str, Any]] = []
    for raw_step in response["steps"]:
        if not isinstance(raw_step, dict):
            continue
        tool = raw_step.get("tool")
        if tool not in available_tools:
            continue
        steps.append(
            {
                "step_id": raw_step.get("step_id", ""),
                "tool": tool,
                "args": raw_step.get("args", {}),
                "reasoning_focus": raw_step.get("reasoning_focus", ""),
            }
        )
    return steps or _fallback_plan(task_payload)


def _candidate_from_plan(
    plan: List[Dict[str, Any]],
    phase_index: int,
    task_payload: Dict[str, Any],
    current_observation,
) -> Dict[str, Any]:
    if phase_index < len(plan):
        candidate = plan[phase_index]
    else:
        candidate = _fallback_plan(task_payload)[phase_index]
    reasoning = candidate.get("reasoning", "")
    if not reasoning:
        reasoning = _fallback_reasoning(
            current_observation.question,
            task_payload["goal"],
            candidate["tool"],
            list(current_observation.visible_state_facts or []),
        )
        if candidate.get("reasoning_focus"):
            reasoning += f" I will emphasize {candidate['reasoning_focus']}."
    return {
        "tool": candidate["tool"],
        "args": dict(candidate.get("args", {})),
        "reasoning": reasoning,
    }


def _self_check(
    env: HackathonEnvironment,
    task_payload: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    current_step = env._session.current_step_contract()
    tools = get_tools_for_role(task_payload["task_object"].role, task_payload["task_object"].domain)
    score, detail = compute_reward(
        agent_output=json.dumps(action),
        tool_name=action["tool"],
        args=action["args"],
        reasoning=action["reasoning"],
        available_tools=list(tools.keys()),
        tool_registry=tools,
        scenario=env._session.public_task_payload(),
        step_context={
            "phase_index": env.state.phase_index,
            "step_id": current_step.step_id,
            "question": current_step.question,
            "context": env._session.current_context(),
            "visible_facts": env._session.current_visible_facts(),
            "failure_count": env.state.failure_counts.get(current_step.step_id, 0),
            "turn_index": env.state.step_count,
        },
        previous_actions=env.state.action_history[-3:],
        task=task_payload["task_object"],
        use_llm_judge=False,
    )
    detail["preview_score"] = score
    return detail


def _revise_action(
    client: Optional[OpenAI],
    env: HackathonEnvironment,
    action: Dict[str, Any],
    preview_detail: Dict[str, Any],
) -> Dict[str, Any]:
    revised = _call_json(
        client,
        prompt=_revision_prompt(
            question=env._session.current_step_contract().question,
            context=env._session.current_context(),
            visible_facts=env._session.current_visible_facts(),
            available_tools=list(env.state.available_tools),
            failed_checks=preview_detail.get("failed_checks", []),
            tips=preview_detail.get("tips", []),
            current_action=action,
        ),
        max_tokens=MAX_REVISION_TOKENS,
    )
    if not revised:
        return action
    tool = revised.get("tool", action["tool"])
    if tool not in env.state.available_tools:
        tool = action["tool"]
    return {
        "tool": tool,
        "args": revised.get("args", action["args"]) if isinstance(revised.get("args", {}), dict) else action["args"],
        "reasoning": revised.get("reasoning", action["reasoning"]) or action["reasoning"],
    }


def _hint_rewrite(task_payload: Dict[str, Any], env: HackathonEnvironment) -> Dict[str, Any]:
    step_contract = env._session.current_step_contract()
    visible_facts = env._session.current_visible_facts()
    reasoning = _fallback_reasoning(
        env._session.current_step_contract().question,
        task_payload["goal"],
        step_contract.required_tool,
        visible_facts,
    )
    if step_contract.hint_templates:
        reasoning += " " + " ".join(step_contract.hint_templates[:2])
    return {
        "tool": step_contract.required_tool,
        "args": dict(step_contract.optimal_args),
        "reasoning": reasoning,
    }


def run_episode(client: Optional[OpenAI], task_payload: Dict[str, Any]) -> Dict[str, Any]:
    env = HackathonEnvironment(
        task_id=task_payload["id"],
        api_base_url=API_BASE_URL,
        api_key=API_KEY,
        judge_model=MODEL_NAME,
        use_llm_judge=bool(API_BASE_URL and API_KEY),
    )

    rewards: List[float] = []
    score = 0.0
    success = False
    steps_taken = 0

    log_start(task=task_payload["name"], env=BENCHMARK_NAME, model=MODEL_NAME)
    initial_observation = env.reset(task_id=task_payload["id"])
    plan = _build_initial_plan(client, task_payload, initial_observation.available_tools)

    try:
        while not env.state.done and env.state.step_count < env.state.max_steps:
            phase_index = env.state.phase_index
            candidate = _candidate_from_plan(plan, phase_index, task_payload, initial_observation if env.state.step_count == 0 else observation)

            preview_detail = _self_check(env, task_payload, candidate)
            retries = 0
            while preview_detail["preview_score"] < PASS_THRESHOLD and retries < MAX_RETRIES_PER_PHASE:
                candidate = _revise_action(client, env, candidate, preview_detail)
                preview_detail = _self_check(env, task_payload, candidate)
                retries += 1

            if preview_detail["preview_score"] < PASS_THRESHOLD and retries >= MAX_RETRIES_PER_PHASE:
                candidate = _hint_rewrite(task_payload, env)
                preview_detail = _self_check(env, task_payload, candidate)

            observation = env.step(
                HackathonAction(
                    tool=candidate["tool"],
                    args=dict(candidate["args"]),
                    reasoning=candidate["reasoning"],
                )
            )
            steps_taken += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            action_label = f"{candidate['tool']}({env._session.contract.steps[min(phase_index, len(env._session.contract.steps) - 1)].step_id})"
            log_step(step=steps_taken, action=action_label, reward=reward, done=observation.done, error=None)

        episode_summary = (observation.reward_breakdown or {}).get("episode_summary", {}) if steps_taken else {}
        score = float(episode_summary.get("final_episode_score", sum(rewards) / max(len(rewards), 1)))
        success = bool(env.state.done and not env.state.hard_failure and score >= 0.70)
    except Exception as exc:
        log_step(step=max(steps_taken, 1), action="benchmark_action", reward=0.0, done=False, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_payload["id"],
        "task_name": task_payload["name"],
        "final_score": score,
        "steps_used": steps_taken,
        "rewards": rewards,
        "success": success,
    }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY else None

    task_payloads = []
    for task in list_tasks():
        payload = task.to_public_dict()
        payload["id"] = task.id
        payload["name"] = task.name
        payload["goal"] = task.goal
        payload["scenario_title"] = task.scenario_title
        payload["scenario_briefing"] = payload["scenario_briefing"]
        payload["task_object"] = task
        task_payloads.append(payload)

    results = [run_episode(client, task_payload) for task_payload in task_payloads]
    average_score = sum(result["final_score"] for result in results) / len(results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "baseline_scores.json").write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK_NAME,
                "model_name": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "tasks": results,
                "average_score": average_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
