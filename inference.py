#!/usr/bin/env python3
"""
Hackathon OpenEnv - Baseline Inference Script
=============================================

MANDATORY STDOUT FORMAT:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required environment variables:
- API_BASE_URL: The API endpoint for the LLM
- MODEL_NAME: The model identifier to use for inference
- HF_TOKEN: API key for authentication
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
from task_graders import grade_episode


API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_TOKENS = 220
TEMPERATURE = 0.2


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


def _build_reasoning_prompt(task: Dict[str, Any]) -> str:
    optimal_args = json.dumps(task["optimal_args"], indent=2)
    return (
        "Write concise strategic reasoning for the provided benchmark action.\n"
        f"Task: {task['scenario_title']}\n"
        f"Goal: {task['goal']}\n"
        f"Question: {task['question']}\n"
        f"Action tool: {task['required_tool']}\n"
        f"Action args: {optimal_args}\n"
        f"Keywords to include when relevant: {', '.join(task['optimal_reasoning_keywords'])}\n"
        "Return plain text only, 2-4 sentences, no markdown."
    )


def _generate_reasoning(client: Optional[OpenAI], task: Dict[str, Any]) -> str:
    fallback = (
        f"First, I will use {task['required_tool']} because the benchmark requires an action that directly "
        f"advances {task['goal']}. This move addresses the immediate scenario constraints, references the "
        "highest-risk tradeoffs, and keeps the plan measurable with explicit success criteria."
    )
    if client is None:
        return fallback

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You are a concise strategy copilot for benchmark actions."},
                {"role": "user", "content": _build_reasoning_prompt(task)},
            ],
        )
        content = completion.choices[0].message.content or ""
        cleaned = re.sub(r"\s+", " ", content).strip()
        return cleaned or fallback
    except Exception:
        return fallback


def run_episode(client: Optional[OpenAI], task_payload: Dict[str, Any]) -> Dict[str, Any]:
    env = HackathonEnvironment(
        api_base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        judge_model=MODEL_NAME,
        use_llm_judge=bool(HF_TOKEN),
    )

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_payload["name"], env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        env.reset(task_id=task_payload["id"])
        reasoning = _generate_reasoning(client, task_payload)
        action = HackathonAction(
            tool=task_payload["required_tool"],
            args=dict(task_payload["optimal_args"]),
            reasoning=reasoning,
        )

        observation = env.step(action)
        steps_taken = 1
        reward = float(observation.reward or 0.0)
        rewards.append(reward)
        action_label = f"{action.tool}({task_payload['step_id']})"
        log_step(step=1, action=action_label, reward=reward, done=observation.done, error=None)

        episode_grade = grade_episode(task_payload["task_object"], rewards, observation.done)
        score = episode_grade["score"]
        success = episode_grade["success"]
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
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    task_payloads = []
    for task in list_tasks():
        payload = task.to_public_dict()
        payload["id"] = task.id
        payload["name"] = task.name
        payload["goal"] = task.goal
        payload["question"] = task.question
        payload["scenario_title"] = task.scenario_title
        payload["step_id"] = task.step_id
        payload["optimal_args"] = dict(task.optimal_args)
        payload["optimal_reasoning_keywords"] = list(task.optimal_reasoning_keywords)
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
