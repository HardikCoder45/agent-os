#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))
BASE_URL = os.environ.get("ENV_URL", f"http://127.0.0.1:{PORT}")
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _run(command: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout_s} seconds.")


def _run_adversarial_regressions() -> Dict[str, Any]:
    script = """
from hackathon_environment import HackathonEnvironment, HackathonAction
import json

TASK_ID = "ceo_fundraise_or_default"
cases = {
    "wrong_tool": HackathonAction(
        tool="call_board_meeting",
        args={"urgency": "urgent_48h", "agenda": "Need input", "desired_outcome": "Help"},
        reasoning="I should do something quickly.",
    ),
    "empty_args": HackathonAction(
        tool="set_strategic_direction",
        args={},
        reasoning="Need fundraising focus.",
    ),
    "placeholder_reasoning": HackathonAction(
        tool="set_strategic_direction",
        args={"focus_area": "fundraising", "rationale": "test", "tradeoffs": "todo", "success_criteria": "pass", "timeline_weeks": 4},
        reasoning="test dummy placeholder todo",
    ),
    "strong_action": HackathonAction(
        tool="set_strategic_direction",
        args={
            "focus_area": "fundraising",
            "rationale": "In the next 48 hours I need the CFO, CTO, co-founder, and the vacationing board member aligned before any VC call because the cap table model is stale, runway is only 10 weeks, revenue is growing 18% MoM, and each term sheet creates a different failure mode: Sequoia ratchet risk, Andreessen 2x liquidation preference, and Tiger down-round morale damage.",
            "tradeoffs": "We will pause noncritical roadmap work, slow two hiring loops, and pull leadership off routine execution for 48 hours so finance can rebuild the dilution model and we can define a shared walkaway line.",
            "success_criteria": "Within 48 hours we have an updated cap table, a side-by-side model for all three term sheets plus the angel bridge, and a company position targeting 12M+ pre-money, no ratchet, max 1x non-participating liquidation preference, preserved product velocity, and protected team morale.",
            "timeline_weeks": 4,
        },
        reasoning="First, I need internal alignment before any VC conversation because negotiating with a stale cap table would let a bad governance or dilution clause slip through. The tradeoff is a short pause on lower-priority shipping and hiring, but that is worth it because we only have 10 weeks of runway and one bad term could damage cap table health, morale, and future fundraising leverage. I will use this sprint to align the CFO, CTO, co-founder, and board member, define explicit walkaway terms, and tie success to measurable financing, governance, and morale outcomes. The downside risk is negotiating blind or letting Tiger's down-round framing leak into the team, so the mitigation is a 48-hour, numbers-first process with clear criteria and stakeholder communication.",
    ),
}
results = {}
for name, action in cases.items():
    env = HackathonEnvironment(task_id=TASK_ID, use_llm_judge=False)
    env.reset(task_id=TASK_ID)
    observation = env.step(action)
    results[name] = {
        "score": observation.reward,
        "failed_checks": observation.reward_breakdown.get("failed_checks", []),
        "done": observation.done,
    }
print(json.dumps(results))
"""
    result = _run([sys.executable, "-c", script], allow_failure=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    data = json.loads(result.stdout)
    (OUTPUT_DIR / "adversarial_regressions.json").write_text(json.dumps(data, indent=2))
    return data


def _run_judge_consistency() -> Dict[str, Any]:
    if not os.environ.get("API_BASE_URL") or not os.environ.get("API_KEY"):
        report = {
            "skipped": True,
            "reason": "API_BASE_URL or API_KEY not set; LLM judge consistency was not run.",
        }
        (OUTPUT_DIR / "judge_consistency_report.json").write_text(json.dumps(report, indent=2))
        return report

    script = """
from benchmark_tasks import get_task
from benchmark_engine import AgentOSSession
from reward_engine import compute_reward
from tool_schemas import get_tools_for_role
import json
import os

task = get_task(task_id="ceo_fundraise_or_default")
session = AgentOSSession(task.contract)
step = session.current_step_contract()
tools = get_tools_for_role(task.role, task.domain)

examples = [
    {
        "label": "strong",
        "tool": "set_strategic_direction",
        "args": {
            "focus_area": "fundraising",
            "rationale": "I need the CFO, CTO, co-founder, and board aligned before any VC call because the cap table is stale and the runway is 10 weeks.",
            "tradeoffs": "Pause lower-priority roadmap work for 48 hours while finance rebuilds the dilution model and we define walkaway terms.",
            "success_criteria": "Updated cap table, explicit walkaway line, no ratchet, max 1x non-participating liquidation preference, and preserved morale within 48 hours.",
            "timeline_weeks": 4,
        },
        "reasoning": "First I need internal alignment because negotiating with a stale cap table can lock in bad governance and dilution. The tradeoff is a short pause on lower-priority work, but the mitigation is a tight 48-hour sprint with measurable success criteria and stakeholder communication.",
    },
    {
        "label": "weak",
        "tool": "set_strategic_direction",
        "args": {},
        "reasoning": "Need fundraising focus.",
    },
]

rows = []
for example in examples:
    score, detail = compute_reward(
        agent_output=json.dumps(example),
        tool_name=example["tool"],
        args=example["args"],
        reasoning=example["reasoning"],
        available_tools=list(tools.keys()),
        tool_registry=tools,
        scenario=session.public_task_payload(),
        step_context={
            "phase_index": 0,
            "step_id": step.step_id,
            "question": step.question,
            "context": session.current_context(),
            "visible_facts": session.current_visible_facts(),
            "failure_count": 0,
            "turn_index": 0,
        },
        previous_actions=[],
        task=task,
        api_key=os.environ["API_KEY"],
        api_base_url=os.environ["API_BASE_URL"],
        judge_model=os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini"),
        use_llm_judge=True,
    )
    rows.append({
        "label": example["label"],
        "deterministic": detail["manual_score"],
        "llm": detail["llm_score"],
        "difference": None if detail["llm_score"] is None else abs(detail["manual_score"] - detail["llm_score"]),
        "final_reward": score,
    })

valid = [row["difference"] for row in rows if row["difference"] is not None]
report = {
    "examples": rows,
    "average_difference": None if not valid else sum(valid) / len(valid),
}
print(json.dumps(report))
"""
    result = _run([sys.executable, "-c", script], allow_failure=True)
    if result.returncode != 0:
        report = {"error": result.stderr.strip() or result.stdout.strip()}
    else:
        report = json.loads(result.stdout)
    (OUTPUT_DIR / "judge_consistency_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report: Dict[str, Any] = {}

    local_validation = _run(["openenv", "validate", str(ROOT), "--json"], allow_failure=True)
    report["local_validation"] = json.loads(local_validation.stdout)

    pytest_result = _run([sys.executable, "-m", "pytest", "-q"], allow_failure=True)
    report["pytest"] = {
        "returncode": pytest_result.returncode,
        "stdout_tail": pytest_result.stdout.strip().splitlines()[-20:],
        "stderr_tail": pytest_result.stderr.strip().splitlines()[-20:],
    }

    report["adversarial_regressions"] = _run_adversarial_regressions()
    report["judge_consistency"] = _run_judge_consistency()

    server = subprocess.Popen(
        [sys.executable, "-m", "server.app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_server(BASE_URL)
        runtime_validation = _run(
            ["openenv", "validate", "--url", BASE_URL],
            allow_failure=True,
        )
        report["runtime_validation"] = json.loads(runtime_validation.stdout)

        root_response = requests.get(BASE_URL, timeout=5)
        tasks_response = requests.get(f"{BASE_URL}/tasks", timeout=5)
        report["root_ping"] = {
            "status_code": root_response.status_code,
            "ok": root_response.status_code == 200,
        }
        report["tasks_ping"] = {
            "status_code": tasks_response.status_code,
            "count": tasks_response.json().get("count") if tasks_response.ok else None,
        }

        inference_result = _run([sys.executable, "inference.py"], allow_failure=True)
        report["inference"] = {
            "returncode": inference_result.returncode,
            "stdout_tail": inference_result.stdout.strip().splitlines()[-20:],
            "stderr_tail": inference_result.stderr.strip().splitlines()[-20:],
        }

        if shutil.which("docker"):
            docker_result = _run(
                ["docker", "build", "-t", "hackathon-openenv-local", "."],
                allow_failure=True,
            )
            report["docker_build"] = {
                "returncode": docker_result.returncode,
                "stderr_tail": docker_result.stderr.strip().splitlines()[-20:],
            }
        else:
            report["docker_build"] = {"skipped": True, "reason": "docker not installed"}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    output_path = OUTPUT_DIR / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
