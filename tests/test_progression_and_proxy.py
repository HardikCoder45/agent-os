from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

import judge_engine
from hackathon_environment import HackathonAction, HackathonEnvironment
from server.app import app


def _strong_first_step_action() -> HackathonAction:
    return HackathonAction(
        tool="set_strategic_direction",
        args={
            "focus_area": "fundraising",
            "rationale": "In the next 48 hours I need the CFO, CTO, co-founder, and the vacationing board member aligned before any VC call because the cap table model is stale, runway is only 10 weeks, revenue is growing 18% MoM, and each term sheet creates a different failure mode: Sequoia ratchet risk, Andreessen 2x liquidation preference, and Tiger down-round morale damage.",
            "tradeoffs": "We will pause noncritical roadmap work, slow two hiring loops, and pull leadership off routine execution for 48 hours so finance can rebuild the dilution model and we can define a shared walkaway line.",
            "success_criteria": "Within 48 hours we have an updated cap table, a side-by-side model for all three term sheets plus the angel bridge, and a company position targeting 12M+ pre-money, no ratchet, max 1x non-participating liquidation preference, preserved product velocity, and protected team morale.",
            "timeline_weeks": 4,
        },
        reasoning="First, I need internal alignment before any VC conversation because negotiating with a stale cap table would let a bad governance or dilution clause slip through. The tradeoff is a short pause on lower-priority shipping and hiring, but that is worth it because we only have 10 weeks of runway and one bad term could damage cap table health, morale, and future fundraising leverage. I will use this sprint to align the CFO, CTO, co-founder, and board member, define explicit walkaway terms, and tie success to measurable financing, governance, and morale outcomes. The downside risk is negotiating blind or letting Tiger's down-round framing leak into the team, so the mitigation is a 48-hour, numbers-first process with clear criteria and stakeholder communication.",
    )


def test_multi_step_progression_requires_more_than_one_successful_action() -> None:
    env = HackathonEnvironment(task_id="ceo_fundraise_or_default", use_llm_judge=False)
    env.reset(task_id="ceo_fundraise_or_default")
    observation = env.step(_strong_first_step_action())
    assert observation.reward > 0.80
    assert observation.done is False
    assert env.state.phase_index == 1
    assert env.state.phase_count == 10


def test_failed_attempt_does_not_advance_phase() -> None:
    env = HackathonEnvironment(task_id="ceo_fundraise_or_default", use_llm_judge=False)
    env.reset(task_id="ceo_fundraise_or_default")
    observation = env.step(
        HackathonAction(
            tool="call_board_meeting",
            args={"urgency": "urgent_48h", "agenda": "Need input", "desired_outcome": "Help"},
            reasoning="I should do something quickly.",
        )
    )
    assert observation.reward < 0.20
    assert observation.done is False
    assert env.state.phase_index == 0


def test_proxy_env_values_drive_runtime_configuration(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://validator.example/proxy")
    monkeypatch.setenv("API_KEY", "validator-key")
    inference = importlib.import_module("inference")
    inference = importlib.reload(inference)
    assert inference.API_BASE_URL == "http://validator.example/proxy"
    assert inference.API_KEY == "validator-key"
    assert judge_engine._runtime_api_base_url() == "http://validator.example/proxy"
    assert judge_engine._runtime_api_key() == "validator-key"

    assert "openrouter.ai/api/v1" not in Path("inference.py").read_text()
    assert "openrouter.ai/api/v1" not in Path("judge_engine.py").read_text()
    assert "openrouter.ai/api/v1" not in Path("hackathon_environment.py").read_text()


def test_root_and_tasks_endpoints_work() -> None:
    client = TestClient(app)
    root = client.get("/")
    tasks = client.get("/tasks")
    assert root.status_code == 200
    assert tasks.status_code == 200
    payload = tasks.json()
    assert payload["count"] == 12
