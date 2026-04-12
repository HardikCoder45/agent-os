from __future__ import annotations

from hackathon_environment import HackathonAction, HackathonEnvironment


TASK_ID = "ceo_fundraise_or_default"


def _run(action: HackathonAction) -> float:
    env = HackathonEnvironment(task_id=TASK_ID, use_llm_judge=False)
    env.reset(task_id=TASK_ID)
    observation = env.step(action)
    return float(observation.reward or 0.0)


def test_wrong_tool_scores_below_point_two() -> None:
    score = _run(
        HackathonAction(
            tool="call_board_meeting",
            args={"urgency": "urgent_48h", "agenda": "Need input", "desired_outcome": "Help"},
            reasoning="I should do something quickly.",
        )
    )
    assert score < 0.20


def test_empty_args_scores_below_point_two() -> None:
    score = _run(
        HackathonAction(
            tool="set_strategic_direction",
            args={},
            reasoning="Need fundraising focus.",
        )
    )
    assert score < 0.20


def test_placeholder_reasoning_cannot_pass() -> None:
    score = _run(
        HackathonAction(
            tool="set_strategic_direction",
            args={
                "focus_area": "fundraising",
                "rationale": "test",
                "tradeoffs": "todo",
                "success_criteria": "pass",
                "timeline_weeks": 4,
            },
            reasoning="test dummy placeholder todo",
        )
    )
    assert score < 0.30


def test_strong_action_scores_above_point_eight() -> None:
    score = _run(
        HackathonAction(
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
    )
    assert score > 0.80
