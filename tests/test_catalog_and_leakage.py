from __future__ import annotations

from benchmark_engine import AgentOSSession, get_episode_contract
from benchmark_tasks import list_tasks
from hackathon_environment import HackathonEnvironment


FORBIDDEN_FIELDS = {
    "required_tool",
    "required_args_hints",
    "optimal_args",
    "counterfactual_tip",
    "success_threshold",
    "scoring_rubric",
    "hint_templates",
}


def test_public_catalog_exposes_12_canonical_tasks_with_10_step_shape() -> None:
    tasks = list_tasks()
    assert len(tasks) == 12
    for task in tasks:
        payload = task.to_public_dict()
        assert payload["step_count"] == 10
        assert payload["max_steps"] >= 10
        assert payload["available_tools"]


def test_public_task_payload_hides_evaluator_fields() -> None:
    for task in list_tasks():
        payload = task.to_public_dict()
        for forbidden in FORBIDDEN_FIELDS:
            assert forbidden not in payload


def test_public_observation_is_leak_free() -> None:
    env = HackathonEnvironment(task_id="ceo_fundraise_or_default", use_llm_judge=False)
    observation = env.reset(task_id="ceo_fundraise_or_default")
    payload = observation.model_dump()
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in payload
    assert payload["visible_state_facts"]


def test_variant_reproducibility_for_same_seed() -> None:
    contract = get_episode_contract(task_id="ceo_fundraise_or_default")
    session_a = AgentOSSession(contract, seed=7)
    session_b = AgentOSSession(contract, seed=7)
    assert session_a.state.variant_id == session_b.state.variant_id
    assert session_a.current_context() == session_b.current_context()
