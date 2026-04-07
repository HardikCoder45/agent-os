# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hackathon Environment Client – Multi-Tool Agent Challenge."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HackathonAction, HackathonObservation


class HackathonEnv(EnvClient[HackathonAction, HackathonObservation, State]):
    """Client for the Hackathon Multi-Tool Agent Challenge environment.

    The agent interacts by:
    - Calling tools (calculator, search, memory, text_process, file_reader)
    - Submitting answers
    - Requesting hints (costs reward)
    - Skipping tasks

    Example:
        >>> with HackathonEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_text)
        ...     result = client.step(HackathonAction(
        ...         action_type="call_tool", tool_name="calculator", arguments="2+2"
        ...     ))
        ...     print(result.observation.tool_output)
    """

    def _step_payload(self, action: HackathonAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[HackathonObservation]:
        obs_data = payload.get("observation", {})
        observation = HackathonObservation(
            event_type=obs_data.get("event_type", ""),
            task_id=obs_data.get("task_id", 0),
            category=obs_data.get("category", ""),
            task_text=obs_data.get("task_text", ""),
            available_tools=obs_data.get("available_tools", []),
            step_limit=obs_data.get("step_limit", 10),
            steps_remaining=obs_data.get("steps_remaining", 0),
            tool_name=obs_data.get("tool_name", ""),
            tool_output=obs_data.get("tool_output", ""),
            tool_success=obs_data.get("tool_success", False),
            correct=obs_data.get("correct"),
            reward=payload.get("reward"),
            explanation=obs_data.get("explanation", ""),
            hint=obs_data.get("hint", ""),
            total_score=obs_data.get("total_score", 0.0),
            tasks_completed=obs_data.get("tasks_completed", 0),
            tasks_attempted=obs_data.get("tasks_attempted", 0),
            episode_done=obs_data.get("episode_done", False),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
