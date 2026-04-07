#!/usr/bin/env python3
"""Smoke test for the reward-based Hackathon environment."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from server.hackathon_environment import HackathonEnvironment
from models import HackathonAction

def main():
    env = HackathonEnvironment(seed=42)
    obs = env.reset()
    print(f"[RESET] event={obs.event_type} tasks={obs.tasks_completed}")
    print(f"  Task {obs.task_id} ({obs.category}): {obs.task_text[:80]}")
    print(f"  Tools: {obs.available_tools}")

    episode_reward = 0.0
    step = 0

    while not obs.episode_done:
        step += 1
        if obs.event_type == "answer" or obs.event_type == "error":
            # Move to next task
            action = HackathonAction(action_type="next_task")
        elif obs.event_type == "observation" or obs.event_type == "task_start":
            # Call the first available tool with a dummy argument
            tool = obs.available_tools[0] if obs.available_tools else ""
            if tool == "calculator":
                action = HackathonAction(action_type="call_tool", tool_name="calculator", arguments="1+1")
            elif tool == "search":
                action = HackathonAction(action_type="call_tool", tool_name="search", arguments="capital of france")
            elif tool == "text_process":
                action = HackathonAction(action_type="call_tool", tool_name="text_process", arguments="uppercase:hello")
            elif tool == "file_reader":
                action = HackathonAction(action_type="call_tool", tool_name="file_reader", arguments="list")
            elif tool == "memory":
                action = HackathonAction(action_type="call_tool", tool_name="memory", arguments="list")
            else:
                action = HackathonAction(action_type="submit_answer", arguments="42")
        elif obs.event_type == "tool_result":
            # Submit an answer after getting tool output
            action = HackathonAction(action_type="submit_answer", arguments="42")
        elif obs.event_type == "hint":
            action = HackathonAction(action_type="call_tool", tool_name=obs.available_tools[0] if obs.available_tools else "memory", arguments="test")
        elif obs.event_type == "skipped":
            action = HackathonAction(action_type="next_task")
        else:
            # Fallback: try to submit
            action = HackathonAction(action_type="submit_answer", arguments="skip")

        obs, rew, done = env.step(action)
        episode_reward += rew

        if step <= 25:
            label = "✓" if obs.correct else ("✗" if obs.correct is False else "·")
            print(f"  [{label:1s}] step={step:3d} event={obs.event_type:20s} rew={rew:+.4f} "
                  f"total={obs.total_score:.4f} hints={obs.hint[:30] if obs.hint else ''}")

        if done or step > 100:
            break

    print(f"\n[DONE] episode_reward={episode_reward:.4f} final_score={obs.total_score:.4f} "
          f"completed={obs.tasks_completed}/{obs.tasks_attempted}")
    print("  " + (obs.explanation or ""))

if __name__ == "__main__":
    main()
