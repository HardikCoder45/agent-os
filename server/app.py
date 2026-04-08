from __future__ import annotations

import argparse
import os

import gradio as gr
from openenv.core.env_server.http_server import create_app

from benchmark_tasks import BENCHMARK_NAME, list_tasks
from hackathon_environment import HackathonAction, HackathonObservation
from server.hackathon_environment import HackathonEnvironment

try:
    from app import demo as agent_os_demo
except Exception:
    agent_os_demo = None


openenv_app = create_app(
    HackathonEnvironment,
    HackathonAction,
    HackathonObservation,
    env_name="hackathon",
    max_concurrent_envs=1,
)


@openenv_app.get("/tasks")
def tasks() -> dict:
    return {
        "benchmark": BENCHMARK_NAME,
        "count": len(list_tasks()),
        "tasks": [task.to_public_dict() for task in list_tasks()],
    }


app = openenv_app
if agent_os_demo is not None:
    app = gr.mount_gradio_app(openenv_app, agent_os_demo, path="/")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
