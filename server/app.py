from __future__ import annotations

import json
import os

from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

from benchmark_tasks import BENCHMARK_NAME, list_tasks
from hackathon_environment import HackathonAction, HackathonObservation
from server.hackathon_environment import HackathonEnvironment


app = create_app(
    HackathonEnvironment,
    HackathonAction,
    HackathonObservation,
    env_name="hackathon",
    max_concurrent_envs=1,
)


@app.get("/tasks")
def tasks() -> dict:
    return {
        "benchmark": BENCHMARK_NAME,
        "count": len(list_tasks()),
        "tasks": [task.to_public_dict() for task in list_tasks()],
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    tasks_payload = json.dumps([task.to_public_dict() for task in list_tasks()])
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Hackathon OpenEnv</title>
    <style>
      :root {{
        --paper: #f7f3ea;
        --ink: #16120d;
        --panel: rgba(255, 251, 245, 0.82);
        --line: rgba(22, 18, 13, 0.12);
        --accent: #d14d2f;
        --accent-soft: rgba(209, 77, 47, 0.12);
        --muted: #62584c;
        --shadow: 0 24px 80px rgba(22, 18, 13, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(209, 77, 47, 0.18), transparent 28%),
          radial-gradient(circle at bottom right, rgba(24, 118, 120, 0.16), transparent 26%),
          linear-gradient(180deg, #fbf8f1 0%, var(--paper) 100%);
        font-family: Georgia, "Times New Roman", serif;
      }}
      .shell {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 40px 20px 64px;
      }}
      .hero {{
        display: grid;
        gap: 22px;
        grid-template-columns: 1.15fr 0.85fr;
        align-items: start;
      }}
      .headline {{
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        border-radius: 28px;
        padding: 28px;
        position: relative;
        overflow: hidden;
      }}
      .headline::after {{
        content: "";
        position: absolute;
        inset: auto -80px -80px auto;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(209, 77, 47, 0.18), transparent 68%);
      }}
      .eyebrow {{
        display: inline-block;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        font-size: 11px;
        color: var(--accent);
        margin-bottom: 14px;
      }}
      h1 {{
        margin: 0 0 14px;
        font-size: clamp(2.4rem, 6vw, 4.7rem);
        line-height: 0.94;
      }}
      .lede {{
        margin: 0;
        max-width: 58ch;
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.65;
      }}
      .stat-grid {{
        display: grid;
        gap: 14px;
      }}
      .stat-card, .workbench, .result-box, .task-list {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
      }}
      .stat-card {{
        padding: 20px 22px;
      }}
      .stat-label {{
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }}
      .stat-value {{
        font-size: 2rem;
        margin-top: 8px;
      }}
      .layout {{
        display: grid;
        gap: 24px;
        grid-template-columns: 0.95fr 1.05fr;
        margin-top: 26px;
      }}
      .task-list {{
        padding: 18px;
      }}
      .task-card {{
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.55);
        cursor: pointer;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      }}
      .task-card:hover, .task-card.active {{
        transform: translateY(-2px);
        border-color: rgba(209, 77, 47, 0.44);
        background: var(--accent-soft);
      }}
      .task-card h3 {{
        margin: 0 0 8px;
        font-size: 1.1rem;
      }}
      .task-meta {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .workbench {{
        padding: 22px;
      }}
      textarea, input {{
        width: 100%;
        border-radius: 14px;
        border: 1px solid var(--line);
        padding: 14px;
        font: inherit;
        background: rgba(255,255,255,0.86);
      }}
      textarea {{
        min-height: 148px;
        resize: vertical;
      }}
      label {{
        display: block;
        margin: 14px 0 8px;
        font-size: 0.86rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      button {{
        appearance: none;
        border: none;
        border-radius: 999px;
        background: var(--ink);
        color: #fff8f1;
        padding: 12px 18px;
        font: inherit;
        cursor: pointer;
        transition: transform 160ms ease, opacity 160ms ease;
      }}
      button:hover {{ transform: translateY(-1px); }}
      .button-row {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 16px;
      }}
      .button-alt {{
        background: transparent;
        color: var(--ink);
        border: 1px solid var(--line);
      }}
      .result-box {{
        padding: 18px 20px;
        margin-top: 18px;
      }}
      pre {{
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.9rem;
      }}
      .tiny {{
        color: var(--muted);
        font-size: 0.93rem;
      }}
      @media (max-width: 920px) {{
        .hero, .layout {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="hero">
        <article class="headline">
          <div class="eyebrow">OpenEnv Hackathon</div>
          <h1>Judge hard.<br />Deploy clean.<br />Score fast.</h1>
          <p class="lede">
            This Space exposes a typed OpenEnv environment with deterministic benchmark graders,
            optional LLM judging, a root-level inference baseline, and Hugging Face-friendly Docker deployment.
            The root URL returns a live control surface so health checks see a real 200, not a dead landing page.
          </p>
        </article>
        <aside class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">Benchmark</div>
            <div class="stat-value">{BENCHMARK_NAME}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Tasks</div>
            <div class="stat-value">{len(list_tasks())}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">API</div>
            <div class="stat-value">`/reset` ` /step ` `/state`</div>
          </div>
        </aside>
      </section>

      <section class="layout">
        <div class="task-list">
          <div class="eyebrow">Benchmark Tasks</div>
          <div id="taskCards"></div>
        </div>
        <div class="workbench">
          <div class="eyebrow">Live Workbench</div>
          <div class="tiny" id="taskSummary">Select a task to inspect it, then trigger a reset or send a graded action.</div>
          <label for="toolInput">Tool</label>
          <input id="toolInput" placeholder="set_strategic_direction" />
          <label for="argsInput">Arguments JSON</label>
          <textarea id="argsInput">{{}}</textarea>
          <label for="reasoningInput">Reasoning</label>
          <textarea id="reasoningInput">First, I want to pick the action that best matches the scenario constraints, protects downside risk, and gives the grader measurable success criteria.</textarea>
          <div class="button-row">
            <button id="resetButton">Reset Task</button>
            <button id="stepButton">Submit Step</button>
            <button class="button-alt" id="stateButton">Fetch State</button>
          </div>
          <div class="result-box">
            <pre id="resultPane">Ready.</pre>
          </div>
        </div>
      </section>
    </div>

    <script>
      const tasks = {tasks_payload};
      let selectedTask = tasks[0];

      const taskCards = document.getElementById("taskCards");
      const taskSummary = document.getElementById("taskSummary");
      const toolInput = document.getElementById("toolInput");
      const argsInput = document.getElementById("argsInput");
      const reasoningInput = document.getElementById("reasoningInput");
      const resultPane = document.getElementById("resultPane");

      function renderTasks() {{
        taskCards.innerHTML = "";
        tasks.forEach((task) => {{
          const card = document.createElement("div");
          card.className = "task-card" + (task.task_id === selectedTask.task_id ? " active" : "");
          card.innerHTML = `
            <h3>${{task.task_id}}</h3>
            <div class="task-meta">${{task.domain}} / ${{task.role}}</div>
            <p>${{task.question}}</p>
          `;
          card.onclick = () => {{
            selectedTask = task;
            toolInput.value = task.required_tool;
            argsInput.value = JSON.stringify({{}}, null, 2);
            taskSummary.textContent = `${{task.scenario_title}} | Goal: ${{task.goal}}`;
            renderTasks();
          }};
          taskCards.appendChild(card);
        }});

        if (selectedTask) {{
          toolInput.value = selectedTask.required_tool;
          taskSummary.textContent = `${{selectedTask.scenario_title}} | Goal: ${{selectedTask.goal}}`;
        }}
      }}

      async function postJson(url, body) {{
        const response = await fetch(url, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(body),
        }});
        const data = await response.json();
        resultPane.textContent = JSON.stringify(data, null, 2);
      }}

      document.getElementById("resetButton").onclick = async () => {{
        await postJson("/reset", {{ task_id: selectedTask.task_id }});
      }};

      document.getElementById("stepButton").onclick = async () => {{
        let parsedArgs = {{}};
        try {{
          parsedArgs = JSON.parse(argsInput.value || "{{}}");
        }} catch (error) {{
          resultPane.textContent = "Arguments must be valid JSON.";
          return;
        }}

        await postJson("/step", {{
          action: {{
            tool: toolInput.value,
            args: parsedArgs,
            reasoning: reasoningInput.value,
          }}
        }});
      }};

      document.getElementById("stateButton").onclick = async () => {{
        const response = await fetch("/state");
        const data = await response.json();
        resultPane.textContent = JSON.stringify(data, null, 2);
      }};

      renderTasks();
    </script>
  </body>
</html>
"""


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
