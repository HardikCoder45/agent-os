"""
app.py — Hackathon Agent Environment · Gradio UI

Preserves original UI structure (colored agent header, scenario briefing,
step question, J1 scoring panels, replay log, sub-agent console)
while adding:
  ✅ OpenRouter API key + custom/typed model in UI
  ✅ LLM judge (multi-dimensional) OR manual judge — toggle
  ✅ Configurable score threshold gate
  ✅ NO "Required tool: X" hints anywhere
  ✅ NO hardcoded sub-agent hints
  ✅ All tools shown freely — agent/user picks
"""

from __future__ import annotations

import copy
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openai import OpenAI

# ── Local imports ──
try:
    from multi_agent_env import MultiAgentEnvironment
    from agents import AGENTS
    from tool_schemas import TOOL_REGISTRY
    from judge_engine import run_llm_judge, run_manual_judge, FinalVerdict
    from reward_engine import blend_scores

    _LOADED = True
except ImportError as _e:
    print(f"[WARN] Preview mode: {_e}")
    _LOADED = False

    class MultiAgentEnvironment:
        def __init__(self, domain):
            self.domain = domain

        def reset(self, agent_role):
            return {
                "scenario_title": "Demo",
                "scenario_briefing": "Briefing text",
                "scenario_goal": "Achieve the goal",
                "step_question": "What do you do?",
                "step_context": "Here is the context",
                "step_index": 0,
                "total_steps": 4,
                "agent_title": "CEO",
                "agent_emoji": "👑",
                "agent_persona": "Persona text",
                "agent_color": "#6c63ff",
                "j1_history": [],
                "cross_agent_log": [],
            }

        def step(self, t, a, r):
            return {
                "j1": 0.5,
                "j2": 0.0,
                "done": False,
                "reasoning_breakdown": {},
                "tool_errors": [],
                "counterfactual": "",
                "cross_agent_log": [],
                "next_step": {"question": "Next?", "context": "ctx", "step_id": "s2"},
                "replay": [],
            }

        def invoke_subagent(self, role, q, ctx):
            return f"[Preview] {role}: Good question."

        def list_agents(self):
            return []

    AGENTS = {
        "tech_startup": {
            "CEO": None,
            "CTO": None,
            "CMO": None,
            "CFO": None,
            "Head_of_Product": None,
        },
        "pharma": {"CEO": None, "CSO": None, "Head_of_Regulatory": None},
        "healthcare": {"CEO": None, "CMO_Medical": None},
        "ecommerce": {"CEO": None, "CMO": None},
    }
    TOOL_REGISTRY = {}

    class _FV:
        instant_zero = False
        instant_zero_reasons = []
        total_score = 0.0
        tool_validity = 0.0
        reasoning_quality = 0.0
        task_alignment = 0.0
        strategic_quality = 0.0
        risk_safety = 0.0
        hard_rule_bonus = 0.0
        overall_feedback = ""
        improvement_tips = []
        judge_confidence = 1.0

    FinalVerdict = _FV

    def run_llm_judge(**kw):
        return None

    def run_manual_judge(s, f):
        v = _FV()
        v.total_score = s
        v.overall_feedback = f
        return v

    def blend_scores(manual_score, llm_score):
        return manual_score, {
            "manual_weight": 1.0,
            "llm_weight": 0.0,
            "agreement_adjustment": 0.0,
            "agreement_gap": None,
            "method": "preview",
        }

# ─────────────────────────────────────────────
#  Global session state
# ─────────────────────────────────────────────
ENV: Optional[MultiAgentEnvironment] = None
OBS: Dict = {}
ACTION_HISTORY: List[Dict] = []
JUDGE_HISTORY: List[Any] = []
STEP_FAILURE_COUNTS: Dict[str, int] = {}

DOMAIN_LIST = list(AGENTS.keys())

SPECIALIST_ROLES = [
    "CTO",
    "CMO",
    "CFO",
    "Head_of_People",
    "Head_of_Legal",
    "Head_of_Product",
    "Head_of_Sales",
    "Head_of_Operations",
    "Head_of_Regulatory",
    "CSO",
    "Head_of_Clinical",
    "Head_of_Marketing",
    "VP_Engineering",
    "Head_of_Risk",
]

DEFAULT_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-pro-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "Custom (type below)"
]

DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODELS[0])


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _roles_for_domain(domain: str) -> List[str]:
    return list(AGENTS.get(domain, {}).keys())


def _all_tools() -> Dict[str, Any]:
    return dict(TOOL_REGISTRY)


def _tools_for_role(role: str) -> Dict[str, Any]:
    all_t = _all_tools()
    result = {}
    for name, schema in all_t.items():
        roles = getattr(schema, "agent_roles", None)
        if roles is None or role in roles or "all" in (roles or []):
            result[name] = schema
    return result


def _tool_arg_guide(schema) -> str:
    if schema is None:
        return "_Select a tool to see its arguments._"
    name = getattr(schema, "name", "?")
    desc = getattr(schema, "description", "")
    args_list = getattr(schema, "args", None)
    if args_list:
        lines = [
            f"#### `{name}`",
            desc,
            "",
            "**Arguments** (✱ = required, ○ = optional):",
        ]
        for a in args_list:
            req = "✱" if a.required else "○"
            opts = f"  options: `{a.options}`" if a.options else ""
            lines.append(f"- {req} **{a.name}** `{a.type}`{opts}  \n  _{a.hint}_")
        return "\n".join(lines)
    input_schema = getattr(schema, "input_schema", {})
    if input_schema:
        props = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        lines = [f"#### `{name}`", desc, "", "**Arguments:**"]
        for pname, pmeta in props.items():
            req = "✱" if pname in required else "○"
            ptype = pmeta.get("type", "any")
            pdesc = pmeta.get("description", "")
            lines.append(f"- {req} **{pname}** `{ptype}` — {pdesc}")
        return "\n".join(lines)
    return f"#### `{name}`\n{desc}"


def _score_bar(score: float) -> str:
    filled = int(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    pct = int(score * 100)
    emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    return f"{emoji} {bar} {pct}%"


def _j1_history_md(history: List[float]) -> str:
    if not history:
        return "_No steps yet._"
    bars = "\n".join(f"Step {i + 1}: {_score_bar(s)}" for i, s in enumerate(history))
    avg = sum(history) / len(history)
    return f"{bars}\n\n**Average Score:** `{avg:.3f}`"


def _render_score_card(
    overall_score: float,
    *,
    score_mode: str,
    rubric_score: Optional[float] = None,
    llm_score: Optional[float] = None,
    human_score: Optional[float] = None,
    feedback: str = "",
    tips: Optional[List[str]] = None,
    blend_detail: Optional[Dict[str, Any]] = None,
    blocked: bool = False,
    block_message: str = "",
) -> str:
    lines = [
        f"## Overall Score: `{overall_score:.3f}`",
        "",
        _score_bar(overall_score),
        "",
        f"**Mode:** {score_mode}",
    ]

    component_parts = []
    if rubric_score is not None:
        component_parts.append(f"Rubric `{rubric_score:.3f}`")
    if llm_score is not None:
        component_parts.append(f"LLM `{llm_score:.3f}`")
    if human_score is not None:
        component_parts.append(f"Human `{human_score:.3f}`")
    if component_parts:
        lines.append(f"**Components:** {' | '.join(component_parts)}")

    if blend_detail and blend_detail.get("llm_weight", 0.0) > 0:
        lines.append(
            f"**Blend:** rubric {blend_detail.get('manual_weight', 0.0):.0%} + "
            f"LLM {blend_detail.get('llm_weight', 0.0):.0%}"
        )
        agreement_gap = blend_detail.get("agreement_gap")
        if agreement_gap is not None:
            lines.append(f"**Agreement Gap:** `{agreement_gap:.3f}`")

    if blocked and block_message:
        lines.extend(["", f"**Threshold Result:** {block_message}"])

    if feedback:
        lines.extend(["", f"**Feedback:** {feedback}"])

    if tips:
        lines.extend(["", "**Improvement Tips:**"])
        for tip in tips[:4]:
            lines.append(f"  -> {tip}")

    return "\n".join(lines)


def _replay_md(replay: List[Dict]) -> str:
    if not replay:
        return "_No actions yet._"
    lines = [
        "| Step | Tool | Rubric Score | Words | Tier |",
        "|------|------|--------------|-------|------|",
    ]
    for r in replay:
        j1 = f"{r.get('j1', 0):.3f}"
        rw = str(r.get("thinking_words", 0)) + "w"
        tier = r.get("rubric_tier", "?")
        lines.append(f"| {r['step']} | `{r['tool']}` | {j1} | {rw} | {tier} |")
    return "\n".join(lines)


def _progress_bar(step: int, total: int) -> str:
    if total == 0:
        return ""
    pct = step / total
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"**Progress:**  Step {step}/{total}  `{bar}`  {int(pct * 100)}%"


def _step_panel(obs: Dict[str, Any], extra_note: str = "") -> str:
    step_idx = int(obs.get("step_index", 0))
    total = int(obs.get("total_steps", 0))
    panel = (
        f"### Step {step_idx + 1} / {total}\n\n"
        f"**{obs.get('step_question', '')}**\n\n"
        f"_{obs.get('step_context', '')}_"
    )
    if extra_note:
        panel += f"\n\n{extra_note}"
    return panel


def _hint_block(obs: Dict[str, Any], counterfactual: str, fail_count: int) -> str:
    hints: List[str] = []
    arg_hints = obs.get("required_args_hints", {}) or {}
    for _, hint_text in list(arg_hints.items())[:3]:
        hints.append(f"- {hint_text}")
    if counterfactual:
        hints.append(f"- Counterfactual: {counterfactual}")
    if not hints:
        hints.append("- Make the reasoning more explicit, scenario-specific, and measurable.")
    return (
        f"**Hint Mode Activated:** {fail_count} failed attempts on this step.\n\n"
        "Use these hints before retrying:\n"
        + "\n".join(hints)
    )


def _manual_feedback(manual_score: float, manual_breakdown: Dict[str, Any], tool_errors: List[str]) -> Tuple[str, List[str]]:
    reasoning_score = float(manual_breakdown.get("reasoning_score", manual_score))
    tool_arg_score = float(manual_breakdown.get("tool_arg_score", manual_score))
    rubric_tier = manual_breakdown.get("rubric_tier", "unknown")
    tool_hints = list(manual_breakdown.get("tool_hints", []) or [])

    if manual_score >= 0.85:
        feedback = "Rubric scoring sees a strong step: the tool choice and reasoning are aligned with the scenario."
    elif manual_score >= 0.65:
        feedback = "Rubric scoring sees a workable answer, but it still needs sharper scenario alignment or fuller arguments."
    elif manual_score >= 0.40:
        feedback = "Rubric scoring sees partial progress only. The action is missing important step-specific detail."
    else:
        feedback = "Rubric scoring sees a weak move for this step. The answer is not grounded enough in the scenario requirements."

    tips: List[str] = []
    if tool_errors:
        tips.extend(tool_errors[:2])
    if tool_arg_score < 0.65:
        tips.extend(tool_hints[:2] or ["Tighten the tool arguments and cover the core required fields."])
    if reasoning_score < 0.65:
        tips.append("Make the reasoning more explicit about tradeoffs, risks, and success criteria.")
    if rubric_tier == "poor":
        tips.append("Re-anchor the answer to the exact question being asked on this step.")
    return feedback, tips[:4]


# ─────────────────────────────────────────────
#  Event handlers
# ─────────────────────────────────────────────


def on_domain_change(domain: str):
    roles = _roles_for_domain(domain)
    return gr.update(choices=roles, value=roles[0] if roles else None)


def on_start(domain: str, role: str):
    global ENV, OBS, ACTION_HISTORY, JUDGE_HISTORY, STEP_FAILURE_COUNTS
    try:
        ENV = MultiAgentEnvironment(domain=domain)
        OBS = ENV.reset(agent_role=role)
        ACTION_HISTORY = []
        JUDGE_HISTORY = []
        STEP_FAILURE_COUNTS = {}

        role_tools = _tools_for_role(role)
        tool_names = list(role_tools.keys())

        color = OBS.get("agent_color", "#6c63ff")
        emoji = OBS.get("agent_emoji", "👤")
        title = OBS.get("agent_title", role)
        persona = OBS.get("agent_persona", "")
        step_idx = OBS.get("step_index", 0)
        total = OBS.get("total_steps", 0)

        agent_header = (
            f'<div style="background:{color};color:#fff;border-radius:10px;'
            f'padding:16px 20px;margin-bottom:4px">'
            f'<span style="font-size:2em">{emoji}</span>'
            f'<span style="font-size:1.3em;font-weight:700;margin-left:10px">{title}</span>'
            f'<div style="font-size:0.85em;margin-top:6px;opacity:0.9">{persona}</div>'
            f"</div>"
        )
        scenario_md = (
            f"## 📋 {OBS.get('scenario_title', '')}\n\n"
            f"**Goal:** {OBS.get('scenario_goal', '')}\n\n"
            f"{OBS.get('scenario_briefing', '')}"
        )
        step_md = (
            f"### Step {step_idx + 1} / {total}\n\n"
            f"**{OBS.get('step_question', '')}**\n\n"
            f"_{OBS.get('step_context', '')}_"
        )
        guide = (
            _tool_arg_guide(role_tools.get(tool_names[0]))
            if tool_names
            else "_No tools available._"
        )

        return (
            agent_header,
            scenario_md,
            step_md,
            gr.update(choices=tool_names, value=tool_names[0] if tool_names else None),
            guide,
            "",
            "",
            "",
            _j1_history_md([]),
            _progress_bar(step_idx + 1, total),
            _replay_md([]),
        )
    except Exception:
        err = traceback.format_exc()
        return (
            f"❌ Error:\n```\n{err}\n```",
            "",
            "",
            gr.update(choices=[]),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )


def on_tool_select(role: str, tool_name: str):
    role_tools = _tools_for_role(role)
    return _tool_arg_guide(role_tools.get(tool_name))


def on_step(
    role: str,
    tool_name: str,
    args_text: str,
    reasoning: str,
    judge_mode: str,
    manual_score_input: float,
    manual_feedback: str,
    pass_threshold: float,
    api_key: str,
    judge_model: str,
):
    global ENV, OBS, ACTION_HISTORY, JUDGE_HISTORY, STEP_FAILURE_COUNTS
    if ENV is None:
        return ("⚠ Start a scenario first.", "", "", "", "")

    # Parse args
    try:
        args = json.loads(args_text) if args_text.strip() else {}
    except json.JSONDecodeError:
        args = {}
        for part in args_text.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip().strip("\"'")

    env_snapshot = copy.deepcopy(ENV)
    obs_snapshot = copy.deepcopy(OBS)
    action_history_snapshot = copy.deepcopy(ACTION_HISTORY)
    judge_history_snapshot = copy.deepcopy(JUDGE_HISTORY)
    step_id = str(obs_snapshot.get("step_id", f"step_{obs_snapshot.get('step_index', 0)}"))

    # Step env
    try:
        result = ENV.step(tool_name, args, reasoning)
    except Exception as e:
        result = {
            "j1": 0.0,
            "j2": 0.0,
            "done": False,
            "reasoning_breakdown": {},
            "tool_errors": [str(e)],
            "counterfactual": "",
            "cross_agent_log": [],
            "next_step": None,
            "replay": ACTION_HISTORY,
        }

    done = result.get("done", False)
    env_j1 = result.get("j1", 0.0)
    replay = result.get("replay", [])
    next_step = result.get("next_step")
    tool_errors = result.get("tool_errors", [])
    counterfactual = result.get("counterfactual", "")

    step_idx = int(obs_snapshot.get("step_index", 0)) + 1
    total = int(obs_snapshot.get("total_steps", 8))

    if done:
        j1_avg = sum(r.get("j1", 0) for r in replay) / max(len(replay), 1)
        step_md = (
            f"## ✅ Scenario Complete!\n\n"
            f"**Average Rubric Score:** {j1_avg:.3f}  |  **J2:** {result.get('j2', 0):.3f}\n\n"
            f"_Review the Replay Log below._"
        )
    elif next_step:
        q = (
            next_step.get("question", "")
            if isinstance(next_step, dict)
            else getattr(next_step, "question", "")
        )
        ctx = (
            next_step.get("context", "")
            if isinstance(next_step, dict)
            else getattr(next_step, "context", "")
        )
        step_md = f"### Step {step_idx + 1} / {total}\n\n**{q}**\n\n_{ctx}_"
    else:
        step_md = f"### Step {step_idx + 1} / {total}\n\n_Continuing…_"

    if tool_errors:
        step_md += "\n\n**⚠ Tool Issues:**\n" + "\n".join(
            f"  • {e}" for e in tool_errors
        )
    if counterfactual:
        step_md += f"\n\n**💡 Counterfactual:** _{counterfactual}_"

    # Judge
    verdict = None
    llm_verdict = None
    llm_score: Optional[float] = None
    human_score: Optional[float] = None
    blend_detail: Optional[Dict[str, Any]] = None
    rubric_score = float(result.get("manual_score", env_j1))
    manual_breakdown = result.get("manual_breakdown", {})
    rubric_feedback, rubric_tips = _manual_feedback(rubric_score, manual_breakdown, tool_errors)
    action_str = json.dumps(
        {"tool": tool_name, "args": args, "reasoning": reasoning}, indent=2
    )
    role_tools = _tools_for_role(role)
    tool_registry_serializable = {}
    for k, v in role_tools.items():
        args_list = getattr(v, "args", [])
        serializable_args = []
        for arg in args_list:
            if hasattr(arg, "__dict__"):
                serializable_args.append({
                    "name": getattr(arg, "name", ""),
                    "type": getattr(arg, "type", ""),
                    "required": getattr(arg, "required", True),
                    "options": getattr(arg, "options", None),
                    "hint": getattr(arg, "hint", ""),
                })
            else:
                serializable_args.append(arg)
        tool_registry_serializable[k] = {
            "description": getattr(v, "description", ""),
            "args": serializable_args,
        }

    llm_requested = judge_mode in {"🤖 LLM Judge", "⚖ Hybrid", "🧩 Combined Judge"}
    if llm_requested and api_key.strip():
        try:
            llm_verdict = run_llm_judge(
                agent_output=action_str,
                tool_registry=tool_registry_serializable,
                scenario={
                    "title": obs_snapshot.get("scenario_title", ""),
                    "goal": obs_snapshot.get("scenario_goal", ""),
                    "briefing": obs_snapshot.get("scenario_briefing", ""),
                    "domain": getattr(ENV, "domain", ""),
                    "role": role,
                },
                step_context={
                    "step": step_idx,
                    "total_steps": total,
                    "step_id": obs_snapshot.get("step_id", ""),
                    "question": obs_snapshot.get("step_question", ""),
                    "context": obs_snapshot.get("step_context", ""),
                    "rubric_score": rubric_score,
                },
                available_tools=list(role_tools.keys()),
                previous_actions=ACTION_HISTORY[-5:],
                api_base_url=DEFAULT_API_BASE_URL,
                api_key=api_key.strip(),
                model=judge_model,
            )
            if llm_verdict and not llm_verdict.instant_zero:
                llm_score = llm_verdict.total_score
        except Exception:
            llm_verdict = None

    feedback = rubric_feedback
    tips = list(rubric_tips)
    mode_label = "Rubric Judge"

    if judge_mode == "✍ Human Override" or judge_mode == "✍ Manual Judge":
        human_score = max(0.0, min(1.0, manual_score_input / 100.0))
        final_score = human_score
        feedback = manual_feedback if manual_feedback.strip() else "Human override is active."
        tips = []
        mode_label = "Human Override"
    elif judge_mode == "🧮 Rubric Judge":
        final_score = rubric_score
        mode_label = "Rubric Judge"
    elif judge_mode == "🤖 LLM Judge":
        if llm_verdict is not None:
            if llm_verdict.instant_zero:
                final_score = 0.0
                llm_score = 0.0
                feedback = llm_verdict.overall_feedback
                tips = list(llm_verdict.instant_zero_reasons) + list(llm_verdict.improvement_tips)
            else:
                final_score = llm_verdict.total_score
                llm_score = llm_verdict.total_score
                feedback = llm_verdict.overall_feedback
                tips = list(llm_verdict.improvement_tips)
            mode_label = "LLM Judge"
        else:
            final_score = rubric_score
            feedback = "LLM judge was unavailable, so the rubric score was used as a fallback."
            mode_label = "LLM Judge (rubric fallback)"
    else:
        if llm_verdict is not None and llm_verdict.instant_zero:
            final_score = 0.0
            llm_score = 0.0
            feedback = llm_verdict.overall_feedback
            tips = list(llm_verdict.instant_zero_reasons) + list(llm_verdict.improvement_tips) + list(rubric_tips)
            blend_detail = {
                "manual_weight": 0.55,
                "llm_weight": 0.45,
                "agreement_adjustment": 0.0,
                "agreement_gap": 1.0,
                "method": "instant_zero",
            }
        else:
            final_score, blend_detail = blend_scores(rubric_score, llm_score)
            if llm_verdict is not None and llm_score is not None:
                feedback = llm_verdict.overall_feedback
                tips = list(dict.fromkeys(list(rubric_tips) + list(llm_verdict.improvement_tips)))
                mode_label = "Combined Judge"
            else:
                feedback = "Combined mode is using the rubric score because the LLM judge is unavailable."
                mode_label = "Combined Judge (rubric fallback)"

    verdict_md = _render_score_card(
        final_score,
        score_mode=mode_label,
        rubric_score=rubric_score,
        llm_score=llm_score,
        human_score=human_score,
        feedback=feedback,
        tips=tips,
        blend_detail=blend_detail,
    )

    threshold_passed = final_score >= float(pass_threshold)

    attempt_record = {
        "step": step_idx,
        "tool": tool_name,
        "args": args,
        "reasoning": reasoning,
        "env_j1": env_j1,
        "rubric_score": rubric_score,
        "llm_score": llm_score,
        "final_score": final_score,
        "judge_score": llm_score,
        "judge_mode": mode_label,
        "passed_threshold": threshold_passed,
        "threshold": float(pass_threshold),
    }
    history_verdict = llm_verdict if llm_verdict is not None else run_manual_judge(final_score, feedback)

    if not threshold_passed:
        ENV = env_snapshot
        OBS = obs_snapshot
        ACTION_HISTORY = action_history_snapshot + [attempt_record]
        JUDGE_HISTORY = judge_history_snapshot + ([history_verdict] if history_verdict else [])
        fail_count = STEP_FAILURE_COUNTS.get(step_id, 0) + 1
        STEP_FAILURE_COUNTS[step_id] = fail_count

        failure_note = (
            f"**Threshold Gate:** Score `{final_score:.3f}` is below the pass threshold "
            f"`{pass_threshold:.2f}`. This step did **not** advance. Revise the action and retry."
        )
        if tool_errors:
            failure_note += "\n\n**⚠ Tool Issues:**\n" + "\n".join(f"  • {e}" for e in tool_errors)
        if counterfactual:
            failure_note += f"\n\n**💡 Counterfactual:** _{counterfactual}_"
        if fail_count >= 3:
            failure_note += f"\n\n{_hint_block(OBS, counterfactual, fail_count)}"
            if tips:
                failure_note += "\n\n**Scoring Hints:**\n" + "\n".join(f"- {tip}" for tip in tips[:3])

        verdict_md += (
            f"\n\n**Threshold Result:** blocked at `{final_score:.3f}` / required `{pass_threshold:.2f}`."
        )
        return (
            _step_panel(OBS, failure_note),
            verdict_md,
            _j1_history_md([item.get("final_score", item.get("env_j1", 0.0)) for item in ACTION_HISTORY]),
            _progress_bar(int(OBS.get("step_index", 0)) + 1, int(OBS.get("total_steps", total))),
            _replay_md(env_snapshot.replay if hasattr(env_snapshot, "replay") else []),
        )

    ACTION_HISTORY = action_history_snapshot + [attempt_record]
    JUDGE_HISTORY = judge_history_snapshot + ([history_verdict] if history_verdict else [])
    STEP_FAILURE_COUNTS.pop(step_id, None)
    if hasattr(ENV, "_build_obs"):
        OBS = ENV._build_obs(event="step")
    else:
        OBS["step_index"] = step_idx

    score_history = [item.get("final_score", item.get("env_j1", 0.0)) for item in ACTION_HISTORY]
    progress_md = (
        _progress_bar(int(OBS.get("step_index", 0)) + 1, int(OBS.get("total_steps", total)))
        if not done
        else f"**Complete!** {total}/{total} steps ✅"
    )

    return (
        _step_panel(OBS) if not done else step_md,
        verdict_md,
        _j1_history_md(score_history),
        progress_md,
        _replay_md(replay),
    )


def on_subagent(
    role: str,
    specialist: str,
    question: str,
    api_key: str,
    judge_model: str,
):
    if not specialist or not question.strip():
        return "Select a specialist and enter a question."
    if ENV is None:
        return "⚠ Start a scenario first."
    try:
        if api_key.strip():
            scenario_ctx = (
                f"Domain: {ENV.domain}. "
                f"Scenario: {OBS.get('scenario_title', '')}. "
                f"Current step: {OBS.get('step_question', '')}."
            )
            sys_prompt = (
                f"You are a {specialist} with deep domain expertise. "
                f"Context: {scenario_ctx} "
                "Give concise, actionable advice in 3-5 sentences. Be specific, cite numbers."
            )
            client = OpenAI(
                base_url=DEFAULT_API_BASE_URL,
                api_key=api_key.strip(),
            )
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=400,
                temperature=0.5,
            )
            answer = resp.choices[0].message.content or ""
            return f"**🤖 {specialist}:**\n\n{answer}"
        else:
            answer = ENV.invoke_subagent(specialist, question, "")
            return f"**🤖 {specialist}:** {answer}"
    except Exception as e:
        return f"❌ Sub-agent error: {e}"


def on_reset(domain: str, role: str):
    global ENV, OBS, ACTION_HISTORY, JUDGE_HISTORY, STEP_FAILURE_COUNTS
    if ENV is None:
        return (
            '<div style="background:#f0f0f0;border-radius:10px;padding:20px;'
            'color:#999;font-style:italic">Start a scenario first, then reset.</div>',
            "_Start or reset a scenario._",
            "_Step question will appear here._",
            gr.update(choices=[], value=None),
            "_Select a tool to see its arguments._",
            "",
            "",
            "",
            "_No steps yet._",
            "",
            "_No actions yet._",
        )
    try:
        ENV = MultiAgentEnvironment(domain=domain)
        OBS = ENV.reset(agent_role=role)
        ACTION_HISTORY = []
        JUDGE_HISTORY = []
        STEP_FAILURE_COUNTS = {}

        role_tools = _tools_for_role(role)
        tool_names = list(role_tools.keys())

        color = OBS.get("agent_color", "#6c63ff")
        emoji = OBS.get("agent_emoji", "👤")
        title = OBS.get("agent_title", role)
        persona = OBS.get("agent_persona", "")
        step_idx = OBS.get("step_index", 0)
        total = OBS.get("total_steps", 0)

        agent_header = (
            f'<div style="background:{color};color:#fff;border-radius:10px;'
            f'padding:16px 20px;margin-bottom:4px">'
            f'<span style="font-size:2em">{emoji}</span>'
            f'<span style="font-size:1.3em;font-weight:700;margin-left:10px">{title}</span>'
            f'<div style="font-size:0.85em;margin-top:6px;opacity:0.9">{persona}</div>'
            f"</div>"
        )
        scenario_md = (
            f"## 📋 {OBS.get('scenario_title', '')}\n\n"
            f"**Goal:** {OBS.get('scenario_goal', '')}\n\n"
            f"{OBS.get('scenario_briefing', '')}"
        )
        step_md = (
            f"### Step {step_idx + 1} / {total}\n\n"
            f"**{OBS.get('step_question', '')}**\n\n"
            f"_{OBS.get('step_context', '')}_"
        )
        guide = (
            _tool_arg_guide(role_tools.get(tool_names[0]))
            if tool_names
            else "_No tools available._"
        )

        return (
            agent_header,
            scenario_md,
            step_md,
            gr.update(choices=tool_names, value=tool_names[0] if tool_names else None),
            guide,
            "",
            "",
            "",
            _j1_history_md([]),
            _progress_bar(step_idx + 1, total),
            _replay_md([]),
        )
    except Exception:
        err = traceback.format_exc()
        return (
            f"❌ Error:\n```\n{err}\n```",
            "",
            "",
            gr.update(choices=[]),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )


def on_get_state():
    if ENV is None:
        return "_No active environment._"
    try:
        state = ENV.get_state() if hasattr(ENV, "get_state") else OBS
        if isinstance(state, dict):
            lines = ["### 🔍 Current Environment State", ""]
            for k, v in state.items():
                if k in ("j1_history", "cross_agent_log"):
                    continue
                if isinstance(v, (dict, list)):
                    lines.append(f"**{k}:**\n```json\n{json.dumps(v, indent=2)}\n```")
                else:
                    lines.append(f"**{k}:** {v}")
            return "\n".join(lines)
        return f"```\n{state}\n```"
    except Exception as e:
        return f"❌ Error getting state: {e}"


def on_history():
    if not ACTION_HISTORY:
        return "_No actions yet._"
    lines = ["| Step | Tool | Final | Rubric | LLM | Status |", "|------|------|-------|--------|-----|--------|"]
    for a in ACTION_HISTORY:
        final_score = f"{a.get('final_score', a.get('env_j1', 0.0)):.3f}"
        rubric_score = f"{a.get('rubric_score', a.get('env_j1', 0.0)):.3f}"
        llm_score = f"{a['llm_score']:.3f}" if a.get("llm_score") is not None else "—"
        status = "pass" if a.get("passed_threshold", True) else "retry"
        lines.append(f"| {a['step']} | `{a['tool']}` | {final_score} | {rubric_score} | {llm_score} | {status} |")
    if JUDGE_HISTORY:
        scores = [getattr(v, "total_score", 0) for v in JUDGE_HISTORY]
        lines.append(f"\n**Average Final Score:** `{sum(scores) / len(scores):.3f}`")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #f4efe6;
  --panel: rgba(255, 252, 247, 0.94);
  --panel-strong: rgba(255, 248, 240, 0.98);
  --ink: #1f2430;
  --muted: #5c6470;
  --line: rgba(94, 88, 80, 0.16);
  --accent: #cb5e3d;
  --accent-deep: #9f4225;
  --accent-soft: #f4cdb7;
  --gold: #d78d2f;
  --sage: #315b51;
}

body,
.gradio-container {
  font-family: 'Manrope', system-ui, sans-serif !important;
  background:
    radial-gradient(circle at top left, rgba(215, 141, 47, 0.14), transparent 26%),
    radial-gradient(circle at top right, rgba(49, 91, 81, 0.12), transparent 24%),
    linear-gradient(180deg, #f8f3eb 0%, #efe6d9 100%) !important;
  color: var(--ink) !important;
}

.gradio-container {
  max-width: 1440px !important;
}

h1, h2, h3, h4, .prose h1, .prose h2, .prose h3 {
  font-family: 'Manrope', system-ui, sans-serif !important;
  letter-spacing: -0.02em;
}

.hero-shell {
  background:
    linear-gradient(135deg, rgba(24, 31, 41, 0.98), rgba(53, 37, 29, 0.96)),
    linear-gradient(90deg, rgba(203, 94, 61, 0.16), rgba(215, 141, 47, 0.1));
  color: #fffaf5;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 24px;
  padding: 24px 28px;
  box-shadow: 0 24px 80px rgba(45, 28, 20, 0.18);
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #ffd7b2;
}

.hero-title {
  margin-top: 10px;
  font-size: 3rem;
  font-weight: 800;
  line-height: 0.95;
}

.hero-subtitle {
  margin-top: 10px;
  max-width: 900px;
  font-size: 1rem;
  line-height: 1.65;
  color: rgba(255, 246, 238, 0.84);
}

.hero-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.hero-stat {
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 14px 16px;
}

.hero-stat-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(255, 226, 204, 0.74);
}

.hero-stat-value {
  margin-top: 4px;
  font-size: 1rem;
  font-weight: 700;
}

.block-title {
  font-size: 1.1rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 6px;
}

.tool-guide-box,
.verdict-box,
.j1-box,
.step-box,
.scenario-box,
.state-box {
  background: var(--panel);
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid var(--line);
  box-shadow: 0 12px 40px rgba(82, 55, 39, 0.06);
  backdrop-filter: blur(12px);
}

.tool-guide-box { border-left: 4px solid var(--accent); }
.verdict-box { border-left: 4px solid var(--sage); }
.j1-box { border-left: 4px solid var(--gold); }
.step-box { border-left: 4px solid #4b657f; }
.scenario-box { border-left: 4px solid #8d5f3a; }
.state-box { border-left: 4px solid #6b7280; }

.tool-guide-box,
.tool-guide-box *,
.verdict-box,
.verdict-box *,
.j1-box,
.j1-box *,
.step-box,
.step-box *,
.scenario-box,
.scenario-box *,
.state-box,
.state-box * {
  color: var(--ink) !important;
}

.hero-shell code,
.tool-guide-box code,
.verdict-box code,
.j1-box code,
.step-box code,
.scenario-box code,
.state-box code,
pre,
pre code {
  color: #1f2430 !important;
  background: rgba(49, 61, 74, 0.08) !important;
  border-radius: 8px;
}

.gr-button {
  border-radius: 14px !important;
  font-weight: 700 !important;
  border: 1px solid rgba(32, 37, 45, 0.08) !important;
}

.gr-button.primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important;
  color: #fff !important;
}

.gr-button.secondary {
  background: linear-gradient(135deg, #f6ddc9, #efd0b2) !important;
  color: #4a3427 !important;
}

.gr-accordion {
  border: 1px solid var(--line) !important;
  border-radius: 18px !important;
  background: var(--panel-strong) !important;
}

.gr-accordion,
.gr-accordion *,
.gr-markdown,
.gr-markdown *,
.gr-form,
.gr-form *,
.gr-panel,
.gr-panel *,
.gr-box,
.gr-box * {
  color: var(--ink) !important;
}

.gr-box,
.gr-form,
.gr-panel {
  border-color: var(--line) !important;
  background: var(--panel-strong) !important;
}

label,
.gr-block-label,
.gradio-container label,
.gradio-container .gr-form .form,
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown li,
.gradio-container .gr-markdown strong,
.gradio-container .gr-markdown em,
.gradio-container .gr-markdown td,
.gradio-container .gr-markdown th,
.gradio-container .gr-markdown h1,
.gradio-container .gr-markdown h2,
.gradio-container .gr-markdown h3,
.gradio-container .gr-markdown h4 {
  color: var(--ink) !important;
}

input,
textarea,
select,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ink) !important;
  background: #fffaf5 !important;
  border-color: rgba(94, 88, 80, 0.22) !important;
}

input::placeholder,
textarea::placeholder {
  color: #7b6f65 !important;
  opacity: 1 !important;
}

.gradio-container table,
.gradio-container tr,
.gradio-container td,
.gradio-container th {
  color: var(--ink) !important;
}

.gradio-container .secondary-text,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li {
  color: var(--ink) !important;
}

textarea,
input,
select {
  font-family: 'IBM Plex Mono', monospace !important;
}

footer { display: none !important; }

@media (max-width: 900px) {
  .hero-title { font-size: 2.3rem; }
  .hero-grid { grid-template-columns: 1fr; }
}
"""


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="AGENT OS") as demo:
    # ── TOP: title + config ──
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(
        f"""
        <section class="hero-shell">
          <div class="hero-kicker">Agentic Evaluation System</div>
          <div class="hero-title">AGENT OS</div>
          <div class="hero-subtitle">
            A stronger multi-agent operating system for scenario execution, interactive LLM judging,
            specialist consultation, replay analysis, and OpenEnv-compatible evaluation. The OpenAI client
            is fixed to the OpenRouter-compatible endpoint: <code>{DEFAULT_API_BASE_URL}</code>.
          </div>
          <div class="hero-grid">
            <div class="hero-stat">
              <div class="hero-stat-label">Eval Depth</div>
              <div class="hero-stat-value">10 steps minimum per scenario</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-label">Judge Stack</div>
              <div class="hero-stat-value">OpenAI SDK + multi-axis LLM rubric</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-label">Operator View</div>
              <div class="hero-stat-value">Goal, situation, step question, replay, state, and sub-agents</div>
            </div>
          </div>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=2, min_width=220):
            api_key = gr.Textbox(
                label="OpenRouter API Key",
                placeholder="sk-or-v1-...",
                type="password",
                info="Used through the OpenAI client with the fixed OpenRouter endpoint.",
            )
        with gr.Column(scale=4):
            with gr.Row():
                judge_model_dd = gr.Dropdown(
                    choices=DEFAULT_MODELS,
                    value=DEFAULT_MODEL_NAME,
                    label="Model Name",
                    allow_custom_value=True,
                    interactive=True,
                    scale=4,
                    info="Type any OpenRouter-compatible model ID directly if it is not listed.",
                )
                judge_mode = gr.Radio(
                    choices=["🧩 Combined Judge", "🤖 LLM Judge", "🧮 Rubric Judge", "✍ Human Override"],
                    value="🧩 Combined Judge",
                    label="Judge Mode",
                    scale=3,
                )
        with gr.Column(scale=2, min_width=220):
            pass_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.60,
                step=0.01,
                label="Pass Threshold",
                info="A step only advances when the final score meets or exceeds this threshold.",
            )

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── ROW 1: Setup + Agent header ──
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="block-title">Scenario Setup</div>')
            with gr.Row():
                domain_dd = gr.Dropdown(
                    choices=DOMAIN_LIST, value=DOMAIN_LIST[0], label="Domain"
                )
                role_dd = gr.Dropdown(
                    choices=_roles_for_domain(DOMAIN_LIST[0]),
                    value=_roles_for_domain(DOMAIN_LIST[0])[0]
                    if _roles_for_domain(DOMAIN_LIST[0])
                    else None,
                    label="Agent Role",
                )
            with gr.Row():
                start_btn = gr.Button("Start Scenario", variant="primary", size="lg")
                reset_btn = gr.Button("Reset Scenario", variant="secondary", size="lg")
        with gr.Column(scale=2):
            agent_header = gr.HTML(
                '<div style="background:#f0f0f0;border-radius:10px;padding:20px;'
                'color:#999;font-style:italic">Start a scenario to see agent info</div>'
            )

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── ROW 2: Scenario brief + Step | Action panel ──
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<div class="block-title">Live Scenario</div>')
            scenario_display = gr.Markdown(
                "_Start a scenario._", elem_classes=["scenario-box"]
            )
            progress_md = gr.Markdown("")
            step_display = gr.Markdown(
                "_Step question will appear here._", elem_classes=["step-box"]
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="block-title">Action Workspace</div>')
            tool_dd = gr.Dropdown(
                choices=[],
                label="Select Tool",
                interactive=True,
                info="All tools for this role — choose freely",
            )
            tool_guide = gr.Markdown(
                "_Select a tool to see its arguments._", elem_classes=["tool-guide-box"]
            )
            args_input = gr.Textbox(
                label="Arguments  (JSON  or  key=value, key=value)",
                placeholder='{"channel": "seo_content", "budget": 35000, "timeline_weeks": 6}',
                lines=4,
            )
            reasoning_input = gr.Textbox(
                label="Reasoning  (explain your decision — more detail = higher score)",
                placeholder=(
                    "I'm choosing this tool because...\n"
                    "First, I need to... because the scenario shows...\n"
                    "The tradeoff is... therefore I will...\n"
                    "This advances the goal by..."
                ),
                lines=5,
            )

            with gr.Accordion("✍ Human Override Settings", open=False):
                manual_score = gr.Slider(
                    0, 100, value=70, step=1, label="Score (0–100)"
                )
                manual_feedback = gr.Textbox(
                    label="Feedback", lines=2, placeholder="Why this score?"
                )

            step_btn = gr.Button("⚡  Submit Action", variant="primary", size="lg")

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── ROW 3: J1 history | Judge verdict ──
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="block-title">Score History</div>')
            j1_display = gr.Markdown("_No steps yet._", elem_classes=["j1-box"])
        with gr.Column(scale=1):
            gr.HTML('<div class="block-title">Judge Verdict</div>')
            verdict_display = gr.Markdown(
                "_Submit an action to see the verdict._", elem_classes=["verdict-box"]
            )

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── State inspector ──
    with gr.Accordion("🛰 Environment State", open=False):
        with gr.Row():
            state_btn = gr.Button("Refresh Current State", variant="secondary")
        state_display = gr.Markdown("_No active environment._", elem_classes=["state-box"])

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── Sub-Agent Console ──
    with gr.Accordion("🧠 Sub-Agent Console  —  Consult Any Specialist", open=True):
        gr.Markdown(
            "Summon any specialist at any time. No hints — you choose who to ask and what to ask."
        )
        with gr.Row():
            specialist_dd = gr.Dropdown(
                choices=SPECIALIST_ROLES, label="Specialist", scale=2
            )
            subagent_q = gr.Textbox(
                label="Question",
                placeholder="e.g. What CAC benchmark should I target for this segment?",
                lines=2,
                scale=4,
            )
            summon_btn = gr.Button("Summon", scale=1, variant="secondary")
        subagent_out = gr.Markdown("")

    gr.HTML("<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>")

    # ── Replay Log ──
    with gr.Accordion("📜 Replay Log", open=False):
        replay_display = gr.Markdown("_No actions yet._")

    # ── Action History ──
    with gr.Accordion("📊 Action History", open=False):
        hist_btn = gr.Button("Refresh", size="sm")
        hist_display = gr.Markdown("_No actions yet._")

    # ── Event wiring ──
    domain_dd.change(on_domain_change, inputs=[domain_dd], outputs=[role_dd])

    start_btn.click(
        on_start,
        inputs=[domain_dd, role_dd],
        outputs=[
            agent_header,
            scenario_display,
            step_display,
            tool_dd,
            tool_guide,
            args_input,
            reasoning_input,
            verdict_display,
            j1_display,
            progress_md,
            replay_display,
        ],
    )

    tool_dd.change(on_tool_select, inputs=[role_dd, tool_dd], outputs=[tool_guide])

    step_btn.click(
        on_step,
        inputs=[
            role_dd,
            tool_dd,
            args_input,
            reasoning_input,
            judge_mode,
            manual_score,
            manual_feedback,
            pass_threshold,
            api_key,
            judge_model_dd,
        ],
        outputs=[
            step_display,
            verdict_display,
            j1_display,
            progress_md,
            replay_display,
        ],
    )

    summon_btn.click(
        on_subagent,
        inputs=[role_dd, specialist_dd, subagent_q, api_key, judge_model_dd],
        outputs=[subagent_out],
    )

    reset_btn.click(
        on_reset,
        inputs=[domain_dd, role_dd],
        outputs=[
            agent_header,
            scenario_display,
            step_display,
            tool_dd,
            tool_guide,
            args_input,
            reasoning_input,
            verdict_display,
            j1_display,
            progress_md,
            replay_display,
        ],
    )

    hist_btn.click(on_history, outputs=[hist_display])
    state_btn.click(on_get_state, outputs=[state_display])


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
