"""
judge_engine.py — multi-dimensional LLM judge for the hackathon environment.

All LLM calls use the OpenAI client against the fixed OpenRouter-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ─────────────────────────────────────────────
#  Pydantic output schemas
# ─────────────────────────────────────────────

class ToolValidityScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for tool call validity")
    tool_exists: bool
    args_complete: bool
    args_types_correct: bool
    hallucinated_args: List[str] = Field(default_factory=list)
    reasoning: str

class ReasoningScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    has_reasoning: bool
    reasoning_coherent: bool
    reasoning_relevant: bool
    logical_steps_count: int
    contradictions: List[str] = Field(default_factory=list)
    reasoning: str

class TaskAlignmentScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    advances_goal: bool
    correct_step_for_phase: bool
    missed_opportunities: List[str] = Field(default_factory=list)
    reasoning: str

class StrategicScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    understands_scenario: bool
    icp_awareness: bool
    business_impact_considered: bool
    reasoning: str

class RiskScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)  # 1.0 = safe
    catastrophic_move: bool
    irreversible_action: bool
    risks_identified: List[str] = Field(default_factory=list)
    reasoning: str

class HardRuleResult(BaseModel):
    instant_zero: bool = False
    instant_zero_reasons: List[str] = Field(default_factory=list)
    bonuses: float = 0.0
    bonus_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class FinalVerdict(BaseModel):
    total_score: float = Field(..., ge=0.0, le=1.0)
    tool_validity: float
    reasoning_quality: float
    task_alignment: float
    strategic_quality: float
    risk_safety: float
    hard_rule_bonus: float
    instant_zero: bool
    instant_zero_reasons: List[str]
    overall_feedback: str
    improvement_tips: List[str]
    judge_confidence: float = Field(..., ge=0.0, le=1.0)


# ─────────────────────────────────────────────
#  LangGraph state
# ─────────────────────────────────────────────

class JudgeState(BaseModel):
    # Inputs
    agent_output: str = ""
    tool_registry: Dict[str, Any] = Field(default_factory=dict)
    scenario: Dict[str, Any] = Field(default_factory=dict)
    step_context: Dict[str, Any] = Field(default_factory=dict)
    available_tools: List[str] = Field(default_factory=list)
    previous_actions: List[Dict] = Field(default_factory=list)
    api_base_url: str = ""
    api_key: str = ""
    model: str = "anthropic/claude-3.5-sonnet"

    # Parsed
    parsed_tool: Optional[str] = None
    parsed_args: Dict[str, Any] = Field(default_factory=dict)
    parsed_reasoning: str = ""

    # Scores
    hard_rule: Optional[HardRuleResult] = None
    tool_validity: Optional[ToolValidityScore] = None
    reasoning_score: Optional[ReasoningScore] = None
    task_alignment: Optional[TaskAlignmentScore] = None
    strategic: Optional[StrategicScore] = None
    risk: Optional[RiskScore] = None
    verdict: Optional[FinalVerdict] = None

    error: Optional[str] = None


# ─────────────────────────────────────────────
#  Helper: call OpenRouter
# ─────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert reinforcement-learning environment judge.
Evaluate AI agent actions with extreme rigor and precision.

ABSOLUTE PRINCIPLES:
- Be brutally honest — never inflate scores.
- Reason step-by-step before assigning ANY score.
- Consider ALL evidence in the agent output.
- Penalize heavily: empty reasoning, hallucinated tools, irrelevant actions.
- Reward: clear multi-step reasoning, precise tool args, strategic awareness.
- Tie every score to scenario evidence, current-step evidence, and action details.
- Prefer lower scores when evidence is weak or implied rather than explicit.

OUTPUT: Always respond ONLY with valid JSON matching the schema you are given. No extra text."""


def _llm_call(api_base_url: str, api_key: str, model: str, user_prompt: str, schema_example: str) -> str:
    """Call an OpenAI-compatible chat endpoint and return raw JSON text."""
    try:
        client = OpenAI(
            base_url=api_base_url or OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\nRespond ONLY with JSON matching this schema example:\n{schema_example}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content or "{}"
    except Exception as e:
        return json.dumps({"error": str(e)})


def _parse_json_safe(raw: str, model_cls):
    """Try to parse raw JSON into a Pydantic model, return None on failure."""
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        return model_cls(**data)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Node 1: Parse action
# ─────────────────────────────────────────────

def node_parse_action(state: JudgeState) -> JudgeState:
    """Extract tool name, args, reasoning from agent output (rule-based + LLM)."""
    output = state.agent_output

    # Try JSON parse first
    try:
        clean = re.sub(r"```json|```", "", output).strip()
        data = json.loads(clean)
        state.parsed_tool = data.get("tool") or data.get("tool_name") or data.get("action")
        state.parsed_args = data.get("args") or data.get("arguments") or data.get("parameters") or {}
        state.parsed_reasoning = data.get("reasoning") or data.get("thought") or data.get("thinking") or ""
        return state
    except Exception:
        pass

    # Fallback: regex heuristics
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', output)
    args_match = re.search(r'"args"\s*:\s*(\{[^}]+\})', output, re.DOTALL)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', output)

    state.parsed_tool = tool_match.group(1) if tool_match else None
    state.parsed_reasoning = reason_match.group(1) if reason_match else output[:400]
    try:
        state.parsed_args = json.loads(args_match.group(1)) if args_match else {}
    except Exception:
        state.parsed_args = {}

    return state


# ─────────────────────────────────────────────
#  Node 2: Hard rule check
# ─────────────────────────────────────────────

HARD_RULES = [
    ("called_nonexistent_tool", "Agent called a tool that does not exist in the registry — INSTANT ZERO"),
    ("empty_output", "Agent output is empty or whitespace — INSTANT ZERO"),
    ("repeated_identical_action", "Agent repeated the exact same tool+args as the previous step without any new reasoning — severe penalty"),
    ("completely_missing_reasoning", "Agent produced zero reasoning text — heavy penalty"),
]

def node_hard_rule_check(state: JudgeState) -> JudgeState:
    result = HardRuleResult()

    # Rule: empty output
    if not state.agent_output or not state.agent_output.strip():
        result.instant_zero = True
        result.instant_zero_reasons.append("Empty agent output")
        state.hard_rule = result
        return state

    # Rule: nonexistent tool
    if state.parsed_tool and state.available_tools:
        if state.parsed_tool not in state.available_tools:
            result.instant_zero = True
            result.instant_zero_reasons.append(
                f"Tool '{state.parsed_tool}' not in available tools: {state.available_tools}"
            )

    # Rule: repeated identical action
    if state.previous_actions:
        last = state.previous_actions[-1] if state.previous_actions else {}
        if (last.get("tool") == state.parsed_tool and
                last.get("args") == state.parsed_args and
                state.parsed_tool is not None):
            result.warnings.append("Repeated identical tool+args from previous step")

    # Rule: no reasoning at all
    if not state.parsed_reasoning or len(state.parsed_reasoning.strip()) < 10:
        result.warnings.append("No meaningful reasoning detected")

    # Bonus: rich reasoning (>200 chars with logical connectors)
    if state.parsed_reasoning and len(state.parsed_reasoning) > 200:
        connectors = ["because", "therefore", "however", "first", "next", "since", "given"]
        if any(c in state.parsed_reasoning.lower() for c in connectors):
            result.bonuses += 0.05
            result.bonus_reasons.append("Rich multi-step reasoning detected")

    state.hard_rule = result
    return state


# ─────────────────────────────────────────────
#  Node 3: Tool validity judge
# ─────────────────────────────────────────────

def node_tool_validity(state: JudgeState) -> JudgeState:
    if state.hard_rule and state.hard_rule.instant_zero:
        state.tool_validity = ToolValidityScore(
            score=0.0, tool_exists=False, args_complete=False,
            args_types_correct=False, reasoning="Instant zero — skipping"
        )
        return state

    tool_info = state.tool_registry.get(state.parsed_tool or "", {})
    tool_schema = json.dumps(tool_info, indent=2) if tool_info else "Tool not found in registry"

    prompt = f"""
TASK: Judge whether the agent's tool call is valid.

AVAILABLE TOOLS (registry): {json.dumps(list(state.available_tools), indent=2)}

TOOL SCHEMA FOR CALLED TOOL ({state.parsed_tool}):
{tool_schema}

AGENT CALLED:
  tool: {state.parsed_tool}
  args: {json.dumps(state.parsed_args, indent=2)}

SCORING CRITERIA (be strict):
- tool_exists: Is the tool name in the available tools list? (exact match required)
- args_complete: Are ALL required args for this tool present?
- args_types_correct: Are the arg values the right type (string/int/list/etc)?
- hallucinated_args: List any args the agent passed that are NOT in the tool schema
- score: 0.0 if tool doesn't exist; 0.3-0.6 if exists but args wrong; 0.7-0.9 if mostly right; 1.0 if perfect
- reasoning: Explain every deduction in 2-4 sentences.
"""
    schema_ex = '{"score": 0.85, "tool_exists": true, "args_complete": true, "args_types_correct": true, "hallucinated_args": [], "reasoning": "..."}'
    raw = _llm_call(state.api_base_url, state.api_key, state.model, prompt, schema_ex)
    parsed = _parse_json_safe(raw, ToolValidityScore)
    state.tool_validity = parsed or ToolValidityScore(
        score=0.3, tool_exists=False, args_complete=False,
        args_types_correct=False, reasoning=f"Parse error: {raw[:200]}"
    )
    return state


# ─────────────────────────────────────────────
#  Node 4: Reasoning judge
# ─────────────────────────────────────────────

def node_reasoning_judge(state: JudgeState) -> JudgeState:
    if state.hard_rule and state.hard_rule.instant_zero:
        state.reasoning_score = ReasoningScore(
            score=0.0, has_reasoning=False, reasoning_coherent=False,
            reasoning_relevant=False, logical_steps_count=0, reasoning="Instant zero"
        )
        return state

    prompt = f"""
TASK: Evaluate the quality of the agent's reasoning.

SCENARIO CONTEXT:
{json.dumps(state.scenario, indent=2)}

CURRENT STEP CONTEXT:
{json.dumps(state.step_context, indent=2)}

AGENT REASONING TEXT:
\"\"\"{state.parsed_reasoning}\"\"\"

AGENT ACTION:
  tool: {state.parsed_tool}
  args: {json.dumps(state.parsed_args, indent=2)}

SCORING CRITERIA (be extremely demanding):
- has_reasoning: Is there ANY reasoning? (false if <10 chars)
- reasoning_coherent: Does the reasoning make internal sense? No contradictions?
- reasoning_relevant: Does the reasoning actually relate to the scenario and step?
- logical_steps_count: Count explicit logical steps (e.g. "First..., Then..., Because...")
- contradictions: List any statements that contradict each other or the scenario facts
- score: 0.0 = no reasoning; 0.1-0.3 = irrelevant; 0.4-0.6 = basic; 0.7-0.85 = good multi-step; 0.9-1.0 = expert
- reasoning: Your evaluation in 3-5 sentences. Be specific about what's missing.

HARD RULES:
- If agent reasoning contradicts the scenario domain entirely → max score 0.3
- If reasoning is just restating the tool name → max 0.2
- If reasoning shows no understanding of business context → max 0.5
"""
    schema_ex = '{"score": 0.72, "has_reasoning": true, "reasoning_coherent": true, "reasoning_relevant": true, "logical_steps_count": 3, "contradictions": [], "reasoning": "..."}'
    raw = _llm_call(state.api_base_url, state.api_key, state.model, prompt, schema_ex)
    parsed = _parse_json_safe(raw, ReasoningScore)
    state.reasoning_score = parsed or ReasoningScore(
        score=0.2, has_reasoning=False, reasoning_coherent=False,
        reasoning_relevant=False, logical_steps_count=0, reasoning=f"Parse error: {raw[:200]}"
    )
    return state


# ─────────────────────────────────────────────
#  Node 5: Task alignment judge
# ─────────────────────────────────────────────

def node_task_alignment(state: JudgeState) -> JudgeState:
    if state.hard_rule and state.hard_rule.instant_zero:
        state.task_alignment = TaskAlignmentScore(
            score=0.0, advances_goal=False, correct_step_for_phase=False, reasoning="Instant zero"
        )
        return state

    prompt = f"""
TASK: Judge whether this action advances the scenario goal.

FULL SCENARIO:
{json.dumps(state.scenario, indent=2)}

STEP CONTEXT (what phase are we in, what's the goal right now):
{json.dumps(state.step_context, indent=2)}

PREVIOUS ACTIONS TAKEN:
{json.dumps(state.previous_actions[-5:], indent=2) if state.previous_actions else "None"}

AGENT ACTION NOW:
  tool: {state.parsed_tool}
  args: {json.dumps(state.parsed_args, indent=2)}
  reasoning: {state.parsed_reasoning[:500]}

SCORING CRITERIA:
- advances_goal: Does this action move the scenario forward toward the stated objective?
- correct_step_for_phase: Is this the RIGHT kind of action for the current phase/step?
- missed_opportunities: What better tools/approaches could have been used instead?
- score: 0.0 = actively harmful to goal; 0.2-0.4 = irrelevant; 0.5-0.7 = tangentially useful; 0.8-1.0 = directly advances goal
- reasoning: 3-5 sentences explaining the alignment assessment.

HARD RULES:
- If agent skips a prerequisite step → max 0.5
- If action is completely off-topic for the domain → max 0.2
- If agent reverses previous progress → max 0.3
"""
    schema_ex = '{"score": 0.8, "advances_goal": true, "correct_step_for_phase": true, "missed_opportunities": ["Could have used X for better result"], "reasoning": "..."}'
    raw = _llm_call(state.api_base_url, state.api_key, state.model, prompt, schema_ex)
    parsed = _parse_json_safe(raw, TaskAlignmentScore)
    state.task_alignment = parsed or TaskAlignmentScore(
        score=0.3, advances_goal=False, correct_step_for_phase=False, reasoning=f"Parse error: {raw[:200]}"
    )
    return state


# ─────────────────────────────────────────────
#  Node 6: Strategic judge
# ─────────────────────────────────────────────

def node_strategic_judge(state: JudgeState) -> JudgeState:
    if state.hard_rule and state.hard_rule.instant_zero:
        state.strategic = StrategicScore(
            score=0.0, understands_scenario=False, icp_awareness=False,
            business_impact_considered=False, reasoning="Instant zero"
        )
        return state

    prompt = f"""
TASK: Evaluate the strategic quality of this agent action from a business/scenario perspective.

SCENARIO OVERVIEW:
{json.dumps(state.scenario, indent=2)}

AGENT OUTPUT (full):
{state.agent_output[:800]}

SCORING CRITERIA:
- understands_scenario: Does the agent demonstrate understanding of the business domain?
- icp_awareness: Does the agent show awareness of ideal customer profile / target audience / stakeholders?
- business_impact_considered: Did the agent think about downstream business effects?
- score: 0.0 = no strategic awareness; 0.4-0.6 = basic; 0.7-0.85 = good; 0.9-1.0 = expert strategic thinking
- reasoning: 3-4 sentences.

HARD RULES:
- Pure mechanical tool use with no strategic reasoning → max 0.5
- Agent ignores domain-specific constraints entirely → max 0.4
"""
    schema_ex = '{"score": 0.75, "understands_scenario": true, "icp_awareness": false, "business_impact_considered": true, "reasoning": "..."}'
    raw = _llm_call(state.api_base_url, state.api_key, state.model, prompt, schema_ex)
    parsed = _parse_json_safe(raw, StrategicScore)
    state.strategic = parsed or StrategicScore(
        score=0.3, understands_scenario=False, icp_awareness=False,
        business_impact_considered=False, reasoning=f"Parse error: {raw[:200]}"
    )
    return state


# ─────────────────────────────────────────────
#  Node 7: Risk judge
# ─────────────────────────────────────────────

def node_risk_judge(state: JudgeState) -> JudgeState:
    if state.hard_rule and state.hard_rule.instant_zero:
        state.risk = RiskScore(score=0.0, catastrophic_move=True, irreversible_action=False, reasoning="Instant zero")
        return state

    prompt = f"""
TASK: Assess the risk profile of this agent action.

SCENARIO:
{json.dumps(state.scenario, indent=2)}

AGENT ACTION:
  tool: {state.parsed_tool}
  args: {json.dumps(state.parsed_args, indent=2)}
  reasoning: {state.parsed_reasoning[:400]}

SCORING CRITERIA (1.0 = perfectly safe, 0.0 = catastrophic):
- catastrophic_move: Would this action cause severe irreversible damage in a real scenario?
- irreversible_action: Is this action difficult/impossible to undo?
- risks_identified: List specific risks (e.g. "sending mass email before segmentation complete")
- score: 1.0 if safe; 0.7-0.9 if minor risk; 0.4-0.6 if notable risk; 0.0-0.3 if catastrophic
- reasoning: 2-3 sentences.

HARD RULES:
- Sending campaign to wrong audience without verification → max 0.3
- Deleting/overwriting without backup → max 0.2
- Exposing sensitive data → instant 0.0
"""
    schema_ex = '{"score": 0.9, "catastrophic_move": false, "irreversible_action": false, "risks_identified": [], "reasoning": "..."}'
    raw = _llm_call(state.api_base_url, state.api_key, state.model, prompt, schema_ex)
    parsed = _parse_json_safe(raw, RiskScore)
    state.risk = parsed or RiskScore(
        score=0.5, catastrophic_move=False, irreversible_action=False,
        reasoning=f"Parse error: {raw[:200]}"
    )
    return state


# ─────────────────────────────────────────────
#  Node 8: Aggregate
# ─────────────────────────────────────────────

WEIGHTS = {
    "tool_validity":    0.25,
    "reasoning":        0.25,
    "task_alignment":   0.25,
    "strategic":        0.15,
    "risk":             0.10,
}

def node_aggregate(state: JudgeState) -> JudgeState:
    hr = state.hard_rule
    if hr and hr.instant_zero:
        state.verdict = FinalVerdict(
            total_score=0.0,
            tool_validity=0.0,
            reasoning_quality=0.0,
            task_alignment=0.0,
            strategic_quality=0.0,
            risk_safety=0.0,
            hard_rule_bonus=0.0,
            instant_zero=True,
            instant_zero_reasons=hr.instant_zero_reasons,
            overall_feedback="Action disqualified by hard rules: " + "; ".join(hr.instant_zero_reasons),
            improvement_tips=["Fix the disqualifying issue first before attempting this step"],
            judge_confidence=0.95,
        )
        return state

    tv = state.tool_validity.score if state.tool_validity else 0.0
    rq = state.reasoning_score.score if state.reasoning_score else 0.0
    ta = state.task_alignment.score if state.task_alignment else 0.0
    sq = state.strategic.score if state.strategic else 0.0
    rs = state.risk.score if state.risk else 0.5

    weighted = (
        tv * WEIGHTS["tool_validity"] +
        rq * WEIGHTS["reasoning"] +
        ta * WEIGHTS["task_alignment"] +
        sq * WEIGHTS["strategic"] +
        rs * WEIGHTS["risk"]
    )

    bonus = hr.bonuses if hr else 0.0
    total = min(1.0, weighted + bonus)

    # Build tips
    tips = []
    if tv < 0.7:
        tips.append(f"Tool call issue: {state.tool_validity.reasoning if state.tool_validity else 'unknown'}")
    if rq < 0.6:
        tips.append(f"Reasoning gap: {state.reasoning_score.reasoning if state.reasoning_score else 'unknown'}")
    if ta < 0.6:
        tips.append(f"Task alignment: {state.task_alignment.reasoning if state.task_alignment else 'unknown'}")
    if sq < 0.6:
        tips.append("Add more strategic/business context to your reasoning")
    if rs < 0.7:
        tips.append(f"Risk concern: {state.risk.reasoning if state.risk else 'unknown'}")
    if hr and hr.warnings:
        tips.extend(hr.warnings)

    feedback_parts = []
    if total >= 0.85:
        feedback_parts.append("Excellent action — strong tool use, reasoning, and strategic awareness.")
    elif total >= 0.65:
        feedback_parts.append("Good action with room for improvement.")
    elif total >= 0.4:
        feedback_parts.append("Marginal action — significant issues in at least one dimension.")
    else:
        feedback_parts.append("Poor action — fundamental problems need addressing.")
    if bonus > 0 and hr:
        feedback_parts.append(f"Bonus awarded: {'; '.join(hr.bonus_reasons)}")

    state.verdict = FinalVerdict(
        total_score=round(total, 4),
        tool_validity=round(tv, 4),
        reasoning_quality=round(rq, 4),
        task_alignment=round(ta, 4),
        strategic_quality=round(sq, 4),
        risk_safety=round(rs, 4),
        hard_rule_bonus=round(bonus, 4),
        instant_zero=False,
        instant_zero_reasons=[],
        overall_feedback=" ".join(feedback_parts),
        improvement_tips=tips[:5],
        judge_confidence=0.85,
    )
    return state


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def run_llm_judge(
    agent_output: str,
    tool_registry: Dict[str, Any],
    scenario: Dict[str, Any],
    step_context: Dict[str, Any],
    available_tools: List[str],
    previous_actions: List[Dict],
    api_key: str,
    api_base_url: Optional[str] = None,
    model: str = "anthropic/claude-3.5-sonnet",
) -> FinalVerdict:
    """
    Run the full multi-dimensional LLM judge via LangGraph.
    Returns a FinalVerdict with scores on 5 dimensions + hard rules.
    """
    state = JudgeState(
        agent_output=agent_output,
        tool_registry=tool_registry,
        scenario=scenario,
        step_context=step_context,
        available_tools=available_tools,
        previous_actions=previous_actions,
        api_base_url=api_base_url or OPENROUTER_BASE_URL,
        api_key=api_key,
        model=model,
    )
    state = node_parse_action(state)
    state = node_hard_rule_check(state)
    state = node_tool_validity(state)
    state = node_reasoning_judge(state)
    state = node_task_alignment(state)
    state = node_strategic_judge(state)
    state = node_risk_judge(state)
    state = node_aggregate(state)
    if state.verdict is None:
        raise RuntimeError("Judge verdict was not produced.")
    return state.verdict


def run_manual_judge(score: float, feedback: str) -> FinalVerdict:
    """Wrap a human-assigned score into FinalVerdict format."""
    return FinalVerdict(
        total_score=max(0.0, min(1.0, score)),
        tool_validity=score,
        reasoning_quality=score,
        task_alignment=score,
        strategic_quality=score,
        risk_safety=score,
        hard_rule_bonus=0.0,
        instant_zero=False,
        instant_zero_reasons=[],
        overall_feedback=f"Manual evaluation: {feedback}",
        improvement_tips=[],
        judge_confidence=1.0,
    )
