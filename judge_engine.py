from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from benchmark_engine import SemanticRule, StepContract


def _runtime_api_base_url(api_base_url: Optional[str] = None) -> str:
    return api_base_url or os.environ.get("API_BASE_URL", "")


def _runtime_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.environ.get("API_KEY", "")


class JudgeChecklistItem(BaseModel):
    rule_id: str
    verdict: str = Field(..., description="pass, partial, or fail")
    evidence: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class ChecklistJudgeVerdict(BaseModel):
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    total_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    items: List[JudgeChecklistItem] = Field(default_factory=list)
    overall_feedback: str = ""
    improvement_tips: List[str] = Field(default_factory=list)
    instant_zero: bool = False
    instant_zero_reasons: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


FinalVerdict = ChecklistJudgeVerdict


JUDGE_SYSTEM = """You are a rigorous semantic checklist judge for an OpenEnv benchmark.

Rules:
- Only evaluate semantic quality, not hidden answer keys.
- Be conservative: weak evidence should score low.
- Use only pass, partial, or fail per checklist item.
- Confidence should reflect how clearly the evidence supports the judgment.
- Output strict JSON only.
"""


def _llm_call(api_base_url: str, api_key: str, model: str, user_prompt: str, schema_example: str) -> str:
    client = OpenAI(
        base_url=_runtime_api_base_url(api_base_url),
        api_key=_runtime_api_key(api_key),
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=1400,
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


def _parse_json_safe(raw: str) -> Optional[Dict[str, Any]]:
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None


def _semantic_rules_payload(step_contract: StepContract) -> List[Dict[str, Any]]:
    payload = []
    for rule in step_contract.semantic_rules:
        payload.append(
            {
                "rule_id": rule.rule_id,
                "description": rule.description,
                "evidence_tokens": rule.evidence_tokens,
                "weight": rule.weight,
            }
        )
    return payload


def _verdict_to_score(verdict: str) -> float:
    normalized = verdict.strip().lower()
    if normalized == "pass":
        return 1.0
    if normalized == "partial":
        return 0.5
    return 0.0


def _fallback_verdict(step_contract: StepContract, error: str) -> ChecklistJudgeVerdict:
    items = [
        JudgeChecklistItem(
            rule_id=rule.rule_id,
            verdict="partial",
            evidence=f"Judge fallback: {error[:140]}",
            confidence=0.0,
        )
        for rule in step_contract.semantic_rules
    ]
    score = 0.5 if items else 0.0
    return ChecklistJudgeVerdict(
        semantic_score=score,
        total_score=score,
        confidence=0.0,
        items=items,
        overall_feedback="LLM judge fallback was used because the semantic judge response could not be parsed.",
        improvement_tips=[
            "Use the deterministic checklist as the primary guide for this step.",
            "Add clearer tradeoffs, stakeholder awareness, and downside management.",
        ],
    )


def _build_prompt(
    *,
    reasoning: str,
    tool_name: str,
    args: Dict[str, Any],
    step_contract: StepContract,
    scenario: Dict[str, Any],
    step_context: Dict[str, Any],
    previous_actions: List[Dict[str, Any]],
) -> str:
    rules = _semantic_rules_payload(step_contract)
    return f"""
TASK: Judge the semantic quality of this action using the checklist items below.

SCENARIO:
{json.dumps(scenario, indent=2)}

STEP CONTEXT:
{json.dumps(step_context, indent=2)}

PREVIOUS ACTIONS:
{json.dumps(previous_actions[-3:], indent=2) if previous_actions else "[]"}

ACTION:
{json.dumps({"tool": tool_name, "args": args, "reasoning": reasoning}, indent=2)}

SEMANTIC CHECKLIST:
{json.dumps(rules, indent=2)}

For each item:
- `pass` only if the evidence is explicit and strong
- `partial` if the idea is present but incomplete
- `fail` if it is missing, weak, or contradicted

Return JSON with:
- items: array of {{rule_id, verdict, evidence, confidence}}
- confidence: 0.0-1.0 overall confidence
- overall_feedback: 2-4 sentences
- improvement_tips: 2-4 concise tips
"""


def run_llm_judge(
    *,
    reasoning: str,
    tool_name: str,
    args: Dict[str, Any],
    step_contract: StepContract,
    scenario: Dict[str, Any],
    step_context: Dict[str, Any],
    previous_actions: List[Dict[str, Any]],
    api_key: str,
    api_base_url: Optional[str] = None,
    model: str = "openai/gpt-4.1-mini",
) -> ChecklistJudgeVerdict:
    schema_example = json.dumps(
        {
            "items": [
                {
                    "rule_id": step_contract.semantic_rules[0].rule_id if step_contract.semantic_rules else "semantic_rule",
                    "verdict": "partial",
                    "evidence": "The answer mentions a tradeoff, but not the downside.",
                    "confidence": 0.71,
                }
            ],
            "confidence": 0.74,
            "overall_feedback": "The action is directionally useful but the semantic reasoning is incomplete.",
            "improvement_tips": [
                "Name the downside and mitigation explicitly.",
                "Tie the action to a measurable target.",
            ],
        }
    )
    prompt = _build_prompt(
        reasoning=reasoning,
        tool_name=tool_name,
        args=args,
        step_contract=step_contract,
        scenario=scenario,
        step_context=step_context,
        previous_actions=previous_actions,
    )
    try:
        raw = _llm_call(api_base_url or "", api_key, model, prompt, schema_example)
    except Exception as exc:
        return _fallback_verdict(step_contract, str(exc))

    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict):
        return _fallback_verdict(step_contract, raw)

    raw_items = parsed.get("items", [])
    items: List[JudgeChecklistItem] = []
    weights_total = 0.0
    weighted_score = 0.0
    rule_map: Dict[str, SemanticRule] = {
        rule.rule_id: rule for rule in step_contract.semantic_rules
    }

    for raw_item in raw_items:
        try:
            item = JudgeChecklistItem(**raw_item)
        except Exception:
            continue
        items.append(item)
        weight = rule_map.get(item.rule_id, SemanticRule(rule_id=item.rule_id, description=item.rule_id)).weight
        weights_total += weight
        weighted_score += _verdict_to_score(item.verdict) * weight

    semantic_score = weighted_score / weights_total if weights_total else 0.0
    confidence = float(parsed.get("confidence", 0.0))
    overall_feedback = str(parsed.get("overall_feedback", "")).strip()
    improvement_tips = [
        str(item).strip()
        for item in parsed.get("improvement_tips", [])
        if str(item).strip()
    ][:4]

    if not items:
        return _fallback_verdict(step_contract, raw)

    return ChecklistJudgeVerdict(
        semantic_score=_clip(semantic_score),
        total_score=_clip(semantic_score),
        confidence=_clip(confidence),
        items=items,
        overall_feedback=overall_feedback or "The semantic judge found partial evidence only.",
        improvement_tips=improvement_tips or [
            "Add stronger tradeoff reasoning.",
            "Address the active stakeholders and downside risks more directly.",
        ],
    )


def run_manual_judge(score: float, feedback: str) -> ChecklistJudgeVerdict:
    score = _clip(score)
    return ChecklistJudgeVerdict(
        semantic_score=score,
        total_score=score,
        confidence=1.0,
        items=[],
        overall_feedback=f"Manual evaluation: {feedback}" if feedback else "Manual override was applied.",
        improvement_tips=[],
    )


def _clip(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)
