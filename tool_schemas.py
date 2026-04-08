from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgSpec:
    name: str
    type: str
    required: bool = True
    options: list[str] | None = None
    min_val: float | None = None
    max_val: float | None = None
    hint: str = ""


@dataclass
class ToolSchema:
    name: str
    description: str
    agent_roles: list[str]
    args: list[ArgSpec] = field(default_factory=list)
    domain: str = "all"
    category: str = "general"


def validate_args(schema: ToolSchema, provided: dict[str, Any]) -> tuple[bool, list[str], float]:
    errors: list[str] = []
    quality_scores: list[float] = []

    for spec in schema.args:
        if spec.required and spec.name not in provided:
            errors.append(f"Missing required argument: `{spec.name}` ({spec.hint})")
            quality_scores.append(0.0)
            continue
        val = provided.get(spec.name)
        if val is None:
            quality_scores.append(0.3)
            continue
        if spec.options and val not in spec.options:
            errors.append(f"`{spec.name}` must be one of {spec.options}, got '{val}'")
            quality_scores.append(0.2)
            continue
        if spec.type == "str" and isinstance(val, str):
            depth_score = min(1.0, len(val.split()) / 8)
            quality_scores.append(0.5 + 0.5 * depth_score)
        elif spec.type == "int" and isinstance(val, (int, float)):
            if spec.min_val is not None and val < spec.min_val:
                errors.append(f"`{spec.name}` too low (min {spec.min_val})")
                quality_scores.append(0.3)
            elif spec.max_val is not None and val > spec.max_val:
                errors.append(f"`{spec.name}` too high (max {spec.max_val})")
                quality_scores.append(0.3)
            else:
                quality_scores.append(0.9)
        elif spec.type == "float" and isinstance(val, (int, float)):
            quality_scores.append(0.9)
        elif spec.type == "enum" and val in (spec.options or []):
            quality_scores.append(1.0)
        else:
            quality_scores.append(0.6)

    overall = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    return len(errors) == 0, errors, round(overall, 4)


TOOL_REGISTRY: dict[str, ToolSchema] = {}


def _reg(t: ToolSchema) -> ToolSchema:
    TOOL_REGISTRY[t.name] = t
    return t


def get_tools_for_role(role: str, domain: str = "all") -> dict[str, ToolSchema]:
    return {
        name: schema
        for name, schema in TOOL_REGISTRY.items()
        if (schema.domain in {"all", domain})
        and ("all" in schema.agent_roles or role in schema.agent_roles)
    }


_reg(ToolSchema("approve_budget", "Approve or deny a budget request with justification",
    agent_roles=["CEO", "CFO"],
    args=[
        ArgSpec("amount", "int", required=True, min_val=0, hint="Dollar amount to approve"),
        ArgSpec("department", "enum", required=True, options=["engineering", "marketing", "sales", "ops", "hr", "r_and_d", "legal"], hint="Which department"),
        ArgSpec("purpose", "str", required=True, hint="Specific purpose - be detailed (what, why, expected outcome)"),
        ArgSpec("timeline_weeks", "int", required=False, min_val=1, max_val=52, hint="Expected spend timeline"),
        ArgSpec("success_metric", "str", required=False, hint="How will you measure ROI on this spend?"),
    ],
    category="finance"
))

_reg(ToolSchema("set_strategic_direction", "Set or update company strategic direction",
    agent_roles=["CEO"],
    args=[
        ArgSpec("focus_area", "enum", required=True, options=["product", "growth", "profitability", "fundraising", "hiring", "partnerships", "survival"], hint="Primary focus for next quarter"),
        ArgSpec("rationale", "str", required=True, hint="Why this direction now? Reference specific company metrics"),
        ArgSpec("tradeoffs", "str", required=True, hint="What are you explicitly NOT prioritizing? Why?"),
        ArgSpec("success_criteria", "str", required=True, hint="What does success look like in 90 days? Be specific"),
        ArgSpec("timeline_weeks", "int", required=True, min_val=4, max_val=52),
    ],
    category="strategy"
))

_reg(ToolSchema("call_board_meeting", "Call an emergency or scheduled board meeting",
    agent_roles=["CEO"],
    args=[
        ArgSpec("urgency", "enum", required=True, options=["emergency_24h", "urgent_48h", "scheduled_1week"], hint="How soon needed?"),
        ArgSpec("agenda", "str", required=True, hint="Specific agenda items — be exhaustive"),
        ArgSpec("desired_outcome", "str", required=True, hint="What decision or approval do you need from the board?"),
        ArgSpec("pre_read_needed", "str", required=False, hint="What materials to distribute before the meeting?"),
    ],
    category="governance"
))

_reg(ToolSchema("make_hire_decision", "Hire or promote for a key role",
    agent_roles=["CEO", "CTO", "Head_of_People", "COO"],
    args=[
        ArgSpec("role_title", "str", required=True, hint="Exact role title"),
        ArgSpec("seniority", "enum", required=True, options=["junior", "mid", "senior", "staff", "principal", "vp", "c_level"], hint="Seniority level"),
        ArgSpec("team", "enum", required=True, options=["engineering", "product", "design", "marketing", "sales", "ops", "hr", "finance", "legal", "research"], hint="Which team"),
        ArgSpec("urgency", "enum", required=True, options=["backfill_critical", "new_headcount", "planned_growth", "strategic_hire"], hint="Why now?"),
        ArgSpec("budget_annual", "int", required=True, min_val=30000, max_val=800000, hint="Total annual comp budget"),
        ArgSpec("justification", "str", required=True, hint="Business case: what problem does this hire solve? How measured?"),
    ],
    category="people"
))

_reg(ToolSchema("respond_to_crisis", "Issue official company response to a PR, legal, or operational crisis",
    agent_roles=["CEO", "CMO", "Head_of_Comms", "COO"],
    args=[
        ArgSpec("crisis_type", "enum", required=True, options=["pr_social_media", "legal_threat", "product_outage", "data_breach", "employee_issue", "customer_complaint", "regulatory", "media_inquiry"], hint="Nature of the crisis"),
        ArgSpec("response_channel", "enum", required=True, options=["public_statement", "private_dm", "press_release", "internal_memo", "social_post", "no_comment", "legal_letter"], hint="How to respond"),
        ArgSpec("stance", "enum", required=True, options=["full_apology", "partial_acknowledgment", "factual_correction", "no_admission", "proactive_transparency", "deflect"], hint="What position to take"),
        ArgSpec("message", "str", required=True, hint="Core message — what exactly are you communicating? Draft it"),
        ArgSpec("follow_up_actions", "str", required=True, hint="What concrete actions follow this response?"),
    ],
    category="comms"
))

_reg(ToolSchema("ship_feature", "Approve and ship a product feature",
    agent_roles=["CTO", "Head_of_Product", "VP_Engineering"],
    args=[
        ArgSpec("feature_name", "str", required=True, hint="Name of the feature"),
        ArgSpec("team_size", "int", required=True, min_val=1, max_val=20, hint="Engineers assigned"),
        ArgSpec("sprint_weeks", "int", required=True, min_val=1, max_val=12, hint="Sprint duration in weeks"),
        ArgSpec("quality_bar", "enum", required=True, options=["mvp_rough", "beta_tested", "production_grade", "enterprise_grade"], hint="Required quality level before ship"),
        ArgSpec("rollout_strategy", "enum", required=True, options=["full_release", "gradual_10pct", "beta_users_only", "internal_only", "flag_gated"], hint="How to roll out"),
        ArgSpec("success_metric", "str", required=True, hint="How do you define success for this feature? Specific KPI"),
        ArgSpec("rollback_plan", "str", required=False, hint="What's the rollback if it breaks?"),
    ],
    category="product"
))

_reg(ToolSchema("manage_technical_debt", "Decision on technical debt: pay down or defer",
    agent_roles=["CTO", "VP_Engineering"],
    args=[
        ArgSpec("area", "enum", required=True, options=["infrastructure", "database", "api_layer", "frontend", "security", "ci_cd", "monitoring", "dependencies"], hint="Debt area"),
        ArgSpec("severity", "enum", required=True, options=["critical_blocking", "high_risk", "medium_nuisance", "low_cosmetic"], hint="Current impact"),
        ArgSpec("decision", "enum", required=True, options=["pay_down_now", "schedule_sprint", "accept_and_document", "architectural_rewrite"], hint="What to do"),
        ArgSpec("eng_weeks_cost", "int", required=True, min_val=1, max_val=52, hint="Engineering weeks this will cost"),
        ArgSpec("risk_if_deferred", "str", required=True, hint="What breaks if you don't fix this? Be specific"),
    ],
    category="engineering"
))

_reg(ToolSchema("launch_campaign", "Launch a marketing campaign",
    agent_roles=["CMO", "Head_of_Marketing", "VP_Growth"],
    args=[
        ArgSpec("channel", "enum", required=True, options=["paid_social", "seo_content", "influencer", "email", "pr_media", "events", "partnerships", "product_led", "community", "out_of_home"], hint="Primary channel"),
        ArgSpec("budget", "int", required=True, min_val=500, max_val=5000000, hint="Campaign budget in dollars"),
        ArgSpec("target_segment", "str", required=True, hint="Exact target audience: demographics, psychographics, job titles, behaviors"),
        ArgSpec("core_message", "str", required=True, hint="One-sentence core message the campaign communicates"),
        ArgSpec("goal_metric", "enum", required=True, options=["brand_awareness", "signups", "revenue", "retention", "nps", "press_coverage", "community_growth"], hint="Primary goal"),
        ArgSpec("target_value", "int", required=True, min_val=1, hint="Numeric target (signups, $ revenue, coverage pieces, etc.)"),
        ArgSpec("timeline_weeks", "int", required=True, min_val=1, max_val=26),
    ],
    category="marketing"
))

_reg(ToolSchema("manage_investor_relations", "Manage LP/investor communication and relationships",
    agent_roles=["CEO", "CFO", "Head_of_IR"],
    args=[
        ArgSpec("communication_type", "enum", required=True, options=["board_update", "monthly_newsletter", "crisis_notification", "fundraise_outreach", "lp_call", "term_sheet_response", "data_room_share"], hint="Type of investor communication"),
        ArgSpec("transparency_level", "enum", required=True, options=["full_disclosure", "headline_metrics_only", "positive_spin", "selective_sharing", "minimal"], hint="How much to share"),
        ArgSpec("key_message", "str", required=True, hint="Core narrative — what do you want investors to take away?"),
        ArgSpec("risks_disclosed", "str", required=True, hint="What risks are you surfacing? (hiding risks scores negatively)"),
        ArgSpec("ask_or_action", "str", required=False, hint="What are you asking from investors, if anything?"),
    ],
    category="finance"
))

_reg(ToolSchema("conduct_clinical_trial", "Initiate or modify a clinical trial phase",
    agent_roles=["CSO", "Head_of_Clinical", "CMO_Medical"],
    domain="pharma",
    args=[
        ArgSpec("phase", "enum", required=True, options=["preclinical", "phase_1", "phase_2", "phase_2b", "phase_3", "phase_4"], hint="Trial phase"),
        ArgSpec("patient_cohort_size", "int", required=True, min_val=10, max_val=10000, hint="Number of patients"),
        ArgSpec("primary_endpoint", "str", required=True, hint="Primary efficacy or safety endpoint being measured"),
        ArgSpec("safety_monitoring", "enum", required=True, options=["standard_dsmb", "enhanced_monitoring", "continuous_realtime", "independent_review"], hint="Safety oversight level"),
        ArgSpec("duration_months", "int", required=True, min_val=1, max_val=60, hint="Trial duration"),
        ArgSpec("stopping_criteria", "str", required=True, hint="Under what conditions would you stop the trial early?"),
    ],
    category="clinical"
))

_reg(ToolSchema("file_regulatory_submission", "File a regulatory submission (IND, NDA, BLA, 510k, etc.)",
    agent_roles=["Head_of_Regulatory", "CSO"],
    domain="pharma",
    args=[
        ArgSpec("submission_type", "enum", required=True, options=["IND", "NDA", "BLA", "sNDA", "510k", "EUA", "MAA", "ANDA"], hint="Type of regulatory filing"),
        ArgSpec("target_agency", "enum", required=True, options=["FDA", "EMA", "PMDA", "Health_Canada", "TGA"], hint="Regulatory body"),
        ArgSpec("data_package_completeness", "enum", required=True, options=["complete", "rolling_submission", "expedited_pathway", "breakthrough_designation"], hint="Submission strategy"),
        ArgSpec("priority_designation", "enum", required=False, options=["standard", "priority_review", "breakthrough_therapy", "fast_track", "accelerated_approval"], hint="Any special designation?"),
        ArgSpec("response_to_deficiencies", "str", required=False, hint="How are you addressing known data gaps?"),
    ],
    category="regulatory"
))

_reg(ToolSchema("summon_subagent", "Summon a sub-agent for expert consultation on a decision",
    agent_roles=["CEO", "COO", "CTO", "CMO", "CFO", "CSO", "Head_of_Operations"],
    args=[
        ArgSpec("agent_role", "enum", required=True,
            options=["CTO", "CMO", "CFO", "Head_of_People", "Head_of_Legal", "Head_of_Product",
                     "Head_of_Sales", "Head_of_Operations", "Head_of_Regulatory", "CSO",
                     "Head_of_Clinical", "Head_of_Marketing", "VP_Engineering", "Head_of_Risk"],
            hint="Which expert to consult"),
        ArgSpec("question", "str", required=True, hint="Specific question or decision you need their expert input on"),
        ArgSpec("context", "str", required=True, hint="What context does the sub-agent need to give useful advice?"),
        ArgSpec("urgency", "enum", required=True, options=["immediate_1h", "today_eod", "this_week", "advisory"], hint="How fast do you need their input?"),
        ArgSpec("decision_authority", "enum", required=True, options=["they_decide", "they_advise_i_decide", "we_decide_together"], hint="Are they deciding or advising?"),
    ],
    category="management"
))

_reg(ToolSchema("negotiate_deal", "Negotiate a business deal: acquisition, partnership, supplier, customer",
    agent_roles=["CEO", "CFO", "Head_of_BD", "Head_of_Sales"],
    args=[
        ArgSpec("deal_type", "enum", required=True, options=["acquisition", "acqui_hire", "licensing", "partnership", "supplier_contract", "customer_enterprise", "joint_venture", "investment"], hint="Type of deal"),
        ArgSpec("our_position", "str", required=True, hint="What we want from this deal — specific terms"),
        ArgSpec("their_likely_position", "str", required=True, hint="What you think they want — show you've thought about their interests"),
        ArgSpec("walkaway_point", "str", required=True, hint="What terms would make you walk away? Be specific"),
        ArgSpec("leverage_points", "str", required=True, hint="What leverage do you have in this negotiation?"),
        ArgSpec("timeline_to_close", "int", required=True, min_val=1, max_val=180, hint="Days to target close"),
    ],
    category="business_development"
))

_reg(ToolSchema("manage_team_health", "Take action on team health, culture, or performance",
    agent_roles=["CEO", "Head_of_People", "COO", "CTO"],
    args=[
        ArgSpec("issue_type", "enum", required=True, options=["burnout", "retention_risk", "cultural_conflict", "performance_gap", "compensation_equity", "layoff", "reorg", "toxic_behavior", "remote_hybrid_policy"], hint="Type of issue"),
        ArgSpec("affected_team", "str", required=True, hint="Which team or individuals are affected"),
        ArgSpec("action", "enum", required=True, options=["1on1_conversations", "compensation_adjustment", "role_change", "pip", "termination", "team_offsite", "policy_change", "leadership_coaching", "mental_health_support", "reorg"], hint="Action to take"),
        ArgSpec("expected_outcome", "str", required=True, hint="What specific improvement do you expect and by when?"),
        ArgSpec("risk_if_no_action", "str", required=True, hint="What happens if you do nothing?"),
    ],
    category="people"
))

_reg(ToolSchema("allocate_production_capacity", "Allocate or change manufacturing production capacity",
    agent_roles=["COO", "Head_of_Operations", "Head_of_Manufacturing"],
    domain="manufacturing",
    args=[
        ArgSpec("line_id", "str", required=True, hint="Which production line or facility"),
        ArgSpec("allocation_change", "enum", required=True, options=["increase_25pct", "increase_50pct", "decrease_25pct", "shutdown_line", "add_shift", "remove_shift", "maintenance_halt"], hint="Change type"),
        ArgSpec("reason", "str", required=True, hint="Business reason for this reallocation"),
        ArgSpec("quality_impact", "enum", required=True, options=["improves", "neutral", "minor_risk", "significant_risk"], hint="Expected quality impact"),
        ArgSpec("worker_impact", "int", required=True, hint="Number of workers affected (positive=hire, negative=reduce)"),
        ArgSpec("timeline_days", "int", required=True, min_val=1, max_val=180),
    ],
    category="operations"
))

_reg(ToolSchema("manage_patient_care", "Make a patient care or resource allocation decision",
    agent_roles=["CMO_Medical", "Head_of_Clinical_Ops", "Head_of_Nursing"],
    domain="healthcare",
    args=[
        ArgSpec("decision_type", "enum", required=True, options=["triage_protocol", "resource_reallocation", "staff_deployment", "equipment_procurement", "care_pathway_change", "capacity_expansion", "discharge_protocol"], hint="Type of care decision"),
        ArgSpec("affected_units", "str", required=True, hint="Which wards, units, or patient populations"),
        ArgSpec("patient_safety_priority", "enum", required=True, options=["critical_safety_first", "balanced", "efficiency_prioritized"], hint="How you're balancing safety vs efficiency"),
        ArgSpec("staffing_change", "int", required=True, hint="Staff delta needed (positive=add, negative=reduce)"),
        ArgSpec("expected_outcome_patients", "str", required=True, hint="Expected patient outcome improvement — specific and measurable"),
        ArgSpec("risk_mitigation", "str", required=True, hint="How are you mitigating risks of this change?"),
    ],
    category="clinical"
))
