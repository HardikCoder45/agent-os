from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class ArgSpec(BaseModel):
    name: str
    type: str
    required: bool = True
    options: Optional[List[str]] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    hint: str = ""

    model_config = ConfigDict(extra="forbid")


class ToolSchema(BaseModel):
    name: str
    description: str
    agent_roles: List[str]
    args: List[ArgSpec] = Field(default_factory=list)
    domain: str = "all"
    category: str = "general"
    intent_tags: List[str] = Field(default_factory=list)
    risk_profile: str = "medium"
    args_model: Optional[Type[BaseModel]] = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


def _annotation_for_arg(spec: ArgSpec):
    mapping = {
        "str": str,
        "int": int,
        "float": float,
        "enum": str,
    }
    annotation = mapping.get(spec.type, Any)
    if not spec.required:
        return Optional[annotation]
    return annotation


def _build_args_model(schema: ToolSchema) -> Type[BaseModel]:
    field_defs: Dict[str, Tuple[Any, Any]] = {}
    for spec in schema.args:
        annotation = _annotation_for_arg(spec)
        default = Field(
            default=... if spec.required else None,
            description=spec.hint or spec.name,
        )
        field_defs[spec.name] = (annotation, default)
    return create_model(
        f"{schema.name.title().replace('_', '')}Args",
        __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
        **field_defs,
    )


def _string_quality(value: str) -> float:
    words = [word for word in value.split() if word]
    if not words:
        return 0.0
    richness = min(1.0, len(words) / 14)
    has_digits = any(char.isdigit() for char in value)
    has_tradeoff = any(
        token in value.lower()
        for token in ("risk", "tradeoff", "metric", "timeline", "because", "therefore", "goal")
    )
    score = 0.45 + 0.35 * richness
    if has_digits:
        score += 0.12
    if has_tradeoff:
        score += 0.08
    return min(score, 1.0)


def _numeric_quality(spec: ArgSpec, value: float) -> float:
    if spec.min_val is not None and value < spec.min_val:
        return 0.0
    if spec.max_val is not None and value > spec.max_val:
        return 0.0
    if spec.min_val is not None and spec.max_val is not None and spec.max_val > spec.min_val:
        midpoint = (spec.min_val + spec.max_val) / 2
        spread = max((spec.max_val - spec.min_val) / 2, 1.0)
        distance = min(abs(value - midpoint) / spread, 1.0)
        return round(0.82 + (1.0 - distance) * 0.18, 4)
    return 0.92


def _format_validation_errors(exc: ValidationError) -> List[str]:
    formatted: List[str] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", ()))
        message = item.get("msg", "Invalid value")
        formatted.append(f"{location}: {message}")
    return formatted


def validate_args(
    schema: ToolSchema,
    provided: Dict[str, Any],
) -> tuple[bool, list[str], float, dict[str, Any], list[str]]:
    if schema.args_model is None:
        schema.args_model = _build_args_model(schema)

    allowed_fields = {spec.name for spec in schema.args}
    hallucinated_args = sorted(set(provided.keys()) - allowed_fields)
    errors = [f"Unexpected argument: `{name}`" for name in hallucinated_args]

    normalized: Dict[str, Any] = {}
    parsed = None
    try:
        parsed = schema.args_model.model_validate(provided)
        normalized = parsed.model_dump(exclude_none=True)
    except ValidationError as exc:
        errors.extend(_format_validation_errors(exc))

    quality_scores: List[float] = []
    values = normalized if normalized else provided
    for spec in schema.args:
        if spec.name not in values:
            if spec.required:
                errors.append(f"Missing required argument: `{spec.name}` ({spec.hint})")
                quality_scores.append(0.0)
            continue

        value = values.get(spec.name)
        if value is None:
            quality_scores.append(0.0 if spec.required else 0.35)
            continue

        if spec.options and str(value) not in spec.options:
            errors.append(f"`{spec.name}` must be one of {spec.options}, got '{value}'")
            quality_scores.append(0.0)
            continue

        if spec.type == "str":
            quality_scores.append(_string_quality(str(value)))
        elif spec.type in {"int", "float"} and isinstance(value, (int, float)):
            numeric_value = float(value)
            if spec.min_val is not None and numeric_value < spec.min_val:
                errors.append(f"`{spec.name}` too low (min {spec.min_val})")
                quality_scores.append(0.0)
            elif spec.max_val is not None and numeric_value > spec.max_val:
                errors.append(f"`{spec.name}` too high (max {spec.max_val})")
                quality_scores.append(0.0)
            else:
                quality_scores.append(_numeric_quality(spec, numeric_value))
        elif spec.type == "enum":
            quality_scores.append(1.0)
        else:
            quality_scores.append(0.5)

    if not quality_scores:
        quality_scores.append(0.0)

    penalty = min(0.45, 0.18 * len(hallucinated_args))
    overall = max(0.0, min(1.0, (sum(quality_scores) / len(quality_scores)) - penalty))
    is_valid = parsed is not None and not errors
    return is_valid, sorted(set(errors)), round(overall, 4), normalized, hallucinated_args


TOOL_REGISTRY: dict[str, ToolSchema] = {}


def _reg(t: ToolSchema) -> ToolSchema:
    t.args_model = _build_args_model(t)
    TOOL_REGISTRY[t.name] = t
    return t


def get_tools_for_role(role: str, domain: str = "all") -> dict[str, ToolSchema]:
    return {
        name: schema
        for name, schema in TOOL_REGISTRY.items()
        if (schema.domain in {"all", domain})
        and ("all" in schema.agent_roles or role in schema.agent_roles)
    }


_reg(ToolSchema(
    name="approve_budget",
    description="Approve or deny a budget request with justification",
    agent_roles=["CEO", "CFO"],
    args=[
        ArgSpec(name="amount", type="int", required=True, min_val=0, hint="Dollar amount to approve"),
        ArgSpec(name="department", type="enum", required=True, options=["engineering", "marketing", "sales", "ops", "hr", "r_and_d", "legal"], hint="Which department"),
        ArgSpec(name="purpose", type="str", required=True, hint="Specific purpose - be detailed (what, why, expected outcome)"),
        ArgSpec(name="timeline_weeks", type="int", required=False, min_val=1, max_val=52, hint="Expected spend timeline"),
        ArgSpec(name="success_metric", type="str", required=False, hint="How will you measure ROI on this spend?"),
    ],
    category="finance",
    intent_tags=["budget_control", "roi", "allocation"],
    risk_profile="medium",
))

_reg(ToolSchema(
    name="set_strategic_direction",
    description="Set or update company strategic direction",
    agent_roles=["CEO"],
    args=[
        ArgSpec(name="focus_area", type="enum", required=True, options=["product", "growth", "profitability", "fundraising", "hiring", "partnerships", "survival"], hint="Primary focus for next quarter"),
        ArgSpec(name="rationale", type="str", required=True, hint="Why this direction now? Reference specific company metrics"),
        ArgSpec(name="tradeoffs", type="str", required=True, hint="What are you explicitly NOT prioritizing? Why?"),
        ArgSpec(name="success_criteria", type="str", required=True, hint="What does success look like in 90 days? Be specific"),
        ArgSpec(name="timeline_weeks", type="int", required=True, min_val=1, max_val=52, hint="How many weeks will you spend executing this direction?"),
    ],
    category="strategy",
    intent_tags=["strategy", "prioritization", "tradeoffs"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="call_board_meeting",
    description="Call an emergency or scheduled board meeting",
    agent_roles=["CEO"],
    args=[
        ArgSpec(name="urgency", type="enum", required=True, options=["emergency_24h", "urgent_48h", "scheduled_1week"], hint="How soon needed?"),
        ArgSpec(name="agenda", type="str", required=True, hint="Specific agenda items — be exhaustive"),
        ArgSpec(name="desired_outcome", type="str", required=True, hint="What decision or approval do you need from the board?"),
        ArgSpec(name="pre_read_needed", type="str", required=False, hint="What materials to distribute before the meeting?"),
    ],
    category="governance",
    intent_tags=["governance", "stakeholder_alignment"],
    risk_profile="medium",
))

_reg(ToolSchema(
    name="make_hire_decision",
    description="Hire or promote for a key role",
    agent_roles=["CEO", "CTO", "Head_of_People", "COO"],
    args=[
        ArgSpec(name="role_title", type="str", required=True, hint="Exact role title"),
        ArgSpec(name="seniority", type="enum", required=True, options=["junior", "mid", "senior", "staff", "principal", "vp", "c_level"], hint="Seniority level"),
        ArgSpec(name="team", type="enum", required=True, options=["engineering", "product", "design", "marketing", "sales", "ops", "hr", "finance", "legal", "research"], hint="Which team"),
        ArgSpec(name="urgency", type="enum", required=True, options=["backfill_critical", "new_headcount", "planned_growth", "strategic_hire"], hint="Why now?"),
        ArgSpec(name="budget_annual", type="int", required=True, min_val=30000, max_val=800000, hint="Total annual comp budget"),
        ArgSpec(name="justification", type="str", required=True, hint="Business case: what problem does this hire solve? How measured?"),
    ],
    category="people",
    intent_tags=["hiring", "team_design", "capacity"],
    risk_profile="medium",
))

_reg(ToolSchema(
    name="respond_to_crisis",
    description="Issue official company response to a PR, legal, or operational crisis",
    agent_roles=["CEO", "CMO", "Head_of_Comms", "COO"],
    args=[
        ArgSpec(name="crisis_type", type="enum", required=True, options=["pr_social_media", "legal_threat", "product_outage", "data_breach", "employee_issue", "customer_complaint", "regulatory", "media_inquiry"], hint="Nature of the crisis"),
        ArgSpec(name="response_channel", type="enum", required=True, options=["public_statement", "private_dm", "press_release", "internal_memo", "social_post", "no_comment", "legal_letter"], hint="How to respond"),
        ArgSpec(name="stance", type="enum", required=True, options=["full_apology", "partial_acknowledgment", "factual_correction", "no_admission", "proactive_transparency", "deflect"], hint="What position to take"),
        ArgSpec(name="message", type="str", required=True, hint="Core message — what exactly are you communicating? Draft it"),
        ArgSpec(name="follow_up_actions", type="str", required=True, hint="What concrete actions follow this response?"),
    ],
    category="comms",
    intent_tags=["crisis_management", "reputation", "stakeholder_trust"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="ship_feature",
    description="Approve and ship a product feature",
    agent_roles=["CTO", "Head_of_Product", "VP_Engineering"],
    args=[
        ArgSpec(name="feature_name", type="str", required=True, hint="Name of the feature"),
        ArgSpec(name="team_size", type="int", required=True, min_val=1, max_val=20, hint="Engineers assigned"),
        ArgSpec(name="sprint_weeks", type="int", required=True, min_val=1, max_val=12, hint="Sprint duration in weeks"),
        ArgSpec(name="quality_bar", type="enum", required=True, options=["mvp_rough", "beta_tested", "production_grade", "enterprise_grade"], hint="Required quality level before ship"),
        ArgSpec(name="rollout_strategy", type="enum", required=True, options=["full_release", "gradual_10pct", "beta_users_only", "internal_only", "flag_gated"], hint="How to roll out"),
        ArgSpec(name="success_metric", type="str", required=True, hint="How do you define success for this feature? Specific KPI"),
        ArgSpec(name="rollback_plan", type="str", required=False, hint="What's the rollback if it breaks?"),
    ],
    category="product",
    intent_tags=["shipping", "rollout", "execution"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="manage_technical_debt",
    description="Decision on technical debt: pay down or defer",
    agent_roles=["CTO", "VP_Engineering"],
    args=[
        ArgSpec(name="area", type="enum", required=True, options=["infrastructure", "database", "api_layer", "frontend", "security", "ci_cd", "monitoring", "dependencies"], hint="Debt area"),
        ArgSpec(name="severity", type="enum", required=True, options=["critical_blocking", "high_risk", "medium_nuisance", "low_cosmetic"], hint="Current impact"),
        ArgSpec(name="decision", type="enum", required=True, options=["pay_down_now", "schedule_sprint", "accept_and_document", "architectural_rewrite"], hint="What to do"),
        ArgSpec(name="eng_weeks_cost", type="int", required=True, min_val=1, max_val=52, hint="Engineering weeks this will cost"),
        ArgSpec(name="risk_if_deferred", type="str", required=True, hint="What breaks if you don't fix this? Be specific"),
    ],
    category="engineering",
    intent_tags=["stability", "reliability", "debt_management"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="launch_campaign",
    description="Launch a marketing campaign",
    agent_roles=["CMO", "Head_of_Marketing", "VP_Growth"],
    args=[
        ArgSpec(name="channel", type="enum", required=True, options=["paid_social", "seo_content", "influencer", "email", "pr_media", "events", "partnerships", "product_led", "community", "out_of_home"], hint="Primary channel"),
        ArgSpec(name="budget", type="int", required=True, min_val=500, max_val=5000000, hint="Campaign budget in dollars"),
        ArgSpec(name="target_segment", type="str", required=True, hint="Exact target audience: demographics, psychographics, job titles, behaviors"),
        ArgSpec(name="core_message", type="str", required=True, hint="One-sentence core message the campaign communicates"),
        ArgSpec(name="goal_metric", type="enum", required=True, options=["brand_awareness", "signups", "revenue", "retention", "nps", "press_coverage", "community_growth"], hint="Primary goal"),
        ArgSpec(name="target_value", type="int", required=True, min_val=1, hint="Numeric target (signups, $ revenue, coverage pieces, etc.)"),
        ArgSpec(name="timeline_weeks", type="int", required=True, min_val=1, max_val=26, hint="Execution horizon in weeks"),
    ],
    category="marketing",
    intent_tags=["go_to_market", "awareness", "pipeline"],
    risk_profile="medium",
))

_reg(ToolSchema(
    name="manage_investor_relations",
    description="Manage LP/investor communication and relationships",
    agent_roles=["CEO", "CFO", "Head_of_IR"],
    args=[
        ArgSpec(name="communication_type", type="enum", required=True, options=["board_update", "monthly_newsletter", "crisis_notification", "fundraise_outreach", "lp_call", "term_sheet_response", "data_room_share"], hint="Type of investor communication"),
        ArgSpec(name="transparency_level", type="enum", required=True, options=["full_disclosure", "headline_metrics_only", "positive_spin", "selective_sharing", "minimal"], hint="How much to share"),
        ArgSpec(name="key_message", type="str", required=True, hint="Core narrative — what do you want investors to take away?"),
        ArgSpec(name="risks_disclosed", type="str", required=True, hint="What risks are you surfacing? (hiding risks scores negatively)"),
        ArgSpec(name="ask_or_action", type="str", required=False, hint="What are you asking from investors, if anything?"),
    ],
    category="finance",
    intent_tags=["investor_communication", "transparency", "trust"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="conduct_clinical_trial",
    description="Initiate or modify a clinical trial phase",
    agent_roles=["CSO", "Head_of_Clinical", "CMO_Medical"],
    domain="pharma",
    args=[
        ArgSpec(name="phase", type="enum", required=True, options=["preclinical", "phase_1", "phase_2", "phase_2b", "phase_3", "phase_4"], hint="Trial phase"),
        ArgSpec(name="patient_cohort_size", type="int", required=True, min_val=10, max_val=10000, hint="Number of patients"),
        ArgSpec(name="primary_endpoint", type="str", required=True, hint="Primary efficacy or safety endpoint being measured"),
        ArgSpec(name="safety_monitoring", type="enum", required=True, options=["standard_dsmb", "enhanced_monitoring", "continuous_realtime", "independent_review"], hint="Safety oversight level"),
        ArgSpec(name="duration_months", type="int", required=True, min_val=1, max_val=60, hint="Trial duration"),
        ArgSpec(name="stopping_criteria", type="str", required=True, hint="Under what conditions would you stop the trial early?"),
    ],
    category="clinical",
    intent_tags=["patient_safety", "trial_design", "evidence"],
    risk_profile="critical",
))

_reg(ToolSchema(
    name="file_regulatory_submission",
    description="File a regulatory submission (IND, NDA, BLA, 510k, etc.)",
    agent_roles=["Head_of_Regulatory", "CSO"],
    domain="pharma",
    args=[
        ArgSpec(name="submission_type", type="enum", required=True, options=["IND", "NDA", "BLA", "sNDA", "510k", "EUA", "MAA", "ANDA"], hint="Type of regulatory filing"),
        ArgSpec(name="target_agency", type="enum", required=True, options=["FDA", "EMA", "PMDA", "Health_Canada", "TGA"], hint="Regulatory body"),
        ArgSpec(name="data_package_completeness", type="enum", required=True, options=["complete", "rolling_submission", "expedited_pathway", "breakthrough_designation"], hint="Submission strategy"),
        ArgSpec(name="priority_designation", type="enum", required=False, options=["standard", "priority_review", "breakthrough_therapy", "fast_track", "accelerated_approval"], hint="Any special designation?"),
        ArgSpec(name="response_to_deficiencies", type="str", required=False, hint="How are you addressing known data gaps?"),
    ],
    category="regulatory",
    intent_tags=["regulatory_strategy", "evidence_package", "compliance"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="summon_subagent",
    description="Summon a sub-agent for expert consultation on a decision",
    agent_roles=["CEO", "COO", "CTO", "CMO", "CFO", "CSO", "Head_of_Operations"],
    args=[
        ArgSpec(name="agent_role", type="enum", required=True, options=["CTO", "CMO", "CFO", "Head_of_People", "Head_of_Legal", "Head_of_Product", "Head_of_Sales", "Head_of_Operations", "Head_of_Regulatory", "CSO", "Head_of_Clinical", "Head_of_Marketing", "VP_Engineering", "Head_of_Risk"], hint="Which expert to consult"),
        ArgSpec(name="question", type="str", required=True, hint="Specific question or decision you need their expert input on"),
        ArgSpec(name="context", type="str", required=True, hint="What context does the sub-agent need to give useful advice?"),
        ArgSpec(name="urgency", type="enum", required=True, options=["immediate_1h", "today_eod", "this_week", "advisory"], hint="How fast do you need their input?"),
        ArgSpec(name="decision_authority", type="enum", required=True, options=["they_decide", "they_advise_i_decide", "we_decide_together"], hint="Are they deciding or advising?"),
    ],
    category="management",
    intent_tags=["consultation", "cross_functional", "coordination"],
    risk_profile="low",
))

_reg(ToolSchema(
    name="negotiate_deal",
    description="Negotiate a business deal: acquisition, partnership, supplier, customer",
    agent_roles=["CEO", "CFO", "Head_of_BD", "Head_of_Sales"],
    args=[
        ArgSpec(name="deal_type", type="enum", required=True, options=["acquisition", "acqui_hire", "licensing", "partnership", "supplier_contract", "customer_enterprise", "joint_venture", "investment"], hint="Type of deal"),
        ArgSpec(name="our_position", type="str", required=True, hint="What we want from this deal — specific terms"),
        ArgSpec(name="their_likely_position", type="str", required=True, hint="What you think they want — show you've thought about their interests"),
        ArgSpec(name="walkaway_point", type="str", required=True, hint="What terms would make you walk away? Be specific"),
        ArgSpec(name="leverage_points", type="str", required=True, hint="What leverage do you have in this negotiation?"),
        ArgSpec(name="timeline_to_close", type="int", required=True, min_val=1, max_val=180, hint="Days to target close"),
    ],
    category="business_development",
    intent_tags=["negotiation", "leverage", "deal_structuring"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="manage_team_health",
    description="Take action on team health, culture, or performance",
    agent_roles=["CEO", "Head_of_People", "COO", "CTO"],
    args=[
        ArgSpec(name="issue_type", type="enum", required=True, options=["burnout", "retention_risk", "cultural_conflict", "performance_gap", "compensation_equity", "layoff", "reorg", "toxic_behavior", "remote_hybrid_policy"], hint="Type of issue"),
        ArgSpec(name="affected_team", type="str", required=True, hint="Which team or individuals are affected"),
        ArgSpec(name="action", type="enum", required=True, options=["1on1_conversations", "compensation_adjustment", "role_change", "pip", "termination", "team_offsite", "policy_change", "leadership_coaching", "mental_health_support", "reorg"], hint="Action to take"),
        ArgSpec(name="expected_outcome", type="str", required=True, hint="What specific improvement do you expect and by when?"),
        ArgSpec(name="risk_if_no_action", type="str", required=True, hint="What happens if you do nothing?"),
    ],
    category="people",
    intent_tags=["retention", "morale", "leadership"],
    risk_profile="high",
))

_reg(ToolSchema(
    name="allocate_production_capacity",
    description="Allocate or change manufacturing production capacity",
    agent_roles=["COO", "Head_of_Operations", "Head_of_Manufacturing"],
    domain="manufacturing",
    args=[
        ArgSpec(name="line_id", type="str", required=True, hint="Which production line or facility"),
        ArgSpec(name="allocation_change", type="enum", required=True, options=["increase_25pct", "increase_50pct", "decrease_25pct", "shutdown_line", "add_shift", "remove_shift", "maintenance_halt"], hint="Change type"),
        ArgSpec(name="reason", type="str", required=True, hint="Business reason for this reallocation"),
        ArgSpec(name="quality_impact", type="enum", required=True, options=["improves", "neutral", "minor_risk", "significant_risk"], hint="Expected quality impact"),
        ArgSpec(name="worker_impact", type="int", required=True, hint="Number of workers affected (positive=hire, negative=reduce)"),
        ArgSpec(name="timeline_days", type="int", required=True, min_val=1, max_val=180, hint="Days to implement"),
    ],
    category="operations",
    intent_tags=["capacity", "operations", "quality"],
    risk_profile="medium",
))

_reg(ToolSchema(
    name="manage_patient_care",
    description="Make a patient care or resource allocation decision",
    agent_roles=["CMO_Medical", "Head_of_Clinical_Ops", "Head_of_Nursing"],
    domain="healthcare",
    args=[
        ArgSpec(name="decision_type", type="enum", required=True, options=["triage_protocol", "resource_reallocation", "staff_deployment", "equipment_procurement", "care_pathway_change", "capacity_expansion", "discharge_protocol"], hint="Type of care decision"),
        ArgSpec(name="affected_units", type="str", required=True, hint="Which wards, units, or patient populations"),
        ArgSpec(name="patient_safety_priority", type="enum", required=True, options=["critical_safety_first", "balanced", "efficiency_prioritized"], hint="How you're balancing safety vs efficiency"),
        ArgSpec(name="staffing_change", type="int", required=True, hint="Staff delta needed (positive=add, negative=reduce)"),
        ArgSpec(name="expected_outcome_patients", type="str", required=True, hint="Expected patient outcome improvement — specific and measurable"),
        ArgSpec(name="risk_mitigation", type="str", required=True, hint="How are you mitigating risks of this change?"),
    ],
    category="clinical",
    intent_tags=["patient_safety", "resource_allocation", "clinical_ops"],
    risk_profile="critical",
))
