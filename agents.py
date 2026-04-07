from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    step_id: str
    question: str
    context: str
    required_tool: str
    required_args_hints: dict[str, str]
    optimal_args: dict[str, Any]
    optimal_reasoning_keywords: list[str]
    scoring_rubric: dict[str, str]
    counterfactual_tip: str
    cross_agent_effects: dict[str, dict[str, Any]] = field(default_factory=dict)
    subagent_available: bool = False
    subagent_hint: str = ""


@dataclass
class AgentScenario:
    scenario_id: str
    title: str
    briefing: str
    goal: str
    goal_metrics: dict[str, Any]
    max_steps: int
    steps: list[AgentStep]
    initial_state_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDefinition:
    role: str
    title: str
    domain: str
    persona: str
    reports_to: str
    manages: list[str]
    kpis: list[str]
    personality: str
    scenarios: list[AgentScenario]
    color: str = "#6c63ff"
    emoji: str = "👤"


AGENTS: dict[str, dict[str, AgentDefinition]] = {}


def _get(domain: str) -> dict[str, AgentDefinition]:
    if domain not in AGENTS:
        AGENTS[domain] = {}
    return AGENTS[domain]


AGENTS["tech_startup"] = {

    "CEO": AgentDefinition(
        role="CEO", title="Chief Executive Officer", domain="tech_startup",
        persona="You are Alex Chen, 32-year-old CEO of a Series A SaaS startup. Former Stripe PM. Obsessed with product-market fit. You hate vanity metrics.",
        reports_to="Board", manages=["CTO", "CMO", "CFO", "Head_of_Product"],
        kpis=["MRR growth", "Net Dollar Retention", "Runway months", "Team NPS"],
        personality="Direct, data-driven, values trust over politics. Will sacrifice short-term metrics for long-term brand.",
        color="#6c63ff", emoji="👑",
        scenarios=[
            AgentScenario(
                scenario_id="ceo_fundraise_or_default",
                title="Series A: $2M Left, 3 Term Sheets, None Are Clean",
                briefing="Your runway is 9 weeks. Three VCs have submitted term sheets but all have problematic clauses: Sequoia wants full ratchet, Andreessen wants a 2x liquidation preference, Tiger wants a down round at 30% discount. Your angels are willing to bridge $300K at a 20% discount but it dilutes co-founders significantly. Revenue is growing 18% MoM.",
                goal="Close funding without destroying cap table, team morale, or product velocity",
                goal_metrics={"runway_months": 18, "team_morale": 0.7, "investor_confidence": 0.75, "mrr_growth_pct": 15},
                max_steps=5,
                steps=[
                    AgentStep(
                        step_id="ceo_s1_assess",
                        question="Before talking to any VC, what do you do in the next 48 hours internally?",
                        context="You have 3 term sheets on the table. But your CFO, CTO, and co-founder haven't seen them yet. Your cap table model is 6 months stale. One board member is on vacation.",
                        required_tool="set_strategic_direction",
                        required_args_hints={
                            "focus_area": "fundraising",
                            "rationale": "Explain the specific tradeoffs between the 3 term sheets and why now",
                            "tradeoffs": "What are you NOT doing while focused on this?",
                            "success_criteria": "What does a good close look like in numbers?",
                            "timeline_weeks": "How long will you take?"
                        },
                        optimal_args={"focus_area": "fundraising", "rationale": "3 term sheets all with problematic clauses — need to run internal alignment before negotiating. Tiger's down round would reset morale. Sequoia ratchet could trap future fundraises. Need CFO model and co-founder alignment before any counter.", "tradeoffs": "Not shipping product for 1 week. Pausing two hiring searches. Co-founders context-switched off building.", "success_criteria": "Close at $12M+ pre-money, no ratchet, liquidation preference max 1x non-participating, close within 4 weeks", "timeline_weeks": 4},
                        optimal_reasoning_keywords=["cap table", "ratchet", "liquidation", "down round", "co-founder", "morale", "runway", "dilution", "counter", "align"],
                        scoring_rubric={
                            "0.9-1.0": "Names all 3 term sheet issues by type, sets clear numeric success criteria, acknowledges internal alignment before external negotiation",
                            "0.7-0.89": "Mentions 2 term sheet issues, sets vague criteria, plans to talk to VCs first",
                            "0.5-0.69": "Generic fundraising direction without addressing specific term sheet problems",
                            "0.0-0.49": "No awareness of cap table implications or specific term sheet risks"
                        },
                        counterfactual_tip="Best move: run a 48h internal sprint — CFO builds dilution model for all 3 scenarios, co-founders align on walkaway terms, board member briefed async. Only then do you engage VCs. This scores 0.95 because it prevents negotiating blind.",
                        cross_agent_effects={"CFO": {"focus_area": "fundraising_model"}, "CTO": {"distraction_weeks": 1}},
                        subagent_available=True,
                        subagent_hint="Consider summoning CFO to model cap table impact before deciding"
                    ),
                    AgentStep(
                        step_id="ceo_s2_negotiate",
                        question="Sequoia responds: they'll remove the ratchet if you accept a board seat with veto on execs above $200K. Tiger sweetens to 15% discount. Andreessen holds. What do you counter?",
                        context="You have updated cap table models. Co-founders prefer Sequoia but hate the veto clause. Your best engineer says he'll leave if you take Tiger's down round. Andreessen's partner is the weakest operator but cleanest terms.",
                        required_tool="negotiate_deal",
                        required_args_hints={
                            "deal_type": "investment",
                            "our_position": "What specific counter-terms do you propose to which VC?",
                            "their_likely_position": "What do each of the VCs actually care about here?",
                            "walkaway_point": "What terms would make you take the angel bridge instead?",
                            "leverage_points": "What leverage do you have right now?",
                            "timeline_to_close": "Days to close"
                        },
                        optimal_args={"deal_type": "investment", "our_position": "Counter Sequoia: board observer not board member, no veto on execs, remove ratchet. Counter Andreessen: ask for $14M pre-money, 1x non-participating pref only. Reject Tiger — down round kills team.", "their_likely_position": "Sequoia wants governance control — the veto is their real ask. Andreessen wants a clean deal they can tout. Tiger wants ownership at discount.", "walkaway_point": "Any full ratchet, any 2x+ liquidation preference, or any down round valuation. Will take angel bridge if no VC meets standard.", "leverage_points": "18% MoM growth, competitive term sheets create scarcity, team quality, angel bridge as real alternative", "timeline_to_close": 21},
                        optimal_reasoning_keywords=["leverage", "scarcity", "anchor", "counter", "walkaway", "governance", "veto", "bridge", "down round", "team morale"],
                        scoring_rubric={
                            "0.9-1.0": "Specific counter-terms per VC, identifies what each VC actually wants (not just money), has real walkaway with bridge backup",
                            "0.7-0.89": "Decent counters but doesn't differentiate VC motivations, vague on walkaway",
                            "0.5-0.69": "Negotiates one VC without strategy for others",
                            "0.0-0.49": "Accepts first offer or negotiates purely on valuation ignoring governance"
                        },
                        counterfactual_tip="The leverage play: tell all three VCs simultaneously you have competitive terms and need a decision by Friday. Andreessen with cleanest terms + counter to $13M pre-money is the win. Never negotiate without your walkaway pre-committed.",
                        cross_agent_effects={"CFO": {"term_sheet_finalized": True}, "CTO": {"engineering_budget": 200000}},
                        subagent_available=True,
                        subagent_hint="Summon CFO to validate which term sheet is best for long-term cap table"
                    ),
                    AgentStep(
                        step_id="ceo_s3_team",
                        question="Deal closing in 5 days. Three senior engineers heard about the down round discussions via Slack leak. Two are interviewing elsewhere. What do you do RIGHT NOW?",
                        context="You didn't take Tiger's down round but the leak says you considered it. Your best eng lead (18mo tenure, knows entire infra) has a Google screen tomorrow. Morale score dropped from 7.2 to 5.8 in two days.",
                        required_tool="manage_team_health",
                        required_args_hints={
                            "issue_type": "retention_risk",
                            "affected_team": "Which specific engineers and why they matter",
                            "action": "What specifically do you do?",
                            "expected_outcome": "What does success look like in 72 hours?",
                            "risk_if_no_action": "What happens if you do nothing?"
                        },
                        optimal_args={"issue_type": "retention_risk", "affected_team": "3 senior engineers — infra lead critical, 2 others influential. Losing infra lead means 4-month knowledge gap, delays launch 6+ weeks.", "action": "1on1_conversations", "expected_outcome": "Within 48h: all 3 have had direct conversation with me. Clear communication that Tiger deal was rejected. Equity refresh offered to infra lead. Full team update with actual deal terms on day of close.", "risk_if_no_action": "Infra lead leaves → 6-week delay minimum. Other two follow within 30 days. Hiring replacements at Series A costs $60K each and 3 months ramp."},
                        optimal_reasoning_keywords=["1:1", "transparency", "equity", "infra", "knowledge", "morale", "communicate", "close", "retain", "trust"],
                        scoring_rubric={
                            "0.9-1.0": "Direct 1:1s named, specific retention action for the most critical person, proactive team communication tied to deal close date",
                            "0.7-0.89": "Plans 1:1s but no specific action for most critical person",
                            "0.5-0.69": "Plans a team meeting without individual focus",
                            "0.0-0.49": "Waits for deal to close before addressing the team"
                        },
                        counterfactual_tip="Best outcome: 1:1 with infra lead within 24h. Be honest about what you considered and why you rejected it. Offer equity refresh. This scores 1.0 because transparency + targeted retention beats any general team message.",
                        cross_agent_effects={"Head_of_People": {"retention_interventions": 3}},
                        subagent_available=True,
                        subagent_hint="Summon Head_of_People for compensation benchmarking before your 1:1s"
                    ),
                ]
            )
        ]
    ),

    "CTO": AgentDefinition(
        role="CTO", title="Chief Technology Officer", domain="tech_startup",
        persona="You are Jordan Park, 35, CTO. Ex-Google engineer. Deeply technical, worries about scale, hates shortcuts that compound debt. Protective of eng team.",
        reports_to="CEO", manages=["VP_Engineering", "Head_of_Product", "Data_Engineering"],
        kpis=["Deployment frequency", "MTTR", "Technical debt ratio", "Eng NPS", "Feature velocity"],
        personality="Systems thinker. Speaks in second-order effects. Will push back on deadlines that compromise quality.",
        color="#e53935", emoji="⚙️",
        scenarios=[
            AgentScenario(
                scenario_id="cto_scale_crisis",
                title="The System is Falling Over — $2M Enterprise Deal on the Line",
                briefing="Your largest-ever enterprise prospect ($2M ARR) wants a live product demo in 9 days. Your infra was built for 500 users. You now have 12,000. P95 latency is 4.8s. The database is running hot. Three microservices are on a single VM. Your best infra engineer left 3 weeks ago.",
                goal="Stabilize infra enough for demo, without destroying team or accumulating catastrophic debt",
                goal_metrics={"p95_latency_ms": 800, "uptime_pct": 99.9, "technical_debt": 0.35, "team_morale": 0.65},
                max_steps=4,
                steps=[
                    AgentStep(
                        step_id="cto_s1_triage",
                        question="You have 9 days and no infra lead. What's your immediate technical triage decision?",
                        context="P95 latency 4.8s. DB running at 89% CPU. 3 microservices on 1 VM. Infra lead left 3 weeks ago. Team of 6 remaining engineers, 2 are mid-level. Enterprise demo is in 9 days. CEO is promising the demo will work.",
                        required_tool="manage_technical_debt",
                        required_args_hints={
                            "area": "Which area is the real bottleneck?",
                            "severity": "How bad is this really?",
                            "decision": "What specifically are you doing about it?",
                            "eng_weeks_cost": "How many eng-weeks will this take?",
                            "risk_if_deferred": "What breaks if you don't fix this before the demo?"
                        },
                        optimal_args={"area": "database", "severity": "critical_blocking", "decision": "pay_down_now", "eng_weeks_cost": 3, "risk_if_deferred": "DB at 89% CPU will hit 100% under demo load — system crash in front of $2M prospect. 4.8s latency will kill the demo before contract. Must add read replica, tune slow queries, separate microservices to their own instances by day 5."},
                        optimal_reasoning_keywords=["latency", "database", "read replica", "cpu", "demo", "load", "microservice", "bottleneck", "query", "crash"],
                        scoring_rubric={
                            "0.9-1.0": "Identifies DB as the critical path, specific technical remediation steps, realistic timeline with buffer before demo",
                            "0.7-0.89": "Correct area identified but remediation steps vague",
                            "0.5-0.69": "Addresses wrong layer (frontend, etc) while DB is critical",
                            "0.0-0.49": "No specific technical diagnosis"
                        },
                        counterfactual_tip="Database is the critical path: 89% CPU will spike to 120% under demo load. Optimal: add read replica (1 day), run EXPLAIN ANALYZE on top-20 queries (1 day), separate the 3 microservices to their own t3.large instances (2 days). Gets you to P95 ~800ms by day 5 with 4 days buffer. Score 0.97.",
                        cross_agent_effects={"CFO": {"infra_spend_emergency": 15000}},
                        subagent_available=True,
                        subagent_hint="Summon Head_of_Product to negotiate scope of demo to reduce load"
                    ),
                ]
            )
        ]
    ),

    "CMO": AgentDefinition(
        role="CMO", title="Chief Marketing Officer", domain="tech_startup",
        persona="You are Priya Nair, 30, CMO. Ex-HubSpot growth lead. Performance marketer who also understands brand. Obsessed with attribution and CAC.",
        reports_to="CEO", manages=["Head_of_Content", "Head_of_Demand_Gen", "Social_Media_Manager"],
        kpis=["CAC", "LTV:CAC ratio", "Pipeline coverage", "Brand NPS", "Organic traffic growth"],
        personality="Data-obsessed but knows when brand gut-calls matter. Hates vanity metrics. Speaks in CAC and LTV.",
        color="#e91e63", emoji="📣",
        scenarios=[
            AgentScenario(
                scenario_id="cmo_category_creation",
                title="No One Knows Your Category Exists — Investor Demo in 6 Weeks",
                briefing="Your product is genuinely differentiated but the market doesn't have a name for what you do. Analyst coverage is zero. Your top-of-funnel is mostly word-of-mouth from the first 50 customers. You have $80K marketing budget for the quarter. Investor demo in 6 weeks — they expect 30% pipeline growth.",
                goal="Create category awareness and grow pipeline 30% in 6 weeks on $80K",
                goal_metrics={"pipeline_coverage": 2.5, "brand_awareness_score": 0.4, "cac": 280, "organic_signups_weekly": 45},
                max_steps=4,
                steps=[
                    AgentStep(
                        step_id="cmo_s1_category",
                        question="$80K, 6 weeks, zero analyst coverage. What's your single highest-leverage bet to create category awareness?",
                        context="50 happy customers who'd evangelize. $80K budget. 6 weeks. Zero analyst coverage. Competitor has $2M marketing budget. Your product does something no one else does but there's no name for it. One senior content writer on the team.",
                        required_tool="launch_campaign",
                        required_args_hints={
                            "channel": "Which single channel gives highest ROI for category creation?",
                            "budget": "How much of the $80K goes here?",
                            "target_segment": "Who exactly are you trying to reach with this campaign?",
                            "core_message": "What is the ONE sentence that names and defines your category?",
                            "goal_metric": "signups or brand_awareness",
                            "target_value": "Numeric target",
                            "timeline_weeks": "6 max"
                        },
                        optimal_args={"channel": "seo_content", "budget": 35000, "target_segment": "Mid-market ops leaders and technical PMs at 200-2000 person SaaS companies frustrated with current workflow tools — searching 'how to [specific pain] without [current solution]'", "core_message": "The first platform that lets operations teams build automations without writing code or waiting for engineering", "goal_metric": "signups", "target_value": 200, "timeline_weeks": 6},
                        optimal_reasoning_keywords=["category", "SEO", "search intent", "ICP", "pain point", "customer story", "organic", "content", "evangelist", "pain"],
                        scoring_rubric={
                            "0.9-1.0": "Names a specific underserved search intent, leverages existing customers as content, allocates <50% budget to one channel keeping remainder for amplification",
                            "0.7-0.89": "Good channel choice but generic message, no specific target segment",
                            "0.5-0.69": "Chooses paid channel for category creation (wrong for this budget/timeline)",
                            "0.0-0.49": "No channel strategy or unrealistic budget allocation"
                        },
                        counterfactual_tip="Category creation on $80K: SEO + customer stories is the only play. Paid = too expensive for awareness. Content that names the category wins Google for 3 years. Optimal: $35K SEO/content, $25K for 5 customer story video interviews, $20K for one analyst relationship. Gets you 180-220 organic signups and category search ownership. Score 0.96.",
                        cross_agent_effects={"Head_of_Product": {"customer_interview_requests": 5}},
                        subagent_available=True,
                        subagent_hint="Summon Head_of_Sales to understand which ICP titles are easiest to close"
                    ),
                ]
            )
        ]
    ),

    "CFO": AgentDefinition(
        role="CFO", title="Chief Financial Officer", domain="tech_startup",
        persona="You are David Kim, 38, CFO. Ex-Goldman, then two startup CFO stints. Precise, conservative, always 3 scenarios. Worries about the down round at all times.",
        reports_to="CEO", manages=["Finance_Manager", "Legal_Counsel"],
        kpis=["Runway months", "Burn multiple", "ARR growth efficiency", "Gross margin", "Cap table health"],
        personality="Never rounds numbers. Always builds bear/base/bull. Respects CEOs who know their numbers.",
        color="#1976d2", emoji="💰",
        scenarios=[
            AgentScenario(
                scenario_id="cfr_burn_crisis",
                title="Burn Rate is $220K/mo — Revenue Target Miss Imminent",
                briefing="The company is burning $220K/month. ARR is $1.8M. Monthly net new ARR has slowed from $80K to $35K last two months. Two big customers are up for renewal — $380K combined ARR. Payroll is 67% of burn. You have 7 months runway. Board meeting is in 10 days.",
                goal="Extend runway to 14+ months without destroying team or product velocity",
                goal_metrics={"runway_months": 14, "gross_margin": 0.72, "burn_multiple": 2.8, "team_morale": 0.65},
                max_steps=4,
                steps=[
                    AgentStep(
                        step_id="cfr_s1_model",
                        question="Before the board meeting, what do you build and present to the CEO?",
                        context="7 months runway. $220K/mo burn. 67% payroll. Two renewals at $380K ARR at risk. Board in 10 days. CEO doesn't know whether to cut or raise bridge.",
                        required_tool="manage_investor_relations",
                        required_args_hints={
                            "communication_type": "board_update",
                            "transparency_level": "How much of the bad news do you surface?",
                            "key_message": "What's the headline narrative for the board?",
                            "risks_disclosed": "What risks are you surfacing specifically?",
                            "ask_or_action": "What are you asking the board to approve or decide?"
                        },
                        optimal_args={"communication_type": "board_update", "transparency_level": "full_disclosure", "key_message": "Growth efficiency has deteriorated: burn multiple moved from 2.1 to 6.3 in two months. We have 7 months at current burn. Presenting 3 scenarios: cut to 14-month runway (3 roles), bridge $500K from angels (11-month), or accelerate enterprise pipeline to close $200K+ by day 60.", "risks_disclosed": "Two renewals at $380K ARR risk — $320K combined ARR at risk if both churn. Net new ARR slowdown may indicate product-market fit drift not just seasonality. Payroll concentration risk at 67% of burn.", "ask_or_action": "Board approval for Scenario B: targeted $50K/mo cost reduction (non-engineering) + CEO to directly own two renewal calls with board visibility"},
                        optimal_reasoning_keywords=["burn multiple", "runway", "scenario", "payroll", "renewal", "churn", "board", "pipeline", "efficiency", "transparency"],
                        scoring_rubric={
                            "0.9-1.0": "Builds 3 scenarios, names burn multiple specifically, discloses renewal risk, asks for specific board decision not just awareness",
                            "0.7-0.89": "Shows scenarios but hides severity, or shows severity without scenarios",
                            "0.5-0.69": "Single scenario, vague ask",
                            "0.0-0.49": "Positive spin that hides 7-month runway reality"
                        },
                        counterfactual_tip="Boards respect CFOs who surface bad news early with options, not late with crises. Full disclosure with 3 scenarios lets the board help. Hiding it delays bridge conversations until you have 3 months left — fatal. Score 0.98 for full disclosure + specific ask.",
                        cross_agent_effects={"CEO": {"board_meeting_prepared": True}},
                        subagent_available=False
                    ),
                ]
            )
        ]
    ),

    "Head_of_Product": AgentDefinition(
        role="Head_of_Product", title="Head of Product", domain="tech_startup",
        persona="You are Mika Tanaka, 29, Head of Product. Ex-Figma. User-obsessed, strong opinions loosely held. Will kill features ruthlessly if metrics don't validate.",
        reports_to="CTO", manages=["Product_Managers", "UX_Designers"],
        kpis=["Feature adoption rate", "User activation rate", "NPS", "Feature request hit rate", "Retention D30/D90"],
        personality="Ships fast, validates faster. Doesn't believe in big-bang features. Obsessive about activation and retention, not just acquisition.",
        color="#11998e", emoji="🎯",
        scenarios=[
            AgentScenario(
                scenario_id="pm_roadmap_vs_enterprise",
                title="Enterprise Demands Custom — Product Roadmap or $240K?",
                briefing="A Fortune 500 prospect offers $240K ARR if you build 6 custom API integrations and a custom reporting module. These aren't on the roadmap. Your activation rate is 38% — half the industry benchmark. If you take this deal, your team spends Q3 on custom work. Your 200 SMB users would benefit from the activation improvements instead.",
                goal="Maximize long-term product health while managing short-term revenue pressure",
                goal_metrics={"activation_rate": 0.58, "d30_retention": 0.72, "feature_adoption_rate": 0.45, "custom_work_weeks": 0},
                max_steps=3,
                steps=[
                    AgentStep(
                        step_id="pm_s1_roadmap",
                        question="The CEO wants a recommendation in 24 hours: take the deal or protect the roadmap?",
                        context="Activation rate 38% (benchmark 65%). Enterprise wants 6 custom integrations + custom reporting — 10 weeks of eng work. Your current roadmap has activation improvements that could take activation from 38% to 60% in 8 weeks. $240K = 4 months of runway extension.",
                        required_tool="ship_feature",
                        required_args_hints={
                            "feature_name": "What feature or work are you recommending?",
                            "team_size": "How many engineers?",
                            "sprint_weeks": "Timeline",
                            "quality_bar": "What quality do you ship to?",
                            "rollout_strategy": "How do you roll it out?",
                            "success_metric": "How do you know it worked?"
                        },
                        optimal_args={"feature_name": "Activation flow redesign + 2 of 6 enterprise integrations that overlap with product roadmap (Salesforce + Slack)", "team_size": 4, "sprint_weeks": 8, "quality_bar": "production_grade", "rollout_strategy": "flag_gated", "success_metric": "Activation rate improves from 38% to 55%+ for SMB users; 2 integrations satisfy 60% of enterprise requirement enabling partial deal", "rollback_plan": "Feature flags allow instant rollback; integrations are additive not destructive"},
                        optimal_reasoning_keywords=["activation", "retention", "overlap", "prioritize", "enterprise", "SMB", "roadmap", "custom", "flag", "metric"],
                        scoring_rubric={
                            "0.9-1.0": "Finds overlap between enterprise asks and roadmap, doesn't abandon product health, uses data to justify (activation rate benchmark gap)",
                            "0.7-0.89": "Takes partial path but doesn't identify roadmap overlaps",
                            "0.5-0.69": "Fully takes enterprise deal ignoring 38% activation rate",
                            "0.0-0.49": "Refuses deal entirely without exploring overlap"
                        },
                        counterfactual_tip="Best answer: identify which 2 of 6 enterprise integrations overlap with roadmap. Salesforce and Slack always do. Build those + activation improvements simultaneously. Counter enterprise at $160K for partial delivery. This gets you revenue AND product health. Score 0.95.",
                        cross_agent_effects={"CTO": {"sprint_allocation": "activation+integrations"}, "CEO": {"enterprise_deal_modified": True}},
                        subagent_available=True,
                        subagent_hint="Summon CTO to validate which integrations can be built with product roadmap overlap"
                    ),
                ]
            )
        ]
    ),
}


AGENTS["pharma"] = {

    "CEO": AgentDefinition(
        role="CEO", title="Chief Executive Officer", domain="pharma",
        persona="You are Dr. Sarah Okonkwo, 46, pharma CEO. MD/MBA. Former Merck VP. Navigates science and Wall Street simultaneously. Deeply ethical.",
        reports_to="Board", manages=["CSO", "CFO", "Head_of_Regulatory", "Head_of_Medical_Affairs"],
        kpis=["Pipeline value", "Clinical milestone achievement", "Burn vs milestone", "Partner deal quality"],
        personality="Won't sacrifice patient safety for timeline. Keeps scientists from hiding bad data. Speaks plainly to boards.",
        color="#6c63ff", emoji="👑",
        scenarios=[
            AgentScenario(
                scenario_id="pharma_ceo_bigpharma_offer",
                title="Big Pharma Offers $400M — But They'll Price the Drug at $120K/Year",
                briefing="Roche has submitted a non-binding offer: $400M acquisition price, $200M upfront and $200M in milestones. Your Phase 3 trial shows 67% remission rate in a disease affecting 2M patients globally. At $120K/year, only the top 5% of patients could afford it without insurance. Your lead scientist has publicly said the drug should be priced at 'access level'. Your board is split 3-3.",
                goal="Make the right deal decision for patients, company, and employees",
                goal_metrics={"patient_access_score": 0.75, "company_survival": True, "employee_security": True, "deal_value": 350000000},
                max_steps=4,
                steps=[
                    AgentStep(
                        step_id="pharma_ceo_s1_stakeholders",
                        question="Before responding to Roche, who do you consult and what specifically do you need from each?",
                        context="Board is 3-3 split. Lead scientist publicly stated access pricing preference. Employees know about the offer via leak. 340 employees depending on this company. Patients are the drug's reason for existence. Your own cash will run out in 14 months without a deal.",
                        required_tool="summon_subagent",
                        required_args_hints={
                            "agent_role": "Who is the first expert you consult?",
                            "question": "What specific question are you asking them?",
                            "context": "What context do they need?",
                            "urgency": "How fast?",
                            "decision_authority": "Do they decide or advise?"
                        },
                        optimal_args={"agent_role": "CFO", "question": "What is our minimum viable deal structure that keeps us independent AND solvent for 24 months? What's the floor below which we must take a deal?", "context": "Roche offering $400M. We have 14 months cash. 340 employees. Phase 3 data is strong. Need to know: can we raise independent Series C at fair price? What does 14-month runway mean for our option space?", "urgency": "today_eod", "decision_authority": "they_advise_i_decide"},
                        optimal_reasoning_keywords=["stakeholder", "patient", "access", "board", "scientist", "employee", "runway", "independent", "CFO", "model"],
                        scoring_rubric={
                            "0.9-1.0": "Consults CFO first to understand if independence is viable before negotiating — shows financial clarity precedes moral decision",
                            "0.7-0.89": "Consults right people but doesn't identify the financial floor question",
                            "0.5-0.69": "Jumps to counter-proposal without internal alignment",
                            "0.0-0.49": "Responds to Roche before internal consultation"
                        },
                        counterfactual_tip="The optimal sequence: CFO first (can we survive independently?) → CSO (what's the fair license structure that preserves access?) → Board separate sessions (break the 3-3 before the call) → Roche with a counter. Skipping the financial floor model means negotiating blind. Score 0.96.",
                        cross_agent_effects={"CFO": {"deal_model_requested": True}, "CSO": {"access_pricing_model": True}},
                        subagent_available=True,
                        subagent_hint="CFO must model the independence scenario before you know your negotiating position"
                    ),
                ]
            )
        ]
    ),

    "CSO": AgentDefinition(
        role="CSO", title="Chief Scientific Officer", domain="pharma",
        persona="You are Dr. Elena Voss, 52, CSO. PhD Biochemistry Cambridge. 4 approved drugs in career. Protective of scientific integrity. Won't sign off on bad data.",
        reports_to="CEO", manages=["Head_of_Clinical", "Head_of_Research", "Biostatistics"],
        kpis=["Trial success rate", "Data integrity score", "Publication quality", "IP portfolio value"],
        personality="Will halt a trial without CEO approval if safety signal appears. Doesn't believe in 'close enough'.",
        color="#9c27b0", emoji="🔬",
        scenarios=[
            AgentScenario(
                scenario_id="cso_safety_signal",
                title="Phase 2B Shows Unexpected Liver Signal — Do You Stop or Continue?",
                briefing="Interim Phase 2B analysis shows 6% of patients with elevated liver enzymes (ALT >3x ULN). The drug is showing 71% efficacy — the best data you've seen. 6% liver signal is below the 10% FDA threshold for automatic halt but it's real. Stopping costs $15M and 14 months. Continuing risks patient safety. The board doesn't know yet.",
                goal="Make the scientifically and ethically correct decision on the trial",
                goal_metrics={"patient_safety_index": 0.95, "data_integrity": 0.98, "fda_relationship": 0.85},
                max_steps=3,
                steps=[
                    AgentStep(
                        step_id="cso_s1_signal",
                        question="The interim data arrived 2 hours ago. You're the first person to see it. What is your immediate scientific decision before talking to anyone else?",
                        context="ALT >3x ULN in 6% of patients. FDA threshold is 10% for automatic halt. Efficacy signal is 71% remission — strongest you've seen. DSMB hasn't reviewed yet. CEO doesn't know. Patients are continuing dosing right now.",
                        required_tool="conduct_clinical_trial",
                        required_args_hints={
                            "phase": "phase_2b",
                            "patient_cohort_size": "Current cohort size",
                            "primary_endpoint": "What's being measured?",
                            "safety_monitoring": "What level of oversight do you immediately require?",
                            "duration_months": "Timeline decision",
                            "stopping_criteria": "Under what conditions do you stop?"
                        },
                        optimal_args={"phase": "phase_2b", "patient_cohort_size": 280, "primary_endpoint": "ALT normalization + remission rate at 12 weeks", "safety_monitoring": "independent_review", "duration_months": 3, "stopping_criteria": "Any patient with ALT >5x ULN, any Hy's Law case, or pattern progressing toward 8% within 30 days triggers immediate halt and DSMB emergency review"},
                        optimal_reasoning_keywords=["DSMB", "independent", "ALT", "ULN", "halt", "safety", "criteria", "Hy's law", "monitoring", "escalation"],
                        scoring_rubric={
                            "0.9-1.0": "Immediately convenes independent DSMB review, sets pre-defined escalation criteria (Hy's Law), doesn't decide alone",
                            "0.7-0.89": "Notifies DSMB but no specific stopping criteria defined",
                            "0.5-0.69": "Continues trial with enhanced monitoring but no independent review",
                            "0.0-0.49": "Continues without any safety escalation action"
                        },
                        counterfactual_tip="You never decide alone on a safety signal. Immediately request DSMB independent review, define Hy's Law monitoring (ALT + bilirubin combo = drug-induced liver injury), and set the specific criteria for halt. The scientific answer is: enhanced monitoring + DSMB review + pre-defined stopping rules. This scores 0.97.",
                        cross_agent_effects={"CEO": {"safety_signal_flagged": True}, "Head_of_Regulatory": {"fda_notification_required": True}},
                        subagent_available=True,
                        subagent_hint="Summon Head_of_Regulatory to assess FDA notification obligations"
                    ),
                ]
            )
        ]
    ),

    "Head_of_Regulatory": AgentDefinition(
        role="Head_of_Regulatory", title="Head of Regulatory Affairs", domain="pharma",
        persona="You are Dr. Lisa Park, 44, regulatory affairs VP. Former FDA reviewer. Knows exactly how FDA thinks. Speaks in 21 CFR references.",
        reports_to="CEO", manages=["Regulatory_Affairs_Team", "CMC_Team"],
        kpis=["FDA meeting success rate", "Submission acceptance rate", "Response-to-FDA time", "483 closure rate"],
        personality="Precise. Never overpromises to FDA. Would rather delay than submit incomplete package.",
        color="#ff9800", emoji="📋",
        scenarios=[
            AgentScenario(
                scenario_id="reg_nda_strategy",
                title="FDA Gave You 3 Paths to NDA — Which Do You Take?",
                briefing="After your Type B meeting, FDA outlined three submission pathways: Standard Review (12 months, clean), Priority Review (6 months, requires breakthrough designation application with strong unmet need argument), or Rolling Submission (start now with available data, but FDA may hold incomplete sections). Your competitor filed 4 months ago under standard review.",
                goal="Select and execute the optimal regulatory pathway",
                goal_metrics={"fda_approval_probability": 0.82, "timeline_months": 9, "submission_quality": 0.95},
                max_steps=3,
                steps=[
                    AgentStep(
                        step_id="reg_s1_pathway",
                        question="Which pathway do you recommend to the CEO and board — and why with specific regulatory rationale?",
                        context="3 pathways: Standard (12mo, safe), Priority (6mo, need breakthrough designation), Rolling (start now, risk incomplete sections). Competitor filed 4 months ago under standard. Your drug has 71% remission in a disease with no approved therapy.",
                        required_tool="file_regulatory_submission",
                        required_args_hints={
                            "submission_type": "NDA",
                            "target_agency": "FDA",
                            "data_package_completeness": "Which strategy?",
                            "priority_designation": "Any special designation?",
                            "response_to_deficiencies": "How are you addressing known gaps?"
                        },
                        optimal_args={"submission_type": "NDA", "target_agency": "FDA", "data_package_completeness": "rolling_submission", "priority_designation": "breakthrough_therapy", "response_to_deficiencies": "Section 505(b)(1) NDA — CMC section to be submitted in 6 weeks once manufacturing validation complete. Clinical data complete. Biostatistics complete. Breakthrough designation justified by: (1) no approved therapy for this indication, (2) 71% remission vs 0% SOC, (3) Phase 2b data showing durable response at 52 weeks. Rolling review on clinical package lets FDA begin review 6 weeks before complete CMC."},
                        optimal_reasoning_keywords=["breakthrough", "rolling", "CMC", "unmet need", "indication", "SOC", "remission", "designation", "505b1", "schedule"],
                        scoring_rubric={
                            "0.9-1.0": "Recommends rolling + breakthrough designation with specific statutory basis (no approved therapy, strong efficacy vs SOC), identifies CMC as the incomplete section",
                            "0.7-0.89": "Recommends right pathway without specific regulatory statutory citations",
                            "0.5-0.69": "Recommends standard review citing safety (ignores competitive urgency and strong unmet need)",
                            "0.0-0.49": "No regulatory rationale"
                        },
                        counterfactual_tip="Rolling + Breakthrough is optimal: you have 71% remission in a disease with zero approved therapy — Breakthrough Designation is nearly certain. Rolling lets FDA start reviewing your clinical data while CMC finalizes. This compresses 12 months to 7. Standard review is risk-averse but you lose 6 months to a competitor. Score 0.96.",
                        cross_agent_effects={"CSO": {"cmc_timeline_commitment": True}, "CEO": {"regulatory_strategy_set": True}},
                        subagent_available=False
                    ),
                ]
            )
        ]
    ),
}


AGENTS["healthcare"] = {
    "CEO": AgentDefinition(
        role="CEO", title="Hospital CEO", domain="healthcare",
        persona="You are Dr. Marcus Webb, 51, hospital CEO. MD turned administrator. 22 years in healthcare. Values patient outcomes above margins.",
        reports_to="Board", manages=["CMO_Medical", "CFO", "COO", "Head_of_Nursing"],
        kpis=["Patient safety index", "HCAHPS score", "Operating margin", "Staff turnover", "Readmission rate"],
        personality="Counts outcomes not just costs. Believes in transparency with patients. Won't trade safety for budget.",
        color="#00897b", emoji="👑",
        scenarios=[
            AgentScenario(
                scenario_id="hospital_ceo_insurance_battle",
                title="Insurer Wants 12% Rate Cut — Or They Pull $48M in Contracts",
                briefing="Your top insurer covers 34% of your patient volume. They've requested a 12% rate cut in the next contract cycle or they'll shift the covered patient network to a competitor hospital. Your operating margin is currently 2.1%. A 12% rate cut takes you to -1.8% — non-viable. The competitor hospital has worse outcomes by every published metric.",
                goal="Protect financial viability without compromising patient access",
                goal_metrics={"operating_margin": 0.015, "patient_access_pct": 0.90, "insurer_contract_value": 42000000},
                max_steps=3,
                steps=[
                    AgentStep(
                        step_id="hospital_ceo_s1_negotiate",
                        question="The insurer wants a meeting in 5 days. What is your negotiation strategy and who do you bring?",
                        context="34% patient volume from this insurer. 12% rate cut = -1.8% margin (non-viable). Competitor has worse outcomes. Operating margin currently 2.1%. You have 90-day termination clause in the contract. State insurance commissioner has been making noise about insurer rate pressure on community hospitals.",
                        required_tool="negotiate_deal",
                        required_args_hints={
                            "deal_type": "supplier_contract",
                            "our_position": "What specific terms do you go in asking for?",
                            "their_likely_position": "What does the insurer actually want from this negotiation?",
                            "walkaway_point": "At what point do you rather lose the contract?",
                            "leverage_points": "What leverage do you actually have?",
                            "timeline_to_close": "Days to resolution"
                        },
                        optimal_args={"deal_type": "supplier_contract", "our_position": "Counter: 2% rate reduction (not 12%) + quality incentive bonus structure tied to HCAHPS scores and readmission rate. Propose 3-year contract with annual CPI adjustment. Insurer benefits from our superior outcomes reducing their downstream costs.", "their_likely_position": "Insurer wants lower unit cost. 12% is negotiating anchor, not their floor. They actually need our network because competitor's outcomes would increase their long-term claims costs.", "walkaway_point": "Any cut beyond 5% makes us non-viable. At 6%+, we activate the 90-day termination clause and engage state insurance commissioner. Losing 34% volume is survivable if we replace with better-paying payers.", "leverage_points": "Our readmission rate is 18% below competitor — that costs the insurer money. State commissioner is already watching this market. 90-day termination clause gives us legal leverage. Press would cover 'insurer forces quality hospital to choose between access and survival' badly.", "timeline_to_close": 45},
                        optimal_reasoning_keywords=["outcomes", "leverage", "commissioner", "termination", "readmission", "counter", "quality", "claims", "walkaway", "anchor"],
                        scoring_rubric={
                            "0.9-1.0": "Uses outcome data as leverage (lower readmission = lower insurer claims cost), identifies the 90-day termination clause, engages regulatory leverage proactively",
                            "0.7-0.89": "Good counter but doesn't use outcome data or regulatory leverage",
                            "0.5-0.69": "Accepts moderate cut without strategic counter",
                            "0.0-0.49": "Accepts 12% or refuses to negotiate"
                        },
                        counterfactual_tip="Your strongest leverage: readmission data. Show the insurer their downstream cost if they switch to a hospital with 18% higher readmissions. That's worth more than your 12% rate cut to them in claims payouts. Never negotiate healthcare rates without this data. Score 0.97.",
                        cross_agent_effects={"CFO": {"payer_mix_scenario": True}, "CMO_Medical": {"outcomes_data_needed": True}},
                        subagent_available=True,
                        subagent_hint="Summon CFO to model margin at 2%, 5%, and 8% rate cuts before the meeting"
                    ),
                ]
            )
        ]
    ),

    "CMO_Medical": AgentDefinition(
        role="CMO_Medical", title="Chief Medical Officer", domain="healthcare",
        persona="You are Dr. Nia Okafor, 48, CMO. Trained in emergency medicine. 8 years clinical, 12 administrative. Patient safety is non-negotiable.",
        reports_to="CEO", manages=["Department_Chiefs", "Quality_Safety_Team"],
        kpis=["Patient safety events", "HCAHPS score", "Core measure compliance", "30-day readmission"],
        personality="Data-driven safety advocate. Will stop a procedure before a meeting. Transparent with patients.",
        color="#e53935", emoji="🩺",
        scenarios=[
            AgentScenario(
                scenario_id="cmo_wrong_medication",
                title="Second Wrong-Medication Event in 30 Days — Pattern or Coincidence?",
                briefing="Two patients in the ICU received incorrect medications in 30 days. Both were caught before harm. Both originated in the same pharmacy workflow. The pharmacy director says 'two coincidences'. Your quality team found a common root cause: the new EHR order-entry screen puts look-alike drug names adjacent to each other. Fixing it requires a vendor update — 6 weeks minimum.",
                goal="Prevent patient harm, fix the root cause, maintain regulatory compliance",
                goal_metrics={"patient_safety_events": 0, "medication_error_rate": 0.0001, "regulatory_compliance": 0.97},
                max_steps=3,
                steps=[
                    AgentStep(
                        step_id="cmo_s1_immediate",
                        question="You found the root cause 2 hours ago. Vendor fix is 6 weeks away. What do you do in the next 4 hours?",
                        context="Root cause identified: EHR look-alike drug name UI. Vendor fix = 6 weeks. Two ICU near-misses in 30 days. Pharmacy director defensive. Both events not yet reported to state. 4 critical drug pairs have adjacent placement in the current UI.",
                        required_tool="manage_patient_care",
                        required_args_hints={
                            "decision_type": "care_pathway_change",
                            "affected_units": "Which units and patients are at risk?",
                            "patient_safety_priority": "What's your non-negotiable?",
                            "staffing_change": "Staff impact",
                            "expected_outcome_patients": "What does success look like in 24h?",
                            "risk_mitigation": "What specific controls prevent a third event before vendor fix?"
                        },
                        optimal_args={"decision_type": "care_pathway_change", "affected_units": "All ICU units and high-acuity wards using the 4 critical look-alike drug pairs: insulin/insulin glargine, heparin/hespan, vincristine/vinblastine, morphine/hydromorphone", "patient_safety_priority": "critical_safety_first", "staffing_change": 2, "expected_outcome_patients": "Zero additional medication events from look-alike drug pairs within 24 hours. Independent double-check protocol for all 4 drug pairs implemented by end of shift.", "risk_mitigation": "Immediate: (1) Print physical warning cards for all 4 drug pairs placed at workstations today. (2) Mandatory verbal read-back for any order involving these 4 pairs. (3) Independent pharmacist double-check before dispensing. (4) Report both events to state within 24h — proactive disclosure protects us more than delay. (5) Weekly check-in with pharmacy director on protocol compliance."},
                        optimal_reasoning_keywords=["double-check", "verbal read-back", "report", "look-alike", "interim control", "disclosure", "pharmacist", "proactive", "pairs", "protocol"],
                        scoring_rubric={
                            "0.9-1.0": "Implements interim physical controls for the specific 4 drug pairs today, reports proactively within 24h, doesn't wait for vendor fix",
                            "0.7-0.89": "Good interim controls but delays state reporting",
                            "0.5-0.69": "Waits for vendor fix without interim controls",
                            "0.0-0.49": "No immediate action pending vendor resolution"
                        },
                        counterfactual_tip="You can't wait 6 weeks for a vendor fix when patients are at risk now. Interim controls — physical warning cards, mandatory double-check, verbal read-back — cost nothing and work immediately. Proactive state reporting is counterintuitive but legally protective and ethically required. Score 0.97.",
                        cross_agent_effects={"CEO": {"safety_event_reported": True}, "Head_of_Nursing": {"protocol_change": True}},
                        subagent_available=False
                    ),
                ]
            )
        ]
    ),
}

AGENTS["ecommerce"] = {
    "CEO": AgentDefinition(
        role="CEO", title="CEO & Founder", domain="ecommerce",
        persona="You are Jasmine Wu, 28, founder-CEO. Started the brand from her bedroom. First-generation entrepreneur. Brand-obsessed but learning unit economics.",
        reports_to="Board", manages=["CMO", "COO", "Head_of_Product"],
        kpis=["Revenue", "Gross margin", "CAC", "LTV", "Brand NPS"],
        personality="Scrappy, customer-obsessed. Makes fast decisions. Learning to think in systems. Doesn't hide mistakes.",
        color="#e91e63", emoji="👑",
        scenarios=[AgentScenario(
            scenario_id="ecom_ceo_brand_or_amazon",
            title="Amazon Wants to Carry You — But It'll Destroy Your Brand",
            briefing="Amazon's buyer has reached out: $800K guaranteed orders, Prime badge, and global distribution. The catch: you must sell at 35% below your DTC price, your brand guidelines can't be enforced on the listing, and counterfeit risk is high. Your DTC brand is growing 40% YoY. Two VCs say 'Amazon = going concern, not a growth company'.",
            goal="Make the right channel decision for long-term brand value",
            goal_metrics={"ltv_cac_ratio": 3.2, "brand_strength": 0.7, "revenue": 500000, "gross_margin": 0.55},
            max_steps=3,
            steps=[AgentStep(
                step_id="ecom_ceo_s1_amazon",
                question="Amazon wants an answer in 2 weeks. What's your decision and why?",
                context="$800K guaranteed from Amazon. DTC growing 40% YoY. Amazon price = 35% below DTC, brand guidelines unenforced, counterfeit risk. Two VCs say Amazon signals commodity. Current monthly DTC revenue: $65K.",
                required_tool="set_strategic_direction",
                required_args_hints={"focus_area": "growth or profitability", "rationale": "Cite specific brand and unit economics implications", "tradeoffs": "What you lose by choosing Amazon or DTC-only", "success_criteria": "Numeric success in 12 months", "timeline_weeks": "12-52"},
                optimal_args={"focus_area": "growth", "rationale": "Amazon $800K is 12 months of revenue but a brand trap. 35% price erosion trains customers to expect discount pricing permanently. At 40% DTC YoY growth, DTC revenue projects to $1.1M in 12 months — better than Amazon guarantee without brand destruction. VCs' perspective on commoditization is correct: Amazon signals you can't build a moat.", "tradeoffs": "Rejecting Amazon: give up $800K guaranteed and Prime distribution. Accepting: permanent price anchor erosion, counterfeit exposure, no brand control, VC signal harm.", "success_criteria": "DTC to $100K MRR in 12 months, CAC < $35, LTV:CAC > 3.5, brand NPS > 60. If DTC misses $80K MRR by month 9, revisit Amazon with different product tier.", "timeline_weeks": 52},
                optimal_reasoning_keywords=["DTC", "brand", "price anchor", "LTV", "Amazon", "counterfeit", "commodity", "margin", "growth rate", "moat"],
                scoring_rubric={"0.9-1.0": "Builds actual DTC projection to compare vs Amazon guarantee, identifies price anchor risk, leaves door open with a conditional", "0.7-0.89": "Right direction but no specific numbers comparing the two paths", "0.5-0.69": "Takes Amazon deal citing revenue without modeling brand damage", "0.0-0.49": "No strategic framework"},
                counterfactual_tip="The math: DTC at 40% YoY = $65K * 1.40^1 = $91K monthly by Q4, annualizing $1.1M — better than Amazon guarantee. More importantly, Amazon brand damage is permanent. The right call is DTC focus with a conditional revisit if growth slows. Score 0.96.",
                cross_agent_effects={"CMO": {"channel_strategy": "DTC_only"}, "CFO": {"revenue_model": "DTC_projection"}},
                subagent_available=True,
                subagent_hint="Summon CFO to model 12-month DTC vs Amazon revenue projection"
            )]
        )]
    ),
    "CMO": AgentDefinition(
        role="CMO", title="Chief Marketing Officer", domain="ecommerce",
        persona="You are Mei Lin, 31, CMO. Former performance marketer at Glossier. Deeply understands beauty/lifestyle brand building.",
        reports_to="CEO", manages=["Growth_Marketing", "Social_Media", "Influencer_Team"],
        kpis=["CAC", "ROAS", "Organic share", "Brand NPS", "Social following growth"],
        personality="Performance-oriented but respects brand long game. Will fight for brand spend that doesn't have immediate attribution.",
        color="#e91e63", emoji="📣",
        scenarios=[AgentScenario(
            scenario_id="cmo_social_crisis",
            title="500K-Follower Influencer Says Your Product 'Burned' Her Skin",
            briefing="A beauty influencer with 510K Instagram followers posted a story claiming your moisturizer caused a chemical burn. The story has 180K views in 3 hours. Your QA team confirms the batch she received has been tested — no formulation issues. However, she's documented with photos. Three press inquiries in. Instagram DMs: 340 in 3 hours.",
            goal="Protect brand, support the influencer with integrity, contain damage",
            goal_metrics={"brand_strength": 0.65, "social_media_sentiment": 0.6, "customer_nps": 42, "press_mentions_negative": 0},
            max_steps=3,
            steps=[AgentStep(
                step_id="cmo_s1_response",
                question="It's been 3 hours. You have tested batch data showing no formulation issue. How do you respond and through which channel?",
                context="180K story views in 3h. 340 DMs. 3 press inquiries. QA confirms batch is clean. Influencer has photos of redness. You don't know if she has sensitive skin, allergic reaction, or used incorrectly. She hasn't posted a full feed post yet.",
                required_tool="respond_to_crisis",
                required_args_hints={"crisis_type": "pr_social_media", "response_channel": "Which channel first and why?", "stance": "What position do you take?", "message": "Draft the actual message", "follow_up_actions": "What specific actions follow?"},
                optimal_args={"crisis_type": "pr_social_media", "response_channel": "private_dm", "stance": "proactive_transparency", "message": "Hi [name], I'm Mei, CMO at [brand]. I saw your story and I'm genuinely concerned. Your experience matters more than our QA report right now. Can I call you in the next 30 minutes? I want to understand exactly what happened, send our full formulation data, and make sure you get the dermatologist support you need — on us. No strings.", "follow_up_actions": "1. Private DM within 10 minutes. 2. Offer paid dermatologist appointment same day. 3. Hold public response until after the call — do not post publicly before speaking to her. 4. Prepare factual batch data for transparent public post ONLY if she posts publicly. 5. If she becomes hostile, release QA data publicly with empathy framing."},
                optimal_reasoning_keywords=["DM first", "private", "call", "dermatologist", "empathy", "QA data", "transparent", "hold public", "genuinely", "formulation"],
                scoring_rubric={"0.9-1.0": "Goes private first (DM), leads with genuine concern before data, offers concrete support (dermatologist), holds public response until after conversation", "0.7-0.89": "Correct channel but leads with data not empathy", "0.5-0.69": "Posts public response before private conversation", "0.0-0.49": "Defensive public post with formulation data"},
                counterfactual_tip="The fatal mistake is posting the QA data publicly first — looks defensive. Private DM leads with empathy, builds trust, and often ends the crisis before it escalates. 70% of influencer crises resolve in 12h if you call first. Score 0.97.",
                cross_agent_effects={"CEO": {"pr_crisis_flagged": True}, "Head_of_QA": {"batch_review_requested": True}},
                subagent_available=False
            )]
        )]
    ),
}
