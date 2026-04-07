from __future__ import annotations
from typing import Any


def get_situation(domain: str, state: dict, step: int, seen_ids: set) -> dict:
    pool = SITUATIONS.get(domain, [])
    candidates = []
    for sit in pool:
        if sit["id"] in seen_ids:
            continue
        if _matches(sit.get("trigger", "always"), state, step):
            candidates.append(sit)
    if not candidates:
        return _fallback(domain, state, step)
    candidates.sort(key=lambda s: (s.get("seq", 99)))
    return candidates[0]


def _matches(trigger: str, state: dict, step: int) -> bool:
    if trigger == "always":
        return True
    parts = trigger.split(":")
    t = parts[0]
    if t == "step_range":
        return int(parts[1]) <= step <= int(parts[2])
    if t == "step_gte":
        return step >= int(parts[1])
    if t == "step":
        return step == int(parts[1])
    if t == "below":
        return state.get(parts[1], float("inf")) < float(parts[2])
    if t == "above":
        return state.get(parts[1], 0) > float(parts[2])
    if t == "range":
        v = state.get(parts[1], 0)
        return float(parts[2]) <= v <= float(parts[3])
    return True


def _fallback(domain: str, state: dict, step: int) -> dict:
    msg = {
        "tech_startup": "The team is heads-down. No major crises today — but the weekly OKR review reveals you're behind on three key results.",
        "pharma": "Research is proceeding. The CSO flags that two competing labs published related papers this week.",
        "interior_design": "The project is moving. The contractor notes the tile delivery will be one day late.",
        "manufacturing": "Line is running. The floor manager reports a 1.2% increase in micro-defect rate on Unit B.",
        "finance": "Markets are calm. Your quant model flags a mild correlation shift in the tech sector.",
        "ecommerce": "Orders flowing steadily. Customer support notes a spike in 'where is my order' tickets.",
        "healthcare": "Ward is stable. The charge nurse reports overnight staffing was 2 people short.",
    }
    return {
        "id": f"fallback_{step}",
        "seq": 99,
        "area": "Operations",
        "narrator_name": "Operations Lead",
        "narrator_role": "Operations",
        "headline": "Steady State — But Watch These Signals",
        "situation": msg.get(domain, "Business continues. Small signals need attention."),
        "question": "How do you respond to these early warning signs?",
        "choices": [
            {"key": "A", "label": "Address them immediately", "description": "Dedicate a team meeting to resolve each signal now.",
             "consequences": {"budget_remaining": -5000, "team_morale": 0.03}, "narrative": "You catch issues early. Team respects the proactiveness.", "intent": "risk", "risk": "low"},
            {"key": "B", "label": "Log and monitor", "description": "Track the signals for 2 more days before acting.",
             "consequences": {}, "narrative": "No action taken. Signals stay at same level.", "intent": "balance", "risk": "low"},
            {"key": "C", "label": "Delegate to the relevant lead", "description": "Assign each signal to its domain owner with a 24-hour resolution window.",
             "consequences": {"team_morale": 0.02, "budget_remaining": -2000}, "narrative": "Owners feel trusted and resolve issues independently.", "intent": "balance", "risk": "low"},
        ]
    }


SITUATIONS: dict[str, list[dict]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# TECH STARTUP
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["tech_startup"] = [

    {
        "id": "cofounder_conflict",
        "seq": 1,
        "trigger": "step_range:1:3",
        "area": "Leadership & Strategy",
        "narrator_name": "Jordan Park",
        "narrator_role": "Co-Founder & CTO",
        "headline": "🔥 Your Co-Founder Wants to Scrap the Product",
        "situation": (
            "Jordan calls an emergency video call at 11 PM. After talking to 20 users this week, "
            "they're convinced you're building the wrong thing entirely. 'We have six weeks until "
            "the investor demo and zero product-market fit signals,' they say. Jordan holds 40% equity "
            "and is your most technically capable person — but they've been wrong before, and you have "
            "a committed roadmap. The team finds out tomorrow regardless."
        ),
        "question": "Jordan wants a decision before the 9 AM standup. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Agree — pivot now before we burn more runway",
                "description": "Restart the product direction entirely. Delay the investor demo by 3 weeks.",
                "consequences": {"product_quality": 0.12, "team_morale": -0.08, "timeline_remaining": -2, "budget_remaining": -25000, "investor_confidence": -0.1},
                "narrative": "The pivot is painful. Three weeks of work is shelved. But Jordan's energy returns and the new direction is sharper. Two engineers quit — they were too committed to the old vision.",
                "intent": "quality", "risk": "high"
            },
            {
                "key": "B", "label": "2-week structured A/B experiment — data decides",
                "description": "Neither of you decides alone. Run both approaches with real users for two weeks.",
                "consequences": {"product_quality": 0.06, "team_morale": 0.05, "timeline_remaining": -1, "budget_remaining": -12000, "investor_confidence": 0.0},
                "narrative": "The data doesn't give a clean answer, but reveals a hybrid approach neither of you had seen. Jordan respects the process. You miss the demo window by one week.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "C", "label": "Hold direction — 6 weeks to demo, no pivot",
                "description": "Acknowledge concerns, log them for post-demo. Ship the roadmap.",
                "consequences": {"product_quality": -0.04, "team_morale": -0.12, "timeline_remaining": 0, "budget_remaining": 0, "investor_confidence": 0.06},
                "narrative": "You ship the demo. Investors are pleased on the surface. But Jordan is visibly disengaged and three engineers notice. The product gap Jordan identified doesn't go away.",
                "intent": "speed", "risk": "medium"
            },
            {
                "key": "D", "label": "Weekend sprint: 15 customer calls together",
                "description": "Block the weekend. You and Jordan do 15 raw customer interviews before any decision.",
                "consequences": {"product_quality": 0.09, "team_morale": 0.1, "timeline_remaining": -0.5, "budget_remaining": -4000, "investor_confidence": 0.03},
                "narrative": "You come out Sunday night aligned. The pivot is smaller than feared. Jordan feels heard. You have a tighter, cleaner thesis to show investors. A good week follows.",
                "intent": "balance", "risk": "low"
            }
        ]
    },

    {
        "id": "viral_negative_tweet",
        "seq": 2,
        "trigger": "step_range:2:6",
        "area": "Marketing & PR",
        "narrator_name": "Priya Nair",
        "narrator_role": "Head of Marketing",
        "headline": "📢 A Thread About Your Bug is Going Viral — 12K Retweets",
        "situation": (
            "A power user with 180K followers discovered a data-loss bug and posted a 15-tweet thread "
            "comparing your product to 'digital shredder'. It's been retweeted 12K times in 4 hours. "
            "Your support inbox has 340 new tickets. Three journalists have emailed for comment. "
            "The bug affects 3% of users. Engineering says a patch takes 8 hours. "
            "Your paid ad campaigns are still running and driving traffic to the negative conversation."
        ),
        "question": "It's 7 AM. You have 2 hours before the story potentially hits tech press. What is your move?",
        "choices": [
            {
                "key": "A", "label": "Pause all ads, post public apology thread + fix ETA",
                "description": "Full transparency. Pause campaigns. CEO posts an apologetic thread with an 8-hour fix commitment.",
                "consequences": {"brand_strength": 0.08, "mau": -200, "budget_remaining": -5000, "nps": 5, "media_coverage": 0.05},
                "narrative": "Your apology thread gets 4K retweets — mostly positive. The tech press runs the story but it's largely 'startup handles crisis well'. Affected users are impressed by the transparency.",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "B", "label": "DM the influencer first — offer personal fix + compensation",
                "description": "Reach out privately. Offer to fix their account first, give them a 3-month free upgrade, and ask for a follow-up tweet.",
                "consequences": {"brand_strength": 0.03, "mau": -300, "budget_remaining": -2000, "nps": -3, "media_coverage": 0.0},
                "narrative": "The influencer accepts. They post a muted follow-up. But screenshots of the DM leak 3 days later, making it look like you tried to bury the story. A second wave follows.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "C", "label": "Say nothing publicly — fix the bug and let it die",
                "description": "Don't amplify. Fix the bug quietly. Respond only to direct support tickets.",
                "consequences": {"brand_strength": -0.1, "mau": -500, "budget_remaining": 0, "nps": -8, "media_coverage": 0.05},
                "narrative": "Three journalists write the story without your comment. The narrative is 'company goes dark on bug'. Brand damage is worse than the bug itself.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "D", "label": "Pause ads + fix + post status page + town hall",
                "description": "Pause campaigns. Post live status page. Ship fix in 8 hours. Host public livestream post-fix explaining what happened.",
                "consequences": {"brand_strength": 0.15, "mau": 200, "budget_remaining": -8000, "nps": 12, "team_morale": 0.05},
                "narrative": "The livestream gets 2K viewers. You show the fix live. Three journalists change their story angle to 'radical transparency'. You gain 800 new signups from the goodwill.",
                "intent": "quality", "risk": "medium"
            }
        ]
    },

    {
        "id": "enterprise_deal",
        "seq": 3,
        "trigger": "above:mrr:1000",
        "area": "Sales & Revenue",
        "narrator_name": "Chris Okafor",
        "narrator_role": "Head of Sales",
        "headline": "💰 $240K Enterprise Deal — But Their Terms Will Break Us",
        "situation": (
            "Chris brings you the biggest deal in company history: a Fortune 500 wants to sign a $240K "
            "annual contract. The catch: they need 14 custom API integrations built exclusively for them, "
            "SOC-2 compliance certification within 60 days, and a dedicated support SLA that requires a "
            "full-time customer success hire. Engineering estimates 3 months of work for one team. "
            "Your CTO says accepting means the core product roadmap stops for a quarter."
        ),
        "question": "Chris needs an answer by end of week. Do you take the deal?",
        "choices": [
            {
                "key": "A", "label": "Take it — $240K solves runway, worry about roadmap later",
                "description": "Sign it. Hire one CS person immediately. Pause the product roadmap for Q3.",
                "consequences": {"budget_remaining": 200000, "mrr": 20000, "product_quality": -0.08, "team_morale": -0.1, "technical_debt": 0.15, "paying_customers": 1},
                "narrative": "The money hits. The team is exhausted doing custom work. Two product engineers quit — they didn't join to build enterprise integrations. Your core product falls 3 months behind competitors.",
                "intent": "cost", "risk": "medium"
            },
            {
                "key": "B", "label": "Counter: $240K but only 4 of the 14 integrations",
                "description": "Negotiate down the scope. Take only the work that also benefits other customers.",
                "consequences": {"budget_remaining": 120000, "mrr": 10000, "product_quality": 0.02, "team_morale": 0.0, "paying_customers": 1},
                "narrative": "They come back at $140K. You accept. The 4 integrations you build end up becoming product features other customers love. Good tradeoff in hindsight.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Pass — enterprise at this stage kills product velocity",
                "description": "Politely decline. You're too early for enterprise complexity. Stay focused on self-serve growth.",
                "consequences": {"budget_remaining": 0, "mrr": 500, "product_quality": 0.06, "team_morale": 0.05, "brand_strength": 0.02},
                "narrative": "The team is relieved. Product velocity increases. You hit a self-serve milestone 6 weeks later that attracts three more SMB customers worth $18K MRR combined.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "D", "label": "Take it — but hire a contract team for the custom work",
                "description": "Sign the deal. Outsource the custom integrations to an agency. Protect the core team.",
                "consequences": {"budget_remaining": 100000, "mrr": 20000, "product_quality": -0.03, "technical_debt": 0.08, "paying_customers": 1, "team_morale": 0.0},
                "narrative": "The agency delivers 70% of what you promised. The integration quality is rough. The enterprise customer complains. You spend months fixing contractor work.",
                "intent": "speed", "risk": "high"
            }
        ]
    },

    {
        "id": "engineering_burnout",
        "seq": 4,
        "trigger": "above:team_size:6",
        "area": "Team & Culture",
        "narrator_name": "Ananya Sharma",
        "narrator_role": "Head of People",
        "headline": "😰 Your Eng Team is Burning Out — 3 Resignation Risk",
        "situation": (
            "Ananya pulls you aside after the all-hands. Three engineers have told her in 1:1s that they're "
            "actively interviewing. The average work week is 62 hours. There's no on-call rotation — "
            "the same two people get paged every night. One engineer's spouse emailed HR directly. "
            "Meanwhile, your two biggest product deadlines are in 6 weeks. You've been publicly "
            "promising those features to customers."
        ),
        "question": "Ananya says if you don't act this week, you'll lose at least two engineers. What do you prioritize?",
        "choices": [
            {
                "key": "A", "label": "Slip both deadlines — give the team real breathing room",
                "description": "Email customers now. Push deadlines 4 weeks. Introduce on-call rotation immediately. No crunch until Q3.",
                "consequences": {"team_morale": 0.2, "budget_remaining": 0, "product_quality": 0.05, "investor_confidence": -0.06, "brand_strength": -0.04},
                "narrative": "Two engineers take you off their list. Morale recovers. Customers are annoyed — one churns. But the product quality in the next sprint is noticeably higher.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "B", "label": "Hire 2 senior engineers immediately + deliver deadline",
                "description": "Start recruiting today. Ship with overtime for 6 more weeks, then transition new hires into on-call.",
                "consequences": {"team_morale": -0.05, "budget_remaining": -120000, "product_quality": 0.03, "team_size": 2, "investor_confidence": 0.04},
                "narrative": "You ship on time. Two engineers still quit despite the promises. New hires take 3 months to ramp up. Net capacity actually drops for 6 weeks after launch.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "C", "label": "Give equity refreshes + meaningful pay increases",
                "description": "Retain the three at-risk engineers with compensation packages. Address on-call rotation now.",
                "consequences": {"team_morale": 0.15, "budget_remaining": -90000, "team_size": 0, "investor_confidence": -0.03},
                "narrative": "All three stay. The pay increase triggers salary compression complaints from longer-tenured team members. You spend a month re-benchmarking everyone.",
                "intent": "cost", "risk": "medium"
            },
            {
                "key": "D", "label": "Host a radical candor session — let the team reprioritize",
                "description": "Kill one of the two features yourself. Let the team vote on which. Publicly own the scope cut.",
                "consequences": {"team_morale": 0.18, "budget_remaining": -5000, "product_quality": 0.08, "investor_confidence": -0.02, "brand_strength": 0.02},
                "narrative": "The team cuts the harder feature. Everyone respects that you let them own it. One engineer withdraws their offer. The remaining feature ships exceptionally well.",
                "intent": "balance", "risk": "low"
            }
        ]
    },

    {
        "id": "competitor_funding",
        "seq": 5,
        "trigger": "above:mau:200",
        "area": "Competition & Strategy",
        "narrator_name": "Mika Tanaka",
        "narrator_role": "Head of Product",
        "headline": "⚔️ Competitor Raised $12M and Copied Your Core Feature",
        "situation": (
            "TechCrunch just published: your closest competitor closed a $12M Series A and launched "
            "a feature set that mirrors your core differentiator — but with a slicker UI and 3 months "
            "of premium support free. Your LinkedIn DMs have three messages from your enterprise prospects "
            "asking if you saw the news. Two customers forwarded the article. Your Series A is 4 months away."
        ),
        "question": "The team is spooked. Investors will call this week. What's your strategic response?",
        "choices": [
            {
                "key": "A", "label": "Accelerate — ship 3 new differentiating features in 6 weeks",
                "description": "Drop everything else. Pure engineering sprint on features the competitor can't copy fast.",
                "consequences": {"product_quality": 0.1, "team_morale": -0.1, "technical_debt": 0.1, "budget_remaining": -20000, "mau": 300, "investor_confidence": 0.08},
                "narrative": "You ship fast. The features are rough but real. Two prospects re-engage. The team is exhausted but investors love the intensity. Technical debt accumulates.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "B", "label": "Don't panic — go deeper on your existing niche",
                "description": "Ignore the competitor. Intensify focus on your existing users. Build what only you can build.",
                "consequences": {"product_quality": 0.14, "nps": 8, "mau": 100, "team_morale": 0.05, "brand_strength": 0.06, "investor_confidence": -0.03},
                "narrative": "You release a deeply researched update. Your NPS jumps 8 points. The competitor acquires volume but your retention is visibly better. Niche dominance builds.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "C", "label": "Call investors now — frame the competition as market validation",
                "description": "Get ahead of the narrative. Brief your investors before they call you.",
                "consequences": {"investor_confidence": 0.12, "brand_strength": 0.04, "budget_remaining": -3000, "mau": 0},
                "narrative": "Two investors call it 'smart positioning'. One says 'now we know the market is real'. You flip a threat into proof. Fundraising timeline moves up by 6 weeks.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "D", "label": "Reach out to the competitor — explore partnership or acqui-hire",
                "description": "Contact their CEO directly. Explore whether combining is smarter than fighting.",
                "consequences": {"investor_confidence": -0.05, "brand_strength": -0.03, "mau": 0, "team_morale": -0.05},
                "narrative": "The competitor is flattered but not interested. The conversation leaks internally. Some team members worry you're giving up. The distraction costs 2 weeks.",
                "intent": "risk", "risk": "high"
            }
        ]
    },

    {
        "id": "series_a_drama",
        "seq": 6,
        "trigger": "below:budget_remaining:300000",
        "area": "Finance & Fundraising",
        "narrator_name": "David Kim",
        "narrator_role": "CFO / Finance Lead",
        "headline": "💸 Lead Investor Wants Board Seat + Ratchet Clause to Lead Series A",
        "situation": (
            "David shares the term sheet from your Series A lead. The good news: $4M at a $20M valuation. "
            "The bad news: they want a board seat with blocking rights on hires above $150K, a 2x ratchet "
            "clause if you miss next year's revenue target, and anti-dilution provisions. Your existing angel "
            "investors say the terms are predatory. Another VC said they're interested but need 8 more weeks. "
            "You have 11 weeks of runway."
        ),
        "question": "David says decide in 72 hours — they have two other deals in queue. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Accept — 11 weeks of runway, you can't negotiate from weakness",
                "description": "Sign it. Protect the company first. You can renegotiate terms at Series B.",
                "consequences": {"budget_remaining": 3500000, "investor_confidence": 0.1, "team_morale": 0.0, "runway_months": 24},
                "narrative": "Money is in. But the ratchet clause becomes a mental weight. Every board meeting has undertones of the target. You hit the number — barely.",
                "intent": "speed", "risk": "medium"
            },
            {
                "key": "B", "label": "Counter: accept $4M but remove ratchet + limit blocking rights",
                "description": "Push back on the predatory terms specifically. Be willing to walk.",
                "consequences": {"budget_remaining": 3500000, "investor_confidence": 0.05, "team_morale": 0.05, "runway_months": 24},
                "narrative": "They remove the ratchet but keep limited blocking rights on C-suite. Acceptable. You close in 2 weeks. Angels respect the backbone.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Wait 8 weeks for the second VC — accept bridge from angels",
                "description": "Ask your angels for a $200K bridge. Buy time for a better term sheet.",
                "consequences": {"budget_remaining": 150000, "investor_confidence": -0.05, "team_morale": 0.0, "runway_months": 4},
                "narrative": "Angels agree to bridge. The second VC's offer is slightly better terms but same valuation. You close with 3 weeks to spare. Stressful but clean terms.",
                "intent": "risk", "risk": "high"
            },
            {
                "key": "D", "label": "Bring both VCs to a bake-off — create competitive tension",
                "description": "Tell the lead you're in final conversations with another firm. Push both for best terms.",
                "consequences": {"budget_remaining": 4200000, "investor_confidence": 0.15, "team_morale": 0.0, "runway_months": 28},
                "narrative": "The lead improves their terms. The second VC raises their offer. You close at $4.5M at a $24M valuation with clean terms. The bake-off worked.",
                "intent": "quality", "risk": "medium"
            }
        ]
    },

    {
        "id": "social_media_stunt",
        "seq": 7,
        "trigger": "above:brand_strength:0.3",
        "area": "Marketing & Growth",
        "narrator_name": "Priya Nair",
        "narrator_role": "Head of Marketing",
        "headline": "🚀 Growth Hack Proposal: Controversial Stunt That Could 10x Signups",
        "situation": (
            "Priya comes with a proposal: your growth team found that posting a 'hot take' comparing "
            "yourselves to a Slack competitor — naming them directly — would likely trigger a Reddit "
            "thread and drive 50K new signups based on a similar case study. It's aggressive, legally grey, "
            "and your CTO is uncomfortable. But you're 3,000 signups short of your investor milestone "
            "with 2 weeks left. Priya says 'this is the only move that gets us there in time.'"
        ),
        "question": "It's a high-wire act. What's your call?",
        "choices": [
            {
                "key": "A", "label": "Approve it — milestone matters more than comfort",
                "description": "Post the comparison. Accept the legal risk. Monitor closely.",
                "consequences": {"mau": 8000, "brand_strength": 0.1, "nps": -3, "budget_remaining": -5000, "investor_confidence": 0.12, "media_coverage": 0.08},
                "narrative": "The post goes viral. 12K new signups in 48 hours. The competitor's legal team emails. You respond publicly with data to back every claim. The press covers the fight.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "B", "label": "Approve a softer version — no direct competitor naming",
                "description": "Hot take without legal risk. Critique the category, not a specific company.",
                "consequences": {"mau": 2000, "brand_strength": 0.06, "nps": 2, "budget_remaining": -2000, "investor_confidence": 0.04},
                "narrative": "Gets traction but not viral. 2,000 signups. Still 1,000 short of milestone but much closer. Investors accept the effort.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Reject it — don't build a brand on cheap shots",
                "description": "Tell Priya to find a different path. Authentic content only.",
                "consequences": {"brand_strength": 0.04, "nps": 3, "team_morale": 0.02, "mau": 500, "investor_confidence": -0.05},
                "narrative": "You miss the milestone. Investors are disappointed but respect the positioning. You write a thoughtful product explainer instead that gets 400 signups organically.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "D", "label": "Let Priya run it as her own opinion — not official brand",
                "description": "Priya posts it from her personal account. Deniable, but morally murky.",
                "consequences": {"mau": 4000, "brand_strength": -0.02, "nps": 0, "budget_remaining": -1000, "team_morale": -0.04},
                "narrative": "It works tactically but the press figures out the connection in 3 days. 'Startup uses contractor as PR puppet' is the next headline. Not great.",
                "intent": "cost", "risk": "high"
            }
        ]
    },

    {
        "id": "glassdoor_crisis",
        "seq": 8,
        "trigger": "below:team_morale:0.5",
        "area": "HR & Employer Brand",
        "narrator_name": "Ananya Sharma",
        "narrator_role": "Head of People",
        "headline": "💣 Ex-Engineer Posted a Scathing Glassdoor Review — Getting Shared in Slack Groups",
        "situation": (
            "A former engineer posted a detailed 800-word Glassdoor review claiming 'CEO gaslights the team, "
            "no real product vision, and HR is a rubber stamp.' It's being shared in three popular Slack "
            "communities for software engineers. Your two current open engineering roles now have zero "
            "applicants after both had 15+ per week before. An investor forwarded it asking 'what's going on?'"
        ),
        "question": "Ananya says this needs a response strategy today. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Post a detailed, honest public response on Glassdoor",
                "description": "Respond publicly. Acknowledge anything valid. Disagree respectfully where untrue.",
                "consequences": {"brand_strength": 0.08, "team_morale": 0.05, "investor_confidence": 0.04, "budget_remaining": -3000},
                "narrative": "Your response is shared in the same communities. Most people find it measured. Two engineers say they'd interview despite the review after reading your reply.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Internal town hall — address it directly with the team",
                "description": "Don't fight publicly. Address internally. What's true? What will change?",
                "consequences": {"team_morale": 0.1, "investor_confidence": 0.02, "budget_remaining": -2000, "brand_strength": 0.03},
                "narrative": "The team is surprised you address it head-on. Two legitimate complaints surface that you act on immediately. Existing team trust goes up. Recruiting stays cold for 2 weeks.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Ask the reviewer to have a private call — hear them out",
                "description": "Reach out directly via LinkedIn. No lawyers. Just a conversation.",
                "consequences": {"brand_strength": 0.05, "team_morale": 0.03, "investor_confidence": 0.0},
                "narrative": "They agree to the call. The conversation is tense but honest. They update their review to be more neutral. The update itself gets positive attention.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "D", "label": "Ignore it — don't amplify by responding",
                "description": "No response. The review cycle continues. Focus energy on showing culture through actions.",
                "consequences": {"brand_strength": -0.06, "team_morale": -0.04, "investor_confidence": -0.05, "budget_remaining": 0},
                "narrative": "More engineers see it. Applications stay at zero for 6 weeks. One investor brings it up in a board call. The silence becomes part of the narrative.",
                "intent": "speed", "risk": "high"
            }
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PHARMA
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["pharma"] = [

    {
        "id": "safety_signal",
        "seq": 1,
        "trigger": "above:trials_passed:0",
        "area": "Clinical & Safety",
        "narrator_name": "Dr. Elena Voss",
        "narrator_role": "Chief Scientific Officer",
        "headline": "🚨 Phase II Data Shows Unexpected Liver Enzyme Elevation in 4% of Patients",
        "situation": (
            "Dr. Voss brings the interim Phase II data. The drug shows strong efficacy — but 4% of patients "
            "show elevated liver enzymes, a potential hepatotoxicity signal. It's below the FDA threshold "
            "for automatic halt, but it's real. You can continue the trial, modify the dosing protocol and "
            "restart, or halt and investigate. Patients in the trial are seeing significant benefit. "
            "Your $1M bond with trial sites depends on completion. The board meets in 3 days."
        ),
        "question": "Dr. Voss needs your directive before the next patient cohort check-in tomorrow morning.",
        "choices": [
            {
                "key": "A", "label": "Halt trial — investigate fully before proceeding",
                "description": "Patient safety first. Halt enrollment. Investigate the mechanism.",
                "consequences": {"safety_signal": -0.2, "fda_relationship": 0.1, "budget_remaining": -200000, "timeline_remaining": -4, "patient_trust": 0.1},
                "narrative": "FDA is informed. The investigation takes 3 months but clears the compound at modified dosing. Patient trust in your process is high. Timeline hit is severe.",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "B", "label": "Modify dosing protocol — reduce dose 20%, continue trial",
                "description": "Lower the dose. Continue with increased monitoring. Brief FDA but don't halt.",
                "consequences": {"safety_signal": -0.1, "drug_potency": -0.08, "fda_relationship": 0.05, "timeline_remaining": -1, "budget_remaining": -80000},
                "narrative": "FDA accepts the protocol modification. Liver markers normalize. Efficacy drops 12% but stays above threshold. The trial continues.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Continue unchanged — 4% is below FDA threshold",
                "description": "Document the signal. Continue monitoring. Report in quarterly data package.",
                "consequences": {"safety_signal": 0.1, "fda_relationship": -0.08, "patient_trust": -0.1, "timeline_remaining": 0, "budget_remaining": 0},
                "narrative": "The trial continues. At week 12, two more patients show elevated markers. FDA issues a clinical hold. You wish you'd acted earlier.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "D", "label": "Bring in independent safety monitoring board",
                "description": "Convene an independent DSMB review within 10 days before any decision.",
                "consequences": {"safety_signal": -0.05, "fda_relationship": 0.12, "patient_trust": 0.08, "budget_remaining": -40000, "timeline_remaining": -1},
                "narrative": "The DSMB recommends modified dosing with enhanced monitoring. FDA receives the report favorably. Your process becomes a case study in responsible trial management.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "researcher_departure",
        "seq": 2,
        "trigger": "step_range:3:10",
        "area": "Team & IP",
        "narrator_name": "Marcus Webb",
        "narrator_role": "VP of R&D",
        "headline": "🧬 Your Lead Researcher Is Leaving — And Wants to Take the Synthesis Method",
        "situation": (
            "Dr. Kim, your lead chemist who developed the core synthesis pathway, has given notice. "
            "She's joining a biotech startup that's willing to pay 2.4x her current salary. "
            "Her exit interview flagged that she considers the synthesis method she developed 'her IP'. "
            "Legal says you have a case under her employment agreement, but it's not airtight. "
            "She trains no one. Without her, Phase III chemistry scale-up is at risk."
        ),
        "question": "You have one week before she's gone. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Counter-offer — match the 2.4x salary",
                "description": "Retain her at any cost. She's irreplaceable for Phase III.",
                "consequences": {"researcher_quality": 0.0, "budget_remaining": -180000, "ip_strength": 0.1, "team_morale": -0.05},
                "narrative": "She stays. She delivers Phase III chemistry. But the retention creates salary compression — two other researchers feel undervalued and signal they're looking.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "B", "label": "Enforce IP agreement — send legal letter",
                "description": "File a formal IP ownership claim. Assert employment agreement. Seek injunction if needed.",
                "consequences": {"ip_strength": 0.15, "researcher_quality": -0.15, "fda_relationship": -0.03, "budget_remaining": -60000, "reputation": -0.05},
                "narrative": "Legal gets you the IP rights. Dr. Kim leaves bitterly. Phase III is delayed 6 months while you recruit replacement chemistry expertise. The case attracts industry press.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "C", "label": "Negotiate IP buyout — pay her fairly for the synthesis method",
                "description": "Acknowledge her contribution with a one-time payment for clean IP transfer.",
                "consequences": {"ip_strength": 0.2, "researcher_quality": -0.08, "budget_remaining": -90000, "reputation": 0.05, "fda_relationship": 0.02},
                "narrative": "You agree on $85K. She does a 3-week knowledge transfer. The IP is clean. The industry respects the approach. Recruiting her replacement is easier because of your reputation.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "D", "label": "Emergency knowledge documentation sprint — extract everything first",
                "description": "Delay her notice period. Pair her with two researchers for intensive documentation.",
                "consequences": {"ip_strength": 0.1, "researcher_quality": -0.05, "budget_remaining": -30000, "timeline_remaining": -0.5},
                "narrative": "Documentation is 70% complete when she leaves. It reduces but doesn't eliminate the knowledge gap. Phase III proceeds with a 3-month delay.",
                "intent": "cost", "risk": "medium"
            }
        ]
    },

    {
        "id": "patient_advocacy_pressure",
        "seq": 3,
        "trigger": "above:trials_passed:1",
        "area": "Patient Relations & Ethics",
        "narrator_name": "Sandra Torres",
        "narrator_role": "Head of Medical Affairs",
        "headline": "🤝 Patient Advocacy Group Demands Compassionate Use Access — Media Watching",
        "situation": (
            "A patient advocacy group representing 200 terminal patients has launched a public campaign "
            "demanding expanded access to your Phase II drug. Three patients have written open letters. "
            "CNN has a camera outside your HQ. The drug shows genuine efficacy. But compassionate use "
            "means uncontrolled real-world data that could contaminate your trial and delay FDA approval — "
            "potentially by a year. Your legal team says there's no obligation to provide access."
        ),
        "question": "Sandra says the story goes national tomorrow regardless. What is your position?",
        "choices": [
            {
                "key": "A", "label": "Grant expanded access to the 200 patients immediately",
                "description": "Do the right thing. Accept the trial data risk. Patients come first.",
                "consequences": {"patient_trust": 0.2, "fda_relationship": 0.05, "media_sentiment": 0.15, "timeline_remaining": -3, "budget_remaining": -150000},
                "narrative": "Your compassionate use program becomes a model. FDA actually expedites review citing the real-world data. It's a net positive — but adds 3 months of data review.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "B", "label": "Enroll them in the trial as an extended cohort",
                "description": "Fast-track 50 of the most critical cases into the trial under compassionate use protocol.",
                "consequences": {"patient_trust": 0.12, "fda_relationship": 0.08, "trials_passed": 0.3, "budget_remaining": -80000, "timeline_remaining": -1},
                "narrative": "A controlled expansion. FDA approves the protocol amendment. 50 patients receive the drug. The advocacy group calls it insufficient but respects the effort.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Deny — protect trial integrity, issue compassionate statement",
                "description": "Decline expanded access. Issue a detailed statement on why trial integrity protects more patients long-term.",
                "consequences": {"patient_trust": -0.15, "media_sentiment": -0.2, "fda_relationship": 0.0, "budget_remaining": -10000, "timeline_remaining": 0},
                "narrative": "The media story is brutal: 'Company Denies Dying Patients Access to Drug'. Congressional inquiry is launched. You're doing the right thing scientifically but it costs you.",
                "intent": "risk", "risk": "high"
            },
            {
                "key": "D", "label": "Partner with advocacy group to design a parallel study",
                "description": "Co-design a 90-day observational parallel access program with the advocacy group.",
                "consequences": {"patient_trust": 0.18, "reputation": 0.1, "media_sentiment": 0.1, "budget_remaining": -100000, "timeline_remaining": -1},
                "narrative": "The advocacy group becomes your champion. Media covers 'company and patients building science together'. FDA finds the parallel data unexpectedly useful.",
                "intent": "quality", "risk": "medium"
            }
        ]
    },

    {
        "id": "bigpharma_offer",
        "seq": 4,
        "trigger": "above:trials_passed:1",
        "area": "Business Development",
        "narrator_name": "Robert Chen",
        "narrator_role": "CFO",
        "headline": "💊 Pfizer Wants to License Your Drug — But They'll Control Pricing",
        "situation": (
            "Robert drops a term sheet from a major pharma company: $50M upfront, $200M in milestones, "
            "and 8% royalties on sales. The deal would eliminate your funding risk entirely. "
            "But the licensing agreement gives them pricing authority — and their drugs typically "
            "launch at 3-5x what smaller biotechs would price. Patient access advocates are already "
            "watching your company. Your lead investor says 'take the money.' Your CSO says 'we lose control.'"
        ),
        "question": "The term sheet expires in 10 days. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Accept — $50M solves everything, their distribution wins",
                "description": "Sign the license. Take the milestone structure. Accept pricing control.",
                "consequences": {"budget_remaining": 45000000, "timeline_remaining": 2, "patient_trust": -0.1, "reputation": -0.05, "fda_relationship": 0.05},
                "narrative": "The money transforms your company. But when the drug launches at $85K/year, patient advocacy groups blast you despite having no control. Your brand takes the hit.",
                "intent": "cost", "risk": "medium"
            },
            {
                "key": "B", "label": "Counter — accept deal but demand pricing cap clause",
                "description": "Sign only if the agreement includes a patient access and affordability clause.",
                "consequences": {"budget_remaining": 35000000, "patient_trust": 0.1, "reputation": 0.1, "timeline_remaining": 1, "fda_relationship": 0.03},
                "narrative": "They reduce the upfront to $35M but accept the pricing cap. Your patient trust score climbs. The deal becomes a case study in ethical partnership.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Reject — go independent to control pricing and access",
                "description": "Decline the offer. Raise your Series B independently. Keep pricing control.",
                "consequences": {"budget_remaining": -200000, "patient_trust": 0.15, "reputation": 0.1, "timeline_remaining": -2, "fda_relationship": 0.0},
                "narrative": "You raise a $30M Series B. The drug launches at $22K/year. Patient access is celebrated. But commercial execution without Big Pharma's network is harder than expected.",
                "intent": "quality", "risk": "high"
            },
            {
                "key": "D", "label": "Counter with co-development — you lead commercial, they fund",
                "description": "Propose a hybrid: they fund Phase III in exchange for co-commercialization with shared pricing authority.",
                "consequences": {"budget_remaining": 20000000, "patient_trust": 0.05, "reputation": 0.05, "timeline_remaining": 0, "fda_relationship": 0.05},
                "narrative": "The negotiation takes 3 weeks. The final deal is more complex but you retain approval rights on list pricing. A genuinely novel partnership structure.",
                "intent": "balance", "risk": "medium"
            }
        ]
    },

    {
        "id": "fda_inspection",
        "seq": 5,
        "trigger": "above:compliance_score:0.5",
        "area": "Regulatory & Compliance",
        "narrator_name": "Dr. Lisa Park",
        "narrator_role": "Head of Regulatory Affairs",
        "headline": "📋 FDA Unannounced Inspection — They Found 3 Documentation Gaps",
        "situation": (
            "An FDA inspector showed up at your facility Monday. The 2-day inspection uncovered three "
            "documentation issues: missing batch record signatures from 4 months ago, a temperature "
            "excursion in storage that was logged but not formally investigated, and a training record "
            "discrepancy for two lab technicians. None are critical, but a Form 483 is coming. "
            "Your NDA submission is planned for 8 weeks from now. This could delay it."
        ),
        "question": "Dr. Park says the FDA gives you 15 days to respond to the 483 observations. What's your plan?",
        "choices": [
            {
                "key": "A", "label": "CAPA response in 15 days — fix all three fully",
                "description": "Drop other priorities. Complete Corrective and Preventive Actions for all three findings in 15 days.",
                "consequences": {"compliance_score": 0.15, "fda_relationship": 0.12, "timeline_remaining": -1, "budget_remaining": -40000},
                "narrative": "FDA acknowledges your thorough response. The 483 is closed with no further action. NDA timeline delayed only 3 weeks. Good outcome given the circumstances.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Respond to 2 of 3 — flag the third as under investigation",
                "description": "Close two findings cleanly. Buy time on the batch record issue, which needs deeper root cause.",
                "consequences": {"compliance_score": 0.08, "fda_relationship": 0.04, "timeline_remaining": -2, "budget_remaining": -25000},
                "narrative": "FDA accepts two closures. The third requires a follow-up meeting 4 weeks later. Additional 4-week delay. Partial credit.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Request a Type A FDA meeting before responding",
                "description": "Request a formal meeting to discuss the observations before submitting your written response.",
                "consequences": {"compliance_score": 0.1, "fda_relationship": 0.15, "timeline_remaining": -2, "budget_remaining": -30000},
                "narrative": "FDA respects the proactive request. The meeting clarifies that two findings are minor. Your response is sharper and closes all three in one cycle.",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "D", "label": "Delay the NDA — fully audit entire facility first",
                "description": "Voluntarily push NDA back 8 weeks. Do a full internal audit to prevent future 483s.",
                "consequences": {"compliance_score": 0.2, "fda_relationship": 0.1, "timeline_remaining": -6, "budget_remaining": -100000},
                "narrative": "The audit finds two more minor issues that would have surfaced later. You address them all. FDA reviewers note the clean submission quality at NDA review.",
                "intent": "risk", "risk": "medium"
            }
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# INTERIOR DESIGN
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["interior_design"] = [

    {
        "id": "scope_creep",
        "seq": 1,
        "trigger": "above:design_progress:0.2",
        "area": "Client Management",
        "narrator_name": "Isabella Reeves",
        "narrator_role": "Senior Project Manager",
        "headline": "🏠 Client Wants to Add a Home Theatre — Mid-Project",
        "situation": (
            "The Garcias are thrilled with progress — so thrilled they want to expand the scope. "
            "A 400 sq ft home theatre with acoustic panels, a 4K projection system, and custom "
            "millwork. Isabella estimates $180K and 8 additional weeks. The project is currently "
            "2 weeks ahead of schedule. Your main contractor has window availability — but only for "
            "the next 3 weeks. Mrs. Garcia says 'money isn't an issue.' Mr. Garcia hasn't weighed in."
        ),
        "question": "Isabella wants your decision before the contractor availability closes. What do you tell the client?",
        "choices": [
            {
                "key": "A", "label": "Accept it — clearly define scope, price, and timeline in writing",
                "description": "Take the additional scope with a signed amendment. Clear documentation of everything.",
                "consequences": {"budget_remaining": 150000, "timeline_remaining": -6, "client_satisfaction": 0.1, "quality_score": 0.05, "team_morale": -0.05},
                "narrative": "Amendment signed. Mr. Garcia sees the invoice and gets cold feet about the cost — even though his wife said yes. You're in the middle of a couple's disagreement with a contract.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "B", "label": "Propose it as a separate Phase 2 project after handover",
                "description": "Protect the current project's quality. Do the theatre as a clean follow-on.",
                "consequences": {"budget_remaining": 0, "client_satisfaction": 0.06, "quality_score": 0.05, "team_morale": 0.05, "timeline_remaining": 0},
                "narrative": "Clients appreciate the honest advice. You finish the main project at the highest quality. The theatre is contracted as Phase 2 — your next $180K project.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "C", "label": "Accept but bring in a specialist sub-contractor",
                "description": "Say yes — but let a dedicated AV and acoustics firm handle the theatre portion.",
                "consequences": {"budget_remaining": 80000, "client_satisfaction": 0.08, "quality_score": 0.04, "timeline_remaining": -4, "team_morale": 0.0},
                "narrative": "The specialist delivers a stunning theatre. Main project quality stays high. Coordination overhead is real but manageable. Both Garcias are delighted.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Get both Garcias together to align before answering",
                "description": "Schedule a joint meeting before committing. Ensure both clients are aligned on cost and timeline.",
                "consequences": {"client_satisfaction": 0.12, "quality_score": 0.03, "budget_remaining": -2000, "timeline_remaining": -0.5},
                "narrative": "The meeting reveals Mr. Garcia is fine with the theatre but wants to redesign one other room instead. You get a cleaner scope from an aligned client pair.",
                "intent": "risk", "risk": "low"
            }
        ]
    },

    {
        "id": "lead_designer_quit",
        "seq": 2,
        "trigger": "above:design_progress:0.4",
        "area": "Team & Talent",
        "narrator_name": "Tom Rivera",
        "narrator_role": "Studio Director",
        "headline": "😱 Your Lead Designer Resigned — With 60% of the Villa Unfinished",
        "situation": (
            "Marco Vitali — your lead designer on the villa project, responsible for the custom furniture "
            "specifications and the client relationship — resigned this morning. He's joining a competitor "
            "studio. He leaves in 2 weeks. The villa is 60% done. Marco owns the design intent files "
            "and has the trusted relationship with Mrs. Garcia who calls him personally. "
            "The remaining 40% includes all the custom joinery, the master suite, and the terrace."
        ),
        "question": "Tom needs a transition plan by end of day. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Promote the junior designer immediately with your oversight",
                "description": "Elevate Sophia from the team. You personally oversee weekly. Retain continuity.",
                "consequences": {"quality_score": -0.06, "client_satisfaction": -0.05, "team_morale": 0.1, "budget_remaining": -10000, "timeline_remaining": -1},
                "narrative": "Sophia is talented but not ready for master-suite joinery decisions alone. You catch three specification errors before they're ordered. Quality holds — barely.",
                "intent": "cost", "risk": "medium"
            },
            {
                "key": "B", "label": "Hire senior freelance designer immediately on contract",
                "description": "Bring in a senior freelancer with villa experience. 2-week handover with Marco.",
                "consequences": {"quality_score": 0.03, "client_satisfaction": 0.0, "team_morale": -0.03, "budget_remaining": -35000, "timeline_remaining": -1},
                "narrative": "The freelancer is technically excellent but takes 3 weeks to match Marco's understanding of the client. Mrs. Garcia notices the change in design voice. She's cautiously okay with it.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "C", "label": "Counter-offer Marco — studio equity stake to stay",
                "description": "Offer Marco a small equity stake in the studio in exchange for seeing this project through.",
                "consequences": {"quality_score": 0.1, "client_satisfaction": 0.1, "team_morale": 0.05, "budget_remaining": -20000},
                "narrative": "Marco accepts a 3% studio stake. He finishes the villa — his best work ever. Mrs. Garcia throws you a referral dinner party. Two new projects come from it.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Tell the client honestly — they deserve to know",
                "description": "Call Mrs. Garcia before she finds out another way. Present your transition plan at the same time.",
                "consequences": {"client_satisfaction": 0.08, "reputation": 0.08, "quality_score": 0.0, "budget_remaining": -5000},
                "narrative": "Mrs. Garcia appreciates the honesty deeply. She says 'Marco told me you run the best studio he's worked at — I trust this will be fine.' Trust maintained.",
                "intent": "risk", "risk": "low"
            }
        ]
    },

    {
        "id": "structural_discovery",
        "seq": 3,
        "trigger": "above:design_progress:0.3",
        "area": "Construction & Operations",
        "narrator_name": "Carlos Mendez",
        "narrator_role": "Head Contractor",
        "headline": "🏚️ Contractor Found Load-Bearing Wall Where the Open Plan Should Go",
        "situation": (
            "Carlos discovered that the wall separating the kitchen and living room — the one your entire "
            "open-plan design depends on — is load-bearing. An engineer is needed to assess options. "
            "Removing it is possible but requires a $35K steel beam installation and pushes the timeline "
            "back 3 weeks. The alternative is redesigning the open-plan concept entirely. "
            "The client's favourite feature of the entire villa concept is that open-plan flow."
        ),
        "question": "Carlos needs a directive this afternoon to avoid idle crew time tomorrow. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Install the steel beam — preserve the design",
                "description": "Absorb the $35K and delay. The open plan is non-negotiable for the client.",
                "consequences": {"budget_remaining": -35000, "timeline_remaining": -2, "client_satisfaction": 0.1, "quality_score": 0.05},
                "narrative": "The beam goes in. Three weeks lost. But the open-plan flows perfectly. At final reveal Mrs. Garcia says it's her favourite room in the world.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Redesign — create a partial-open design with columns",
                "description": "Reframe the constraint as a design opportunity. Structural columns become a feature.",
                "consequences": {"budget_remaining": -10000, "timeline_remaining": -0.5, "client_satisfaction": -0.05, "quality_score": 0.04, "reputation": 0.05},
                "narrative": "Your team designs a colonnade that echoes the villa's architecture. The client is initially disappointed — then genuinely loves the unexpected result. It gets published.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Present both options to the client with full cost/time impact",
                "description": "Don't decide alone. Show both paths with full transparency. Let them choose.",
                "consequences": {"client_satisfaction": 0.12, "quality_score": 0.02, "budget_remaining": -2000, "timeline_remaining": -0.5},
                "narrative": "The Garcias choose the beam without hesitation. Mrs. Garcia says 'please don't try to protect us from cost information — we trust you more when you're direct.'",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "D", "label": "Bring in a structural engineer today for creative options",
                "description": "Don't rush to either answer. Get an engineer in tomorrow for a third-option assessment.",
                "consequences": {"budget_remaining": -8000, "timeline_remaining": -0.5, "quality_score": 0.06, "client_satisfaction": 0.06},
                "narrative": "The engineer finds a cantilever solution that costs only $18K and adds 2 weeks. Better than either option you had. Worth the extra day to explore.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "instagram_complaint",
        "seq": 4,
        "trigger": "above:design_progress:0.5",
        "area": "Social Media & Reputation",
        "narrator_name": "Nina Osei",
        "narrator_role": "Studio Marketing Lead",
        "headline": "📱 Client's Daughter is Live-Posting 'Construction Disaster' on Instagram — 40K Followers",
        "situation": (
            "The clients' 23-year-old daughter Zoe has 40K Instagram followers and has been posting "
            "daily stories calling the renovation a 'construction disaster'. She posted a video of "
            "dusty floors and construction noise with the caption 'thanks for ruining our house'. "
            "Three of your studio's prospects have messaged asking if the stories are real. "
            "The Garcias are embarrassed but haven't asked you to address Zoe directly."
        ),
        "question": "Nina wants a communication strategy today. How do you handle Zoe and the client?",
        "choices": [
            {
                "key": "A", "label": "Call Mrs. Garcia — ask her to handle Zoe privately",
                "description": "This is a family matter. Keep it out of professional channels.",
                "consequences": {"client_satisfaction": 0.0, "reputation": -0.03, "brand_strength": -0.03},
                "narrative": "Mrs. Garcia has the conversation. Zoe goes quiet online. But Zoe tells three friends 'the designer went crying to my mom'. A different kind of reputation.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "B", "label": "Invite Zoe to a site tour — make her part of the story",
                "description": "Reach out to Zoe directly. Offer a behind-the-scenes tour with her documenting it.",
                "consequences": {"brand_strength": 0.12, "client_satisfaction": 0.1, "reputation": 0.08, "budget_remaining": -3000},
                "narrative": "Zoe's tour posts get more engagement than her complaints. She posts a reel calling the project 'actually insane — in a good way'. Your studio gets 800 new followers.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "C", "label": "Post a professional response thread showing project progress",
                "description": "Counter-narrative. Post your own project documentation with context and timelines.",
                "consequences": {"brand_strength": 0.06, "reputation": 0.04, "budget_remaining": -5000, "client_satisfaction": 0.04},
                "narrative": "Your posts get good engagement. Prospects see both sides and call you anyway. Three say 'your calm response sold me more than the portfolio.'",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "D", "label": "Do nothing — Zoe's audience isn't your client base",
                "description": "Stay professional. Don't engage with a 23-year-old's opinion on social media.",
                "consequences": {"brand_strength": -0.06, "reputation": -0.04, "client_satisfaction": -0.03},
                "narrative": "One of the posts gets picked up by a local home design blog. The story runs as 'luxury reno gone wrong?'. You spend two weeks doing damage control.",
                "intent": "speed", "risk": "high"
            }
        ]
    },

    {
        "id": "worker_injury",
        "seq": 5,
        "trigger": "step_gte:4",
        "area": "Safety & Legal",
        "narrator_name": "Carlos Mendez",
        "narrator_role": "Head Contractor",
        "headline": "🚑 Worker Fell from Scaffolding — Broken Wrist, Threatening Legal Action",
        "situation": (
            "A scaffolding installer slipped and broke his wrist this afternoon. He's at hospital, "
            "stable condition. Carlos confirms the scaffolding met code requirements. The worker "
            "says he wasn't given proper grip gloves. You carry liability insurance with a $50K "
            "deductible. The worker's friend is already calling it 'negligence' on a WhatsApp group "
            "your client's daughter is in. Your insurance adjuster wants a call in 2 hours."
        ),
        "question": "This is happening now. What are your first three moves?",
        "choices": [
            {
                "key": "A", "label": "Insurance first — document everything, let them lead",
                "description": "Follow your insurer's process exactly. Say nothing to the worker until legal advises.",
                "consequences": {"budget_remaining": -50000, "reputation": -0.05, "safety_record": -0.1, "client_satisfaction": -0.03},
                "narrative": "Process works but the worker's family feels stonewalled. Local press runs a brief. Your client finds out from the news, not from you.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "B", "label": "Visit the hospital personally — human first, legal second",
                "description": "Go to the hospital with a human gesture. Legal processes follow, but the person is priority one.",
                "consequences": {"reputation": 0.08, "safety_record": -0.05, "budget_remaining": -55000, "client_satisfaction": 0.02, "team_morale": 0.1},
                "narrative": "The worker's family is moved. The legal team is initially alarmed, but the visit defuses the situation. The worker ultimately doesn't pursue action. Goodwill wins.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "C", "label": "Call your client immediately to tell them before they find out",
                "description": "Proactive transparency with the Garcias. Tell them what happened, what you're doing.",
                "consequences": {"client_satisfaction": 0.08, "reputation": 0.05, "budget_remaining": -50000, "safety_record": -0.08},
                "narrative": "The Garcias are upset but respect the call. Mrs. Garcia says 'I appreciate you not hiding this.' The project continues. The worker is fairly compensated.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "D", "label": "Full site safety audit — halt work for 48 hours",
                "description": "Stop the site. Full safety re-inspection. Document compliance on all equipment before restarting.",
                "consequences": {"safety_record": 0.15, "reputation": 0.08, "budget_remaining": -20000, "timeline_remaining": -1, "team_morale": 0.05},
                "narrative": "The audit finds two other equipment issues. You fix them. The Garcias appreciate the visible commitment to safety. No legal action is taken.",
                "intent": "quality", "risk": "low"
            }
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MANUFACTURING
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["manufacturing"] = [

    {
        "id": "quality_recall",
        "seq": 1,
        "trigger": "above:production_rate:0.4",
        "area": "Quality & Customer",
        "narrator_name": "Janet Wu",
        "narrator_role": "VP of Quality Assurance",
        "headline": "⚠️ 340 Shipped Units Have a Defect — Customer is Asking Questions",
        "situation": (
            "Janet found a batch defect during routine QA review. The affected units are already at 12 "
            "customer sites across 3 states. The defect — a faulty seal that can cause fluid leakage — "
            "is not dangerous under normal conditions but violates spec. Two customers reported issues "
            "before you found the pattern. Your biggest client has 80 of the 340 units. "
            "Replacing them costs $320K and takes 2 weeks. Staying silent costs nothing today."
        ),
        "question": "Janet needs your direction before the end of day today. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Proactive recall — contact all 12 customers today",
                "description": "Reach out to all affected customers before they discover it. Own it completely.",
                "consequences": {"customer_satisfaction": 0.15, "reputation": 0.12, "defect_rate": -0.01, "budget_remaining": -320000, "quality_score": 0.08},
                "narrative": "Customers are initially frustrated but respond with significant respect for the transparency. Your biggest client says 'this is why we stayed with you'. No contract loss.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Quietly fix units that have complaints — don't proactively announce",
                "description": "Respond to complaints as they come in. Replace on request only.",
                "consequences": {"customer_satisfaction": -0.1, "reputation": -0.15, "defect_rate": 0.0, "budget_remaining": -80000},
                "narrative": "Two more customers report issues. One posts on an industry forum. The pattern goes public. Now it's a cover-up story, not a defect story.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "C", "label": "Replace the biggest client's 80 units now — handle others as needed",
                "description": "Protect the relationship that matters most. Manage others reactively.",
                "consequences": {"customer_satisfaction": 0.04, "reputation": -0.04, "budget_remaining": -90000, "quality_score": 0.0},
                "narrative": "Your biggest client is satisfied. But a smaller client finds out you prioritized by account value. They leave and post about it publicly.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Halt line, root cause analysis, replace all units within 2 weeks",
                "description": "Stop production. Find the root cause. Full systematic replacement with RCA report to customers.",
                "consequences": {"quality_score": 0.12, "customer_satisfaction": 0.18, "reputation": 0.1, "budget_remaining": -420000, "production_rate": -0.05, "timeline_remaining": -1},
                "narrative": "The RCA finds a supplier sub-component issue. You change suppliers. The systematic response earns two new customer references. A quality crisis becomes a quality story.",
                "intent": "quality", "risk": "medium"
            }
        ]
    },

    {
        "id": "union_strike",
        "seq": 2,
        "trigger": "below:worker_satisfaction:0.5",
        "area": "Labor & HR",
        "narrator_name": "Robert Osei",
        "narrator_role": "Head of HR",
        "headline": "✊ Workers Threatening Strike — Production Stops in 72 Hours",
        "situation": (
            "The union representing your 80 floor workers has issued a 72-hour strike notice. "
            "Their demands: 12% wage increase, improved break room facilities, and mandatory safety "
            "training refresh every 6 months instead of yearly. You have $250K worth of customer orders "
            "due for shipment next week. Management's position has been to offer 5% with improved facilities. "
            "The union rep says 5% is 'insulting' given last quarter's profitability announcement."
        ),
        "question": "Robert says the union meets again in 24 hours. What do you offer?",
        "choices": [
            {
                "key": "A", "label": "Meet their demands — 12%, facilities, 6-month training",
                "description": "Give them what they asked for. Preserve the relationship and hit your orders.",
                "consequences": {"worker_satisfaction": 0.2, "budget_remaining": -180000, "production_rate": 0.05, "reputation": 0.06, "customer_satisfaction": 0.08},
                "narrative": "Strike averted. Workers feel respected. Productivity goes up 8% post-settlement — the data surprises management. $250K orders ship on time.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Counter at 8% + facilities — hold firm on training schedule",
                "description": "Move significantly from 5% but don't concede everything. Negotiate.",
                "consequences": {"worker_satisfaction": 0.1, "budget_remaining": -120000, "production_rate": 0.0, "timeline_remaining": -1},
                "narrative": "The union accepts after 18 hours of back-and-forth. $250K orders delayed one week. Customer understanding is mixed — one docks your performance rating.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Hold at 5% — bring in temp labor if they strike",
                "description": "Don't negotiate under threat. If they strike, use temporary workers.",
                "consequences": {"worker_satisfaction": -0.2, "budget_remaining": -40000, "production_rate": -0.15, "reputation": -0.1, "customer_satisfaction": -0.1},
                "narrative": "Strike happens. Temp labor delivers 40% of normal output. Customers are angry. Local press covers it. The strike ends after 2 weeks but relations are permanently damaged.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "D", "label": "Invite workers to open books — show them the actual financials",
                "description": "Radical transparency. Share the P&L with the union rep before making any offer.",
                "consequences": {"worker_satisfaction": 0.12, "budget_remaining": -140000, "reputation": 0.08, "production_rate": 0.03},
                "narrative": "The union rep is disarmed by the transparency. Negotiations shift from adversarial to collaborative. You agree on 9% with a performance bonus tied to profitability.",
                "intent": "risk", "risk": "medium"
            }
        ]
    },

    {
        "id": "automation_decision",
        "seq": 3,
        "trigger": "above:production_rate:0.5",
        "area": "Technology & Workforce",
        "narrator_name": "Priya Shah",
        "narrator_role": "Head of Engineering",
        "headline": "🤖 AI Automation Could Replace 20 Workers — and Double Throughput",
        "situation": (
            "Priya presents a robotics vendor proposal: an automated line system that would double "
            "production throughput, reduce defect rate by 60%, and pay back in 14 months. "
            "The system replaces 20 of your 100 floor workers. Your union contract has a no-automation "
            "clause that expires in 5 months. The vendor says the price increases 30% after Q2 due to "
            "demand. This decision cannot wait — but 20 jobs hang in the balance."
        ),
        "question": "The board wants a recommendation this week. What do you propose?",
        "choices": [
            {
                "key": "A", "label": "Buy the system now — redeploy workers to QA and logistics",
                "description": "Automation is inevitable. Invest now. Create a retraining program for displaced workers.",
                "consequences": {"production_rate": 0.2, "defect_rate": -0.01, "automation_level": 0.3, "budget_remaining": -800000, "worker_satisfaction": -0.1, "reputation": -0.03},
                "narrative": "Throughput doubles. 12 workers are retrained successfully. 8 leave voluntarily. Local newspaper runs a mixed story. The union files a grievance but the contract has expired.",
                "intent": "speed", "risk": "medium"
            },
            {
                "key": "B", "label": "Wait 5 months for union contract expiry — then move cleanly",
                "description": "Respect the contract. Negotiate worker retraining into the new agreement.",
                "consequences": {"production_rate": 0.0, "automation_level": 0.0, "budget_remaining": -20000, "worker_satisfaction": 0.08, "reputation": 0.06},
                "narrative": "You wait. Contract expires. You negotiate a retraining package with the union before announcing automation. 18 workers choose retraining. 2 take severance. Smooth transition.",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "C", "label": "Propose partial automation — 10 robots, 10 jobs protected",
                "description": "Phase 1: automate only the highest-risk tasks. Protect jobs explicitly.",
                "consequences": {"production_rate": 0.1, "defect_rate": -0.005, "automation_level": 0.15, "budget_remaining": -450000, "worker_satisfaction": 0.05},
                "narrative": "Workers see you not automating everything and trust goes up. Productivity improves. Full automation becomes easier 18 months later when workers have seen it work.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "D", "label": "Co-design with workers — let them propose where automation helps",
                "description": "Hold workshops with floor workers to identify which tasks they'd want automated first.",
                "consequences": {"worker_satisfaction": 0.15, "automation_level": 0.1, "budget_remaining": -300000, "production_rate": 0.08, "reputation": 0.1},
                "narrative": "Workers identify 5 tasks they hate doing. Automating those first earns their trust. The union endorses the phased plan. This process is written about in a manufacturing journal.",
                "intent": "quality", "risk": "low"
            }
        ]
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# FINANCE
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["finance"] = [

    {
        "id": "black_swan",
        "seq": 1,
        "trigger": "step_range:1:5",
        "area": "Market Risk",
        "narrator_name": "Sarah Chen",
        "narrator_role": "Head of Risk Management",
        "headline": "📉 Market Down 11% in One Day — Clients Calling, Portfolio Bleeding",
        "situation": (
            "A surprise central bank announcement triggered a 11% single-day market drop. "
            "Your portfolio is down $110K on paper. Three clients have called demanding you sell everything. "
            "Your quant model signals this is a temporary overreaction — historically, 87% of similar "
            "drops recover within 6 weeks. But it's your clients' money and their emotional state right now "
            "is what's real. Two of them threaten to withdraw funds if you don't act today."
        ),
        "question": "Sarah needs a portfolio directive and a client communication plan within the hour.",
        "choices": [
            {
                "key": "A", "label": "Hold and send a calm data-driven letter to all clients",
                "description": "Don't sell. Send an evidence-based communication with historical context.",
                "consequences": {"portfolio_value": 30000, "client_trust": 0.08, "return_ytd": 0.03, "lp_satisfaction": 0.05},
                "narrative": "Two clients withdraw $50K. The portfolio recovers 9% over 6 weeks. The clients who stayed make 7% on their full position. Your model was right.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "B", "label": "Sell 30% and rebuy once dust settles",
                "description": "Partially de-risk. Protect capital. Redeploy after initial panic passes.",
                "consequences": {"portfolio_value": -20000, "client_trust": 0.04, "return_ytd": -0.02, "lp_satisfaction": 0.02},
                "narrative": "You lock in some losses. The reentry point is 4% higher than your sell price. Clients feel protected even though mathematically you left return on the table.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Buy more — add to positions on the dip",
                "description": "Lean into the quant signal. Buy the dip across core positions.",
                "consequences": {"portfolio_value": 80000, "client_trust": -0.05, "return_ytd": 0.08, "lp_satisfaction": -0.05},
                "narrative": "The market recovers and your returns are exceptional. But two clients who didn't expect you to buy into a crash feel you acted without mandate.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "D", "label": "Call every client personally before making any move",
                "description": "No portfolio action until you've spoken directly to each client about their risk tolerance.",
                "consequences": {"portfolio_value": -5000, "client_trust": 0.15, "return_ytd": -0.01, "lp_satisfaction": 0.12},
                "narrative": "Clients feel heard. Three change their instructions during the call. You trade in line with each mandate. Lower return but highest trust rating of the year.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "rogue_trader",
        "seq": 2,
        "trigger": "step_range:3:8",
        "area": "Ethics & Compliance",
        "narrator_name": "Marcus Webb",
        "narrator_role": "Head of Compliance",
        "headline": "🕵️ Top Trader Exceeded Risk Limits — But Made You $80K",
        "situation": (
            "Marcus caught David, your best-performing trader, exceeding his position limits by 3x "
            "on a tech play last Thursday. The trade made $80K. David says he 'had high conviction' "
            "and the rules 'didn't contemplate this kind of opportunity'. Marcus wants him terminated. "
            "David generates 40% of your firm's returns. Three other junior traders are watching "
            "how you handle this very closely."
        ),
        "question": "Marcus is in your office. David is waiting for your call. What is your decision?",
        "choices": [
            {
                "key": "A", "label": "Terminate David immediately — the rules exist for everyone",
                "description": "Zero tolerance. David is fired. The $80K is kept as a windfall but the precedent is set.",
                "consequences": {"regulatory_compliance": 0.15, "team_quality": -0.15, "return_ytd": -0.05, "lp_satisfaction": -0.04, "media_sentiment": 0.0},
                "narrative": "David leaves. The junior traders are shocked but impressed. Two months later one tells you 'we all trust the rules now.' Your risk framework is tighter.",
                "intent": "risk", "risk": "low"
            },
            {
                "key": "B", "label": "Suspend David for 30 days, ban the specific strategy, keep him",
                "description": "Penalize but retain. Close the rule gap. Make an example without losing the talent.",
                "consequences": {"regulatory_compliance": 0.05, "team_quality": -0.02, "return_ytd": 0.03, "lp_satisfaction": 0.0},
                "narrative": "David returns chastened. The junior traders read it as 'results shield you'. Two of them take slightly bigger risks six months later.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Reward the $80K result — raise his risk limit retroactively",
                "description": "He was right. Reward results, adjust the framework, don't punish winning.",
                "consequences": {"regulatory_compliance": -0.15, "team_quality": 0.02, "return_ytd": 0.04, "lp_satisfaction": -0.08},
                "narrative": "Two months later an LP asks about risk controls. Marcus shows the incident. The LP pulls $500K over 'culture of rule bending'. A $480K net loss.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "D", "label": "Let David present his case to the full risk committee",
                "description": "Transparent due process. Committee decides — not just you. Sets a precedent for how cases are handled.",
                "consequences": {"regulatory_compliance": 0.08, "team_quality": 0.0, "return_ytd": 0.0, "lp_satisfaction": 0.05},
                "narrative": "Committee votes 3-1 for a formal warning. The process itself becomes the lesson. Your governance model is cited in a LP due diligence as 'institutional grade'.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "esg_pressure",
        "seq": 3,
        "trigger": "above:return_ytd:0.05",
        "area": "ESG & Investor Relations",
        "narrator_name": "Rebecca Liu",
        "narrator_role": "Head of Investor Relations",
        "headline": "🌱 Institutional LP Demands You Divest 'Unethical' Holdings or Pull $2M",
        "situation": (
            "Rebecca brings you a letter from your second-largest LP — a university endowment with "
            "$2M invested. They want you to divest three holdings they label as ESG-incompatible: "
            "a defense contractor (7% of portfolio), a palm oil company (3%), and a private prison REIT (2%). "
            "Together they account for 12% of portfolio and have been outperforming this quarter by 14%. "
            "Divesting now means locking in gains but exiting strong momentum. Other LPs haven't raised ESG."
        ),
        "question": "Rebecca says the endowment wants a written commitment within 10 days. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Divest all three — client mandate overrides performance",
                "description": "Clean exit from all three holdings. Acknowledge ESG as a portfolio principle going forward.",
                "consequences": {"esg_score": 0.2, "lp_satisfaction": 0.1, "return_ytd": -0.03, "portfolio_value": -30000},
                "narrative": "The endowment is satisfied. Word spreads in foundation/endowment circles. You receive two new LP inquiries from university funds within 2 months.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Divest the prison REIT only — draw the line at other holdings",
                "description": "Compromise. Exit the most controversial holding. Stand by the others on financial grounds.",
                "consequences": {"esg_score": 0.08, "lp_satisfaction": 0.0, "return_ytd": -0.01, "portfolio_value": -10000},
                "narrative": "The endowment calls it 'insufficient'. They reduce to $500K but don't fully exit. Two other LPs are watching and stay neutral.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Hold firm — your mandate is financial return, not ESG",
                "description": "Decline the request. Offer to create an ESG-screened sub-portfolio if they want one.",
                "consequences": {"esg_score": -0.05, "lp_satisfaction": -0.12, "return_ytd": 0.02, "portfolio_value": 20000},
                "narrative": "The endowment withdraws $2M. The holdings continue to outperform for 8 more weeks then revert. You're up financially short-term but your LP base is now less diversified.",
                "intent": "cost", "risk": "medium"
            },
            {
                "key": "D", "label": "Create a formal ESG policy and present it to all LPs",
                "description": "Turn one LP's request into a firm-wide policy discussion. Let all investors weigh in together.",
                "consequences": {"esg_score": 0.12, "lp_satisfaction": 0.08, "return_ytd": -0.01, "portfolio_value": -15000},
                "narrative": "All LPs receive the policy questionnaire. 60% support ESG screening. You transition the portfolio over one quarter. You position your fund as ESG-aware without being ESG-constrained.",
                "intent": "risk", "risk": "low"
            }
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# E-COMMERCE
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["ecommerce"] = [

    {
        "id": "influencer_disaster",
        "seq": 1,
        "trigger": "step_range:1:5",
        "area": "Marketing & PR",
        "narrator_name": "Mei Lin",
        "narrator_role": "Head of Marketing",
        "headline": "📦 Mega-Influencer's Unboxing Video: 'Worst Product I've Ever Received'",
        "situation": (
            "A beauty influencer with 2.1M followers received a damaged unit and posted a 8-minute "
            "'honest review' that's already at 890K views. The thumbnail is your product next to the word "
            "'SCAM'. The comments are brutal. Your conversion rate dropped 18% in 6 hours. "
            "Three other influencers have shared it. Mei tracked down the issue: the unit was damaged "
            "in shipping, not a product defect. But the influencer doesn't know that yet."
        ),
        "question": "The video is 8 hours old. Mei needs a response strategy immediately.",
        "choices": [
            {
                "key": "A", "label": "DM the influencer with full evidence — shipping damage proof",
                "description": "Send photos, shipping records, proof of product quality. Ask her to verify.",
                "consequences": {"brand_strength": 0.1, "social_media_sentiment": 0.12, "conversion_rate": 0.005, "budget_remaining": -5000},
                "narrative": "She pins a follow-up video: 'The Brand Reached Out and I Was Wrong.' 1.4M views. Net positive for your brand. Conversion recovers fully in 5 days.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Send a new product + full refund + a handwritten apology",
                "description": "No argument. Full service recovery. Make her a brand advocate.",
                "consequences": {"brand_strength": 0.08, "social_media_sentiment": 0.08, "budget_remaining": -800, "customer_nps": 4},
                "narrative": "She posts an unboxing of the new product calling it 'genuinely impressive customer service'. 620K views. Three competitor brands DM you asking how you handled it.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Post a detailed rebuttal video from the CEO",
                "description": "Go public with your counter-evidence. Direct and factual response.",
                "consequences": {"brand_strength": -0.04, "social_media_sentiment": -0.08, "conversion_rate": -0.003, "budget_remaining": -8000},
                "narrative": "The internet sides with the influencer. 'Brand attacks creator' is the new headline. Your CEO video gets 200K views — mostly negative.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "D", "label": "Pause all paid traffic — let the storm pass organically",
                "description": "Cut ad spend. Don't amplify. Trust that 2.1M followers will forget in a week.",
                "consequences": {"brand_strength": -0.06, "revenue": -8000, "conversion_rate": -0.004, "budget_remaining": 15000},
                "narrative": "The video keeps circulating for 12 days. You save on ad spend but organic growth stalls. Marketplace rating drops 0.3 stars.",
                "intent": "cost", "risk": "medium"
            }
        ]
    },

    {
        "id": "demand_spike",
        "seq": 2,
        "trigger": "above:demand_index:0.5",
        "area": "Operations & Inventory",
        "narrator_name": "Felix Oduya",
        "narrator_role": "Head of Operations",
        "headline": "📈 TikTok Trend Makes Your Product Go Viral — 10x Normal Orders Overnight",
        "situation": (
            "A TikTok trend using your product went viral at 11 PM. By 8 AM you have 4,200 orders — "
            "your usual weekly volume is 420. Felix's nightmare: you have 800 units in stock. "
            "Supplier leadtime is 3 weeks minimum. Demand may spike further or die by tomorrow. "
            "You can backorder (commit to 3-week delivery), cancel excess orders, or rush-source from "
            "a secondary supplier at 3x unit cost. Your return rate spikes when customers wait too long."
        ),
        "question": "Felix needs a decision in the next 2 hours before more orders come in. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Backorder everything — ride the full wave at normal margin",
                "description": "Accept all orders. Be transparent about 3-week delivery. Bank on demand holding.",
                "consequences": {"revenue": 180000, "customer_nps": -8, "return_rate": 0.04, "inventory_level": -800, "brand_strength": -0.04},
                "narrative": "Orders flood in. But 38% cancel after 10 days of waiting. Chargebacks hit hard. The TikTok customers are impulse buyers — they don't wait.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "B", "label": "Cap orders at 800 — close the store when stock is gone",
                "description": "Sell what you have. Close orders. Reopen when restocked.",
                "consequences": {"revenue": 35000, "customer_nps": 6, "brand_strength": 0.08, "inventory_level": -800},
                "narrative": "You sell out in 4 hours. FOMO posts appear: 'sold out already'. Second wave of interest when you restock. Scarcity created genuine brand heat.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "C", "label": "Rush-source from secondary supplier at 3x cost for 2K units",
                "description": "Pay the premium. Source 2,000 extra units. Ship in 5 days.",
                "consequences": {"revenue": 90000, "customer_nps": 5, "budget_remaining": -60000, "inventory_level": 1200},
                "narrative": "You ship 2,800 orders in 5 days. Customers are impressed. Margin on the 2K rush units is thin. Net profit is lower but NPS holds and brand gains.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Launch waitlist with email capture + 10% discount for patience",
                "description": "Convert the spike into an email list. Offer a discount to committed waitlisters.",
                "consequences": {"revenue": 15000, "customer_nps": 4, "brand_strength": 0.1, "demand_index": 0.1, "budget_remaining": -5000},
                "narrative": "3,200 waitlist signups. When you restock, 64% convert. Better margin than rush-sourcing. Email list becomes your top-performing marketing channel.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "pricing_war",
        "seq": 3,
        "trigger": "above:revenue:20000",
        "area": "Competitive Strategy",
        "narrator_name": "Mei Lin",
        "narrator_role": "Head of Marketing",
        "headline": "💲 Competitor Drops Price 40% — Your Conversion Rate Fell 25% This Week",
        "situation": (
            "Your main competitor launched at 40% below your price point. They're VC-backed — they can "
            "bleed money. Your conversion rate dropped 25% in one week. Mei ran a survey: 60% of "
            "abandoned carts mention price. Your product is genuinely higher quality — but in an online "
            "thumbnail, they look the same. Your margin is 42%. Matching their price means 18% margin. "
            "You have $300K left in budget."
        ),
        "question": "Mei needs a pricing and positioning response this week. What's your strategy?",
        "choices": [
            {
                "key": "A", "label": "Match the price — compete on parity, win on quality",
                "description": "Drop to their price. Bet that retention will differentiate you over time.",
                "consequences": {"revenue": 12000, "conversion_rate": 0.01, "brand_strength": -0.08, "budget_remaining": 0, "customer_nps": -3},
                "narrative": "Conversion recovers 15%. But margin squeeze means less money for marketing. You're playing their game — on their turf. The VC competitor drops prices again.",
                "intent": "speed", "risk": "high"
            },
            {
                "key": "B", "label": "Raise prices 10% — lean into premium positioning hard",
                "description": "Move away from the comparison. Make price an asset, not a liability.",
                "consequences": {"revenue": 8000, "conversion_rate": -0.003, "brand_strength": 0.12, "customer_nps": 5, "marketplace_rating": 0.2},
                "narrative": "Conversions drop further short-term. But customers who buy leave glowing reviews. In 6 weeks, your premium positioning starts differentiating meaningfully.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "C", "label": "Launch a budget tier — fight on price with a new SKU",
                "description": "Create a lower-cost version. Compete at their price without diluting your flagship.",
                "consequences": {"revenue": 20000, "conversion_rate": 0.005, "brand_strength": -0.03, "budget_remaining": -30000, "inventory_level": 200},
                "narrative": "The budget SKU captures price-sensitive buyers. But it cannibalises 20% of your flagship. You win the volume war but lose premium brand equity.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Double down on content — make quality visible, not just claimed",
                "description": "Produce comparison content. Side-by-side video. User-generated quality evidence.",
                "consequences": {"brand_strength": 0.1, "conversion_rate": 0.006, "budget_remaining": -20000, "social_media_sentiment": 0.1, "customer_nps": 6},
                "narrative": "A comparison video gets 180K views. The quality difference becomes visible. Return rate data in your favor. People start posting 'get the real one' comments.",
                "intent": "quality", "risk": "low"
            }
        ]
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HEALTHCARE
# ─────────────────────────────────────────────────────────────────────────────
SITUATIONS["healthcare"] = [

    {
        "id": "near_miss",
        "seq": 1,
        "trigger": "step_range:1:4",
        "area": "Patient Safety",
        "narrator_name": "Dr. Nia Okafor",
        "narrator_role": "Chief Medical Officer",
        "headline": "🩸 Near-Miss: Wrong Blood Type Almost Administered to ICU Patient",
        "situation": (
            "A nurse caught a blood type mismatch at the bedside — seconds before administration. "
            "The error originated in the lab, where two patients' samples were mislabeled 4 hours earlier. "
            "The patient is safe. No harm occurred. But this is a Serious Reportable Event under JCAHO guidelines. "
            "Dr. Okafor says you must report to the state health department within 24 hours. "
            "Three nurses on the floor know. The patient's family doesn't."
        ),
        "question": "Dr. Okafor needs your directive on disclosure and protocol response within 2 hours.",
        "choices": [
            {
                "key": "A", "label": "Full disclosure — tell the family today, report to JCAHO today",
                "description": "Transparency. Immediate family notification. Regulatory report filed proactively.",
                "consequences": {"patient_safety_score": 0.1, "regulatory_compliance": 0.12, "patient_satisfaction": 0.08, "reputation": 0.06},
                "narrative": "The family is shaken but appreciative of the honesty. The JCAHO report demonstrates a functioning safety culture. No legal action taken. The nurse who caught it is formally commended.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "File JCAHO report but delay family notification until RCA is complete",
                "description": "Comply with regulatory requirement. Get root cause facts before family discussion.",
                "consequences": {"patient_safety_score": 0.06, "regulatory_compliance": 0.08, "patient_satisfaction": -0.04, "reputation": -0.02},
                "narrative": "RCA takes 3 days. The family hears fragments from staff before your call. When you call, trust is already eroded.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "C", "label": "Internal investigation only — near-miss with no harm means no report",
                "description": "Handle internally. The patient wasn't harmed. Regulatory report not strictly required.",
                "consequences": {"patient_safety_score": -0.08, "regulatory_compliance": -0.15, "reputation": -0.1},
                "narrative": "Three months later, a JCAHO auditor reviews records and finds the incident. 'Failure to report' is far worse than the original near-miss. Your accreditation is reviewed.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "D", "label": "Halt similar procedures hospital-wide — system-wide safety sweep",
                "description": "Beyond disclosure: stop all similar blood administration procedures until the lab protocol is re-validated.",
                "consequences": {"patient_safety_score": 0.15, "regulatory_compliance": 0.1, "budget_remaining": -30000, "wait_time_minutes": 8, "reputation": 0.1},
                "narrative": "48-hour pause catches one more mislabeling from the same root cause. The sweep prevents a second near-miss. JCAHO cites your response as a model.",
                "intent": "risk", "risk": "low"
            }
        ]
    },

    {
        "id": "star_surgeon_leaving",
        "seq": 2,
        "trigger": "above:staff_morale:0.5",
        "area": "Talent & Retention",
        "narrator_name": "Marcus Webb",
        "narrator_role": "Head of HR",
        "headline": "💉 Your Top Cardiac Surgeon Is Leaving for a Private Hospital — 2x Salary",
        "situation": (
            "Dr. Hassan is your highest-volume cardiac surgeon — 240 procedures last year, 98.2% success rate. "
            "He's been offered a $1.2M package at a private hospital (you pay $580K). "
            "He's given 60-day notice. His departure affects 3 other cardiac team members who were recruited specifically to work with him. "
            "Marcus says if he goes, you'll need to refer 40% of complex cardiac cases to other hospitals for 8 months minimum. "
            "Patient outcomes will be affected."
        ),
        "question": "Dr. Hassan meets with you tomorrow. What's your position?",
        "choices": [
            {
                "key": "A", "label": "Counter at $950K — push hospital board tonight",
                "description": "Emergency board meeting. Make the financial case for retention based on procedure volume.",
                "consequences": {"staff_morale": 0.1, "budget_remaining": -370000, "patient_safety_score": 0.08, "reputation": 0.05},
                "narrative": "Board agrees. Hassan accepts $950K. The 3 supporting staff stay. Revenue from his procedure volume justifies every dollar. Good decision.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "B", "label": "Let him go — build the next generation of the program",
                "description": "Invest the counter-offer budget into training two junior cardiac surgeons instead.",
                "consequences": {"staff_morale": -0.1, "budget_remaining": -150000, "patient_safety_score": -0.12, "reputation": -0.05, "team_size": 0},
                "narrative": "Eight months of reduced cardiac capacity. Two patient outcomes are worse than they would have been. The junior surgeons develop but it takes 18 months.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "C", "label": "Negotiate a partnership — he works here AND the private hospital",
                "description": "Propose a split-week arrangement. 3 days here, 2 days there.",
                "consequences": {"staff_morale": 0.05, "budget_remaining": -120000, "patient_safety_score": 0.04, "reputation": 0.03},
                "narrative": "Hassan accepts the hybrid. Private hospital is initially resistant but agrees. His productivity actually increases. Unusual arrangement becomes a model for other specialties.",
                "intent": "balance", "risk": "medium"
            },
            {
                "key": "D", "label": "Ask Hassan what would actually make him stay — beyond salary",
                "description": "One meeting. No numbers yet. Just listening.",
                "consequences": {"staff_morale": 0.08, "budget_remaining": -80000, "patient_safety_score": 0.1, "reputation": 0.06},
                "narrative": "Hassan's real issue: administrative load and a lack of research budget. You restructure his admin support and commit $80K to a research programme. He stays at his current salary.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "viral_wait_times",
        "seq": 3,
        "trigger": "above:wait_time_minutes:35",
        "area": "Patient Experience & PR",
        "narrator_name": "Sophia Nguyen",
        "narrator_role": "Head of Patient Experience",
        "headline": "📱 TikTok Video of 4-Hour Wait Time Going Viral — 600K Views",
        "situation": (
            "A patient filmed their 4-hour wait in the ER waiting room and posted it. "
            "The video has 600K views and the local news is asking for comment. "
            "Sophia pulled the data: the long wait was caused by an abnormal patient surge that day — "
            "a local event led to 3x normal ER volume. But the video doesn't show that context. "
            "The comments are brutal. Three city council members have retweeted it."
        ),
        "question": "Sophia has drafted three possible responses. Which path do you take?",
        "choices": [
            {
                "key": "A", "label": "Post the full data — volume surge explained transparently",
                "description": "Release the day's ER data showing 3x normal volume. Context is the defence.",
                "consequences": {"patient_satisfaction": 0.08, "media_sentiment": 0.1, "reputation": 0.06},
                "narrative": "Media covers the context. Several doctors and hospital workers share it. The narrative shifts from 'bad hospital' to 'overwhelmed system needs support'.",
                "intent": "quality", "risk": "low"
            },
            {
                "key": "B", "label": "Apologize publicly, no context — commit to improvement",
                "description": "Simple: sorry, working on it. No defensive data, no excuses.",
                "consequences": {"patient_satisfaction": 0.06, "media_sentiment": 0.04, "reputation": 0.04, "budget_remaining": -15000},
                "narrative": "Mostly positive response. Some journalists criticize the vagueness. But the apology lands better than any data-heavy response would have.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Reach out to the patient privately — offer VIP service next visit",
                "description": "Contact the patient. Personalise the response.",
                "consequences": {"patient_satisfaction": 0.04, "media_sentiment": -0.04, "reputation": -0.03},
                "narrative": "The patient posts about your DM. 'Hospital tries to buy off viral patient' becomes the new story. Worse than the original.",
                "intent": "cost", "risk": "high"
            },
            {
                "key": "D", "label": "Hold a community open day — show the real operations",
                "description": "Invite local media, patients, and council members for a transparent hospital tour.",
                "consequences": {"patient_satisfaction": 0.12, "media_sentiment": 0.12, "reputation": 0.1, "budget_remaining": -20000},
                "narrative": "Open day generates 40 pieces of positive local coverage. Three council members become advocates after seeing the understaffing firsthand. Budget ask gets approved faster.",
                "intent": "quality", "risk": "low"
            }
        ]
    },

    {
        "id": "insurance_denial",
        "seq": 4,
        "trigger": "above:patient_load:160",
        "area": "Patient Advocacy & Operations",
        "narrator_name": "Dr. Nia Okafor",
        "narrator_role": "Chief Medical Officer",
        "headline": "📑 Insurance Company Denied a Critical Cancer Patient's Treatment — Your Team is Furious",
        "situation": (
            "A 54-year-old patient's insurer denied coverage for a $45K immunotherapy treatment. "
            "Your oncologist says it's their best option. The denial is technically correct under the "
            "policy but medically unjustifiable. The patient's daughter posted about it — 8K shares. "
            "Your legal team says fighting the denial takes 6 months. The patient may not have 6 months. "
            "The board has a 'don't fight insurers' rule to protect revenue relationships."
        ),
        "question": "Dr. Okafor wants your support. What do you do?",
        "choices": [
            {
                "key": "A", "label": "Administer the treatment and absorb the cost — patient first",
                "description": "Give the treatment. Fight the insurer for reimbursement after.",
                "consequences": {"patient_safety_score": 0.12, "patient_satisfaction": 0.2, "budget_remaining": -45000, "reputation": 0.15, "media_sentiment": 0.1},
                "narrative": "Patient receives treatment. Three months later, they post a recovery video. Your hospital gets $2M in donation pledges. The insurer settles under press pressure.",
                "intent": "quality", "risk": "medium"
            },
            {
                "key": "B", "label": "File an expedited appeal with an independent physician reviewer",
                "description": "Pursue the fastest legal path. Independent review can decide in 3 weeks.",
                "consequences": {"patient_safety_score": 0.05, "budget_remaining": -8000, "reputation": 0.05, "patient_satisfaction": 0.08},
                "narrative": "The independent reviewer overturns the denial in 18 days. Treatment begins. You establish a fast-appeal protocol for all future denials.",
                "intent": "balance", "risk": "low"
            },
            {
                "key": "C", "label": "Escalate to state insurance commissioner and media simultaneously",
                "description": "Full public advocacy. Go to regulators and press together.",
                "consequences": {"patient_safety_score": 0.1, "reputation": 0.12, "media_sentiment": 0.15, "budget_remaining": -12000},
                "narrative": "Commissioner opens an investigation. Media coverage is intense. Insurer reverses denial within 72 hours. Three other patients in similar situations also get overturned.",
                "intent": "risk", "risk": "medium"
            },
            {
                "key": "D", "label": "Refer to the insurer's exception process — stay within bounds",
                "description": "Follow the board rule. File the exception. Don't fight publicly.",
                "consequences": {"patient_safety_score": -0.05, "patient_satisfaction": -0.1, "reputation": -0.06, "media_sentiment": -0.08},
                "narrative": "The exception is denied. The patient's story goes viral without your support. Staff morale craters. Two oncologists say privately they're looking elsewhere.",
                "intent": "cost", "risk": "high"
            }
        ]
    },
]
