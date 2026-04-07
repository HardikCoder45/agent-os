from __future__ import annotations

try:
    from .base_environment import BaseEnvironment
except ImportError:
    from base_environment import BaseEnvironment


class TechStartupEnvironment(BaseEnvironment):
    domain = "tech_startup"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("mrr", 0) > 25000 and s.get("team_morale", 0) > 0.6
                and s.get("product_quality", 0) > 0.65 and s.get("budget_remaining", 0) > 0)


class PharmaEnvironment(BaseEnvironment):
    domain = "pharma"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("trials_passed", 0) >= 3 and s.get("compliance_score", 0) > 0.7
                and s.get("patient_trust", 0) > 0.6 and s.get("budget_remaining", 0) > 0)


class InteriorDesignEnvironment(BaseEnvironment):
    domain = "interior_design"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("design_progress", 0) > 0.9 and s.get("client_satisfaction", 0) > 0.75
                and s.get("safety_record", 1) > 0.8 and s.get("budget_remaining", 0) > 0)


class ManufacturingEnvironment(BaseEnvironment):
    domain = "manufacturing"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("production_rate", 0) > 0.75 and s.get("quality_score", 0) > 0.8
                and s.get("worker_satisfaction", 0) > 0.6 and s.get("budget_remaining", 0) > 0)


class FinanceEnvironment(BaseEnvironment):
    domain = "finance"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("return_ytd", 0) > 0.1 and s.get("client_trust", 0) > 0.75
                and s.get("regulatory_compliance", 0) > 0.85)


class EcommerceEnvironment(BaseEnvironment):
    domain = "ecommerce"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("revenue", 0) > 200000 and s.get("brand_strength", 0) > 0.6
                and s.get("customer_nps", 0) > 45 and s.get("budget_remaining", 0) > 0)


class HealthcareEnvironment(BaseEnvironment):
    domain = "healthcare"

    def _goal_achieved(self) -> bool:
        s = self.state
        return (s.get("patient_safety_score", 0) > 0.9 and s.get("staff_morale", 0) > 0.7
                and s.get("wait_time_minutes", 999) < 25 and s.get("budget_remaining", 0) > 0)


__all__ = [
    "TechStartupEnvironment", "PharmaEnvironment", "InteriorDesignEnvironment",
    "ManufacturingEnvironment", "FinanceEnvironment", "EcommerceEnvironment", "HealthcareEnvironment",
]
