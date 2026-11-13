"""Connect V3 tracking to the pinned P2-next planner. Packaging verifies its hash."""

from agents.boss_p2_next import Agent as StructuredAgent
from agents.boss_p2_next import PlannerAdapter as BackendPlannerAdapter
from agents.vision_v3_types import EpisodeBudget

BACKEND_SOURCE = "agents/boss_p2_next.py"
BACKEND_SHA256 = "6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd"
__all__ = ["BACKEND_SHA256", "BACKEND_SOURCE", "PlannerAdapter", "StructuredAgent"]


class PlannerAdapter(BackendPlannerAdapter):
    """Adapt image-agent budgets to the pinned structured planner."""
    def __init__(self, budget: EpisodeBudget, mode: str = "p2a"):
        super().__init__(
            budget,
            mode=mode,
            reference_initial_deadline=min(
                budget.initial_deadline, budget.started_at + 2.0
            ),
        )
