"""V4 binding to the user-selected, validated frozen P3 backend.

The standalone packager vendors the exact parent bytes identified below.
This module does not read source files or import any vision library at runtime.
"""

from agents.v4_backend import Agent as StructuredAgent
from agents.v4_backend import PlannerAdapter as BackendPlannerAdapter
from agents.v4_backend import reference_state_key
from agents.v4_types import EpisodeBudget

BACKEND_SOURCE = "agents/v4_backend.py"
BACKEND_SHA256 = "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4"
__all__ = [
    "BACKEND_SHA256",
    "BACKEND_SOURCE",
    "PlannerAdapter",
    "StructuredAgent",
    "reference_state_key",
]


class PlannerAdapter(BackendPlannerAdapter):
    def __init__(self, budget: EpisodeBudget, mode: str = "p2a"):
        super().__init__(
            budget,
            mode=mode,
            reference_initial_deadline=min(
                budget.initial_deadline, budget.started_at + 2.0
            ),
        )
