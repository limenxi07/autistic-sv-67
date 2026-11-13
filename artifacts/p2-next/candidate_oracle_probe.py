"""Diagnostic wiring only; never selects a backend in the active V3 release."""

from time import perf_counter
from agents.p2_next import EpisodeBudget, PlannerAdapter


class Agent(PlannerAdapter):
    def __init__(self):
        started = perf_counter()
        budget = EpisodeBudget(started, started + 3.0, started + 3.5)
        super().__init__(budget, reference_initial_deadline=min(
            budget.initial_deadline, started + 2.0))
