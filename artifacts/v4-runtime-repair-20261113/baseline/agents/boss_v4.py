"""Boss-mode entry point with a 15-second budget for either observation format."""

from agents.vision_v4 import Agent
from agents.vision_v4_backend import Task3Agent
from agents.vision_v4_types import EpisodeBudget


class BossAgent(Agent):
    """Give Task 3 longer deadlines while retaining the original construction time."""
    def __init__(self):
        super().__init__()
        # Include cold setup time; only this entry point selects the boss budget.
        started = self._budget.started_at
        self._budget = EpisodeBudget(started, started + 11.0, started + 14.65)
        self._planning_budget = EpisodeBudget(started, started + 11.0, started + 12.0)
        self._structured = Task3Agent(self._planning_budget)
