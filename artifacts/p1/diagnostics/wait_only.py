"""Debugging loader probe: deliberately waits and loses; not a solution."""

from grid_adventure.step import Action


class Agent:
    def step(self, state: object) -> Action:
        return Action.WAIT
