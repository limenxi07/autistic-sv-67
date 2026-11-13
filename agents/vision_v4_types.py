"""Shared V4 records; importing this module does not load a vision library."""

from typing import NamedTuple

from grid_adventure.grid import GridState
from grid_adventure.step import Action


class EpisodeBudget(NamedTuple):
    """Absolute timestamps shared by perception, planning and execution."""
    started_at: float
    initial_deadline: float
    cutoff: float


class CellScores(NamedTuple):
    """Per-cell class probabilities with board dimensions."""
    probabilities: object
    width: int
    height: int


class BoardHypothesis(NamedTuple):
    """A possible board interpretation retained for planning."""
    state: GridState
    confidence: float
    unresolved: tuple


class ObservationCheck(NamedTuple):
    """Result of comparing a prediction with the current observation."""
    consistent: bool
    changed_cells: tuple
    mismatches: tuple
    observed_scores: object = None


class Emission(NamedTuple):
    turn: int | None
    action: Action
    reason: str
    prediction_valid: bool


class PerceptionEvidence(NamedTuple):
    scores: CellScores
    disagreements: tuple
    policy: str


class PerceptionPolicy(NamedTuple):
    temperature: float
    min_confidence: float
    min_margin: float
    view: str
