"""V3 runtime records. This module deliberately has no vision-library import."""

from typing import NamedTuple

from grid_adventure.grid import GridState
from grid_adventure.step import Action


class EpisodeBudget(NamedTuple):
    started_at: float
    initial_deadline: float
    cutoff: float


class CellScores(NamedTuple):
    probabilities: object
    width: int
    height: int


class BoardHypothesis(NamedTuple):
    state: GridState
    confidence: float
    unresolved: tuple


class ObservationCheck(NamedTuple):
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
