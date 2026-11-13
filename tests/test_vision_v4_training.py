import unittest

import numpy as np

from agents.vision_v4_types import PerceptionPolicy
from tools.train_vision_v4 import (
    GroupSampler,
    choose_shared_epoch,
    score_policy,
    score_raw,
    validate_config,
)


class TrainingTests(unittest.TestCase):
    def test_policy_selection_requires_actual_board_contracts(self):
        from tools.train_vision_v4 import select_policy

        logits = np.zeros((1, 14))
        with self.assertRaisesRegex(ValueError, "board contracts"):
            score_policy(
                logits,
                logits,
                np.array([0]),
                np.array([0]),
                PerceptionPolicy(1, 0.85, 0.1, "full"),
            )
        with self.assertRaisesRegex(ValueError, "board contracts"):
            select_policy({"arrays": {}}, {})

    def test_production_loads_actual_contracts_before_any_fit(self):
        from unittest.mock import patch

        from tools.train_vision_v4 import fit_production

        with (
            patch(
                "tools.train_vision_v4.load_board_contracts",
                side_effect=ValueError("contract sentinel"),
            ) as contracts,
            patch("tools.train_vision_v4.train_fold") as fit,
        ):
            with self.assertRaisesRegex(ValueError, "contract sentinel"):
                fit_production(
                    {
                        "fold": "production",
                        "epochs": 14,
                        "data": "corpora/production",
                        "output": "unused",
                    },
                    {"epoch": 14, "downstream_selection_complete": True},
                )
            contracts.assert_called_once()
            fit.assert_not_called()

    def test_class_then_group_balancing_ignores_duplicate_augmentation_count(self):
        labels = np.array([0] * 1001 + [1] * 2)
        groups = np.array(["large"] * 1000 + ["small", "a", "b"])
        sampler = GroupSampler(labels, groups, seed=7, class_count=2)
        indices = sampler.draw(20000)
        self.assertEqual(np.sum(labels[indices] == 0), 10000)
        fraction = np.mean(groups[indices[labels[indices] == 0]] == "small")
        self.assertAlmostEqual(fraction, 0.5, delta=0.025)
        other = GroupSampler(labels, groups, seed=7, class_count=2)
        np.testing.assert_array_equal(indices, other.draw(20000))

    def test_fold_from_scratch_and_final_exclusion(self):
        base = {
            "fold": 0,
            "epochs": 20,
            "seed": 20261109,
            "checkpoint": None,
            "class_order": list(range(14)),
        }
        with self.assertRaisesRegex(ValueError, "class order"):
            validate_config(base)
        from tools.train_vision_v3 import CLASSES

        base["class_order"] = list(CLASSES)
        validate_config(base)
        for field, value in [
            ("checkpoint", "artifacts/v3/runs/best.pt"),
            ("train", "data/final-300"),
        ]:
            with self.assertRaises(ValueError):
                validate_config({**base, field: value})
        with self.assertRaisesRegex(ValueError, "20"):
            validate_config({**base, "epochs": 21})

    def test_shared_epoch_uses_all_three_folds_and_same_epoch(self):
        def row(epoch, exact, recall, nll):
            return {
                "epoch": epoch,
                "exact_boards": exact,
                "boards": 10,
                "class_true_positive": [recall] * 14,
                "class_support": [10] * 14,
                "nll_sum": nll,
                "cells": 140,
            }

        folds = [{"history": [row(1, 9, 9, 10), row(2, 10, 10, 1)]} for _ in range(3)]
        self.assertEqual(choose_shared_epoch(folds), 2)
        folds[2]["history"].pop()
        self.assertEqual(choose_shared_epoch(folds), 1)
        with self.assertRaisesRegex(ValueError, "three"):
            choose_shared_epoch(folds[:2])

    def test_temperature_preserves_argmax_and_wrong_gem_stays_error(self):
        from tools.train_vision_v3 import CLASSES

        logits = np.zeros((2, 14))
        logits[:, CLASSES.index("coin")] = 20
        labels = np.array([CLASSES.index("gem"), CLASSES.index("coin")])
        for temperature in (0.25, 1, 8):
            r = score_raw(
                logits,
                logits,
                labels,
                np.array([0, 0]),
                PerceptionPolicy(temperature, 0.85, 0.2, "full"),
            )
            self.assertEqual(r["errors"], 1)
            self.assertEqual(r["class_true_positive"][CLASSES.index("gem")], 0)
        r = score_raw(
            logits,
            logits,
            labels,
            np.array([0, 0]),
            PerceptionPolicy(1, 0.85, 0.2, "full"),
        )
        self.assertEqual(r["high_confidence_errors"], 1)

    def test_production_cannot_start_before_downstream_selection(self):
        from unittest.mock import patch

        from tools.train_vision_v4 import fit_production

        selection = {
            "epoch": 1,
            "policy": {
                "temperature": 1,
                "min_confidence": 0.85,
                "min_margin": 0.2,
                "view": "full",
            },
            "downstream_selection_complete": False,
        }
        with patch("tools.train_vision_v4.train_fold") as fit:
            with self.assertRaisesRegex(ValueError, "downstream"):
                fit_production(
                    {"fold": "production", "epochs": 1, "output": "unused"}, selection
                )
            fit.assert_not_called()

    def test_exact_board_selection_counts_tracker_ambiguity_limit(self):
        from grid_adventure.entities import AgentEntity, CoinEntity, ExitEntity
        from grid_adventure.grid import GridState
        from grid_adventure.movements import MOVEMENTS
        from grid_adventure.objectives import OBJECTIVES

        from tests.vision_v3_fixtures import vision_observation
        from tools.train_vision_v3 import CLASSES

        kinds = ["human", "coin", "coin", "coin", "exit"]
        labels = np.array([CLASSES.index(k) for k in kinds])
        probabilities = np.full((5, 14), 1e-12)
        probabilities[0, CLASSES.index("human")] = 1
        probabilities[4, CLASSES.index("exit")] = 1
        for i in (1, 2, 3):
            probabilities[i, CLASSES.index("coin")] = 0.9
            probabilities[i, CLASSES.index("gem")] = 0.1
        logits = np.log(probabilities)
        state = GridState(
            width=5,
            height=1,
            movement=MOVEMENTS["cardinal"],
            objective=OBJECTIVES["collect_gems_and_exit"],
        )
        state.add((0, 0), AgentEntity())
        state.add((4, 0), ExitEntity())
        for x in (1, 2, 3):
            state.add((x, 0), CoinEntity())
        result = score_policy(
            logits,
            logits,
            labels,
            np.zeros(5, dtype=int),
            PerceptionPolicy(1, 0.95, 0.1, "full"),
            board_records={
                0: {
                    "width": 5,
                    "height": 1,
                    "info": vision_observation(state)["info"],
                    "labels": [[[kind] for kind in kinds]],
                }
            },
        )
        self.assertEqual(result["exact_boards"], 0)
        self.assertEqual(result["raw_exact_boards"], 1)
        self.assertEqual(result["boards_exceeding_ambiguity_cap"], 1)

    def test_exact_board_uses_retained_view_winners_not_fused_top_class(self):
        from grid_adventure.entities import PhasingPowerUpEntity

        from tests.vision_v3_fixtures import vision_observation
        from tests.planner_p1_fixtures import corridor
        from tools.train_vision_v3 import CLASSES

        state = corridor()
        state.add((1, 0), PhasingPowerUpEntity())
        full = np.full((3, 14), 1e-12)
        inner = full.copy()
        full[0, CLASSES.index("human")] = inner[0, CLASSES.index("human")] = 1
        full[2, CLASSES.index("exit")] = inner[2, CLASSES.index("exit")] = 1
        full[1, CLASSES.index("coin")] = 0.51
        full[1, CLASSES.index("ghost")] = 0.49
        inner[1, CLASSES.index("gem")] = 0.51
        inner[1, CLASSES.index("ghost")] = 0.49
        labels = np.array([CLASSES.index(k) for k in ("human", "ghost", "exit")])
        records = {
            0: {
                "width": 3,
                "height": 1,
                "info": vision_observation(state)["info"],
                "labels": [[["human"], ["ghost"], ["exit"]]],
            }
        }
        result = score_policy(
            np.log(full),
            np.log(inner),
            labels,
            np.zeros(3, dtype=int),
            PerceptionPolicy(1, 0.85, 0.1, "inset_005"),
            board_records=records,
        )
        self.assertEqual(result["raw_exact_boards"], 1)
        self.assertEqual(result["exact_boards"], 0)

    def test_redundant_margin_grid_has_one_runtime_representative(self):
        from tools.vision_v4_selection import collapse_margin_candidates

        candidates = {}
        for margin in (0.1, 0.2, 0.3):
            key = f"full/.90/{margin}"
            candidates[key] = {
                "key": key,
                "policy": {
                    "temperature": 1.0,
                    "min_confidence": 0.9,
                    "min_margin": margin,
                    "view": "full",
                },
            }
        representatives, mapping = collapse_margin_candidates(candidates)
        self.assertEqual(len(representatives), 1)
        self.assertEqual(set(mapping.values()), {"full/.90/0.1"})
