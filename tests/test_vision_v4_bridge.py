import unittest
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
from grid_adventure.entities import GemEntity, KeyEntity, ShieldPowerUpEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.vision_v4_bridge import (
    decode_metadata,
    reconstruct,
    split_cells,
)
from tests.vision_v3_fixtures import image_env, observable
from tests.planner_p1_fixtures import corridor, mechanic
from tests.test_planner_p1_state import player
from tools.benchmark_vision_v3 import oracle_labels


class BridgeTests(unittest.TestCase):
    def test_unique_agent_identity_survives_unexpected_observed_position(self):
        state = corridor()
        obs, _ = image_env(state).reset()
        previous = reconstruct(oracle_labels(state), obs["info"]).state
        labels = [["floor", "human", "exit"]]
        rebuilt = reconstruct(labels, obs["info"], previous)
        self.assertEqual(player(rebuilt.state).entity_id, player(previous).entity_id)

    def test_duplicate_static_walls_have_equivalent_observable_successors(self):
        from grid_adventure.entities import WallEntity

        original, actions = mechanic("ghost")
        original.add((2, 1), WallEntity())
        original.add((2, 1), WallEntity())
        predicted = None
        wall_components = (("blocking", ()),)

        def semantic(value):
            info, cells = observable(value)
            return info, tuple(
                tuple(
                    e
                    for i, e in enumerate(cell)
                    if e != wall_components or wall_components not in cell[:i]
                )
                for cell in cells
            )

        for action in (None,) + actions[:9]:
            if action is not None:
                original = step(original, action)
                assert predicted is not None
                predicted = step(predicted, action)
            obs, _ = image_env(original).reset()
            labels = oracle_labels(original)
            labels = [
                [
                    tuple(
                        k
                        for i, k in enumerate(cell)
                        if k != "wall" or "wall" not in cell[:i]
                    )
                    for cell in row
                ]
                for row in labels
            ]
            predicted = reconstruct(labels, obs["info"], predicted).state
            for candidate in Action:
                self.assertEqual(
                    semantic(step(original, candidate)),
                    semantic(step(predicted, candidate)),
                )

    def test_fractional_cells_cover_image_once(self):
        image = np.arange(457 * 640 * 4).reshape(457, 640, 4)
        obs = {"image": image, "info": {"config": {"width": 7, "height": 5}}}
        rows = split_cells(obs)
        self.assertEqual([len(row) for row in rows], [7] * 5)
        np.testing.assert_array_equal(
            np.concatenate([np.concatenate(row, axis=1) for row in rows]), image
        )
        self.assertEqual(
            sum(c.shape[0] * c.shape[1] for row in rows for c in row), 292480
        )

    def test_objective_alias_preserves_gem_requirement(self):
        state = corridor()
        state.add((1, 0), GemEntity())
        obs, _ = image_env(state).reset()
        meta = decode_metadata(obs["info"])
        self.assertEqual(meta["objective"], "collect_gems_and_exit")
        rebuilt = reconstruct(oracle_labels(state), obs["info"])
        self.assertFalse(rebuilt.unresolved)
        state = step(step(rebuilt.state, Action.RIGHT), Action.RIGHT)
        self.assertFalse(state.win)

    def test_reidentified_all_action_observable_successors(self):
        for kind in ("key_door", "coin", "gem", "boots", "ghost", "shield", "box_coin"):
            state, actions = mechanic(kind)
            # Use sparse non-location-based identities, deliberately permuted.
            entities = [e for col in state.grid for cell in col for e in cell]
            for i, entity in enumerate(entities):
                entity.entity_id = 100003 + (len(entities) - i) * 37
            predicted = None
            for action in (None,) + actions[:8]:
                if action is not None:
                    state = step(state, action)
                    assert predicted is not None
                    predicted = step(predicted, action)
                obs, _ = image_env(state).reset()
                rebuilt = reconstruct(oracle_labels(state), obs["info"], predicted)
                self.assertFalse(
                    rebuilt.unresolved, (kind, state.turn, rebuilt.unresolved)
                )
                predicted = rebuilt.state
                for candidate in Action:
                    with self.subTest(kind=kind, turn=state.turn, action=candidate):
                        self.assertEqual(
                            observable(step(state, candidate)),
                            observable(step(predicted, candidate)),
                        )

    def test_inventory_and_effect_order_survive_tracking(self):
        state = mechanic("key_door")[0]
        # Stay outside constructor-generated IDs, even in a long full-suite run.
        base = 1 << 40
        state.add((0, 1), KeyEntity(entity_id=base + 1005))
        state.add((0, 1), KeyEntity(entity_id=base + 1037))
        state = step(state, Action.PICK_UP)
        shield = cast(Any, ShieldPowerUpEntity(entity_id=base + 900))
        shield.collectible = None
        shield.usage_limit = replace(shield.usage_limit, amount=2)
        player(state).status_list = [ShieldPowerUpEntity(entity_id=base + 800), shield]
        player(state).status_list[0].collectible = None
        obs, _ = image_env(state).reset()
        meta = decode_metadata(obs["info"])
        self.assertEqual(meta["inventory"], obs["info"]["agent"]["inventory"])
        self.assertEqual(meta["effects"], obs["info"]["agent"]["effects"])
        rebuilt = reconstruct(oracle_labels(state), obs["info"])
        self.assertFalse(rebuilt.unresolved)
        for action in Action:
            self.assertEqual(
                observable(step(state, action)), observable(step(rebuilt.state, action))
            )
        obs2 = deepcopy(obs)
        obs2["info"]["agent"]["effects"][0]["limit_amount"] = 1
        updated = reconstruct(oracle_labels(state), obs2["info"], rebuilt.state)
        self.assertEqual(player(updated.state).status_list[0].usage_limit.amount, 1)
        self.assertEqual(player(rebuilt.state).status_list[0].usage_limit.amount, 5)

    def test_hidden_later_layer_stays_unresolved_without_history(self):
        state = corridor()
        obs, _ = image_env(state).reset()
        obs["info"]["status"]["turn"] = 4
        labels = [["human", "floor", "exit"]]
        self.assertTrue(reconstruct(labels, obs["info"]).unresolved)
