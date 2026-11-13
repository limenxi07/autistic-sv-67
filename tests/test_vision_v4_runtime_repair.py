import unittest
from time import perf_counter
from unittest.mock import patch

from agents.vision_v4_backend import grid_step
from agents.vision_v4_planner import PlannerAdapter
from agents.vision_v4_types import EpisodeBudget
from tests.planner_p1_fixtures import corridor


class NativeConversionCostTests(unittest.TestCase):
    def test_plain_scenery_does_not_scan_all_component_fields(self):
        from grid_universe.grid.convert import to_state
        from grid_universe.grid.entity import BaseEntity

        from agents import vision_v4_search as search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        expected = to_state(state)
        original = BaseEntity.iter_components
        calls = []

        def counted(entity):
            calls.append(entity)
            return original(entity)

        with patch.object(BaseEntity, "iter_components", counted):
            actual = getattr(search, "v4_native_state", to_state)(state)
        self.assertEqual(actual, expected)
        self.assertLess(len(calls), 30)

    def test_native_converter_matches_public_maps_and_orders_during_routes(self):
        from grid_universe.grid.convert import to_state

        from agents import vision_v4_search as search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        convert = getattr(search, "v4_native_state", to_state)
        for label, build in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
            state = build()
            state.turn_limit = 150
            now = perf_counter()
            planner = PlannerAdapter(EpisodeBudget(now, now + 10, now + 10))
            planner.step(state)
            active = planner._reference or planner
            self.assertTrue(active._actions, label)
            for action in active._actions:
                expected, actual = to_state(state), convert(state)
                self.assertEqual(actual, expected, label)
                for name, value in vars(expected).items():
                    if isinstance(value, dict):
                        self.assertEqual(
                            tuple(getattr(actual, name)), tuple(value), (label, name)
                        )
                state = grid_step(state, action)
            self.assertTrue(state.win, label)

    def test_reference_wiring_and_nonplain_entities_use_public_converter(self):
        from types import SimpleNamespace

        from grid_adventure.entities import WallEntity
        from grid_universe.grid.convert import to_state

        from agents import vision_v4_search as search

        state = corridor()
        wall = WallEntity()
        state.add((0, 0), wall)
        wall.blocking = SimpleNamespace(extra=42)
        self.assertEqual(search.v4_native_state(state), to_state(state))
        wall.portal_pair_ref = wall
        with patch.object(search, "v4_to_state", wraps=to_state) as original:
            actual = search.v4_native_state(state)
        self.assertEqual(original.call_count, 1)
        self.assertEqual(actual, to_state(state))

    def test_wall_subclass_keeps_custom_component_iterator(self):
        from grid_adventure.entities import WallEntity
        from grid_universe.components.properties import Damage
        from grid_universe.grid.convert import to_state

        from agents import vision_v4_search as search

        class CustomWall(WallEntity):
            def iter_components(self):
                yield from super().iter_components()
                yield "damage", Damage(2)

        state = corridor()
        wall = CustomWall()
        state.add((0, 0), wall)
        expected = to_state(state)
        actual = search.v4_native_state(state)
        self.assertEqual(actual, expected)
        self.assertIn(wall.entity_id, actual.damage)


class CompactObservationCostTests(unittest.TestCase):
    def test_plain_scenery_avoids_general_component_decoding(self):
        from agents import vision_v4_backend as backend
        from agents import vision_v4_search as search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        state.turn_limit = 150
        board, _, _ = search.v4_encode_proposal(state)
        expected = backend.observe_key(board, state)
        observe = getattr(search, "v4_observe_key", backend.observe_key)
        with (
            patch.object(
                search, "_components", wraps=backend._components
            ) as components,
            patch.object(backend, "_components", components),
        ):
            actual = observe(board, state)
        self.assertEqual(actual, expected)
        self.assertLess(components.call_count, 30)

    def test_keys_match_backend_across_public_routes(self):
        from agents import vision_v4_backend as backend
        from agents import vision_v4_search as search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        compared = 0
        for label, build in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
            state = build()
            state.turn_limit = 150
            board, _, _ = search.v4_encode_proposal(state)
            now = perf_counter()
            planner = PlannerAdapter(EpisodeBudget(now, now + 10, now + 10))
            planner.step(state)
            active = planner._reference or planner
            self.assertTrue(active._actions, label)
            for action in active._actions:
                self.assertEqual(
                    search.v4_observe_key(board, state),
                    backend.observe_key(board, state),
                    label,
                )
                state = grid_step(state, action)
                compared += 1
        self.assertGreater(compared, 500)

    def test_mutations_preserve_backend_rejection_and_keys(self):
        from copy import deepcopy
        from types import SimpleNamespace

        from agents import vision_v4_backend as backend
        from agents import vision_v4_search as search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        original = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        original.turn_limit = 150
        board, _, _ = search.v4_encode_proposal(original)
        cases = []
        for kind, field, value in (
            ("wall", "blocking", None),
            ("wall", "blocking", SimpleNamespace(extra=2)),
            ("wall", "unknown", SimpleNamespace()),
            ("wall", "entity_id", -12345),
            ("wall", "appearance", None),
            ("exit", "exit", None),
            ("exit", "exit", SimpleNamespace(extra=2)),
            ("human", "inventory_list", []),
        ):
            state = deepcopy(original)
            entity = next(
                e
                for col in state.grid
                for cell in col
                for e in cell
                if e.appearance.name == kind
            )
            setattr(entity, field, value)
            cases.append((kind + "/" + field, state))
        for field, value in (
            ("turn_limit", 151),
            ("step_cost", 4),
            ("score", 1.5),
            ("unexpected", True),
        ):
            state = deepcopy(original)
            setattr(state, field, value)
            cases.append((field, state))
        state = deepcopy(original)
        wall = next(
            e
            for col in state.grid
            for cell in col
            for e in cell
            if e.appearance.name == "wall"
        )
        state.grid[1][1].append(wall)
        cases.append(("duplicated wall", state))
        state = deepcopy(original)
        x, y = next(
            (x, y)
            for x, col in enumerate(state.grid)
            for y, cell in enumerate(col)
            if cell and cell[0].appearance.name == "wall"
        )
        state.grid[1][1].append(state.grid[x][y].pop(0))
        cases.append(("moved wall", state))
        for label, state in cases:
            with self.subTest(label=label):
                try:
                    expected = backend.observe_key(board, state)
                except backend.UnsupportedState as error:
                    with self.assertRaises(backend.UnsupportedState) as actual:
                        search.v4_observe_key(board, state)
                    self.assertEqual(str(actual.exception), str(error))
                else:
                    self.assertEqual(search.v4_observe_key(board, state), expected)
