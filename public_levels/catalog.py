from collections.abc import Callable

from grid_universe.grid.gridstate import GridState

from public_levels import challenges, intro


LevelBuilder = Callable[..., GridState]

PUBLIC_TURN_LIMIT = 150
PUBLIC_GAMEPLAY_TIME_LIMIT = 5
PUBLIC_BOSS_TIME_LIMIT = 15

PUBLIC_LEVEL_BUILDERS: dict[str, LevelBuilder] = {
    "A0 Basic Movement": intro.build_level_basic_movement,
    "A1 Maze Turns": intro.build_level_maze_turns,
    "A2 Optional Coin Path": intro.build_level_optional_coin,
    "A3 Multiple Required Gems": intro.build_level_required_multiple,
    "A4 Key and Door": intro.build_level_key_door,
    "A5 Hazard Detour": intro.build_level_hazard_detour,
    "A6 Pushable Box": intro.build_level_pushable_box,
    "A7 Shield Powerup": intro.build_level_power_shield,
    "A8 Ghost Powerup": intro.build_level_power_ghost,
    "A9 Boots Powerup": intro.build_level_power_boots,
    "A10 Combined Mechanics": intro.build_level_combined_mechanics,
    "B0 Thread the Needle": challenges.build_level_thread_the_needle,
    "B1 Hypnosis": challenges.build_level_hypnosis,
    "B2 The Vault": challenges.build_level_the_vault,
    "B3 So Close Yet So Far": challenges.build_level_so_close_yet_so_far,
    "B4 Measure Twice and Cut Once": challenges.build_level_measure_twice_and_cut_once,
    "B5 Ow Oof Owie": challenges.build_level_ow_oof_owie,
    "B6 Warehouse": challenges.build_level_warehouse,
    "B7 Fire Resistance": challenges.build_level_fire_resistance,
    "B8 Choose Wisely": challenges.build_level_choose_wisely,
    "B9 Manhattan": challenges.build_level_manhattan,
    "B10 Khoa Was Here": challenges.build_level_khoa_was_here,
    "C0 Everything Everywhere All at Once": intro.build_level_boss,
    "C1 The Heist": challenges.build_level_the_heist,
}

PUBLIC_BOSS_LEVEL_LABELS = (
    "C0 Everything Everywhere All at Once",
    "C1 The Heist",
)

PUBLIC_GAMEPLAY_LEVEL_BUILDERS: dict[str, LevelBuilder] = {
    label: builder
    for label, builder in PUBLIC_LEVEL_BUILDERS.items()
    if label not in PUBLIC_BOSS_LEVEL_LABELS
}

PUBLIC_BOSS_LEVEL_BUILDERS: dict[str, LevelBuilder] = {
    label: PUBLIC_LEVEL_BUILDERS[label]
    for label in PUBLIC_BOSS_LEVEL_LABELS
}

PUBLIC_LEVEL_MAX_REWARDS: dict[str, int] = {
    "A0 Basic Movement": -9,
    "A1 Maze Turns": -27,
    "A2 Optional Coin Path": -21,
    "A3 Multiple Required Gems": -63,
    "A4 Key and Door": -33,
    "A5 Hazard Detour": -39,
    "A6 Pushable Box": -21,
    "A7 Shield Powerup": -42,
    "A8 Ghost Powerup": -48,
    "A9 Boots Powerup": -27,
    "A10 Combined Mechanics": -84,
    "B0 Thread the Needle": -81,
    "B1 Hypnosis": -78,
    "B2 The Vault": -65,
    "B3 So Close Yet So Far": -315,
    "B4 Measure Twice and Cut Once": -387,
    "B5 Ow Oof Owie": -210,
    "B6 Warehouse": -213,
    "B7 Fire Resistance": -111,
    "B8 Choose Wisely": -93,
    "B9 Manhattan": -63,
    "B10 Khoa Was Here": -162,
    "C0 Everything Everywhere All at Once": -68,
    "C1 The Heist": -158,
}


__all__ = [
    "LevelBuilder",
    "PUBLIC_BOSS_LEVEL_BUILDERS",
    "PUBLIC_BOSS_TIME_LIMIT",
    "PUBLIC_GAMEPLAY_LEVEL_BUILDERS",
    "PUBLIC_GAMEPLAY_TIME_LIMIT",
    "PUBLIC_LEVEL_BUILDERS",
    "PUBLIC_LEVEL_MAX_REWARDS",
    "PUBLIC_TURN_LIMIT",
]
