from collections.abc import Sequence

from grid_universe.grid.gridstate import GridState

from grid_adventure.constants import STEP_COST
from grid_adventure.entities import (
    BoxEntity,
    CoinEntity,
    ExitEntity,
    GemEntity,
    KeyEntity,
    LavaEntity,
    LockedDoorEntity,
    PhasingPowerUpEntity,
    ShieldPowerUpEntity,
    SpeedPowerUpEntity,
    WallEntity,
    create_agent_entity,
)
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES


TURN_LIMIT = 150


def _build_ascii_level(
    layout: Sequence[str], seed: int, *, agent_health: int = 5
) -> GridState:
    if not layout or not layout[0]:
        raise ValueError("Level layout cannot be empty")

    width = len(layout[0])
    if any(len(row) != width for row in layout):
        raise ValueError("Level layout must be rectangular")

    symbols = "".join(layout)
    unknown_symbols = set(symbols) - set(".#ABCDEGKLPSx")
    if unknown_symbols:
        raise ValueError(f"Unknown level symbols: {sorted(unknown_symbols)}")
    if symbols.count("A") != 1 or symbols.count("E") != 1:
        raise ValueError("Level layout must contain exactly one agent and one exit")

    gridstate = GridState(
        width=width,
        height=len(layout),
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=TURN_LIMIT,
        step_cost=STEP_COST,
    )

    for y, row in enumerate(layout):
        for x, symbol in enumerate(row):
            position = (x, y)
            if symbol == "#":
                gridstate.add(position, WallEntity())
            elif symbol == "x":
                gridstate.add(position, BoxEntity())
            elif symbol == "B":
                gridstate.add(position, SpeedPowerUpEntity())
            elif symbol == "C":
                gridstate.add(position, CoinEntity())
            elif symbol == "G":
                gridstate.add(position, GemEntity())
            elif symbol == "K":
                gridstate.add(position, KeyEntity())
            elif symbol == "L":
                gridstate.add(position, LavaEntity())
            elif symbol == "S":
                gridstate.add(position, ShieldPowerUpEntity())
            elif symbol == "P":
                gridstate.add(position, PhasingPowerUpEntity())
            elif symbol == "D":
                gridstate.add(position, LockedDoorEntity())
            elif symbol == "A":
                gridstate.add(position, create_agent_entity(health=agent_health))
            elif symbol == "E":
                gridstate.add(position, ExitEntity())

    return gridstate


def build_level_thread_the_needle(seed: int = 200) -> GridState:
    layout = (
        "A......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        "...............",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#.......",
        ".......#......E",
    )
    return _build_ascii_level(layout, seed)


def build_level_hypnosis(seed: int = 201) -> GridState:
    layout = (
        "###############",
        "#.....#.......#",
        "#.###########.#",
        "..#.........#.#",
        "#.#.##.####.#.#",
        "#...#.....#.#.#",
        "#.#.#.###.#.#.#",
        "#.#.#.#A..#...#",
        "....#.#####.#.#",
        "#.#.#.......#.#",
        "#.#.######.##.#",
        "#.#...........#",
        "#.#######.#####",
        "#.............E",
        "#######.#######",
    )
    return _build_ascii_level(layout, seed)


def build_level_the_vault(seed: int = 202) -> GridState:
    layout = (
        "A..............",
        "...............",
        "..#####.#####..",
        "..#CCC#CC##C#..",
        "..#C#CCC#CCC#..",
        "..##C#CC##CC#..",
        "..##CC#CCCC##..",
        "..##CCC#C#C##..",
        "..##CC#CC#C##..",
        "..###CCCCCCC#..",
        "..##CCCC#C#C#..",
        "..#CC#CC###C#..",
        "..#####.#####..",
        "...............",
        "..............E",
    )
    return _build_ascii_level(layout, seed)


def build_level_so_close_yet_so_far(seed: int = 203) -> GridState:
    layout = (
        "AE.............",
        ".#############.",
        ".#G.....#....#.",
        ".#.#.#.#.#.#.#.",
        ".#..#.....G..#.",
        ".#.#.#.#.#.#.#.",
        ".#........#..#.",
        ".#.#.#G#.###.#.",
        ".#..#.....#..#.",
        ".#.#.#.#.#.#.#.",
        ".#......#...G#.",
        ".#.#.#.#.#.#.#.",
        ".#..G........#.",
        ".###########.#.",
        "...............",
    )
    return _build_ascii_level(layout, seed)


def build_level_measure_twice_and_cut_once(seed: int = 204) -> GridState:
    layout = (
        "A.#############",
        "..#K...........",
        "..############.",
        "...............",
        "....###########",
        "....#..........",
        "....#.#####D###",
        "....#K#........",
        "....#.#.#D#####",
        "....#.#.#...K..",
        "....#.#.#.#####",
        "....#.#.#.#....",
        "....#.#K#.#.#D#",
        "...KD.#.#.D.#..",
        "....#.#.#.#.#.E",
    )
    return _build_ascii_level(layout, seed)


def build_level_ow_oof_owie(seed: int = 205) -> GridState:
    layout = (
        "AL.####E....###",
        ".#...L.####....",
        ".#.#.#...L.###.",
        ".#.#.#.#.#...L.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        ".#.#.#.#.#.#.#.",
        "...#...#...#...",
    )
    return _build_ascii_level(layout, seed)


def build_level_warehouse(seed: int = 206) -> GridState:
    layout = (
        ".#A#...........",
        ".#x#.######.#x#",
        ".#.#........#.#",
        "#..########.#.#",
        "#...........#.#",
        "##########..#.#",
        "......#..####..",
        "......#........",
        "......#.#######",
        "......#.#......",
        "......#.#......",
        "......#.###....",
        "......#...#....",
        "#######.#.#....",
        "E.........#....",
    )
    return _build_ascii_level(layout, seed)


def build_level_fire_resistance(seed: int = 207) -> GridState:
    layout = (
        "A..............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...S...........",
        "..........S....",
        "S..............",
        "...............",
        ".........#L####",
        ".........#L#L.L",
        ".........#.#.#.",
        ".........#L#E#L",
        ".........#.###L",
        ".........#L.LL.",
    )
    return _build_ascii_level(layout, seed, agent_health=1)


def build_level_choose_wisely(seed: int = 208) -> GridState:
    layout = (
        "AP....#########",
        "#.#..#........#",
        "...##########.#",
        "#..#........#.#",
        "..########.#..#",
        ".#.......#...##",
        "########.#.#..#",
        "#....#...#....P",
        "#.#.##.###..##.",
        "#.#....#...#...",
        "#...###..#...#.",
        ".#.......#..#..",
        "P.#.####...#...",
        "...#.....#..#.#",
        ".....#...P...#E",
    )
    return _build_ascii_level(layout, seed)


def build_level_manhattan(seed: int = 209) -> GridState:
    layout = (
        "A.........#....",
        ".#.#.#.#.#.#.#.",
        "......#...B.#..",
        ".#.#.#.#.#.#.#.",
        "..#............",
        ".#.#.#.#.#.#.#.",
        "..#...B.......#",
        ".#.#.#.#.#.#.#.",
        "#...........#..",
        ".#.#.#.#.#.#.#.",
        "....#.......B..",
        ".#.#.#.#.#.#.#.",
        "B.#.........#..",
        ".#.#.#.#.#.#.#.",
        "........#.#...E",
    )
    return _build_ascii_level(layout, seed)


def build_level_khoa_was_here(seed: int = 210) -> GridState:
    layout = (
        "A......L.......",
        ".#...#.L.#...#.",
        ".#..#..L.#...#.",
        ".###...L.#####.",
        ".#..#..L.#...#.",
        ".#...#.L.#...#.",
        "......SLK......",
        "LLLLLLLLLLLLLLL",
        ".......L.......",
        ".#####.L..###..",
        ".#...#.L.#G..#.",
        ".#.E.#.L.###D#.",
        ".#...#.L.#...#.",
        ".#####.L.#...#.",
        "......PL.......",
    )
    return _build_ascii_level(layout, seed, agent_health=1)


def build_level_the_heist(seed: int = 211) -> GridState:
    layout = (
        "A..........B.P.",
        "...............",
        "K.LLLLLLLLLLL..",
        "..L#########L..",
        "..L#C....C.#L..",
        "..L#.......#L..",
        "..L#.C###..#L..",
        "..L#..#G#..#L..",
        "..L#..#D#C.#L..",
        "..L#C......#L..",
        "..L#P...C..#L..",
        "..L#########L..",
        "..LLLLLLLLLLL..",
        "...............",
        "..............E",
    )
    return _build_ascii_level(layout, seed, agent_health=1)


__all__ = [
    "TURN_LIMIT",
    "build_level_thread_the_needle",
    "build_level_hypnosis",
    "build_level_the_vault",
    "build_level_so_close_yet_so_far",
    "build_level_measure_twice_and_cut_once",
    "build_level_ow_oof_owie",
    "build_level_warehouse",
    "build_level_fire_resistance",
    "build_level_choose_wisely",
    "build_level_manhattan",
    "build_level_khoa_was_here",
    "build_level_the_heist",
]
