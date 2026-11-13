"""Instrumented diagnostic; results must not be used as acceptance timing."""

import cProfile
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path.cwd()))
from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS
from tools.benchmark_p1 import load_agent, provenance

source = "artifacts/p2/p2a-ready/task1_agent.py"
module = sys.modules[load_agent(source).__module__]
module.reference_PROFILE = True
rows = []
for label, builder in PUBLIC_BOSS_LEVEL_BUILDERS.items():
    state = builder()
    state.seed = 42
    state.turn_limit = 150
    original = module.grid_step
    first_win = []
    began = perf_counter()

    def counted_step(current, action):
        successor = original(current, action)
        if successor.win and not first_win:
            first_win.append(perf_counter() - began)
        return successor

    module.grid_step = counted_step
    profiler = cProfile.Profile()
    profiler.enable()
    result = module.reference_search(state, deadline=began + 11.0, mode=module.reference_PRODUCTION_MODE)
    profiler.disable()
    module.grid_step = original
    functions = []
    for item in profiler.getstats():
        code = item.code
        functions.append({"function": code if isinstance(code, str) else code.co_name, "calls": item.callcount, "own_seconds": item.inlinetime, "total_seconds": item.totaltime})
    functions.sort(key=lambda item: -item["own_seconds"])
    row = {"level": label, "profiled": True, "search_seconds": 11.0, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "first_reference_winning_successor_seconds": first_win[0] if first_win else None, "first_win_scope": "public-step winning successor, not separately timed path verification", "stop_reason": result.stop_reason, "winning": result.winning, "reward": result.final_score, "expansions": result.expanded, "transitions": result.simulations, "peak_states": result.peak_states, "peak_records": result.peak_records, "simulation_seconds": result.simulation_seconds, "key_seconds": result.key_seconds, "functions": functions, "provenance": provenance()}
    rows.append(row)
    print(label, row["stop_reason"], row["winning"], row["transitions"], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + "\n")
