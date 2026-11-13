"""Diagnostic public-case simulator-call counts, separate from acceptance timing."""
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import agents.p1 as planner
from grid_adventure.grid import step
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

outcomes = []
for label in ('B5 Ow Oof Owie', 'B6 Warehouse'):
    state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()
    state.turn_limit = 150
    calls = [0]
    first = []
    start = perf_counter()

    def transition(current, action):
        calls[0] += 1
        successor = step(current, action)
        if successor.win and not first:
            first.append({'simulation_calls': calls[0], 'seconds': perf_counter() - start,
                          'reward': successor.score})
        return successor

    with patch.object(planner, 'grid_step', transition):
        result = planner.search(state, deadline=600, mode='guided',
                                clock=lambda: float(calls[0]))
    outcomes.append({'level': label, 'first_win': first, 'winning': result.winning,
                     'best_reward': result.final_score, 'simulation_calls': calls[0]})

report = {'diagnostic_only': True, 'source_sha256': sha256(Path('agents/p1.py').read_bytes()).hexdigest(),
          'scope': 'Public transition calls under a deterministic600-call budget; seconds are local diagnostic timing.',
          'results': outcomes}
Path('artifacts/p1/detour-ordering/first-win.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
