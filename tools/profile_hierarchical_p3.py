"""Profile primitive search separately from acceptance timing."""

import argparse
import cProfile
import hashlib
import json
import pstats
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from tests.boss_p2_next_fixtures import boss_like
from tools.benchmark_planner_p1 import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, nargs="+", default=[5, 6, 7, 9, 10, 11])
    a = p.parse_args()
    assert not a.output.exists()
    m = sys.modules[load_agent(a.source).__module__]
    rows = []
    for i in a.cases:
        obs = boss_like(i, split="heldout")[0]
        obs.seed = 42
        b, s, score = m.encode(obs)
        counts = Counter()
        seen = set()
        values = {s: score}
        verification = {"calls": 0, "seconds": 0.0, "statuses": []}
        original = m.fast_step

        def step(
            board,
            state,
            action,
            original=original,
            counts=counts,
            seen=seen,
            score=score,
            b=b,
            s=s,
            values=values,
        ):
            n, d = original(board, state, action)
            counts["transitions"] += 1
            next_value = values[state] + d
            counts["duplicates"] += (n, next_value) in seen
            seen.add((n, next_value))
            if n in values and values[n] != next_value:
                counts["nonunique_state_score"] += 1
            values[n] = next_value
            counts["max_turn"] = max(counts["max_turn"], n.turn)
            counts["max_gems"] = max(
                counts["max_gems"], (b.gems & ~n.items).bit_count()
            )
            counts["max_items"] = max(counts["max_items"], n.inventory.bit_count())
            counts["doors_opened"] += n.doors != s.doors
            counts["loss"] += n.lose
            return n, d

        m.fast_step = step
        profile = cProfile.Profile()
        start = perf_counter()
        deadline = start + 11

        def verify(
            actions, keys, b=b, obs=obs, deadline=deadline, verification=verification
        ):
            before = perf_counter()
            result = m.verify_plan(b, obs, actions, keys, deadline=deadline)
            verification["calls"] += 1
            verification["seconds"] += perf_counter() - before
            verification["statuses"].append(result.status)
            return result

        profile.enable()
        r = m.search(
            b,
            s,
            score,
            deadline=deadline,
            mode="p2a",
            verify_candidate=verify,
        )
        profile.disable()
        m.fast_step = original
        stats = pstats.Stats(profile)
        top = [
            {
                "function": str(k),
                "calls": v[1],
                "self_seconds": v[2],
                "cumulative_seconds": v[3],
            }
            for k, v in sorted(
                stats.stats.items(), key=lambda p: p[1][3], reverse=True
            )[:25]
        ]
        rows.append(
            {
                "case": i,
                "seconds": perf_counter() - start,
                "metrics": dict(counts),
                "verification": verification,
                "result": r._asdict(),
                "profile": top,
            }
        )
        print(i, r.stop_reason, r.winning, dict(counts), flush=True)
    a.output.write_text(
        json.dumps(
            {
                "instrumented": True,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "source_sha256": hashlib.sha256(
                    Path(a.source).read_bytes()
                ).hexdigest(),
                "results": rows,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
