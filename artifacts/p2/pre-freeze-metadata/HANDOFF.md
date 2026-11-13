	# P2 handoff

P2a and P2b are implemented and locally qualified. **Use P2a as the recommended checkpoint.** P2b is a separately preserved experimental variant: its safe reward bounds reduce planning time, but the paired suites show identical rewards and no demonstrated first-win improvement. This did not meet the plan's P2b promotion benefit gate.

| Exact artifact | SHA256 | Public | New heldout | P1 regressions |
|---|---|---|---|---|
| [Recommended P2a](candidate/task1_agent.py) | `e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb` | 66/66 | 200/200 | 100/100 |
| [P2b](p2b-ready/task1_agent.py) | `551ee00377ffe42a343d18c887b1c7f36ef4baf4fa0fff6a22b0053c9845edd0` | 66/66 | 200/200 | 100/100 |

Both public sweeps comprise three repetitions of all 22 cases, each at catalog maximum reward (effective performance 1.0). Neither variant loses any frozen-baseline win or per-case reward on paired wins. The new maze/branch heldout suite has 200 cases across ten mechanic families, including 15x15 and low-health maps; P1 won 197/200. Generated optima are unknown: use raw rewards, not estimated marks. The earlier shared-topology suite is preserved as regression evidence, not independent heldout evidence.

P2a public mean/max: 0.740/3.381 seconds; peak RSS 282.4 MB. P2b: 0.602/3.378 seconds; peak RSS 281.8 MB. Frozen P1 public mean was 3.069 seconds. All episodes stayed below 4.25 seconds and 1 GB. Qualification used fresh workers, a five-second evaluation watchdog, one P-core per runner and one library thread. Linux identifies CPUs 0–7 as P-cores; 8–23 are E-cores. Sources import in about 0.12 seconds in fresh pinned processes; construction is a few microseconds. Torch is not imported by the standalone agent, and no GPU/training is used.

Each exact variant passes **128,912 differential comparisons**, over 1,000 development layouts plus 100 sparse-ID resource traces and every public P1 winning trace. All mandatory categories have coverage. No known mismatches remain within the declared fast schema. The differential oracle projects public components independently, including key-sensitive inventory order. All 113 unit tests, Ruff lint/format and ty with warnings as errors pass. A fresh independent review found three important issues; the single fix pass and subsequent public-trace audit are documented in [review.md](review.md). There are no deferred minor findings.

The fast simulator supports standard cardinal movement, coins/gems/keys, doors, box groups and box/item overlaps, hazards/health, boots, phasing, and positive active effect stacks. It deliberately routes simultaneous ground pickups, overlapping locks, nonpositive/noninteger power counters, unfamiliar schemas, and keyed episodes with more than four possible inventory items to the namespaced frozen P1 backend. B4 uses that reference path. This costs planning time on those configurations; all searches, repairs and verification still share the original episode cutoff. Coin/gem order is canonicalized only when no keys remain. Source and [rulings](rulings.md) describe these limits. No Task 2 image parser is included.

Only reference-replayed winning fast plans can be dispatched or establish a pruning threshold. Every intermediate semantic key and score must match; a failed/expired upgrade preserves a previously verified incumbent. Caps remain 4,096 frontier states and 32,768 retained records. Results stopped by caps/deadlines are best found, not claims of optimality.

The source in `agents/p2.py` matches the recommended candidate byte-for-byte. The only policy difference between the final P2a and P2b files is the production mode constant; both retain both search modes and a self-contained reference backend. The package parses as Python 3.10 grammar, loads after the grader prelude, and runs without registering its module. Earlier `p2a`, `p2b`, `*-final` and `*-release` directories are **superseded development evidence**; use only `*-ready`/`candidate` sources for this handoff.

**P2 website Debugging remains pending.** Paste the entire selected standalone source, including imports/helpers. No website or graded submission was performed. P1 remains unchanged at `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5`, with its user-reported 22/22 Debugging pass and frozen rollback bundle intact.

Reproduce from the repository's locked environment:

```sh
uv run --frozen --no-sync python -m unittest discover -s tests -p 'test_p*.py' -v
ruff check agents tests tools
ruff format --check agents tests tools
ty check agents tests tools --error-on-warning
uv run --frozen --no-sync python tools/differential_p2.py --source artifacts/p2/candidate/task1_agent.py --layouts 1000 --comparisons 100000 --seed 20261108 --output /tmp/p2-differential.json
# Linux: the worker pins to the one allowed CPU and limits library threads.
taskset -c 0 uv run --frozen --no-sync python tools/benchmark_p2.py --agent artifacts/p2/candidate/task1_agent.py --suite public --repeat 3 --output /tmp/p2-public.json
```

Qualification reports: [P2a](p2a-ready/qualification.json), [P2b](p2b-ready/qualification.json). Raw reports contain executable action traces, resource measurements, package revisions, watchdogs and source hashes. The frozen v2 manifest and correctness-revalidation receipt separate initial selection from the later public-triggered fixes. Offline evidence is also retained at `/home/enxi/coding/p2-fast-planner-20261108` on omarchy. Changes are left in the shared detached checkout; no unrelated files were staged, committed, pushed or removed.
