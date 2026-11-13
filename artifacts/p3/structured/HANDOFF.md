# P3 structured search handoff

Candidate: `agents/p3.py`, byte-identical standalone export `candidate/task1_agent.py`.
SHA256: `a8efb9c0ce3a151d3170785a5ceb69ec76851c5b7a7eb27983d6b70f9b6554f1`; 93,486 bytes; Python 3.10 grammar.
Parent: `agents/p2_next.py`, SHA256 `6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd`.

Final qualification: **PASS** (`qualification.json`). Omarchy Performance profile, P-core CPU0, one library thread, sequential fresh workers. All 326 final episodes win; source/export/fixtures/CPU/thread/watchdog/RSS/replay/deadline/coverage gates pass. There are no remaining failures in the measured suites. No platform submission or active V3 backend selection was changed.

| Suite | Wins | Median episode s | Max episode s | Peak RSS MiB |
|---|---:|---:|---:|---:|
| boss | 6/6 | 0.6462 | 0.9873 | 241.8 |
| public | 66/66 | 0.3726 | 3.2958 | 288.6 |
| generated | 200/200 | 0.3995 | 3.1605 | 292.0 |
| stress | 36/36 | 0.6110 | 3.0310 | 244.8 |
| holdout | 18/18 | 1.1148 | 3.2756 | 245.2 |

All 66 public, 200 generated and six boss rewards pair exactly with P2 next. Public bosses preserve C0=-68 and C1=-163 across seeds 42/43/44. All six previous stress winners preserve reward. The six previous failures win in all three repeats, with one search call per episode (primitive pass plus event rescue), no fallback, and no WAIT/restart between phases.

| Reproduced failure | Parent raw reward | P3 raw reward | Median episode s | Median first verified search win s |
|---|---:|---:|---:|---:|
| case-005 | -450 | -78 | 0.7137 | 0.6762 |
| case-006 | -450 | -143 | 3.0147 | 2.9295 |
| case-007 | -450 | -90 | 0.9434 | 0.8922 |
| case-009 | -450 | -148 | 2.9807 | 2.8948 |
| case-010 | -450 | -108 | 1.5538 | 1.4881 |
| case-011 | -450 | -102 | 1.1574 | 1.0943 |

The holdout wins 18/18 at episode seed 42; independent layout-generation seeds are frozen in `holdout-freeze.json`. Raw rewards in case order 000–017: -108, -174, -159, -123, -123, -123, -174, -129, -123, -153, -147, -123, -106, -132, -99, -129, -123, -103. Unknown optima mean these are not optimality claims. See raw reports for setup, fresh-process wall time, executable traces, package revisions and per-episode state counts. Historical baseline/early experiment power modes were not captured, so their times are not a controlled speedup comparison; the declared benefit is reproduced-loss resolution with preserved paired rewards.

The frozen-source differential gate has 257,467 comparisons across 36,781 frames and 1,000 fully compared many-item layouts, zero mismatches. The extra witness check has 16,296 all-action comparisons. Both are independent public transition/projection checks; old differential corpus is reused regression data.


## Change and measured trigger

The six stress losses reproduce on the parent. Each primitive search hits 4,096 frontier states after reaching only turn 22–35, discards its frontier and rebuilds ordering state on later WAIT observations. Six cases restart 16–21 times without a verified incumbent. Separately profiled diagnostics show 69–80% cumulative cost in ordering, 62–71% duplicate successors and no verification work because no win is proposed. See `diagnostics/root-final-local.json` and `reproduced.json`.

A bounded 16,384-frontier/131,072-record experiment solves three cases, still loses 007/010/011, and peaks at 265.0 MiB. Giving required-gem progress the existing key/door progress credit also solves only three cases. The combined bounded primitive probe still loses 011 (local diagnostic, no acceptance timing claim). Sources and unsuccessful outcomes remain under `experiments/` and `diagnostics/`; `ledger.md`, `decisions.json` and `search-addendum.md` record the decision gates.

P3 first runs the unchanged primitive search. A winning result returns unchanged, including reward-refinement behavior. Only a capped/exhausted nonwinner triggers resource-event search, without dispatching WAIT or renewing any clock. Event goals are item consumption, door removal, changed box configuration or terminal win; coordinates guide ordering only. Each endpoint retains the full FastState and score, including inventory IDs/order, exact turns, health, effect IDs/timers/usage, boxes and terminal flags. Local paths preserve incidental pickups and simulate phasing expiry. Every proposed full primitive win passes public replay and prefix/score comparisons before installation/dispatch. Invalid or expired verification exposes no actions.

Global event frontier/records: 1,024/8,192. One local query at a time: frontier/records 1,024/8,192, depth 24, four complete alternatives per goal. Caps truncate heuristically; there is no completeness/dominance claim. All seven primitive actions, including WAIT, remain eligible. Query ordering caches are bounded as in the parent and discarded with each query; no cross-observation cache/ID renaming is introduced. Rescue returns its first fully verified win; it does not optimize that reward further.

## Evidence boundaries

The simulator/schema and primitive-search body are AST-identical to the parent (`parent-ast-audit.json`); the existing adapter receives only a harmless Optional-backend guard. Existing unsupported schemas still use reference fallback. Full repository discovery: 235/235 tests locally. P3 focused tests: 31/31. Source lint/type/package/grammar checks pass. New endpoint tests independently public-replay doors, boxes, effect expiry and all-action outcomes, reject invalid/expired verification, and preserve an existing primitive winner exactly.

The existing twelve stress cases, public levels and 200 generated cases are reused development/regression data. The 18-case disjoint holdout was frozen before tuning, then evaluated only after the candidate freeze. It uses independently seeded vertical switchbacks, cross corridors and comb rooms; layouts and generation seeds are disjoint; switchback/comb motifs are rotated relatives of development families, so topology motifs and public mechanics are partly reused. `holdout-fixture-frozen.py` preserves initial bytes, and `holdout-format-only.json` proves later import sorting/formatting did not change the normalized generator AST. All 30 development/holdout witnesses independently win; all seven actions at their frames yield 16,296 matching public/compact transitions (`witnesses.json`). Witness validity is not planner success.

Unknown generated optima imply raw rewards, not optimality claims. Event rescue can produce lower rewards than successful primitive experiments on individual former failures. The four alternatives and short local horizon can miss useful resource sequences, safe phasing routes or box arrangements. Large multiple-key inventories, simultaneous pickups/overlapping locks and other guarded schemas remain reference-backed. No learned-perception, image boss or platform result is claimed. `mode="p2b"` remains interface-compatible; acceptance qualifies the default p2a mode.

## Integration

```python
from agents.p3 import Agent, EpisodeBudget, PlannerAdapter, Task3Agent

# Default Task1/2: initial +3.0, cutoff +3.5, external watchdog 5 s.
agent = Agent()

# Explicit structured Task3 entrypoint: initial +11, cutoff +12;
# caller must enforce a 15 s whole-episode watchdog.
boss = Task3Agent()

# A caller-owned budget is never renewed on construction, repair or fallback.
planner = PlannerAdapter(budget, mode="p2a", reference_initial_deadline=None)

# Future V3 handoff ONLY, not integrated here:
planner = PlannerAdapter(
    budget, mode="p2a",
    reference_initial_deadline=min(budget.initial_deadline, budget.started_at + 2.0),
)
```

All search, verification, resource-ID suffix rebinding, fallback, recovery and dispatch use the original absolute deadlines. Cached dispatch is blocked at expiry. Task selection never uses a label or board size. No runtime GPU or local asset imports.

## Reproduce

The dedicated Omarchy directory is `/tmp/p3-structured-20261109`; its environment is the existing `/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python`. `tools/run_p3_acceptance.sh` serializes all five suites under `/tmp/grid-adventure-timing.lock`, pins fresh workers to P-core CPU0 and sets all five library thread limits to one. It refuses to overwrite individual benchmark outputs; move to a fresh result directory for a repeat. `COMPUTE.md` describes V3 reservation/power-profile coordination. Per-suite dependencies, package revisions, profile and per-episode EPP stability are recorded in raw JSON.

```sh
ssh enxi@omarchy
cd /tmp/p3-structured-20261109
bash tools/run_p3_acceptance.sh
```

Local reproducible checks and diagnostic commands are in `commands.sh`. Profiles are always separate from acceptance timings. Run `tools.qualify_p3` against the delivered raw reports to recheck source hashes, exact paired coverage/rewards, watchdogs, one CPU/thread, independent replay and resource limits. Preserve `p2_next.py` as rollback. V3 integration is a later task, with its caller's deadline reserve and perception tests still required.
