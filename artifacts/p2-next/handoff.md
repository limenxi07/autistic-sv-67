# P2 next: structured boss and latency handoff

**Candidate:** `agents/p2_next.py` (standalone, 84,390 bytes, Python 3.10 grammar).

`SHA256 6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd`

The smallest validated change fixes C1 without P3 or larger caps. Frozen P1/P2, existing evidence and active V3 source are untouched (22 protected hashes checked). No platform submission or automatic V3 integration occurred.

## Measured boss results

Omarchy, Intel Core Ultra 9 275HX, P-core CPU 0; seeds 42/43/44; sequential fresh episode workers with one library thread. Every trace independently reproduces reward/status. All episode acceptance is unprofiled and uses the exact hashed source. Setup has a separate 30-second watchdog; boss evaluation has a 15-second watchdog.

| Source / initial-cutoff | Boss | Wins | Reward | Median episode s | Median fresh-process wall s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| P2a 3.0/3.5 | C0 | 3/3 | -70 | 3.0313 | 3.6833 | 286.3 |
| P2a 3.0/3.5 | C1 | 0/3 | -450 | 3.7383 | 4.3924 | 296.2 |
| P2a budget-only 11/12 | C0 | 3/3 | -68 | 10.0062 | 10.7242 | 380.5 |
| P2a budget-only 11/12 | C1 | 0/3 | -450 | 12.2557 | 12.9483 | 425.4 |
| P2 next Task3 11/12 | C0 | 3/3 | -68 | 0.3325 | 1.0750 | 241.6 |
| P2 next Task3 11/12 | C1 | 3/3 | -163 | 1.0585 | 1.7948 | 241.5 |

Candidate first **fully reference-verified** wins occur at median **0.041378 s (C0)** and **0.588161 s (C1)** from the search origin. This includes public replay and prefix/score comparisons. Setup medians are 0.6867/0.6841 s. Fresh-process wall includes interpreter startup/import, evaluation and teardown; OS caches were not flushed. Candidate episode maxima are 0.3367/1.0857 s; fresh-process maxima 1.1394/1.8481 s.

Both candidate bosses use one fast search, no reference fallback, and finish on `state_cap` while retaining an independently verified winning incumbent. C0: 22,200 transitions, 7,280 records. C1: 23,333 transitions, 7,425 records. Both peak at the original 4,096-state frontier; record limit stays 32,768. C1 is five reward points below catalog -158; no optimum claim.

Frozen P2a uses reference fallback on both bosses. The separately hashed budget-only copy changes four deadline constants, not the search policy. More reference time alone never wins C1. See [baseline report](baseline-report.md) for all rejection sites, setup/RSS/stop details and separately instrumented first-win diagnostics. The diagnostic replay adds overhead (short C0 reward -72 versus acceptance -70); its first-win times are not exact-source acceptance timings.

## Changes justified by evidence

1. Support one-key/many-item inventories. Inventory order is canonical only with at most one key, or after all keys are gone. A lock then has at most one matching key identity; coins/gems cannot change key choice. IDs, quantities, resource positions and key consumption remain exact. Multiple-key inventories above the original four-resource threshold still fall back. Sparse-ID/permuted-order and mixed-inventory probes validate this quotient.
2. Preserve key bottlenecks in compact primitive ordering even when an uncollected ghost exists. The old C1 search reached its frontier cap after 18,032 transitions without carrying a key or opening a door. A fourfold frontier experiment still lost and slowed C0; it was rejected. The ordering change finds C1 at the original caps. It changes ordering only, with no extra pruning.
3. Fix two public-probed simulator cases found by the new corpus: boxes cannot be pushed onto lava; phasing still pushes movable boxes and overlaps them when a push is blocked. The first failures and red/green tests are retained. [Box probes](box-probes.json) cover 24 phased/unphased destination cases.
4. Add a structural absolute-budget adapter and explicit Task3 entrypoint. Search, verification, rebinding, reference recovery and dispatch share the original cutoff. Checks after observation/search prevent crossing the cutoff during dispatch preparation. A late reference constructor immediately uses the original recovery phase. Cached suffixes are installed atomically only after public verification.

Effect IDs and individual timers/usage counters, health, exact turns, box positions and terminal flags remain in state. Simultaneous pickups, overlapping locks, nonstandard amounts/counters, unknown schemas and live state changes retain conservative reference fallback. All 24 explicit rejection sites are listed in [coverage-final.json](coverage-final.json). Initial eligibility is not proof of full-trace equivalence.

## Validation

| Gate | Result |
|---|---|
| Existing repository tests | 161 pass; 19 targeted candidate tests pass |
| Differential public transitions | 257,467 comparisons, 36,781 reachable frames; zero mismatches |
| New many-item layouts | 1,000 initially supported and 1,000 fully compared, sparse/permuted IDs |
| Public gameplay | 66/66 wins; every paired reward equals frozen P2a and P2b; max episode 3.2455 s |
| Existing generated regressions | 200/200 wins; every paired reward equals frozen P2a and P2b; max episode 3.2133 s |
| Unchanged V3 oracle | 22/22 wins and paired P2a rewards; max episode 2.8412 s |
| Separate candidate adapter oracle probe | 22/22 wins and paired P2a rewards; max episode 2.9427 s |
| Source/package | 84,390 bytes; Python 3.10 grammar; standalone unregistered-module/grader-prelude load |
| Resources | All acceptance rows one CPU/thread; maximum structured-suite RSS 305,426,432 bytes; no runtime GPU |

Oracle checks validate the integration boundary using supplied labels; they do not establish perception accuracy. The candidate oracle wrapper is diagnostic-only: [candidate_oracle_probe.py](candidate_oracle_probe.py). Its dependency hashes are in [candidate-v3-manifest.json](candidate-v3-manifest.json) and [remote-source-hashes.txt](remote-source-hashes.txt). The unchanged oracle runner reports its own absolute remote filename; the evidence verifier maps that checkout prefix to the local repository before hash comparison.

| Re-encoded legacy corpus | Frozen eligible frames | Candidate eligible frames | Remaining rejection |
|---|---:|---:|---|
| Initial public24 | 21/24 | 23/24 | B4 large multiple-key inventory |
| Public22 traces | 736/803 | 736/803 | 67 large multiple-key frames |
| P2-v2 generated traces | 6716/6856 | 6776/6856 | 80 simultaneous-pickup/overlapping-lock frames |
| Frozen boss traces | 0/178 | 178/178 | none |
| Candidate boss winning traces | — | 86/86 | none |

A backend selected as reference stays reference; frame eligibility does not imply automatic switching. New differential coverage includes all seven actions, key consumption, mixed inventories, stacked effects, expiry, boxes, overlaps, existing traces and both winning boss traces.

## Unfamiliar boss-like cases and remaining failures

Twelve separately seeded snake/chamber/bay layouts were evaluated once at episode seed 42. Optima are unknown: these are raw rewards. Every layout has a public-replayed winning witness, and every action at its witness frames matches the final simulator under seed 42 ([witness validation](stress-witness-seed42.json)).

| Case | Result | Raw reward | Episode s | Searches | Final stop |
|---|---|---:|---:|---:|---|
| case-000 | win | -55 | 0.2766 | 1 | state_cap |
| case-001 | win | -48 | 0.4005 | 1 | state_cap |
| case-002 | win | -50 | 0.3432 | 1 | state_cap |
| case-003 | win | -73 | 0.4422 | 1 | state_cap |
| case-004 | win | -50 | 0.3982 | 1 | state_cap |
| case-005 | loss | -450 | 12.1478 | 21 | deadline |
| case-006 | loss | -450 | 12.1783 | 16 | deadline |
| case-007 | loss | -450 | 12.1728 | 18 | deadline |
| case-008 | win | -84 | 0.5101 | 1 | state_cap |
| case-009 | loss | -450 | 12.1737 | 16 | deadline |
| case-010 | loss | -450 | 12.1900 | 16 | deadline |
| case-011 | loss | -450 | 12.2284 | 17 | deadline |

Six stress losses remain. They stay on the fast backend, reach the original frontier cap, repeat 16–21 primitive searches, find no verified incumbent and finally stop at the original deadline. They are search/generalization failures, not claimed solver coverage. No hierarchy was needed for C0/C1. These cases justify a future measured comparison of continued primitive search/ordering before resource-aware subgoals, under the existing [conditional P3 design](../../docs/superpowers/specs/2026-11-09-p3-hierarchical-planner-design.md). No P3 code was introduced.

Existing public/generated outcomes and early development/stress outcomes were reused after correctness fixes. Final stress results are revalidation, not an untouched holdout. No heuristic tuning used final stress rewards. Unknown generated optima are never presented as optimal rewards.

## Integration interface

Use `time.perf_counter()` for all budget fields. The caller reserves execution/perception time after cutoff; callers may tighten but never extend the original values. Default `Agent()` keeps Task1/2 initial +3.0/cutoff +3.5 under the external five-second episode limit. `Task3Agent()` explicitly chooses +11/+12 under the external 15-second limit. Pass an existing budget when construction occurs after episode start.

```python
from agents.p2_next import EpisodeBudget, PlannerAdapter, Task3Agent

# Structured Task 3: started_at is the ORIGINAL episode origin.
boss = Task3Agent(EpisodeBudget(started_at, started_at + 11, started_at + 12))
action = boss.step(structured_observation)

# Optional future V3 wiring; no active V3 file was changed.
planner = PlannerAdapter(
    budget, mode="p2a",
    reference_initial_deadline=min(budget.initial_deadline, budget.started_at + 2.0),
)
action = planner.step(reconstructed_state)
```

The adapter has no vision imports. It rechecks remaining primitive suffixes against public transitions after resource-ID changes; verification is conditional on the supplied structured state. Mode `p2b` remains available, but this checkpoint qualifies the P2a production policy. Board size and catalog labels never select a budget.

## Reproduce

The Omarchy isolation directory and existing environment are retained. Commands refuse to overwrite benchmark evidence; use new output paths for another run. The harness sets all five library thread environment variables to one before spawning workers.

```sh
ssh enxi@omarchy
cd /home/enxi/coding/p2-next-20261109
PY=/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python
sha256sum agents/p2_next.py
taskset -c 0 "$PY" -m tools.benchmark_boss --source agents/p2_next.py --entrypoint Task3Agent --repeat 3 --output /tmp/p2-next-boss-recheck.json
taskset -c 0 "$PY" -m tools.benchmark_boss --source agents/p2_next.py --suite public --repeat 3 --output /tmp/p2-next-public-recheck.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 timeout 600 taskset -c 0 "$PY" -m tools.differential_p2_next --source agents/p2_next.py --output /tmp/p2-next-diff-recheck.json
taskset -c 0 "$PY" -m tools.benchmark_boss --source agents/p2_next.py --suite heldout-v2 --repeat 1 --output /tmp/p2-next-generated-recheck.json
taskset -c 0 "$PY" -m tools.benchmark_boss --source agents/p2_next.py --entrypoint Task3Agent --suite stress --count 12 --repeat 1 --output /tmp/p2-next-stress-recheck.json
taskset -c 0 "$PY" -m tools.benchmark_v3 --bridge-only --output /tmp/p2-next-original-oracle-recheck.json
taskset -c 0 "$PY" -m tools.benchmark_v3 --observation oracle --source artifacts/p2-next/candidate_oracle_probe.py --output /tmp/p2-next-candidate-oracle-recheck.json
```

From the local repository, audit the delivered evidence and rerun tests:

```sh
.venv/bin/python -m tools.verify_p2_next_evidence --output /tmp/p2-next-evidence-recheck.json
MPLCONFIGDIR=/tmp/p2-next-mpl .venv/bin/python -m unittest discover -s tests
```

See [verified-summary.json](verified-summary.json) for distributions/coverage, [decision ledger](decision-ledger.json) for rejected experiments and review fixes, and `*-v3.json` for final exact-source episode evidence. Earlier v1/v2 files are intentionally retained as diagnostic/failed checkpoints. Frozen baseline first-win diagnostics are separate from acceptance evidence.

The `v3` suffix in new candidate/result filenames denotes the third P2 validation revision, separate from the V3 vision release.
