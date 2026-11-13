# P3 Measured Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or superpowers:subagent-driven-development if the user selects delegation. Steps use checkbox syntax for tracking. Do not execute a conditional branch before its evidence gate passes.

**Goal:** Resolve measured structured boss limitations while preserving P2's Task 1 results and V3's deadline/resource-rebinding contract.

**Architecture:** Start from the exact P2 checkpoints and isolate budget allocation from solver changes. Preserve the qualified reference fallback while investigating the large-keyed compact-schema exclusion; introduce new search machinery only if post-change measurements require it. A standalone P3 adapter accepts caller-owned absolute deadlines and independently revalidates cached suffixes after resource IDs change.

**Tech Stack:** Existing locked Python environment, public Grid Adventure APIs, standard-library unittest and profiling; Omarchy single-P-core acceptance runs. No new dependency or GPU work.

**Spec:** [Design](../specs/2026-11-09-p3-hierarchical-planner-design.md).

## Global constraints

- Tasks 1/2: 5 seconds per whole episode. Structured Task 3: 15 seconds per whole episode. All: 150 turns, 1.5 GB memory, one CPU core, no GPU.
- Source below 2,000,000 bytes, self-contained, Python 3.10 grammar, standard library and allowed Grid Adventure imports; grader-prelude and unregistered-module compatible.
- Preserve P1/P2 frozen bytes/evidence and V3/V4 perception. Implement in new P3 files; integrate through a separately tested backend contract.
- Use public constructors, fields and black-box `grid_step`; do not inspect installed engine implementation source.
- Caller owns `started_at`, `initial_deadline` and `cutoff`. Initial search, fallback, verification, suffix rebinding and repair share the original absolute deadlines; they may tighten but never extend them. Expiry also prevents cached dispatch.
- Task 1 default remains initial +3.0 s / cutoff +3.5 s. V3 preserves its supplied budget and initial reference cap `min(initial_deadline, started_at + 2.0)`. Structured boss budget-only diagnostics use initial +11.0 s / cutoff +12.0 s under the 15.0 s watchdog, leaving at least 3.0 s after cutoff.
- Local acceptance targets: Tasks 1/2 <4.25 s; structured boss <13.5 s; peak RSS <1,000,000,000 bytes. Hard watchdogs remain 5/15 s. Image boss budget tuning is out of scope.
- Submitted code must not identify tasks by catalog label or infer them from board size. Task-specific callers/entry points provide the budget; standalone Task 1 retains its safe default.

## Review focus

- Late-created fallback receives a fresh clock or loses the caller's reserve: exact deadline propagation tests in Task 2.
- Public pickup changes resource IDs and an old cached action is dispatched: compact/reference suffix rebinding and failure tests in Task 2.
- Large inventories appear equivalent by counts but consume different keys: sparse-ID, ordering/history and all-action tests in Task 3.
- Time-budget changes are misreported as search improvements, or cap stops as losses: exact-source and phase-separated comparisons in Tasks 1/5.
- Candidate correctness/reward is tested only on its own model or old holdouts: independent public replay, generated provenance and bridge regression gates in Tasks 4/6.

## Evidence already obtained

[Baseline receipt](../../../artifacts/p3/baseline/summary.md), 24 fresh-worker episodes and independent trace replay:

| Case | Unchanged P2a/P2b | Budget-only P2a/P2b | Consequence |
|---|---|---|---|
| C0 | Each 3/3 wins, reward -70 | Each 3/3 wins, catalog reward -68 | Longer internal allowance fixes the measured reward gap |
| C1 | Each 0/3, reward -450 | Each 0/3, reward -450, 150 WAIT actions | More time alone does not solve it |

Both bosses reject initial fast encoding due to `Large keyed inventory iteration history` and run the same reference search. The separately profiled C1 root search spends 9.671/11.004 seconds in reference simulation, stops at its deadline, peaks at 575/4096 frontier states and 907/32768 records, and finds no winning successor. This selects **large-keyed coverage/reference transition cost** as the first investigation. It does not prove that faster simulation alone solves C1. Increasing caps or implementing hierarchy is not the next unconditional task.

## File responsibilities

| File | Responsibility |
|---|---|
| `artifacts/p3/baseline/` | Completed raw boss/coverage/profile evidence, exact variants, runner scripts and checksums |
| `agents/p3.py` | New standalone runtime fork; caller-owned budget adapter; only evidence-qualified solver changes |
| `tests/test_p3_adapter.py` | Backend substitution contract, deadline propagation, atomic suffix rebind/repair |
| `tests/p3_fixtures.py` | Public-constructor resource-order counterexamples and later disjoint stress families |
| `tests/test_p3_resources.py` | Resource ordering and successor equivalence regressions |
| `tools/probe_p3_resources.py` | Black-box characterization, minimized counterexamples and schema decision report |
| `tools/benchmark_p3.py` | Structured public/boss runner, caller budget profiles and separate diagnostics |
| `tools/qualify_p3.py` | Task-aware exact-source/reward/resource/coverage gates |
| `tests/test_p3_qualification.py` | Fail-closed evidence validation and wrong-budget rejection |
| `tests/test_p3_package.py` | Standalone export, loader and dependency checks |
| `artifacts/p3/decisions.json` | Measured trigger, selected experiment, outcome and stop reason for each change |

Use a scoped worktree/checkout at execution time only if it can include the current untracked inputs. The shared repository already contains unrelated untracked P1/P2/V3 work; do not stage all files, reset, switch branches or commit someone else's work. Each task is a logical commit boundary; stage only files owned by that task after confirming provenance.

## Task 1: Preserve the baseline and activate the measured branch

**Files:** `artifacts/p3/baseline/*`; create `artifacts/p3/decisions.json` at implementation.

**Interfaces:** Existing `tools.benchmark_p1.run_case(agent_spec, level_label, seed, *, evaluation_limit, setup_limit=30.0, profile=False) -> dict`; completed runner temporarily extends the catalog in its own process to include bosses, leaving source catalogs untouched.

- [x] Verify frozen hashes locally/remotely, live CPU/load and P-core affinity; run unchanged and four-constant budget variants under a 15-second watchdog, three repeats each.
- [x] Replay all 24 boss traces plus 22 public and 200 generated legacy traces at recorded seeds; compare final win/reward exactly. Audit every frame's encoding eligibility and preserve raw rejection reasons.
- [x] Profile root reference searches separately; label instrumentation, first public winning successor versus independently timed verification, and cap/deadline stops.
- [ ] Record the selected first branch in `decisions.json`: `large_keyed_coverage`, triggered by C1 loss and reference simulation cost. Record C0 as `budget_only_reward_gap_resolved`; keep state-cap/hierarchy work inactive.
- [ ] Recheck current source/runner hashes before implementation. The preserved diagnostic commands are in the receipt/runner; do not rerun all episodes unless source, environment or a specific concern changed.

**Deliverable:** A baseline that distinguishes the existing constraint, measured C1 failure, C0 budget fix and untested remedies.

## Task 2: Implement a caller-budget adapter with resource-ID recovery

**Files:** create `agents/p3.py`, `tests/test_p3_adapter.py`; read `agents/v3_planner.py:2110`, `agents/v3_types.py`, `tests/test_v3_harness.py`. Do not edit the existing V3 adapter during this task.

**Interfaces:** Define a local `EpisodeBudget(started_at: float, initial_deadline: float, cutoff: float)` record, accepting any caller object with those fields. Export `PlannerAdapter(budget, mode="p2a", reference_initial_deadline=None)`, `step(GridState) -> Action`, and `info() -> dict[str, str]`. Preserve parent `encode`, `read_state`, `fast_step`, `search`, `verify_plan` and reference backend interfaces. Standalone `Agent()` remains Task-1-safe; a structured boss entry point supplies its explicit budget.

- [ ] Add tests first: `test_delayed_construction_preserves_caller_deadlines`, `test_late_fallback_receives_original_deadlines`, `test_v3_reference_reserve_is_preserved`, `test_budget_tightening_is_monotonic`, `test_expired_budget_blocks_cached_dispatch`. Inject a deterministic clock and assert initial/cutoff never exceed supplied values, fallback initial <= caller-provided reference deadline, and expired dispatch returns WAIT without starting work.
- [ ] Add `test_compact_pickup_ids_rebind_without_new_search`, `test_reference_key_ids_rebind_without_new_search`, `test_effect_ids_and_counters_survive_rebinding`, and `test_invalid_or_expired_rebind_never_installs_stale_suffix`. Reuse the public metadata/reconstruction flow from V3's existing tests; assert the successful sequence `[RIGHT, PICK_UP, RIGHT]`, one original search, public replay win, and bounded repair on failures.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_p3_adapter -v`; confirm missing P3 implementation is the failure, not environment setup.
- [ ] Fork frozen P2a into `agents/p3.py`, record its parent hash, and port the already-working V3 adapter behavior with explicit caller-provided reference reserve. Add an injectable clock consistently across search, reference fallback, replay and rebinding. Preserve search policy/schema; a budget adapter is not a solver rewrite.
- [ ] Install rebindings atomically only after full public replay wins. On failure, clear stale actions/keys before bounded repair. Pass the original initial/cutoff into every nested operation; construction time never creates extra allowance. A tighter runtime budget must also clamp already-created fallback instances.
- [ ] Run the adapter tests plus existing `tests.test_v3_harness`; require all to pass. Keep V3 source selection unchanged. Archive an adapter-only checkpoint and a focused diff against its parent.

**Deliverable:** A backend can be substituted without losing deadline reserves or public-ID recovery. No claim of better search.

## Task 3: Characterize the large-keyed boundary before widening it

**Files:** create `tests/p3_fixtures.py`, `tests/test_p3_resources.py`, `tools/probe_p3_resources.py`; output `artifacts/p3/resources/{probes,counterexamples,schema-decision}.json` and a concise `schema-contract.md` if support is proved.

**Interfaces:** `resource_cases() -> iterable[(case_id, GridState, tuple[Action, ...])]`; `probe(source: Path, output: Path) -> dict`. Reports contain source/package hashes, case/seed, action prefix, all-action public outcomes, full ordered component projection, mismatch details and `decision` equal to `supported_extension` or `keep_guarded`.

- [ ] Reproduce the guard on C0, C1 and B4, then construct reachable histories with 5, 8, 16 and 32 carried/available coin/gem/key items, up to five keys/doors, sparse/reordered public IDs, repeated pickups/key use, adjacent/current-cell doors, and live re-identification. Include collisions and order changes using public IDs; do not read engine internals.
- [ ] Compare all seven actions at every selected reachable prefix using an independent public-field projection. Preserve identity-to-component association, inventory/effect order, counters, score and terminal flags. Test pairs with equal counts/masks but different histories; count equivalence is never assumed.
- [ ] First tests assert frozen P2's exact rejection reason and the replay validity of every fixture. Run `uv run --frozen --no-sync python -m unittest tests.test_p3_resources -v` and verify deterministic probes before testing a candidate extension.
- [ ] Investigate the least invasive representation/transition change that accounts for observed ordering. If public observable state is insufficient to determine successors, keep those cases reference-backed. Do not remove the >4 guard merely because one ordering recipe passes a few probes.
- [ ] Write `schema-decision.json` and `schema-contract.md`: eligible schemas, exact state fields/token semantics, transition order, fallback conditions, independent test evidence, and known limitations. Minimize every mismatch into a fixture with its action trace.
- [ ] Gate Task 4 on `supported_extension`. If characterization remains unresolved after one focused development pass, preserve guards and record the alternative: separately investigate reducing reference transition overhead through supported public APIs, or prepare the conditional search experiment in Task 5. No automatic broad rewrite.

**Deliverable:** A concrete, reviewed semantic contract or a justified retained guard. This task is allowed to conclude that full compact support is not yet safe.

## Task 4: Implement only the characterized coverage improvement

**Files:** modify `agents/p3.py`, `tests/test_p3_resources.py`; add targeted regressions to P3 tests only. Reuse `tools/differential_p2.py` as a read-only independent oracle or create a P3 wrapper if report field names require it.

**Interfaces:** The existing `encode(observation) -> (Board, FastState, score)`, `read_state(board, observation) -> (FastState, score)`, and `fast_step(board, state, action) -> (FastState, reward_delta)` stay compatible. Any necessary new state fields are fixed by Task 3's contract before code changes, and retained by keys, live reads and replay. No schema widening outside that contract.

- [ ] Convert minimized failing supported-extension examples into RED tests, with exact reference successor/score/flags/order assertions. Preserve overlapping-box, nonpositive-counter, overlapping-lock, simultaneous-pickup, timer and sparse-ID regressions from P2.
- [ ] Implement the single characterized change, leaving unsupported schemas guarded and preserving the working reference policy. Every new executable candidate still passes public prefix replay; invalid upgrades preserve a compatible verified incumbent.
- [ ] Run P3 resource/adapter tests and the full existing P1/P2 suite. Run independent differential qualification on exact candidate bytes: >=100,000 comparisons, >=1,000 layouts, 22 public traces and the new large-keyed corpus. Require zero known supported-schema mismatches and report unsupported prefixes explicitly.
- [ ] Re-measure both bosses at the same 11/12/15 budget and unchanged caps. If C1 still fails, capture first-win/refinement profiles before selecting any second change. Preserve the coverage-only checkpoint separately.

**Deliverable:** A correctly widened fast path with measured impact, or an explicitly rejected experiment. It is not assumed that this alone solves C1.

## Task 5: Select remaining search work from post-coverage measurements

**Files:** create `tools/benchmark_p3.py`, `tests/test_p3_qualification.py`; update `artifacts/p3/decisions.json`. Any new search design is saved as a linked addendum before its implementation.

**Interfaces:** `benchmark(source: Path, suite: str, budget_profile: str, repeat: int, output: Path) -> None`, with suites `public`, `boss`, `p1-regression`, `p2-v2`, `p3-development`, `p3-heldout`; profiles `task1`, `boss15`, `v3-oracle`. CLI exposes those exact values. Diagnostic modes must explicitly mark instrumentation and record actual absolute-deadline allocations.

- [ ] Add evidence-validation tests rejecting a boss run with a 5-second watchdog, a 15-second watchdog falsely labelled as changed internal budgets, mixed source hashes, missing pairs/seeds, profiled acceptance timing, or hidden inner fallback metrics.
- [ ] Capture per-backend first verified win, best reward over time, transition/ordering/replay costs, and frontier/record limits. Keep acceptance agents uninstrumented; separately hashed diagnostic variants may expose extra metrics.
- [ ] If cap exhaustion now demonstrably blocks C1, run one controlled cap-doubling experiment while holding policy/budget fixed and verifying RSS. If ordering now dominates, test bounded exact-state/phase/origin caches. Do not implement both in one experiment.
- [ ] If C1 wins at the required reward after budget/coverage changes, stop adding machinery. If first-win/reward deficits persist, write a focused subgoal-search addendum using the spec's conditional design: complete resource-sensitive endpoints, all primitive actions, verified incumbents, bounded local alternatives and original deadlines. Fix exact worker interfaces/scheduling and acceptance criteria in that addendum before implementation.
- [ ] For every branch, predeclare the affected cases and benefit criterion, then compare paired results. Failed experiments remain experimental artifacts; do not weaken thresholds after inspecting outcomes.

**Deliverable:** The minimum measured change needed, with a causal comparison. Hierarchy is implemented only through a justified follow-on plan.

## Task 6: Exact-artifact qualification and replacement handoff

**Files:** create `tools/qualify_p3.py`, `tests/test_p3_package.py`; output `artifacts/p3/candidate/`, `qualification.json`, `HANDOFF.md`. Use `tools/package_p1.package` for byte-preserving standalone exports. Keep frozen P2 and current V3 selection intact until replacement qualification succeeds.

**Interfaces:** `qualify(candidate_report: dict, baselines: list[dict], manifest: dict, profile: str) -> dict`, with explicit per-task budget/resource limits and paired coverage checks; return `passed`, `checks`, `regressions`, `evidence_scope`.

- [ ] Add tests for Python 3.10 parsing, prelude/no-module-registration load, no local runtime imports/files, exact export hash, and rejection of mixed hashes/empty or incomplete reports. Assert compatible deadline and re-identification behavior on the exported bytes too.
- [ ] Freeze development and new heldout definitions before tuning; include actual large-keyed interactions and any measured C1 deficiency, plus mixed mechanics, low health and 15x15 boards. Separate geometry families/seeds; generated rewards stay raw unless independently proved optimal.
- [ ] Freeze the candidate and all budget/routing choices before final results. Run public22 x3, both bosses x3, P1's 100 and P2-v2's 200 cases, and the new declared heldout suite sequentially on one recorded P-core.
- [ ] Require no lost paired wins/rewards against frozen P2a/P2b and the best budget-only boss baseline, public catalog rewards, targeted C1 failure resolved, no timeouts/errors/mismatches, and task-aware time/RSS limits. Any unresolved boss failure blocks a claim of complete P3 solver success.
- [ ] Inject the candidate through a test-only planner factory into the existing V3 oracle flow, retaining its original budget and +2.0 s reference reserve. Require 22/22 and identical paired P2a rewards; rerun both compact/reference resource-ID tests. This checks backend compatibility without adding perception features or claiming learned-image performance.
- [ ] Run the full relevant unittest suite, Ruff/format and ty checks as in the P2 handoff. Verify exact source hashes again after testing. Package source, decisions, fixtures, logs, manifests and qualification; report local versus user-reported platform outcomes separately.
- [ ] Hand over the candidate and retain the frozen P2 rollback. Do not submit graded attempts or change V3's default backend as part of planning/diagnostic work.

## Execution order and stop rules

Task 1 diagnostics are complete. Execute Task 2, then Task 3; Task 4 depends on proved coverage support. Task 5 selects only remaining measured work, and Task 6 qualifies the final candidate. One focused characterization pass precedes any decision to pursue a substantially different backend. Keep a usable P2 checkpoint throughout and account for the 13 November assignment deadline.

The expected first implementation is **budget/rebinding compatibility plus large-keyed coverage characterization**, not a hierarchical rewrite. No claim is made that the planned accelerator or optional hierarchy already works.
