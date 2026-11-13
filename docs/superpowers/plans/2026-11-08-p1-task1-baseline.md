# P1 Task 1 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a small, reproducible Task 1 agent that completes levels within the episode budget and can serve as our first submission candidate.

**Architecture:** Retain the public `grid_adventure.grid.step` simulator. First repair the starter's search with a complete state key, parent pointers and cached plans; then evaluate bounded best-first ordering and reward improvements against that checkpoint. Keep all submitted logic in one source module and put benchmarking, fixtures and packaging outside it.

**Tech Stack:** Python 3.13, standard library, the repository's pinned Grid Adventure/Grid Universe packages, `unittest`, and the existing public evaluation utilities. CPU only; no training, GPU dependencies or new runtime dependencies.

**Spec:** [P1 and scoring analysis](../../approaches.md), [assignment and starter](../../../capstone-project.py), [tutorial](../../../tutorial.py), [evaluation utilities](../../../utils.py), [public level catalog](../../../public_levels/catalog.py).

## Global Constraints

- Task 1 only: structured `GridState` observations. Preserve `Agent()` and `step(self, state: GridState | ImageObservation) -> Action`; always return an `Action`. Image solving is outside this milestone.
- Assignment limits: **150 turns, 5 seconds per entire instance, 1.5 GB memory, 2 MB upload, one CPU core, no GPU**. Do not interpret five seconds as a per-call budget.
- Use the documented public transition interface and black-box experiments; do not inspect installed engine implementation source or replace its dynamics in P1.
- Preserve existing notebook behavior and dependencies. Do not hard-code public maps, action sequences, level names, catalog rewards or asset paths into the submitted agent.
- A candidate is a Task 1 artifact, not a Task 2/Boss solution. No GPU rental or M5080 access is required.
- Local tests do not establish hidden-test marks. Public timeouts/errors have effective performance zero; the local evaluator diagnoses elapsed time after completion rather than interrupting a run.
- Planning targets, subject to measurement: initial search stops after **3.0 s**, all search/repair stops by **3.5 s from agent construction**, and the remainder is reserved for observation checks and execution. Target measured whole episodes below **4.25 s** on the validation CPU; this is headroom, not a grader guarantee.
- Initial resource caps: **4,096 retained simulator states** and **32,768 compact search records**. Stop expansion at a cap and preserve the incumbent. Tune downward if measured peak RSS exceeds **1.0 GB**; do not raise caps without a measured memory margin below the assignment limit.

## Review Focus

1. Identical-looking grids can have different inventory, health, effect counters or remaining turns; they must not be incorrectly merged (Task 2).
2. A winning path can arrive after the useful deadline, or memory can fill before a win; termination and fallback must remain bounded (Tasks 3–4).
3. A cached action sequence can become stale; compare the complete predicted state before consuming its next action (Task 4).
4. The shortest win can earn less than a longer coin route; compare final reward and effective performance, not wins alone (Tasks 1 and 5).
5. A locally importable agent can depend on repository files or behave differently after packaging; test the exact isolated artifact (Task 6).

## Deliverables and order

| Stage | Deliverable | Decision gate |
|---|---|---|
| Tasks 1–2 | Reproducible measurements and sufficient state representation | Correct scoring and no false state merges in mechanic fixtures |
| Tasks 3–4: P1a | Bounded cached BFS baseline | Complete episodes, valid actions, verified plan reuse |
| Task 5: P1b | Optional best-first/reward improvement | Promote only on measured improvement without reliability regressions |
| Task 6 | Frozen single-file Task 1 candidate and evidence | Local gates, then Coursemology Debugging validation before a graded attempt |

Estimate **2–3 focused engineering days**: roughly half a day for harness/key validation, one day for search/cache, and half to one day for tuning, adversarial cases and packaging. This is a planning estimate, not a commitment that P1 will solve every challenge map. Stop after the measured P1 candidate rather than making a custom simulator a prerequisite.

## File map

- `agents/__init__.py`: empty package marker.
- `agents/p1.py`: self-contained submitted logic: state key, bounded search, plan cache and `Agent`.
- `tools/benchmark_p1.py`: sequential isolated episode workers, baseline loading, measurements and JSON output.
- `tools/package_p1.py`: copy the self-contained source, enforce size, calculate hashes and create a manifest.
- `tests/p1_fixtures.py`: small public-API-built mechanic cases, known winning witnesses and deterministic layout variations.
- `tests/test_p1_harness.py`, `test_p1_state.py`, `test_p1_search.py`, `test_p1_agent.py`, `test_p1_package.py`: meaningful behavioral tests.
- `artifacts/p1/`: generated measurements, traces, manifests and frozen source. Keep generated artifacts separate from source and include the relevant build/test files with the eventual supplementary submission.

All commands below run from the repository root using the existing environment. `uv run` uses the locked project dependencies; no GPU installation is part of this plan.

### Task 1: Establish a bounded and correctly scored baseline

**Files:** Create `tools/benchmark_p1.py`, `tests/p1_fixtures.py`, `tests/test_p1_harness.py`.

**Interfaces:**
- `load_agent(spec: str) -> type`: accept `starter`, `agents.p1:Agent`, or an artifact `.py` path. For `starter`, extract only the existing import/Agent cell with Python AST; never execute the marimo app or its evaluation cells.
- `run_case(agent_spec: str, level_label: str, seed: int) -> dict[str, object]`: run one fresh worker, return status, reward, effective performance, timing, memory and provenance.
- `fixture_cases(split: str) -> dict[str, tuple[Callable[[], GridState], tuple[Action, ...]]]` in `tests/p1_fixtures.py`: return builders and verified witness actions; `split` is `mechanics`, `tuning` or `heldout`.
- CLI: `--agent`, `--suite intro|task1|generated`, `--repeat`, `--output`, and `--split tuning|heldout` (required for `generated`). Generated fixtures come from `fixture_cases`; public cases come from the catalog.

- [x] Write harness tests. Check the existing formula with `O=-20, M=-30`: a win at `R=-20` yields `1.0`, at `R=-25` yields `0.5`, and at `R=-30` yields `0.0`. A loss, exception or timeout yields effective performance `0.0` even if an intermediate reward looks good. Test a deliberately hanging worker is terminated.
- [x] Run `uv run python -m unittest discover -s tests -p 'test_p1_harness.py' -v`; confirm the new behavior is initially missing, then implement the loader/worker and rerun to pass.
- [x] Reuse `create_env`, `evaluate` and `get_performance` from `utils.py`, and rewards/builders from `public_levels/catalog.py`. Override public builders' default turn limits with **150**. Measure the evaluation interval with `perf_counter`; report import/environment setup separately. Match `evaluate_level` results on a quick case before extending instrumentation.
- [x] Add a parent watchdog outside the agent. Worker signals when timed evaluation begins; interrupt at **5.0 s** from that signal and report timeout. Allow at most **30 s** for import/environment setup, with a distinct setup-error result. Run workers sequentially, never parallelize the submitted agent. Record the elapsed interval and whether termination was local-watchdog or evaluator-detected.
- [x] Record unrounded evaluation seconds, peak worker RSS with platform-correct units, status, raw reward, effective performance, turns, host/CPU, Python/package versions, source SHA256 and seed. Mark unavailable measurements explicitly rather than recording zero. If instrumented P1 statistics are available, include expansions, key/simulation time and cache hits. Use a separate profiling run for detailed timers so profiling overhead does not contaminate acceptance timings.
- [x] Run `uv run python tools/benchmark_p1.py --agent starter --suite task1 --repeat 1 --output artifacts/p1/starter.json`. Preserve all 22 outcomes, including timeouts. Do not reuse historical numbers from `approaches.md` as new measurements.

**Acceptance:** A hanging baseline cannot hang the full benchmark; scoring matches the repository; fresh measurements exist for all 22 public Task 1 cases.

### Task 2: Build a sufficient, compact state key

**Files:** Create `agents/__init__.py`, `agents/p1.py`, `tests/test_p1_state.py`; extend `tests/p1_fixtures.py`.

**Interfaces:** In `agents/p1.py`, define `StateKey = tuple[Hashable, ...]` and `state_key(state: GridState) -> StateKey`. Both search deduplication and cached-state verification use this function.

- [x] Add small fixtures using tutorial constructors and public transitions: a picked-up key, an opened door, consumed coin/gem, pushed box, damaged agent, active/expiring boots and ghost, and depleted shield. Include a box covering a collectible and a key consumed by unlocking. Fixtures contain witness actions for black-box replay, not engine source-derived expected internals.
- [x] Write `test_copy_has_same_key`, `test_each_dynamic_component_changes_key`, `test_overlapping_entities_are_retained`, and `test_equal_keys_have_equal_successors`. On paired copies, assert key equality and then equal successor keys, reward deltas and terminal flags for **every `Action`**. On controlled one-component variants, assert unequal keys.
- [x] Run `uv run python -m unittest discover -s tests -p 'test_p1_state.py' -v`; confirm failure before implementing `state_key`, then implement and rerun.
- [x] Encode grid dimensions/configuration, board entities and positions, locked/open door state, remaining collectibles, agent health, inventory and active effects with remaining counters, turn/turn limit, terminal flags, and **score** initially. Traverse every entity in overlapping cells. Use sorted immutable primitive tuples, explicit component fields and stable semantic identifiers where inventory/effect references require them; never use `str(grid)`, object addresses or display names alone.
- [x] Establish the exact inventory/effect field mapping from public documentation and runtime observations before writing those encoders. The current public object exposes `inventory_list` and `status_list`; empty initial lists are not sufficient evidence for the active-state schema. Document every encoded field and any intentionally omitted cosmetic field beside the encoder. Preserve uncertain gameplay-relevant fields conservatively until equivalence is demonstrated.
- [x] Retain score and exact remaining turns in P1's key, even though this reduces merging. Removing score into a dominance table, merging different effect timers, or discarding `WAIT` successors requires a separate correctness argument and is outside the initial patch. Measure key generation separately from simulator cost.

**Acceptance:** Mechanically distinct states remain distinct; copied states and all tested action successors agree. Tests are evidence for the supported schema, not a mathematical proof of completeness.

### Task 3: Replace repeated path copying with bounded search

**Files:** Extend `agents/p1.py`; create `tests/test_p1_search.py`.

**Interfaces:**
- `SearchResult`: `actions: tuple[Action, ...]`, `expected_keys: tuple[StateKey, ...]`, `winning: bool`, `final_score: float | None`, `stop_reason: str`, `expanded: int`.
- `search(start: GridState, *, deadline: float, mode: str = "bfs", clock: Callable[[], float] = perf_counter) -> SearchResult`.
- For a winning path of length N, `expected_keys` has N+1 entries, starting with `state_key(start)`. No winning plan is represented by `winning=False` and empty action/key tuples; never label a partial route as a solution.

- [x] Write tests for a short solvable map, an unsolvable map, a near-turn-limit map, a deadline reached before expansion, deadline reached after finding a win, and both resource caps. Use an injected fake clock for deterministic budget tests; actual wall-clock behavior belongs in the benchmark.
- [x] Assert that replaying returned actions through `grid_step` reaches `win`, matches every expected key and matches `final_score`. Assert deadline/cap exits preserve an existing winning result. Assert the parent chain reconstructs the correct action order without retaining a full path list at each node.
- [x] Run `uv run python -m unittest discover -s tests -p 'test_p1_search.py' -v`; confirm missing functionality, implement BFS and rerun.
- [x] Store compact parent/action/key records and only the simulator states still needed by the frontier. Remove popped states after expansion. Count every retained simulator state, including transient successors, toward the state cap; count visited/parent records toward the record cap. Check deadlines before and after each simulator call and bound reconstruction to the remaining turn horizon.
- [x] Enumerate the public `Action` enum, use `grid_step` for all transitions, reject losing successors, and deduplicate only by the complete key. BFS returns its first winning path. No hand-written movement, pickup, unlock or effect-expiry logic is introduced.
- [x] Include a tiny exhaustive fixture oracle in tests only. Under a generous test budget, BFS should find the minimum-action win; do not assert it maximizes reward. If no win is found, return an explicit failure result for Task 4 to handle.

**Acceptance:** Search is bounded by time, turns and retained records; every reported solution is a replayable reference-simulator win. A single slow simulator call cannot be interrupted inside the agent, so measured call latency must fit the reserve.

### Task 4: Cache a complete plan and enforce the episode budget

**Files:** Extend `agents/p1.py`; create `tests/test_p1_agent.py`.

**Interfaces:** `Agent.__init__(self)`, `Agent.step(self, state: GridState | ImageObservation) -> Action`, and optional development statistics through `Agent.info(self) -> dict[str, str]`. Constructor starts the episode clock; search consumes the absolute deadlines in Global Constraints.

- [x] Write `test_one_search_for_matching_episode`: roll a deterministic fixture forward with the public simulator and assert one search call, valid actions and a final win. Write `test_mismatch_discards_suffix`, `test_repair_does_not_reset_budget`, `test_terminal_returns_action` and `test_no_plan_returns_action`.
- [x] Run `uv run python -m unittest discover -s tests -p 'test_p1_agent.py' -v`; confirm failures, implement cache handling and rerun.
- [x] At the first observation, search until the initial deadline. Retain actions and expected keys from `SearchResult`, then release the frontier and search tables. At each subsequent observation, compare its complete key with the expected pre-action key before consuming the cached action.
- [x] On mismatch, discard the stale suffix and perform a bounded repair only if time remains before the absolute **3.5 s** search cutoff. Never restart the clock on a mismatch. Tests create a fresh `Agent()` for each episode, matching the evaluator.
- [x] If no verified winning plan is available and no search time remains, return `Action.WAIT` in constant time. This is a bounded legal-action fallback, **not** a claimed recovery strategy. Likewise return a valid action for terminal input; image input may return `WAIT` but must not be reported as supported solving.
- [x] Run the public benchmark for `agents.p1:Agent` and preserve the result and exact source bytes as **P1a** before tuning. Compare against starter wins, effective performance, runtime, memory and expansions, not only elapsed search time.

**Acceptance:** A matching plan is searched once and reused; divergent observations cannot trigger stale actions or renewed five-second budgets. The baseline checkpoint remains recoverable if subsequent tuning regresses.

### Task 5: Improve search order and reward within the same budget

**Files:** Extend `agents/p1.py`, `tests/test_p1_search.py`, `tests/p1_fixtures.py`.

**Interfaces:** Add `mode="best_first"` to `search`; preserve its result/deadline contract. Keep production mode as a source constant so the exported candidate has a fixed, recorded configuration.

- [x] Add a fixture where the shortest win has lower reward than a longer coin route. An exhaustive tiny-map oracle establishes achievable final rewards under the fixture's turn limit. Assert that replacing an incumbent requires a replayable win with strictly higher final score, and that an expired deadline cannot discard the incumbent.
- [x] Implement a heap ordered by `(depth + relaxed_route_length, -state.score, insertion_counter)`. Define `relaxed_route_length` as Manhattan distance to the closest required target plus the Manhattan minimum spanning tree over remaining gems and exit; with no gems, use distance to exit. Compute it from observed entities, never level identity. This is an ordering heuristic only: boots, phasing and rewards invalidate naive optimality assumptions.
- [x] Continue bounded exploration after the first win and retain the highest-reward winning path encountered. Do not apply A* optimality stopping rules, prune on an unproved heuristic, merge resource states, or implement P2 branch-and-bound/custom simulation in this task.
- [x] Generate **100 deterministic solvable layout variations** from small mechanic templates, with a reference-simulator-verified witness for each. Split **50 tuning / 50 held-out** before tuning. Vary dimensions, positions, obstacles, keys/doors, health and effect opportunities; changing only a rendering seed is not structural generalization. Exclude unsolved/unverified generated maps from win-rate denominators and report their count separately.
- [x] Compare P1a and P1b on all 22 public maps plus the tuning split, including simulation/key/heap costs. Promote best-first only if there are no new timeouts/errors or lost previously solved cases and mean public effective performance improves. Then evaluate the frozen choice once on the held-out split; report its outcomes without claiming unknown optimal scores.
- [x] If the public wrapper dominates and P1 remains weak, record the bottleneck and retain the stronger checkpoint. A future experiment using another documented transition interface or a custom simulator is a separate decision; do not silently expand P1 to P2.

**Acceptance:** The chosen candidate improves measured score without sacrificing baseline reliability, or we deliberately retain P1a. Failure to improve is a valid experiment outcome.

### Task 6: Package and qualify the first submission candidate

**Files:** Create `tools/package_p1.py`, `tests/test_p1_package.py`; generate `artifacts/p1/candidate/` and benchmark reports; update the P1 outcome in `docs/approaches.md` during implementation.

**Interfaces:** `package(source: Path, output_dir: Path) -> Path` copies `agents/p1.py` unchanged to `task1_agent.py` and writes a manifest with source hash, byte count, configuration and environment provenance. No bundler is needed because the submitted module is self-contained.

- [x] Write packaging tests: source/artifact bytes and SHA256 agree; size is **less than 2,000,000 bytes**; isolated import finds `Agent`; fresh `Agent().step(...)` returns `Action`. Run from a temporary directory with the project root removed from the import path, while retaining installed allowed dependencies. Reject imports of local helpers, tests, catalog data and runtime model/assets files.
- [x] Run `uv run python -m unittest discover -s tests -p 'test_p1*.py' -v`; all behavioral tests must pass before packaging. Record test output with the candidate.
- [x] Run `uv run python tools/package_p1.py --source agents/p1.py --output-dir artifacts/p1/candidate`, then benchmark **that exported path** with `--suite task1 --repeat 3`. Repetitions measure timing variation, not three hidden scoring cases. Record cold import/setup separately as well as full evaluation time and memory.
- [x] First-candidate local gate: **11/11 intro maps solved**, all 22 public maps attempted, no exceptions or whole-episode timeouts, every public episode under the **4.25 s planning target**, and peak RSS below **1.0 GB** on the recorded validation host. Aim for intro mean effective performance **at least 0.85** and report its per-level distribution; this is a local target, not the grader's grouping or a mark prediction. Report B-map and held-out wins/performance separately; do not require 22/22 before preserving a first candidate.
- [x] If a gate fails, fix the specific issue or report the unmet gate; do not label the artifact submission-ready. Replay and inspect failing mechanic traces before enlarging search budgets. A GPU does not solve a single-core inference bottleneck.
- [x] Prepare the exact source file, supplementary source/test/build files, manifest and benchmark summary for **Task 1 in Capstone Project (Debugging)**. Debugging success must be observed before recommending a graded attempt. Keep graded submission and any required comments as an explicit user-facing handoff; this planning request does not authorize an upload or consume an attempt.
- [x] After debugging validation, freeze that exact candidate hash and record the observed result separately from local estimates. Any subsequent change requires revalidation of the changed artifact. Preserve a previous working candidate throughout later Task 2 development.

**Acceptance:** The user has one identifiable candidate, reproducible evidence and a clear debugging-to-graded handoff. Until platform validation is observed, its status is **locally validated candidate**, not a confirmed graded result.

## Compute allocation

Use the existing CPU environment for implementation and sequential benchmarks. Training cost is **$0 for P1**; local CPU time is additional but requires no rental. A three-repeat public sweep has at most **330 seconds of timed evaluation** at the watchdog limit, plus interpreter/environment setup and reporting; starter and generated-map sweeps are additional. The M5080 is unnecessary for this plan. Reserve it for a later V3 image-parser experiment after Task 1 is stable, and identify its exact hardware/software then rather than assuming CUDA capability from its name.

## Plan review

P1 requirements map to Tasks 2–4; scoring and bounded measurements to Tasks 1/5; generalization checks to Task 5; and exact-artifact validation to Task 6. All five review risks have assigned tests. No training, custom simulator, Task 2 parsing or actual submission has been started by writing this plan.

## Execution status — 8 November 2026

Tasks 1–6 are complete. The user reported all 22 Debugging tests passed and requested freezing P1; the exact validated source and freeze record are preserved under `artifacts/p1/frozen/`.

Current exact source: `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5` (37,287 bytes). All 68 tests, Ruff lint/format and ty with warnings as errors pass. Three exact-export public sweeps reached **22/22 wins and catalog maximum rewards each** (66 wins; mean effective performance 1.0), with no errors/timeouts, maximum episode 3.199105 s and peak RSS 322.6 MB. Fixed tuning and heldout corpora each replayed 50/50 wins; these are revalidation of the existing corpus, not new independent heldout evidence. The user reported all 22 Debugging tests passed for the latest delivered source; P1 is now frozen. See `artifacts/p1/candidate/platform-validation.json`.

Platform progression: generic loader failure, then 10/22 after compatibility changes, then 20/22 after guided ordering (latest delivered ce5a7034; submitted hashes not explicitly confirmed). The remaining two cases were Ow Oof Owie and Warehouse. Stronger first-win hazard/box ordering reduced simulator work 491→119 and 720→210 respectively. Reward refinement retains the earlier box cost to avoid an experimentally observed A6 coin regression. Stable carried-effect IDs are now included in the state key after a black-box stacked-shield counterexample. All new fixes have regression evidence.

Task 5 deviation: P1c uses cached wall/resource distances, small target tours and lazy all-action expansion in place of the original Manhattan-only ordering. This responds to measured public/platform failures; estimates remain ordering-only, with no pruning or handwritten simulator. The full-state key, action enumeration, public transition API, 3.0/3.5 s deadlines and 4096/32768 caps are unchanged in purpose. Independent reviews and scope rulings are preserved under `artifacts/p1/`.

P1 is complete and frozen at the hash above. The actual user-reported Debugging pass is recorded in `artifacts/p1/candidate/platform-validation.json`; source and freeze metadata are in `artifacts/p1/frozen/`. The graded submission remains a user handoff; no graded attempt was performed by the agent. Future work starts from a separate source so this P1 baseline remains recoverable.
