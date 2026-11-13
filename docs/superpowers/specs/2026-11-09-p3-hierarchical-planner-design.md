# P3: measured solver improvements and backend compatibility

Status: proposed implementation design, 9 November 2026. User confirmed the scope. Structured baseline and coverage diagnostics are authorized; solver changes remain a plan. Hierarchical search is conditional, not a predetermined deliverable.

## Intent and evidence boundary

Measure structured boss performance under the proper 15-second budget, then address fallback coverage, search scaling and reward according to what actually limits performance. Keep image perception in V3/V4. Preserve caller-owned deadlines and resource re-identification behavior when integrating a replacement backend. Resolved simulator bugs remain regression tests, not additional P3 features.

The user confirmed no newer P2 solver failures before this baseline. Both frozen P2 variants have user-reported 22/22 Debugging passes. V3's latest oracle bridge records 22/22 wins and identical paired P2a rewards; that is not learned-perception validation. P2b showed no reward regression, but no demonstrated reward or first-win improvement. Boss performance was previously unmeasured; fallback guards and old deadlines were constraints, not evidence of boss losses.

Frozen source hashes verified locally and on Omarchy:
- P2a: `artifacts/p2/frozen/p2a/task1_agent.py`, `e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb`.
- P2b: `artifacts/p2/frozen/p2b/task1_agent.py`, `551ee00377ffe42a343d18c887b1c7f36ef4baf4fa0fff6a22b0053c9845edd0`.

Saved reports: both won 66/66 public and 200/200 P2-v2 cases. P2a/P2b public mean times were 0.740/0.602 s. B4 used reference search, maxima 3.381/3.378 s; 40/200 v2 cases used reference fallback. Winning `state_cap` stops: P2a 54/66 public, 110/200 v2; P2b 18/66 public, 8/200 v2. These stops alone do not prove missed solutions. Saved B5/B6 profiles show substantial ordering cost; those are instrumented diagnostics.

New boss evidence and its exact scope are recorded in `artifacts/p3/baseline/summary.md`, raw `boss.json`, `coverage.json` and `manifest.json`. Those measurements determine the next experiment; do not substitute expectations for the receipt.

Completed baseline: every variant wins C0 in all three runs; extended budgets reach catalog reward -68 versus frozen P2's -70. Every variant loses C1 at -450 with 150 WAIT actions. Both bosses enter reference search through the large-keyed-inventory guard. A separate instrumented C1 root search spends about 88% of its 11-second search in reference simulation, stops at the deadline and remains below both caps. Prioritize large-keyed coverage/transition-cost characterization; do not infer that lifting a guard safely or accelerating simulation alone will solve C1.

## Constraints

- Tasks 1/2: 5 seconds per whole episode. Structured Task 3: 15 seconds per whole episode. All: 150 turns, 1.5 GB memory, one CPU core, no GPU.
- Source below 2,000,000 bytes, self-contained, Python 3.10 grammar, standard library and allowed Grid Adventure imports; grader-prelude and unregistered-module compatible.
- Preserve P1/P2 frozen bytes/evidence and V3/V4 perception. Implement in new P3 files; integrate through a separately tested backend contract.
- Use public constructors, fields and black-box `grid_step`; do not inspect installed engine implementation source.
- Caller owns `started_at`, `initial_deadline` and `cutoff`. Initial search, fallback, verification, suffix rebinding and repair share the original absolute deadlines; they may tighten but never extend them. Expiry also prevents cached dispatch.
- Task 1 default remains initial +3.0 s / cutoff +3.5 s. V3 preserves its supplied budget and initial reference cap `min(initial_deadline, started_at + 2.0)`. Structured boss budget-only diagnostics use initial +11.0 s / cutoff +12.0 s under the 15.0 s watchdog, leaving at least 3.0 s after cutoff. This is an experimental allocation, not a measured optimum.
- Local acceptance targets: Tasks 1/2 <4.25 s; structured boss <13.5 s; peak RSS <1,000,000,000 bytes. Hard watchdogs remain 5/15 s. Image boss budget tuning is out of scope.
- Submitted code must not identify tasks by catalog label or infer them from board size. Task-specific callers/entry points provide the budget; standalone Task 1 retains its safe default.

## Stage 0: baseline and coverage audit

Run C0 Everything Everywhere All at Once and C1 The Heist at seeds 42/43/44, sequential fresh workers on a verified idle Omarchy P-core with one library thread. Compare unchanged P2a/P2b and separately hashed copies changing only four outer/reference deadline constants to 11/12 seconds. All four variants use a 15-second whole-episode watchdog. Raising only the watchdog leaves P2's short internal deadlines unchanged.

Record win/loss/timeout/error, reward, turns, executable actions, exact hashes, host/CPU/affinity, package revisions, RSS and whole-episode time. Report setup/import separately. Replay traces outside acceptance timing. No runtime policy patching or profiled timing in acceptance results.

Audit encoding at initial public24 states and every reachable state in captured public22, P2-v2 and boss traces. Record exact rejection reasons and first unsupported turn, initial versus whole-trace coverage, and replay score/status. Initial eligibility alone does not establish coverage. Existing all-action differential evidence remains separate.

Where a deficit remains, use a separately labelled diagnostic to expose fallback's own stop reason, expansions/records/frontier, first verified win, reward trajectory, ordering and verification cost. Outer `unsupported` and zero counters are not inner-search evidence. Diagnostic instrumentation cannot establish acceptance runtime.

## Stage 1: measured decision gates

| Outcome | Next experiment | Boundary |
|---|---|---|
| Budget-only variant fixes boss reward/win gap | Budget-aware entry point and integration contract | No new search machinery without a remaining measured deficit |
| Deficient cases spend time in reference fallback | Rank exact exclusion reasons by impact; probe one schema extension or earlier refinement stop | Guards remain until independent all-action probes pass |
| Cap/deadline prevents first win or leaves a measured reward gap | One bounded cap experiment or cached/resumable primitive search | Winning cap stops alone do not activate this branch |
| Ordering dominates a deficient case | Preindexed descriptors and bounded exact-state/phase/origin caches | Judge whole episodes and paired reward |
| Deficit persists after correct budgets and cheaper primitive search | Resource-aware subgoal candidates, then a bounded portfolio | Hierarchy must show reproducible improvement |
| No deficit remains | Keep existing search; ship only useful integration/budget work | Record that additional machinery was unnecessary |

Do not blanket-lift large-keyed-inventory, simultaneous-pickup, overlapping-lock or power-counter exclusions. Characterize one reason through public API probes before widening support. No undocumented set-history emulation as incidental cleanup. Default reference fallback stays the working P2 policy; replacing it or changing reference deduplication is a separately measured experiment.

P2b bounds are valid only for the proved standard schema. Unknown rewards, movement or effect semantics receive no assumed pruning bound. Keep exact turns, effects, inventory identity/order and box configuration. Existing simulator fixes are regression coverage.

## Stage 2: replacement-backend contract

If evidence warrants a runtime change, fork the qualified parent into standalone `agents/p3.py`. Expose `PlannerAdapter(budget, mode="p2a", reference_initial_deadline=None)` accepting the same structural budget fields as V3, with no vision imports. The V3 caller supplies `min(budget.initial_deadline, budget.started_at + 2.0)`; structured callers choose their own reserve. Clamp all deadlines monotonically against the caller's original time origin. No late constructor creates fresh planning allowance.

On cache mismatch, try the remaining primitive suffix from the current observation. Fast path: rebuild Board slots from new public resource IDs, produce predicted prefixes and compare every prefix/score against public replay. Reference path: retain actual GridStates and replay with public transitions. Atomically install board, actions, keys and cursor only after a verified terminal win. Failed/expired replay invalidates the incompatible suffix and triggers repair only while the same cutoff permits it. Never blindly rename IDs or dispatch stale actions.

A previously verified incumbent remains usable only when compatible with the current observation. Resource identity/order and counters are significant. Image-derived state replay is conditional verification, not perception evidence.

Test both backends for delayed construction, late fallback, tightened deadlines, expired initial search, verification/rebinding crossing cutoff, cached dispatch after expiry, public pickup re-identification, sparse key IDs and consumption order, effect identities/counters, and failed rebind followed by bounded repair. Use injected clocks to assert propagated deadlines and atomic installation. Preserve V3's existing oracle 22/22 wins and identical paired P2a rewards before changing backend selection.

## Conditional search design

If selected by Stage 1, first try exact-context caching or one bounded cap experiment. Retain P2's 4,096 frontier/32,768 record limits as control; any increase requires measured RSS and useful-search improvement. A resumable worker must retain frontier, pending actions, parents, incumbent and caches across slices, including expensive ordering work.

Only if deficits persist, propose hierarchical goals for gems/exit and useful keys, powers, doors, boxes and coins. Goal completion is a resource event, not arrival at a coordinate. A required-gem subset DP may order relaxed routes; it cannot prune resource-dependent alternatives without proof. Local sequences carry complete endpoint states, preserve alternatives with different health/timers/inventory/boxes, simulate incidental pickups and stop on terminal states. Verify the full winning primitive sequence before dispatch or incumbent replacement.

Initial experimental limits may be four endpoint alternatives per goal, 24 primitive actions per local query, and an extra 1,024 frontier/8,192 record pool including local queries. They are heuristic truncations, not dominance/completeness guarantees. Preserve all seven primitive actions, WAIT included, and an independently retained incumbent. Cache limits, slice allocation and reference-state merging rules must be specified in a measured follow-on design before implementing this branch. Do not treat this option as already approved implementation.

## Qualification and handoff

Before tuning a selected change, freeze a fresh stress/heldout manifest with disjoint seeds and topology families targeted at measured deficiencies. Existing P1/P2/V3 outcomes are regression data. Witnesses must exercise intended mechanics. Unknown generated optima imply raw rewards only. Freeze candidate hash and improvement thresholds before opening final outcomes; disclose reused holdouts/revalidation.

Common gates: no lost paired wins/reward regressions versus frozen P2a/P2b and the best budget-only boss baseline; public22 x3 at catalog rewards; both bosses x3 without timeouts/errors and with the targeted deficit resolved; zero supported-schema mismatches; unchanged V3 oracle 22/22 paired reward; package/time/RSS limits. A partial diagnostic checkpoint may remain experimental if boss gates fail.

For schema changes, rerun >=100,000 differential transitions over >=1,000 layouts, all public traces and targeted sparse-ID/order fixtures with an independent projection. For performance, compare sequential fresh workers on the same P-core across three repeats, using whole-episode distributions.

Declare the applicable benefit gate before candidate outcomes: fix reproduced loss/timeout with replay-valid wins; improve reward on declared deficient cases without paired regression; or reduce median time >=20% on a predeclared slow subset without >10% aggregate p95 regression. Compatibility-only work claims compatibility, not solver improvement. P2b's non-regression and any newly measured first-win/reward benefit remain distinct.

Deliver a decision ledger tying each change to evidence, exact-source manifests, raw/summary results, runner/profile flags, freeze records and limitations. No automatic graded submission. Keep P2 deployable and V3/V4 perception independent.

## Builder review

A fresh reviewer identified five decisions: distinguish watchdog/internal budgets; activate changes only from measured shortfalls; preserve caller and V3-specific reserves; retain fallback policy and bound eligibility; audit reachable states as well as initial encoding. This design resolves all five. New search architecture remains conditional on baseline diagnostics.
