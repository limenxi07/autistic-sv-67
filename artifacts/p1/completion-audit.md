# P1 requirement audit

Current exact source: `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5` (37,287 bytes). All 68 tests, Ruff lint/format and ty with warnings as errors pass. Three exact-export public sweeps reached **22/22 wins and catalog maximum rewards each** (66 wins; mean effective performance 1.0), with no errors/timeouts, maximum episode 3.199105 s and peak RSS 322.6 MB. Fixed tuning and heldout corpora each replayed 50/50 wins; these are revalidation of the existing corpus, not new independent heldout evidence. The user reported all 22 Debugging tests passed for the latest delivered source; P1 is now frozen. See `artifacts/p1/candidate/platform-validation.json`.

| Requirement | Evidence | Status |
|---|---|---|
| Correct loss/error/timeout scoring and AST/file/module loading | harness tests and `tools/benchmark_p1.py` | Passed |
| Fresh sequential workers,150 turns,5 s evaluation/30 s setup watchdogs, provenance | report rows, hang/setup tests, qualification tool | Passed locally |
| Unchanged starter and recoverable P1a/P1b checkpoints | frozen sources/reports/selection and baseline manifest | Preserved |
| Sufficient supported state schema: health, inventory, locks, overlaps, effects/counters, score and exact turn | state tests and encoder | Passed observed fixtures; not a proof for arbitrary extensions |
| Stable carried reference IDs; canonical ordering; equal-key successors for every Action | stacked-shield RED→GREEN, public roundtrip and focused review | Passed |
| Replayable simulator wins, BFS oracle, bounded parent reconstruction | search tests | Passed |
| Deadlines/caps preserve incumbent;4096 states including start/current/transient,32768 records | injected-clock/cap tests and independent weak-reference review | Enforced |
| All actions retained, no heuristic pruning/dominance/custom dynamics | source, all-action regression, guided/lazy review | Passed |
| Complete expected-key cache; bounded repair; legal fallback | Agent tests and all 66 winning episode cache stats | Passed |
| Higher-reward coin routes, strict incumbent replacement | tiny oracle and A2/A6/B2/B7 regressions | Passed |
| Targeted slower-host first-win margin | deterministic150/250-call regressions, `detour-ordering/first-win.json` | B5=119,B6=210 calls |
|100 structural layouts with fixed 50/50 split and verified witnesses | generated manifest/tests/reports |100 verified, zero excluded; repeated corpus is revalidation |
| Measured ordering promotion without lost wins/timeouts | original P1b selection; current exact66 plus generated 100 | Current public mean1.0 |
| Self-contained byte-identical <2 MB source; isolated import and real step | packaging/compatibility tests, manifest/hash | Passed |
|68 tests, Ruff/ty, scoped independent reviews | candidate test logs, checks, `review.md`, `guidance/review.md` | Passed; later small weights tested/measured |
| Exact22×3, all intros, no error/timeout, <4.25 s,<1 GB | public-repeat3 and recomputed qualification | Passed: max 3.199105 s, RSS 322.6 MB |
| Simulator call latency fits observed reserve | current `latency.json` |10,080 samples,max 1.056 ms; unseen states unbounded |
| Exact artifact, supplementary source/build/test files and evidence digests | current manifest/summary, evidence receipt and ZIP | Prepared and digest-verified |
| Actual current-source website Debugging result | user reported all22 passed; candidate/platform-validation.json | Passed, user-reported |
| Post-Debugging frozen hash and graded handoff | frozen/task1_agent.py, frozen/freeze.json, plan final checkbox and HANDOFF | Complete; no graded attempt by agent |

Task 5 follow-up deliberately extends the original Manhattan-only baseline ordering with wall/resource geometry, target tours and lazy expansion in response to actual platform failures. Every transition still uses the public simulator; no public maps/names/rewards/routes are embedded in the agent. Exact state/turn/score keys and resource/time bounds remain binding. The latest stacked-reference fix and targeted detour penalties were regression-tested; the rejected all-phase box16 experiment remains preserved with its A6 score regression.

No engine implementation source was inspected. No GPU, training, Task 2 image solving or graded attempt was introduced. Unknown generated optima, hidden marks, custom closures, unsupported mechanics and adversarial sandbox security remain outside these claims. The user-reported Debugging pass is recorded separately from local qualification in candidate/platform-validation.json; graded performance remains unmeasured.
