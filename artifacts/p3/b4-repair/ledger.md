# P3 B4 dispatch repair

User-reported debugging result: Measure Twice and Cut Once, performance0, reward-450, loss, no timeout/error, runtime3.79s. Frozen P3 parent a8efb9c0ce3a151d3170785a5ceb69ec76851c5b7a7eb27983d6b70f9b6554f1.

Root diagnosis: B4 uses guarded reference fallback. Local instrumented search finds a130-action/-387 incumbent after213 public transitions, well before3s, then refines until3s. Existing Omarchy trace consumes~0.2s during dispatch, fitting cutoff3.5. Deterministic slower public transitions at5.5ms each reproduce loss-450 at3.828s: search already holds winning130-action route at3.003s; it dispatches only91 actions before cutoff and WAITs to death. This is an execution-reserve failure, not failure to find a route. The controlled cost model is diagnostic, not grader hardware evidence.

Change gate: retain original initial/cutoff deadlines; while a winning reference incumbent exists, tighten refinement using measured per-transition cost and retained route length. Reserve for full independent public replay and subsequent dispatch. No simulator/schema/task-label/board-size changes, no fresh time allowance, no V3 integration.

Old P3 frozen source/evidence remain intact under structured/. New release/evidence live here. Any old holdout rerun is regression revalidation, not a new untouched holdout.

Implementation: reference_search optional dispatch_cutoff; reserve0.1+4*retained path length*observed elapsed/search-transition cost after an incumbent exists; ordering shares tightened deadline. Reference winning cache installation now requires independent full public prefix replay. Invalid/cross-cutoff replay leaves cache empty. Plain Agent fallback inherits original origin/initial/cutoff and calls backend.step, avoiding step_until clamping away the dispatch reserve. Review caught the direct late-fallback corner, with red/green test retained.

All mechanics/schema/fast/event search AST blocks unchanged. New source frozen d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4,96,112bytes. Reused holdout is explicitly regression data. Reviewer found no remaining blocking issue after deadline fix; new tests add invalid and expired replay atomic-install coverage.

An earlier full-suite process imported the intermediate source before the late-fallback correction and failed that one new test (246 cases). Its log is preserved as full-suite.log. A fresh full-suite process on final source is recorded separately as full-suite-final.log.

Final exact-source acceptance PASS: all326 episodes win; rewards pair exactly with both prior P3 and P2-next public/generated/boss baselines. B4=-387 on seeds42/43/44, with verified public replay and130 dispatched actions. 247/247 current repository tests pass;257,467 exact-source public/compact comparisons pass. All86 protected P1/P2 files and71 old P3 evidence files match prior immutable archive. Shared Omarchy lock released and V3 notified; user-selected Performance left unchanged. Platform Debugging rerun remains pending.
