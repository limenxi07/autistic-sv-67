# P3 B4 dispatch repair

Current source: `agents/p3.py`; standalone byte-identical export: `candidate/task1_agent.py`.
SHA256: `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4` (96,112 bytes).
Previous source a8efb9c0 remains frozen under `../structured/candidate/`; its original evidence/archive is unchanged.

## Failure and correction

The user reported Measure Twice and Cut Once: reward -450, loss, no timeout/error, runtime 3.79 s. This is a platform debugging result, superseding any suggestion that the original P3 was platform-qualified.

B4 uses the reference backend. Its search finds a 130-action winning route at reward -387, then refines until the initial 3 s deadline. The old cache dispatcher enforces the 3.5 s cutoff even when already holding that route. On a slower execution path it therefore changes to WAIT before reaching the exit. A deterministic 5.5 ms/public-transition clock reproduces reward -450 at 3.828 s, despite an existing winning incumbent; only 91 route actions are dispatched. This controlled model reproduces the mechanism, not the grader's exact CPU/overhead.

The repair tightens refinement only after a winning reference incumbent exists. It estimates per-step cost from elapsed search time/public transition count, and reserves `0.1 + 4 * route_length * observed_cost` seconds before the original dispatch cutoff. The reserve includes independent replay, subsequent observation/dispatch overhead and a safety factor. Initial and cutoff deadlines only tighten; there is no new time allowance. Ordering receives the same tightened deadline.

Before caching, the reference incumbent is independently replayed through public transitions and every prefix/key/score/terminal win is checked. Failed or expired replay installs no actions or partial keys. Plain Agent fallback also inherits its original constructor origin and deadlines, matching the structural adapter; a late constructor cannot renew the reserve. The simulator/schema, compact primitive/event search and task-budget constants are unchanged (`ast-audit.json`).

Defaults remain initial +3.0/cutoff +3.5; explicit Task3 remains +11/+12 under a 15 s watchdog. No task identification by board size/label, runtime GPU, P2/frozen changes, V3 integration or platform submission.

## Validation scope

`dispatch-reproduced.json` records the old controlled failure. `dispatch-red.log`, `late-fallback-red.log`, `dispatch-final.log` and `dispatch-cost-sweep-final.json` retain red/green evidence. The 5.5 ms model now wins at -387 in 2.6015 virtual seconds. Models at 4, 6.5 and 7.2 ms also win, with the last completing in 3.4056 virtual seconds. These are deterministic clock tests, not wall-time acceptance results and not an unlimited-hardware guarantee.

The source was frozen before the repaired acceptance run. All original public/generated/boss/stress/holdout fixtures are now reused regression data. No new generalization claim is made. Exact-source acceptance is on Omarchy Performance, one P-core/thread, sequential fresh workers, with V3 using the shared timing lock in a separate reservation. Final receipt: `qualification.json`.

Final qualification: **PASS**, 326/326 episodes win. Every reward also pairs exactly with the previous P3 release (`parent-pairs.json`), including B4=-387 for all three seeds. B4 episodes took 3.169–3.175 wall seconds on Omarchy; independent reference verification was recorded as successful for every run. Exact-source mechanics: 257,467 comparisons over 36,781 frames and 1,000 many-item layouts, zero mismatches. Full repository tests: 247/247 pass. Source lint/type/grammar/standalone checks pass.

| Suite | Wins | Max episode s | Peak RSS MiB |
|---|---:|---:|---:|
| boss | 6/6 | 0.9280 | 241.7 |
| public | 66/66 | 3.1748 | 288.0 |
| generated | 200/200 | 3.2330 | 292.8 |
| stress | 36/36 | 2.9812 | 244.7 |
| holdout | 18/18 | 3.2749 | 245.2 |


The target platform must still rerun Debugging on this new hash; no 22/22 platform pass is claimed.

## Reproduce

The artifact bundle is a repository overlay using the existing locked environment. The standalone candidate itself has only standard-library and allowed Grid Adventure imports.


```sh
# Old loss and repaired outcome, using deterministic virtual CPU costs:
.venv/bin/python -m tools.probe_p3_dispatch --source artifacts/p3/structured/candidate/task1_agent.py --output /tmp/p3-parent-dispatch-recheck.json
.venv/bin/python -m tools.probe_p3_dispatch --source agents/p3.py --output /tmp/p3-dispatch-recheck.json
MPLCONFIGDIR=/tmp/p3-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest tests.test_p3_dispatch
.venv/bin/python -m tools.qualify_p3 --root artifacts/p3/b4-repair --output /tmp/p3-b4-recheck.json
```

Omarchy isolation directory: `/tmp/p3-b4-repair-20261109`, environment `/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python`. Its `bash tools/run_p3_acceptance.sh` owns `/tmp/grid-adventure-timing.lock`, pins CPU0 and sets all five library thread limits to one. Use fresh output paths for another run. V3 and P3 default backend selection remain independent.
