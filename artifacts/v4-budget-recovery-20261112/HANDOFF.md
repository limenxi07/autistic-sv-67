# V4 bounded cold-start recovery repair

The repaired complete Agent will be saved under [frozen/task2_agent.py](frozen/task2_agent.py).
Source SHA256 `d18cfbecdb6d9425349a2463b8af94bb5ab456b4787285c0ccc7be8c453aded4`,
693960 bytes. This source includes the earlier notebook-scope configuration repair.
Do not use the earlier attempt-state-only candidate or either slow-start wrapper.

## Evidence and cause

The user's platform diagnostic intentionally raised after its first image action.
Basic movement reported no underlying exception, threads1/1, a56us constructor
interval, one admissible hypothesis, and WAIT/planner_no_verified_plan after
3.092559s. The diagnostic did not measure perception separately or prove that
its admissible board was exact.

An actual local cold run of the same95e43962 source reproduced that branch:
perception3.241486s, whole step3.250763s, initial window already spent, no search
result, WAIT and planner discarded. Each later frame constructed a new planner
with its initial-attempt flag reset, despite the absolute initial deadline being
in the past. The remaining recovery interval was therefore never used.

This is a concrete matching local failure. It does not establish that every
platform case has exactly the same first divergence. The source's actual
platform retest remains necessary.

## Repair

A late-created V4 PlannerAdapter now enters recovery if the original initial
window has already elapsed. It can find a plan on that same observation without
spending a turn on WAIT. A planless WAIT whose prediction was successfully
committed retains the spent attempt for the next observed state. Every other
fallback still clears the planner, and ordinary visual/metadata reconciliation
continues. A failed prediction or expired cutoff clears all executable state.

No deadline is reset or extended: initial planningt0+3.0s, recovery/cutofft0+3.5s,
and reference initial refinementt0+2.0s. Frozen P3 backend, CNN weights, selected
temperature/confidence/margin/view, perception policy and training fit count are
unchanged. No consumed final corpus was rerun.

## Checks

The old code failed the deterministic cold-start regression. The first partial
repair retained the attempt but still emitted an unnecessary initial WAIT;
the tighter no-extra-WAIT regression also failed. The complete repair moves
RIGHT immediately after perception at originalt0+3.05s, continues correctly,
and retains the same budget. Separate emission coverage checks planless WAIT
retention with no cached suffix; existing cutoff and fallback tests still pass.

The complete repository suite passed352/352 in178.073s. Focused47/47 tests,
Ruff, format, type checks and byte-identical repackaging passed. All5889
protected parent files match. See checks.json, the RED/GREEN logs and patches.

Complete exact-source public module and notebook-scope batches each passed66/66
wins and exact boards with rewards equal to original frozen V4. Raw reports and
measured resource maxima are under complete-repair/.

The controlled slow-start comparison injects first-perception delay to original
t0+3.05s in explicitly diagnostic-only source wrappers. Its failed rows,
watchdog exits and previous-source comparison are retained separately. Those
injected timings are not source qualification or platform results.

## Preserved boundaries

Original frozen V4fb978e3b8bc7 and notebook repair95e43962 remain unchanged. The
original299/300 final result belongs tofb978e3b8bc7; it is not a new final result
for this repair. Unseen-art performance is unproven. The user's earlier11/22
platform image result and subsequent diagnostic are preserved as reports.

The attempt-state-only sourcef7d0204d and its complete public reports remain
historical. Its original startup diagnostic stopped when the watchdog killed a
worker and the old runner rejected the absent returned hash. That stopped
attempt is retained, not merged into the later complete comparison. The later
comparison keeps such timeout rows as failures and never increases its5s limit.

## Reproduction

Restore the original V4 supplementary parts, then the notebook compatibility
repair, then this repair's supplementary archive into a NEW directory. Do not
overwrite concurrent edits in the live workspace. The standalone has no runtime
file or network dependencies. Only the frozen standalone is intended for pasting.

```sh
MPLCONFIGDIR=/tmp/v4-mpl UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python -m unittest tests.test_v4_agent tests.test_v4_tracking tests.test_v4_package tests.test_v4_backend -v
MPLCONFIGDIR=/tmp/v4-mpl UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python -m unittest discover -s tests -v
```

The public and controlled-startup runner scripts preserve the established CPU0,
one-thread, fresh-worker, Performance/EPP and shared timing-lock recipe. Use new
output directories for later runs. They do not unlock or evaluate final data.
