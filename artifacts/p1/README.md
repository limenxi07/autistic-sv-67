# P1 Task 1 evidence

P1 is frozen after the user reported **22/22 Debugging tests passed**. The exact source is preserved in `frozen/task1_agent.py`.

Current exact source: `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5` (37,287 bytes). All 68 tests, Ruff lint/format and ty with warnings as errors pass. Three exact-export public sweeps reached **22/22 wins and catalog maximum rewards each** (66 wins; mean effective performance 1.0), with no errors/timeouts, maximum episode 3.199105 s and peak RSS 322.6 MB. Fixed tuning and heldout corpora each replayed 50/50 wins; these are revalidation of the existing corpus, not new independent heldout evidence. The user reported all 22 Debugging tests passed for the latest delivered source; P1 is now frozen. See `artifacts/p1/candidate/platform-validation.json`.

The frozen source is `frozen/task1_agent.py`; `candidate/task1_agent.py` is byte-identical. Preserve the entire file, including imports and helpers. `candidate/HANDOFF.md` gives the website instructions. Source/evidence digests are in `candidate/evidence-sha256.json`; supplementary source, tests, build inputs and reports are in `p1-supplementary.zip`.

## Results and platform distinction

| Checkpoint | Public local wins | Public local mean | Reported website result |
|---|---:|---:|---|
| Starter |11/22|0.474026|not recorded|
| P1a bounded BFS |10/22|0.428571|not recorded|
| P1b compatibility checkpoint |15/22|0.659441|10/22 on latest delivered source; submitted hash unconfirmed|
| Guided score-refined checkpoint ce5a7034 |22/22 ×3|1.0|20/22; Ow Oof Owie and Warehouse failed; submitted hash unconfirmed|
| Frozen P1 |22/22 ×3|1.0|22/22 passed, user-reported|

The previous candidate's two remaining failures returned -450, lose=True, timeout=False and error=False at 4.20/4.07 s. Planning exhaustion is consistent with those outcomes and local restricted-work experiments; no website action trace was supplied. Current first-win diagnostics require 119 simulator calls for Ow Oof Owie (previous 491) and 210 for Warehouse (previous 720). All actual transitions still use the public simulator. See `platform-results.json` and `detour-ordering/`.

A stronger box penalty applied to every search phase regressed A6's coin score to 0.428571. That experiment was rejected and preserved in `pre-box-refinement/`. The final source uses stronger hazard/box penalties to find a first win, then restores the earlier box penalty and neutral hazard cost during score refinement. A matched 3,000-call regression verifies A6's maximum reward. No public names, maps, catalog scores or action routes appear in submitted source.

The independent 0.75/0.85-second planning snapshot under `guidance-score/` solved 22/22 with 21 maximum rewards; it has a different source hash and is diagnostic evidence for that earlier snapshot. It is not acceptance evidence for this file.

## Bounds, checks and review

Validation host: Omarchy, Intel Core Ultra 9 275HX, Linux x86_64, Python 3.13; workers sequential with CPU affinity [0], numerical-library threads set to 1, 150-turn episodes and 5 s evaluation/30 s setup watchdogs. Maximum evaluation: 3.199105 s against the 4.25 s local target; maximum RSS: 322,584,576 bytes against the 1 GB local target. Import/environment setup was measured separately at 0.617002–0.625914 s. The private grader's timing boundary remains unverified.

Search keeps the 3.0 s initial and absolute 3.5 s repair deadlines, 4096 retained simulator states and 32768 compact records. All Actions remain available, deduplication uses complete score/turn/resource keys, and cached steps compare predicted full states. WAIT after exhausted planning is a legal fallback, without a recovery claim. Geometry estimates order work and never certify or prune a route.

The carried-reference audit found that exchanging usage counters between two shield IDs can change the next public transition even when the counter multiset is unchanged. The key now preserves each carried entity's stable reference ID and components. New RED→GREEN tests, public conversion roundtrips and a focused independent review verify this fix; cosmetic board IDs remain omitted. See `guidance/review.md` and `tests/test_p1_state.py`.

68 tests passed locally (32.099 s) and on Omarchy (20.252 s); Ruff lint/format and ty with warnings as errors pass. Packaging checks cover Python 3.10 grammar, unregistered-module execution, a prepended grader prelude, byte identity, size, isolation and a real step. The guided/lazy planner was independently reviewed at 64 tests with no findings, including weak-reference cap probes. Later score/weight changes were covered by regression tests and exact-source measurement. Original P1b review/provenance findings and frozen sources remain preserved; see `review.md` and `completion-audit.md`. Starter notebook behavior and dependency pins are unchanged.

All 100 generated layouts have public-simulator-verified witnesses and a fixed 50/50 split created before P1b selection. The corpus has been replayed after algorithm changes, so it is no longer fresh evidence of generalization. Unknown generated optimal rewards remain unavailable.

## Diagnostic costs

The current separate profile attributes 85.8% of aggregate episode time to public simulation, 8.7% to state keys and 1.5% to timed heap/child-priority work. The 4.0% residual includes action ordering, setup, reheap and other work. Profile episodes are excluded from acceptance timing; see `candidate/profile-summary.json`. Earlier P1a/P1b profiles retain their own source hashes.

The current-source latency probe sampled 10,080 transitions: maximum 1.056 ms and p99 0.256 ms. Maximum key latency was 0.125 ms. Observed reserve is not a bound for other hosts or unseen states.

## Reproduction

```sh
uv sync --frozen
ruff check agents tools tests
ruff format --check agents tools tests
ty check agents tools tests --error-on-warning
uv run --frozen python -m unittest discover -s tests -p 'test_p1*.py' -v
uv run --frozen python tools/package_p1.py --source agents/p1.py --output-dir artifacts/p1/candidate
uv run --frozen python tools/benchmark_p1.py --agent artifacts/p1/candidate/task1_agent.py --suite task1 --repeat 3 --output artifacts/p1/candidate/public-repeat3.json
uv run --frozen python tools/qualify_p1.py qualify --public artifacts/p1/candidate/public-repeat3.json --manifest artifacts/p1/candidate/manifest.json --output artifacts/p1/candidate/qualification.json
```

On the sandboxed Mac, prepend `MPLCONFIGDIR=/private/tmp/p1-matplotlib UV_CACHE_DIR=/private/tmp/p1-uv-cache` and use `uv run --frozen --no-sync` with the existing environment. Worker cleanup exitcode-15 after a completed result does not replace its recorded outcome. Watchdog results leave unavailable reward/turn/RSS fields null.

Debugging attempts are unlimited; graded attempts are separate. No graded attempt or website upload was performed by the agent. Local qualification does not establish hidden-test marks, image support, arbitrary custom closures or unsupported engine versions. Historical reports retain actual hashes; no old report certifies the changed source.

Frozen archive: `p1-frozen-a51380b9.zip`; verify it with the adjacent `.zip.sha256` file. Future milestone work must preserve the frozen P1 bytes.
