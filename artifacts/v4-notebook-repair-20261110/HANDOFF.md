# V4 notebook compatibility repair

Use [frozen/task2_agent.py](frozen/task2_agent.py) as the complete replacement Agent source.
Repaired source:693446bytes, SHA256
`95e43962f2357c89f27880f493b7ea2aad4fa33c5142406620a15c27e4ecb501`.
The original V4 source `fb978e3b8bc7` is preserved unchanged.

## Diagnosed boundary and repair

Function-wrapped pasted source captured cell-local perception settings while
`configure()` wrote different module-global bindings. The model stayed
unconfigured, its RuntimeError was recovered as WAIT, and the agent made no
progress. Module execution worked, so the previous package checks missed this
notebook scope. This matching local reproduction is saved in diagnosis.json and
the failing regression log; the platform implementation itself was not inspected.

The repair mutates one shared perception configuration object. Model loading,
calibration and policy now observe the same settings in both scopes. CNN weights,
selected policy, exact P3 backend, tracking/search behavior, original clocks and
resource limits are unchanged. There was no retraining or final-corpus replay.

## Verification

The new regression first failed with two WAIT actions and
`unsupported_observation:RuntimeError`. It now emits two correct RIGHT actions,
loads the real embedded model once, retains the selected policy and produces
exactly equal full probability tensors in notebook and module execution.
Focused tests15/15; complete repository350/350 in164.845s; scoped Ruff, format
and type checks pass. Repackaging is byte-identical. All5889 protected parent
files match their original hashes. See checks.json and the preserved logs.

Comparable complete public execution-scope validation is recorded in
public-summary.json, module-public-66.json and notebook-cell-public-66.json.
The run uses Omarchy CPU0, one thread, user-selected Performance/EPP, fresh
spawned workers, the original3.0/3.5-second budget and strict5-second watchdog
starting before process creation. Timing and source hashes are retained per row.

The complete Performance batches passed all checks:

| Scope | Wins | Exact boards | Max episode | Max strict cold |
|---|---:|---:|---:|---:|
| Module |66/66|66/66|3.312057s|3.420471s|
| Notebook cell |66/66|66/66|3.306085s|3.414091s|

Every per-case reward equals frozen V4. Peak RSS329011200bytes (313.8MiB)
across both scopes. All rows used unchanged original deadlines, one CPU thread,
stable Performance/EPP and fresh spawned workers; no diagnostic trace was active.

## Earlier lower-power diagnostic

The first batch ran with CPU EPP=power. Module and notebook execution each
parsed66/66 boards exactly and won60/66. B3 and B4 each failed all three seeds
at the planning cutoff. Unchanged original frozen V4 also failed both seed2
controls in that mode. These complete reports and their failed gate summary
remain under power-mode/. They are not merged with the separate Performance
batch. The agent, model, policy and deadlines were unchanged between batches;
the user plugged Omarchy in and selected Performance before the latter run.

## Evidence boundary

The original source's299/300 fresh-layout result belongs to its original hash;
it is not a new final result for this repair. The already-consumed final corpus
was not rerun. No new unseen-art or graded-performance claim is made. The user's
platform failures are preserved in user-report.json; a platform retest of these
repaired bytes remains pending.

## Reproduction

Restore the original V4 supplementary parts into a NEW directory first. Then
apply this repair's supplementary archive, which contains the changed runtime
module/tests, repaired standalone, source bindings, before/after evidence and
public reports. The standalone itself needs no runtime files or network.

From the repository root:

```sh
MPLCONFIGDIR=/tmp/v4-mpl UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python -m unittest tests.test_v4_package tests.test_v4_perception -v
MPLCONFIGDIR=/tmp/v4-mpl UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python -m unittest discover -s tests -v
```

run-public.py records the public-only execution-scope recipe; it does not use
final data. Use a new evidence output tree for any later run. The notebook-cell
wrapper mechanically indents the exact repaired source into a function and
returns Agent; candidate-binding.json binds both files. Do not paste the wrapper;
paste the linked frozen standalone source.
