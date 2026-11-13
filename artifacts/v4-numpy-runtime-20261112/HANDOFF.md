# V4 same-weight NumPy runtime

Frozen runtime: [frozen/task2_agent.py](frozen/task2_agent.py),679147 bytes.
SHA256 `832a8450b61fec09076d66bb03a324b10452a6bae0932d5fbb455238c9862c2e`.
This is a separate runtime revision. Original frozen V4 and the notebook repair
remain unchanged. Platform retesting is pending.

The actual platform diagnostic showed a first WAIT after3.092559s with no
underlying exception, one admissible hypothesis and planner_no_verified_plan.
A real local cold Torch run reproduced perception taking3.241486s and spending
the initial planning interval. A late planner then repeatedly restarted its
expired initial attempt. The bounded recovery fix addresses that loop without
resetting clocks, but the controlled3.05s-startup experiment still loses longer
episodes because little execution time remains. Those failures are preserved.

This revision also removes Torch from image inference. It exports exactly the
existing104526 trained FP32 parameters to an in-memory compressed NPZ and runs
the same CNN with NumPy. No model fitting, architecture change, new training
input, threshold change or calibration change occurred. The new policy file
changes only export-binding metadata. Export report records every tensor hash,
original checkpoint/TorchScript lineage and byte equality on reload.

Convolution, ReLU, adaptive2x2 average pooling, flattening and linear layers match
the original network. Im2col intermediates are bounded in32-crop chunks. NumPy
FP32 operation order introduces small rounding differences: maxlogiterror
2.3365020751953125e-5 on random batches1/7/225. Across88 real public initial
boards/13656cells, maxprobabilityerror2.384185791015625e-7, zero changed argmax,
and zero changed confidence/margin admissibility decisions.

The notebook-scope configuration repair remains included. A late-created planner
uses the remaining original recovery interval immediately, avoiding a needless
WAIT; only a successfully committed planless WAIT retains its spent attempt.
Initialt0+3.0s, reference initialt0+2.0s and hard cutofft0+3.5s are unchanged.
Every state-changing action still passes prediction/reconciliation. Frozen P3
backend bytes are unchanged.

The package loads only embedded BytesIO data with allow_pickle=False. Tests deny
Torch imports, local module imports, file access and network access while checking
real image actions and one lazy model load. Function-scoped notebook execution
retains model/policy/calibration and matches module probabilities exactly.

The final full suite passed354/354 in159.914s, including the reviewer-requested
unsafe-np.load regression. The earlier353/353 suite is preserved too. A subsequent
style-only validator cleanup passed the package suite again. Scoped Ruff/type/format and
byte-identical repackaging pass. All5889 protected parent files match. Read the
checks and review receipts for final verification status.

Default-thread first-action diagnostics on this Mac: Basic Movement.110249s,
So Close Yet So Far.841038s, Measure Twice2.287178s; no Torch imported. The full
local seed42 diagnostic won20/22 with equal rewards on wins; B4 and B10 hit the
cutoff and remain explicit failures. These are local behavior diagnostics,
not comparable Omarchy timing or platform results.

The detached target-host batch uses fresh workers, CPU0, one thread and stable
Performance/EPP under the shared timing lock. It tests public module66,
notebook66, exact debugging seed42 across22levels, and retired300 regression
jobs. Raw results belong in this directory; do not restart or merge batches after
connection loss. Host-side run.exit and validation-summary.json determine actual
completion. No consumed final data is rerun.

The original299/300 fresh-layout result belongs to the oldfb978 source, not this
revision. Unseen-art performance remains unproven. Platform-default Linux BLAS
thread behavior still requires actual platform retesting; default local behavior
and target evaluation with thread limits do not establish that outcome.

For reproduction, restore the original V4 parts and previous notebook repair
into a NEW directory, then apply this revision's supplementary part. It contains
the new runtime modules, exporter, tests, export/report/policy and evidence. The
standalone itself needs no runtime files or network. Existing run outputs must
not be overwritten, and the consumed final corpus must not be used for tuning.

## Completed target validation

| Batch | Wins | Exact initial boards | Reward comparison |
|---|---:|---:|---|
| Public module |66/66|66/66|Every reward equal to frozen V4|
| Public notebook scope |66/66|66/66|Every reward equal to frozen V4|
| Debugging public seed42 |22/22|22/22|Every effective performance1.0|
| Retired development corpus |300/300|300/300|Every reward equal to frozen V4|

All checks passed on the single saved complete target batch. Maxepisode
3.396148612955585s, maxstrictcold3.5059287060284987s,
peakRSS131903488bytes (125.8MiB). All rows retained original budgets, stable
Performance/EPP, fresh workers and no Torch import. Network loss delayed result
retrieval; no row was restarted or replaced. Original job exit0 is preserved.

The earlier controlled3.05s startup comparison is complete too: previous Torch
source0/22wins; budget-only repair10/22wins. Its failures explain why the budget
repair alone was not delivered as the final fix. The NumPy runtime was evaluated
without injected latency, since it removes the expensive import itself.

Status: ready for an actual ungraded platform retest. No graded/private outcome
or new final/unseen-art qualification is claimed.
