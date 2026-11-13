# Fresh implementation review

Reviewer: gpt-6-astra, ultra reasoning, fresh context, read-only; one whole-implementation review before final unlock. No sub-reviewers or remote work. Review covered the V4 inventory, binding plan/spec, ledger rulings and protected parents.

No Critical findings. One Important: standard fold-selection/default policy-screen/production-transfer entrypoints could omit actual board contracts, despite the corrected reselection supplying them. A fused ghost winner absent from both retained view winners could incorrectly count exact. Fixed by requiring parser contracts for policy admission, separate explicit raw epoch/NLL scoring, automatic contracts in ordinary entrypoints, and contract loading before fitting. Tests `test_policy_selection_requires_actual_board_contracts` and `test_production_loads_actual_contracts_before_any_fit` observed RED then GREEN. No additional training; selected weights/policy unchanged. Whole suite after this fix346/346 passes132.746s (before subsequent diagnosed runtime fix).

Minor deferred: diagnostic `trace_agent` wraps old `classify_initial`, not V4 `classify_evidence`. First-frame model/perception time is included in total elapsed but missing from named stage attribution. It does not alter cold qualification timing. Later `check_observation`/reconcile/planning stages are measured.

Strengths confirmed: transactional WAIT/deadline handling, independent evidence/ambiguity bounds, exact frozen P3 backend, all5889 protected files unchanged, corrected120/120 production transfer with four fits/epoch14, cold/corpus provenance and honest exposed-art boundary.

Accepted ledger rulings: temperature grid correction; independent implementation interleaving; development-only warm screens; saved-logit actual-parser reselection; proven normalized-probability margin equivalence; review before final unlock with results/delivery pending.

Declined to judge and executor dispositions:

- Final model/host results: awaiting actual exact-source gates; no sign-off inferred from transfer or unit tests. Cost of error: false performance claim.
- Remote hardware/jobs: reviewer did not inspect host; executor's per-row live affinity/power/library provenance governs. Cost: incomparable timing.
- Frozen P3 redesign: unchanged bytes are binding; only integration is evaluated. Cost: inherited planner limitations remain.
- Unseen-art generalization: no untouched reserve exists, so status stays unproven and original gate false. Cost: limited generalization evidence.
- Commits/archive/HANDOFF: scoped checkpoints are authorized; final delivery must preserve actual data/runs/evidence. Cost: reproducibility gap if archives omitted.

Assessment: implementation ready with Important fix; no qualification sign-off until measured gates and archive complete. Subsequent unchanged-ambiguity repair is independently diagnosed/tested and does not change model selection.

Update after target-host cold failure: the diagnostic-stage omission became
material to failure diagnosis and was fixed. A real standalone Agent trace test
failed without `classify_evidence` attribution and passed after adding that name
to the diagnostic wrapper. Full suite349/349 passed132.593s; scoped Ruff/ty pass.
The original qualification runner is preserved at
`checkpoints/benchmark-qualified-runner.py`; candidate/model/policy bytes did not
change. There are no remaining deferred Minor findings.
