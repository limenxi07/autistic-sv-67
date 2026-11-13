# B4 timeout repair

Paste the complete `submission/task2_agent.py` into the Task 2 Agent cell.
SHA256: `0b61f5c7c594541b12656995aa2389642ce37a48ad9f5a2bc3544360c45d92d5`. Size: 719,790 bytes.

The previous submission eb417a551f9c was user-reported 21/22, with Measure Twice and Cut Once timing out at five seconds. Its large-keyed-inventory guard routed B4 through expensive reference search.

V4 now tries a compact proposal for this specific unsupported inventory schema. All other schema checks remain. Proposal search is bounded to 0.75 seconds within the existing deadline. Every proposed state must match a complete public-simulator replay and the route must end in a win before dispatch. Rejected or unsuccessful proposals fall back to the reference planner. Replay uses the original absolute planning cutoff; the 0.75-second bound covers search, not replay. Existing image and metadata checks, execution deadlines, model parameters and frozen structured P3 remain unchanged. No level-name or route lookup is used.

## Validation

- Omarchy exact standalone: 66/66 wins, exact boards and performance 1.0.
- Omarchy function-scoped notebook: 22/22 wins, exact boards and performance 1.0.
- Maximum strict cold completion across both batches: 0.644 seconds. Reports and recorded exit 0 are retained.
- Full repository suite: 379/379 in 162.010 seconds. Ruff and type checks pass. Independent reviewer found no actionable issues and separately passed 39 focused tests.
- Sixfold elapsed-clock stress plus minimum 14ms environment cost: 22/22 wins, 20/22 target rewards. B4 wins at -387. Repeated B4 comparison: previous submission 0/3 wins (all exceeded five seconds); repaired submission 3/3 wins in 4.417–4.511 seconds. This is synthetic timing, not platform certification.
- Regression was observed failing before implementation: B4 required 606 public transitions; the test now requires no more than 150 while independently replaying the winning 130-action route. Rejected-proposal fallback also passes.

The `submission` bytes reproduce `bounded` exactly. `candidate` is an earlier 0.25-second search experiment, not the final source. Previous submission and frozen P3 hashes are preserved in protected-hashes.json. No retraining or final-holdout rerun. Platform 22/22 remains pending retest.
