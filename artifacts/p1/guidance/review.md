# Independent P1c review

Reviewer: gpt-6-astra/high, read-only review of the guided/lazy planner snapshot
in guidance/experiment_agent.py (906lines) and its64-test suite.

No Critical, Important or Minor findings. All64 tests passed. Additional
weak-reference probes at state caps3/5/8/12/30 found live GridStates below the
reported peaks and caps. Lazy aliases conservatively overcount states. Tiny
coin-map exhaustive rewards matched; cap exits preserved highest encountered
replayable win. No discarded pending action, heuristic pruning, incumbent loss
or uncounted state reference was established.

Final score follow-up after that review: risk-neutral wall-aware score
refinement plus minimum protective-pickup tours were added. The new65th test
failed for B2/B7 before those changes and passed at2000simulations afterward;
the whole65-test suite passed and a full public sweep reached all22 catalog
scores. These small ordering changes were verified by tests/measurements,
not a second reviewer. Current source/table/caps unchanged in purpose; every
reported solution still comes from the public transition function.

Rulings on declined topics:

- Platform/hidden reliability requires actual external results; local passes
  do not establish them. Cost: Debugging retry remains a handoff.
- Heuristic optimality/admissibility is not promised; estimates order work,
  never prune. Cost: bounded search may miss better or any solution elsewhere.
- Conservative alias counting is intentional. Cost: may stop below real cap.
- Geometry estimates approximate effects/resources; simulator alone defines
  transitions. Cost: poor estimates can waste time, not certify false wins.
- Arbitrary closures, duplicate-ID/malformed wrappers and unsupported mechanics
  remain outside the declared schema. Cost: no qualification for those inputs.
- Engine internals remain uninspected per plan. Cost: black-box evidence only.
- Packaging checks are guardrails, not an adversarial Python sandbox.

A subsequent focused review of carried-reference identity is recorded separately in reference-review.md. Current source also adjusts first-win hazard/box penalties with RED/GREEN regression evidence; these small weight changes were not part of the original64-test review. Historical review scope is preserved.
