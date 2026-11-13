# Independent P1 review

Reviewer: gpt-6-astra, high reasoning, fresh context, read-only review of source,
tests, plan/spec and public API behavior. No installed engine source inspected.

Critical findings: none. Minor findings: none.

Important finding: qualification accepted missing/incorrect CPU-affinity,
numerical-library thread counts, turn limits and setup/evaluation watchdog
provenance. Current measured runs used correct settings, but the gate could
certify unrelated measurements. Fixed with mandatory execution and host/package
provenance checks. `test_execution_provenance_required_and_enforced` first
failed across 11 invalid-field subcases, then passed after the fix; it also
checks missing fields. The complete suite subsequently passed (43 tests).

Reviewer additionally probed every mechanic trace through public
`from_state(to_state(...))` and every action: state keys and successor keys
matched. Reordered distinct-ID stacked effects also preserved successor
keys. No concrete planner, cap-accounting, incumbent or cache defect found.

Rulings on matters the reviewer set aside:

- Tuning, profiling, held-out and exact-artifact repeat measurements are the
  implementer's remaining evidence work; they must exist before qualification.
- Platform acceptance and hidden-test results require actual platform evidence;
  do not infer them from local passes. A Debugging handoff remains required.
- Arbitrary objective/movement closures are outside the documented built-in
  configuration support; same-named custom closures need richer identity.
- Image solving and mechanics outside the assignment schema remain outside P1.
- Duplicate numeric entity IDs are malformed wrappers, not supported gameplay.
- Static packaging checks are guardrails for this reviewed source, not a
  security sandbox for deliberately evasive Python.
- Engine internals remain uninspected under the plan's public-API restriction.

Cost of the scope rulings: arbitrary/custom/malformed inputs and adversarial
Python are not qualified. Public runtime tests cannot establish hidden-test
marks or platform acceptance.
