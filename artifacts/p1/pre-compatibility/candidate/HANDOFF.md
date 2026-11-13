# Task 1 candidate handoff

Use `task1_agent.py`, SHA256:
`3c56f6514b1839d444666b4ccc2efecf98c4f8f49452fcb5375314553021a7cd`

The 13,014-byte file is self-contained and preserves `Agent()` and
`step(state) -> Action`. It uses the Python standard library and provided
Grid Adventure/Grid Universe packages. No model, runtime asset, GPU or
repository helper is required. Image solving is outside P1.

Local qualification passed: all 43 tests, Ruff lint/format, ty with warnings
as errors, three public sweeps with 15/22 wins each and perfect intro results,
50/50 heldout replay, max episode 3.704s and peak RSS312.5MB.
See `qualification.json`, `summary.json`, test logs and the parent README.

Search uses the public simulator, a 3.0 s initial deadline, absolute 3.5 s repair
cutoff, 4096 retained states and 32768 compact records. Full dynamic state is
checked before cache reuse. WAIT after exhausted planning is a legal bounded
fallback, not a recovery claim. Seven challenge losses remain visible.

Platform status: **Debugging pending. No upload/graded attempt was made.**
Run this exact file in **Capstone Project (Debugging), Task 1**, and retain the
reported status, errors, runtime/performance and applicable limits together
with this hash. Record that evidence separately from local measurements.
The plan requires observed Debugging success before recommending a graded
attempt. Any source change needs revalidation; a graded upload remains an
explicit user handoff. Prior qualified versions remain preserved.
