# Task 1 candidate handoff

Use `task1_agent.py`, SHA256:
`55d3ae5655e5c4d6d289e7a8812a0a2bee2b37ae137008472539d5d405c5600a`

The 12,922-byte file is self-contained and preserves `Agent()` and
`step(state) -> Action`. It uses the Python standard library and provided
Grid Adventure/Grid Universe packages. No model, runtime asset, GPU or
repository helper is required. Image solving is outside P1.

Local qualification passed: all 46 tests, Ruff lint/format, ty with warnings
as errors, three public sweeps with 15/22 wins each and perfect intro results,
50/50 heldout replay, max episode 3.705 s and peak RSS 312.4 MB.
See `qualification.json`, `summary.json`, test logs and the parent README.

Search uses the public simulator, a 3.0 s initial deadline, absolute 3.5 s repair
cutoff, 4096 retained states and 32768 compact records. Full dynamic state is
checked before cache reuse. WAIT after exhausted planning is a legal bounded
fallback, not a recovery claim. Seven challenge losses remain visible.

Platform status: **Compatibility retry pending. A prior Debugging attempt failed generically;
no graded attempt was made by the agent.**
Run this exact file in **Capstone Project (Debugging), Task 1**, and retain the
reported status, errors, runtime/performance and applicable limits together
with this hash. Record that evidence separately from local measurements.
The plan requires observed Debugging success before recommending a graded
attempt. Any source change needs revalidation; a graded upload remains an
explicit user handoff. Prior qualified versions remain preserved.

Replace the entire editor with this file. Compatibility regressions now cover
Python 3.10 grammar and grader-style exec/prelude loading. See the parent
compatibility.json and platform-results.json; the actual grader cause is still
unconfirmed. Debugging attempts are unlimited. If the generic error persists,
the optional parent diagnostics/wait_only.py probe should evaluate normally
with zero performance and can isolate the minimal loader/API path.
