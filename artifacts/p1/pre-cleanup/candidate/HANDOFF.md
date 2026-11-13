# Task 1 candidate handoff

The exact single-file agent is `task1_agent.py`. It uses only Python standard
library and the assignment's installed Grid Adventure/Grid Universe packages.
No training, GPU, model, runtime asset file or repository helper is required.
`Agent()` and `Agent.step(state) -> Action` are preserved. Images are outside P1.

Selected configuration: best-first search using the public simulator, 3.0 s
initial deadline and an absolute 3.5 s repair cutoff, 4096 retained states and
32768 compact records. An unplanned state after the cutoff receives WAIT;
that is a legal bounded fallback, not a recovery claim. Exact score/turns,
health, keys and timed/usage-limited effects participate in the state key.

Source SHA256:
`620f04dfdaeaaf8edad9a64feb84fbe2dbd66fafb41712d0c06162573b45d23f`

`manifest.json` records source bytes/configuration/build provenance.
`qualification.json` records local gates against three exact-artifact public
sweeps; profiling timings are excluded. `heldout.json` is one evaluation of
the 50 previously frozen held-out layouts. Generated scores have no invented
optimum: report wins and raw rewards, with normalized performance unavailable.

Platform status: **Debugging validation pending. No attempt has been uploaded.**
The plan requires observed success in **Capstone Project (Debugging), Task 1**
before recommending a graded attempt. Use the exact source above for that
check and retain the result together with its hash. Share the platform's
status, errors, runtime/performance and any displayed limits so they can be
recorded separately from local evidence. A changed source needs revalidation.
The graded submission remains an explicit user handoff; local public wins
and averages do not predict hidden-test marks.

P1a and P1b checkpoints remain preserved in the parent directory. Unsolved
public challenge maps are reported as losses, not hidden by the intro gate.
See the parent README for complete measurements and reproduction commands.
