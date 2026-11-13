# Runtime repair: Task2 and Task3 selected

The user reported platform Debugging passes for this exact Task2 candidate:
B3 performance1.0, reward-315, runtime3.69s; B4 performance1.0, reward-387,
runtime4.58s. Both won without errors or timeouts. Task2 was promoted to
`submissions/task2_agent.py`; its previous bytes and historical artifact links
were preserved. See [reported results](platform-debugging-results.json) and
[promotion receipt](task2-promotion.json). These two results do not certify
all22 image cases or the graded test sets. The user subsequently reported that the Task3 candidate passed Debugging in both GridState and ImageObservation modes. Task3 was also promoted; detailed scores and runtimes were not supplied. See [boss report](platform-boss-debugging-results.json) and [promotion receipt](task3-promotion.json).

Use the clean candidate files in **Debugging first**:

- [Task 2 image candidate](candidate-4/task2/task2_agent.py)
- [Task 3 boss candidate](candidate-4/task3/task3_agent.py)

Paste the whole appropriate file into its Agent cell. Start with B4 **Measure
Twice and Cut Once** and B3 **So Close Yet So Far**, then run all 22 public image
Debugging cases. Test the Task 3 candidate with both boss input modes. These
public aliases do not establish which hidden layouts belong to graded image set 4.

No scored submissions were made by the agent. Task1 remains unchanged. Previous
submission bytes are preserved under `baseline/submissions`; the old Task2 is
also archived under its full hash in `submissions/releases/`. The previous Task3 is archived there as well.

## Changes

The image agent avoids full-board tile slicing when only selected cells are
needed, duplicate actor/pushable scans, redundant floor-template conversion,
and repeated general decoding of plain wall/exit components. Background updates
copy the current frame and restore occluded tiles with identical integer pixel
boundaries. Full dynamic validation, visual checks, reference replay, model
weights and calibration are retained. Expiry during perception now reports
`cutoff` rather than misleadingly reporting `unresolved_perception`.

The boss structured adapter retains its 12-second planning cutoff and can
dispatch an already verified matching suffix until 14.65 seconds. It handles
crossing the planning cutoff inside the delegated call without emitting a
spurious WAIT. It cannot consume a planned WAIT twice, reset deadlines, or start
a new search after the planning deadline.

## Validation

- Full repository suite: **415/415** passed; Ruff, production-module type checks,
  package reproducibility, and diff checks passed.
- Omarchy, exact final source: **66/66** public image episodes, **22/22** notebook
  image episodes, **12/12** boss episodes, and **4/4** boss notebook episodes.
  All reached their full threshold-adjusted score. These are development runs.
- B4 mean agent time: macOS **0.596 -> 0.417 seconds** (30% lower); Omarchy
  **0.346 -> 0.258 seconds** (25% lower). Full helper overhead was included in
  episode results; these agent-only comparisons isolate the changed code.
- Final macOS eightfold boss slowdown: **12/12**, maximum **12.662 seconds**.
- Final macOS sixfold image slowdown, with 14 ms minimum environment cost:
  **21/22**. B4 still crossed the 4.65-second execution cutoff before finishing,
  first cutoff at turn 108 and hard timeout after 122 returned actions. Those
  actions include fallback WAITs. This remains a timing limitation.

The initial full run had three harness setup failures while Matplotlib built its
font cache. The isolated harness rerun passed 10/10 and the final full run passed
415/415. No harness source change was needed.

Pixel and state equivalence tests cover odd image boundaries, malformed inputs,
custom entity subclasses, mutated scenery/components, public-route conversion
and observation keys, and B4's complete 130-action pixel-equivalent replay.
Package tests verify both module and notebook scope and preserve frozen P3's
original bindings. See [review](review.md), [summary](summary.json), and the
`final-*` JSON/log files for exact evidence and provenance.

The harness's generic `evidence_boundary` text still says macOS in remote
reports; per-row `provenance` correctly records the Omarchy host. Source loading
is timed separately; episode timing includes reset, construction, cold model
inference, public history conversion, environment rendering, and final history.

## If Debugging still times out

[Optional timing probe](task2_end_debug_only.py) contains the exact Task 2
candidate followed by the existing minimal end-report wrapper. It intentionally
raises `TASK2_END_291207ab` before a predicted final action or on the first
fallback. Use it only in Debugging to obtain timing through an error-only UI.
An intentional diagnostic exception is not a gameplay pass or a normal candidate
error. Never use this probe for grading. Its source binding is recorded in
[diagnostic-binding.json](diagnostic-binding.json).

## Frozen candidates

Task 2: `291207ab939644f7c06ebf6e6f340b10f7665ee0d051804408fe10cda7106e9c`
(731,887 bytes).

Task 3: `ff7b47d1f2e6d97148cec07eac5ddcd2e6b959bac64192f763ec8c47e6c2f1c3`
(734,984 bytes).

Both rebuild byte-for-byte. Frozen P3 remains
`d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`.
Each candidate directory contains its source, model, policy, and runtime-source
hash manifest. Earlier candidate directories are experimental receipts, not
files for submission.
