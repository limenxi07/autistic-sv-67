# V3 execution record

Plan: `docs/superpowers/plans/2026-11-09-v3-vision.md`.
Implementation authorized by the user's 9 November request to get started.

## Task 1

Bridge, frozen-P2a-derived adapter, evaluator modes and metadata probes implemented.
Oracle gate: **22/22 wins with exact paired P2a rewards** on Omarchy CPU 0,
single library thread. Maximum oracle evaluation 2.754 s; maximum worker RSS
110,800,896 bytes. This excludes learned perception; no image-agent or graded
success is claimed. See `bridge/public.json` for exact evaluated source hashes.
The archive includes earlier failed attempts rather than replacing them.

Local regression: **123 tests pass** in `bridge/all-tests.log`. The first baseline
attempt had three harness setup failures due to cold Matplotlib cache setup;
the writable-cache rerun passed all 113 pre-existing tests. Ruff and ty pass on
Task 1 changed files. Engine implementation source was not inspected.

Decisions and experimental findings:

- Public resource metadata exposes IDs, key types, effect types, counters and
  multipliers. Preserve complete entries. Board-only identities are locally
  assigned negative slots; after pickup use the explicitly observed resource
  identifier. No unobserved engine ID or environment handle enters the bridge.
  This qualifies the design's local-ID wording: inventing arbitrary resource
  integers changes Python-set key consumption semantics.
- Re-identification invalidates cached state keys. The V3 adapter independently
  replays and rebinds the remaining winning suffix against its current conditional
  board, bounded by the original cutoff. This is not proof of correct perception.
- B4 uses the existing large-inventory reference fallback. Its original initial
  refinement could consume the entire repair allowance. The adapter sets original
  deadlines explicitly and caps first reference refinement at t0+2.00 s, within
  the plan's t0+3.00 s ceiling. A measured B4 probe retained -387 and completed in
  2.745 s; the full paired suite then passed. No P2 schema expansion was made.
- A missing predicted history on a later covered cell is explicitly unresolved.
  Full layered labels exist only in the oracle diagnostic; production image
  interfaces receive the unmodified image and public metadata.

## Remaining

Task 2 is complete: selected `data/manifest-v2.json`, 200,000 verified training
crops, 14,000 validation crops, 600 real training boards, three validation axes
of 40 boards each, and 300 final boards. All 14 classes are represented; audits
found zero split/provenance violations. The v1 corpus is retained after an
initial-label audit found the opened-door class absent from its full boards.
No model was evaluated on either final corpus. Actual v2 data bytes are archived
in `checkpoints/task2-datasets-v2.tar`, with a separate tooling checkpoint.

Task 3 is in progress. CNN, lazy inference, standalone tensor export and the
audited class-balanced training loop pass six focused tests. The full suite
passed 138 tests before the user's torch.compile addition. CUDA 12.8/Torch
2.7.0+cu128 forward/backward works in a separate Omarchy environment on the
16 GB RTX 5080 Laptop GPU. Compiled-training measurements are in progress;
checkpoints retain the original eager model weights for CPU export.

Pilot/full training, calibration, tracker, standalone packaging and final
qualification remain. Frozen P1/P2 source hashes were rechecked unchanged.
No submission has been made.
