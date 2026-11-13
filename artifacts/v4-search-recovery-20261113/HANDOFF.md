# V4 platform timing repair

Use `release/task2_agent.py` in the Agent cell. This is normal runnable source, not the diagnostic that intentionally raises.

Source SHA256: `5f260f30163bd5378a1d693a21ee366a313138cda17703356a90c9a507e37962` (705,450 bytes).

## Diagnosis and changes

User platform diagnostics for required multiple and the vault return a valid first action at approximately 3.03 and 3.06 seconds, with correct visible board layouts and no reported exceptions. V4 has an absolute 3.5-second cutoff. Consuming almost the entire budget before dispatch leaves too little time to complete and then yields WAIT. The first-step trace does not independently prove every later state; full platform retest remains necessary.

- Bound refinement to 25 ms after the first winning incumbent; propagate that deadline to ordering. Fast incumbents are publicly verified before the window begins. Reference incumbents retain independent replay before dispatch.
- Enable the existing admissible reward-bound pruning mode for image planning.
- Cache CNN logits by exact FP32 crop bytes, deduplicating within batches and across calls. Bound the cache to 1,024 entries, reset with each model, and return independent output arrays.
- Retain a single complete same-turn predicted state only after independent image and metadata checks corroborate it. Ambiguous and contradicted observations retain reconciliation.
- Reuse public simulator successors from the safety check when committing the emitted action, avoiding a duplicate transition.

The original trained parameters and calibration remain unchanged. Structured dispatch still uses the frozen P3 Agent. The original 3.0-second initial and 3.5-second absolute cutoff remain unchanged. Prior frozen artifacts remain intact.

## Exact release evidence

Omarchy fresh-worker public images: 66/66 wins, 66/66 exact initial boards. Maximum episode 1.801 s; maximum cold start through completion 2.021 s.

Omarchy pasted-function notebook: 22/22 wins, 22/22 exact initial boards. Maximum episode 1.759 s; maximum cold start 1.978 s. Peak across both runs 102,928,384 bytes (98.16 MiB).

CNN parity on 88 public boards / 13,656 cells: no changed argmax or admissibility decision; maximum probability difference 2.384185791015625e-7.

## Limits and tradeoffs

The Vault reward falls from -65 to -81 in the public runs because refinement is deliberately bounded. Thus 22/22 wins is not a claim of 22/22 maximum-reward platform test passes.

Accelerated-clock diagnostic experiments still lose long episodes at fourfold elapsed time; latest pre-deadline-review cached experiment was 20/22 (B3 and B4 failed). These are synthetic local stress diagnostics, not platform results or a guarantee of safety margins. Full target validation is source-bound in release-summary.json. No consumed final holdout was rerun and no old holdout result is transferred to these bytes.

Repository release test suite: 360/360 tests pass in 135.615 seconds; see release-all-tests.log. Focused tests, Ruff and type checks pass. Review findings and intermediate attempts are retained alongside this file.

Platform outcome: pending user retest. This repair is not promoted under the original final qualification gate.
