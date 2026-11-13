# V3 vision qualification: not promoted

**The frozen candidate fails the exact-initial-board gate.** Final holdout: **295/300 wins (98.33%)**, but **288/300 exact initial boards (96.00%)**, below the required 99% (297/300). The win-rate target is 95%. Every other named check passes. No release approval or platform submission is implied.

Executable: [frozen-reviewed/task2_agent.py](frozen-reviewed/task2_agent.py), **668,698 decimal bytes**.
SHA256: `71c0fd0e84d55dd76dc785ce8aba298766f6e1ca709f8f5c1d9a87e131f9ef1d`.

The user selected finishing this frozen P2-next qualification before P3 integration. Backend: `agents/p2_next.py`, SHA256 `6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd`. No P3 integration was performed. P1/P2 frozen sources remain unchanged. The V3 structured dispatch won 266/266 with **exactly equal per-case rewards** to this parent; the oracle bridge matched all 22 paired rewards.

## Measured qualification

Omarchy, Intel Core Ultra 9 275HX, P-core CPU0; sequential fresh workers under `/tmp/grid-adventure-timing.lock`, user-selected Performance/EPP performance stable before and after each episode. One CPU affinity and one library/inference thread. CPU Torch 2.7.0, no GPU inference. The lock was released after all runs finished.

| Suite | Wins | Exact initial boards | Max episode s | Max strict cold s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Public image | 66/66 | 66/66 | 3.1883 | 3.2949 | 313.9 |
| Final image holdout | 295/300 | 288/300 | 3.8028 | 3.9287 | 316.7 |
| Structured regression | 266/266 | n/a | 3.1183 | 3.2105 | 106.4 |
| Oracle bridge | 22/22 | n/a | 2.7455 | 2.8523 | 107.5 |

Public image seeds: 2, 43, 20261109; minimum effective performance **1.0**. Final-v5: 100 disjoint layouts, three held-out artwork assignments each, ten mechanic families and all 14 classes. Generated optima are unknown; those rows report raw rewards, not optimality. Every episode fits the 4.25-second target and 1 GB RSS gate, and every parent-measured strict cold episode fits 5 seconds. Maximum strict cold time is 3.9287 seconds; peak RSS is 332,054,528 bytes.

[Machine qualification](qualification-reviewed/qualification.json) · [Per-family/class summary and all losses](qualification-reviewed/summary.json) · [Raw final outcomes](qualification-reviewed/final-300.json) · [Raw public outcomes](qualification-reviewed/public-66.json) · [Paired regressions](qualification-reviewed/regression-paired.json).

## Why promotion failed

There are 23 initial cell errors across 12 boards: **15 gems classified as coins, 7 gems as ghosts, and 1 box as a wall**. Gem recall is 98/120 (81.67%); the aggregate win rate masks this weakness. Multi-gem: 25/30 exact initial boards and 29/30 wins. Combined mechanics: 23/30 exact boards and 26/30 wins. The other eight families each have 30/30 exact boards and wins.

Five loss rows remain in the raw report. One starts from an exact board but ends with repeated unsuccessful observation reconciliation (`unsupported_observation:ValueError`); the other four record `cutoff`. These reason strings are observed diagnostics, not a claim that all root causes have been isolated. Some initially misparsed boards still win after later observations; a reconstructed-board replay alone never proves perception correct.

The final-v5 corpus has now been evaluated. Any tuning using these results must retire it to regression and freeze a fresh final corpus before another acceptance attempt. The next diagnosed work is gem-style discrimination and the failed temporal reconciliation, not a claim that changing the planner to P3 fixes image parsing. No training or runtime tuning followed this final evaluation.

## Model, runtime and validation

The agent embeds a 104,526-parameter CPU TorchScript CNN, preserves symbolic history and compares real pixels plus public inventory/effect metadata before continuing a conditional plan. Actor subviews and an episode-local observed actor template help with overlaps. A validation-selected 5% inset view distinguishes the gem/shield pair; it did not cover the final gem/coin and gem/ghost failures.

One absolute episode clock begins before perception: initial planning through +3.00s, verification/repair cutoff +3.50s, initial reference refinement capped at the same origin +2.00s. Structured input never loads Torch. Runtime uses embedded model bytes and no external files/assets/network. Unsupported ambiguity produces a legal WAIT with a reason; this is not a safety guarantee. `Agent.parse` can reject a known exit covered by the actor; this deferred visualization limitation is recorded separately from gameplay outcomes.

Training used an isolated RTX 5080 Laptop CUDA environment and `torch.compile(mode="reduce-overhead", dynamic=False)`. Four initial runs were retained: one ten-minute diagnostic pilot and three full 20-epoch seeds. Each full run used 200,000 class-balanced augmented crops plus 72,600 real-board crops. Selection used validation only. Selected seed 20261111 achieved 14,000/14,000 validation crops and 120/120 validation boards (40 each: known-layout/unseen-art, new-layout/known-art, new-layout/unseen-art), all 14 class recalls 1.0. These validation numbers do not replace the failed final-holdout gate.

Checkpoint SHA256: `30a48c263e04528a6bdefc8bc094f6f9517c2e591efc535357c85d28f3059175`.
Model export SHA256: `298fe663f9ba98cc79b9656a450179cb9c9208b8c94e639c961da9314eb44a9e`.
Temperature: 0.25. FP32 eager/export logits match exactly for batches 1, 7, and 225. CPU cold import/load/225-cell forward measured 0.527s, above the aspirational 0.50s initial target; complete-episode gates are separately reported above.

[Selected training run](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/v9l4ucrv) · [Seed 20261110](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/dbfnmwxc) · [Seed 20261109](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/bkkvujr1) · [Pilot](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/stnlrddw). All local configs, histories, calibration and checkpoints are archived. No credentials or W&B SDK are in the submitted source.

## Verification and reproducibility

Fresh repository suite: **243/243 tests pass**; focused V3 suite: **80/80 pass**. Scoped ty and Ruff pass, with an explicit I001 import-order suppression only for the exact generated model loader. Its bytes were preserved. Fresh standalone tests cover an unregistered grader namespace/prelude, Python 3.10 grammar, in-memory TorchScript loading, runtime file/network denial and lazy structured dispatch. Repackaging produced byte-identical standalone source. These are software/package checks, not evidence that the failed image-accuracy target passed.

[Check receipts](qualification-reviewed/checks.json) · [Implementation decisions](qualification-reviewed/decisions.md) · [Execution ledger](qualification-reviewed/execution-ledger.md). The fresh review found two important issues (retained contradicted hypotheses and unbound holdout reports); both were fixed before this freeze with failing-then-passing tests. The covered-exit visualization limitation remains deferred. The required parent inputs are untracked in a detached checkout; work is preserved in scoped archives, with no broad staging, merge, push or submission.

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/v3-mpl UV_CACHE_DIR=/tmp/v3-uv-cache uv run --frozen --no-sync python -m unittest discover -s tests -p 'test_v3_*.py' -v
PYTHONPATH=. UV_CACHE_DIR=/tmp/v3-uv-cache uv run --frozen --no-sync python artifacts/v3/reproduce/assemble-reviewed.py
```

The second command intentionally exits **1** and regenerates the failed qualification report from the saved evidence. It does not rerun or tune the final holdout. The exact remote execution recipe is preserved as `reproduce/run-qualification-reviewed.sh`; do not overwrite the saved final reports by rerunning it. Keep the repository CPU lock separate from the recorded CUDA training environment.

## Supplementary parts

These are distinct from the 668,698-byte executable:

- [Reviewed code/model/evidence archive](supplementary/code-models-reviewed.tar.gz), including configs, checkpoints, histories, source, tests, raw reports and a per-file SHA256 manifest.
- [Actual training/validation dataset archive](checkpoints/task2-datasets-v2.tar), 1,593,968,640 bytes; SHA256 `4dc7dca7ab61958e692ea0933b0b0a81975f6a5e0b8188798b80aff2c5f87452`.
- [Actual final-v5 dataset archive](checkpoints/final-datasets-v5.tar.gz), 43,659,268 bytes; SHA256 `7a58712c444280053be98b07dbd930c99f27d20e24f4c5b201bb5a1bd1f14e39`.
- [Part sizes and hashes](supplementary/reviewed-parts.json) and [SHA256SUMS](supplementary/SHA256SUMS-reviewed).

Final-v5 canonical manifest SHA256: `6dc8e2c0cbbc05b706d613472d321264755d5484c83743225d8d08f5d554be52`; frozen record SHA256: `234d320ff4a1b7218083837d72e63ea0545cec5addcdd5dfec8e3ee9a5e1ce60`. Earlier candidates and failed/power-saver diagnostics remain preserved. Prior final-v2/v3/v4 corpora were superseded before any final model outcomes; v5 was frozen before this evaluation.

All evidence is local public/synthetic evaluation. It does not establish private-artwork generalization, user-run Debugging passes, or graded performance for V3.
