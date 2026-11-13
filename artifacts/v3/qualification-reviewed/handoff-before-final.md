# V3 vision candidate

**Qualification is pending. Do not treat this file as a release approval yet.**

Current executable: [frozen-v2/task2_agent.py](frozen-v2/task2_agent.py), 667,198 decimal bytes. SHA256 `c921753682174ce03d0e311de13c3e2e230e93fde4da08fbcb4141bf11c68f5e`. The structured and image paths use the user-selected P2-next backend, SHA256 `6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd`. Frozen P1/P2 bytes are preserved.

The image agent embeds a 104,526-parameter CPU TorchScript CNN, keeps symbolic history and checks actual pixels and public inventory/effect metadata before continuing a conditional plan. Small independent actor views and an episode-local actor template help with overlaps. A 5% shape view handles the diagnosed gem/shield confusion pair. Reconstruction and planning remain conditional on perception. Unsupported ambiguity returns a legal WAIT and an explicit reason; recovery is counted only after an independent following observation agrees.

A single episode clock starts before perception: initial deadline 3.00 seconds, final planning/repair cutoff 3.50 seconds; reference refinement is capped at the same origin +2.00 seconds. Deployment is CPU-only and forces one Torch intra-op/inter-op thread. Structured input does not load Torch. No external model or asset file is needed by the submitted source.

## Training and selection

Training ran on Omarchy's RTX 5080 Laptop in an isolated CUDA environment, using `torch.compile(mode="reduce-overhead", dynamic=False)`. The measured 100-update comparison was 4.744ms eager versus 3.761ms compiled (1.261x). All four initial runs are retained: one ten-minute diagnostic pilot and three full 20-epoch seeds. Selection used validation boards, then class recall; no final-corpus outcomes were used to select weights.

Selected seed: 20261111. Validation: 14,000/14,000 crops, all 14 class recalls 1.0, and 120/120 boards across the three layout/art axes. Temperature 0.25. Checkpoint SHA256 `30a48c263e04528a6bdefc8bc094f6f9517c2e591efc535357c85d28f3059175`; CPU export `298fe663f9ba98cc79b9656a450179cb9c9208b8c94e639c961da9314eb44a9e`. FP32 eager/export logits match exactly for batches 1, 7 and 225. Measured CPU cold Torch import/load/225-cell forward: 0.527 seconds; full-episode qualification is separate.

W&B project: `enxi-nanochat/autistic-sv-67`.

- [Selected seed 20261111](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/v9l4ucrv)
- [Seed 20261110](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/dbfnmwxc)
- [Seed 20261109](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/bkkvujr1)
- [Diagnostic pilot](https://forge.coreweave.com/wandb/enxi-nanochat/autistic-sv-67/runs/stnlrddw)

Logs include loss, gradient norm, learning rate, throughput/update time, per-class precision/recall/F1/confusion, exact-board accuracy by validation axis, CPU/GPU memory/system statistics, compile timing, dataset manifests and checkpoint artifacts. W&B is development-only; no credential or SDK is bundled in the executable.

## Evidence boundaries and pending work

The first frozen candidate won 65/66 public image cases; one high-confidence gem/shield error caused the failure. Its source and report remain intact. The revised source corrected all 66 initial public boards, but a run under power-saver hit the original cutoffs on long cases. Unchanged old and new sources both ran about twice as slowly under that host profile. The user then selected Performance locally. The unchanged candidate is queued for complete qualification, with the shared timing lock and before/after CPU power-state receipts.

Final-v2/v3 corpora were retired before any model outcomes: v3 corrected one-gem generation in the multi-gem family; v4 provides fresh layouts after the public-policy revision. Final-v4 contains 100 disjoint layouts, ten families ×30 episodes, three reserved appearance assignments per layout, and all 14 initial classes. It is audited with zero violations and remains untested by a model at this draft. Manifest SHA256 `8c1769673c62b45a0b414bcc70774a74db01aa97628d618091e7daaaae4ffc51`. Public-library and synthetic holdouts cannot establish performance on private artwork or private grading.

The planned acceptance reports are public66, structured266 versus the frozen parent, oracle22, final300 and the strict qualification result. No platform submission, user Debugging result, or graded outcome is claimed.

## Reproduction and supplementary files

Use the repository's locked CPU environment. The CUDA training environment and command recipes are recorded separately; do not replace the deployment lock with CUDA dependencies.

```sh
MPLCONFIGDIR=/tmp/v3-mpl UV_CACHE_DIR=/tmp/v3-uv-cache uv run --frozen --no-sync python -m unittest discover -s tests -p 'test_v3_*.py' -v
MPLCONFIGDIR=/tmp/v3-mpl UV_CACHE_DIR=/tmp/v3-uv-cache uv run --frozen --no-sync python -m tools.package_v3 --source-root . --export artifacts/v3/models/selected-export --output-dir /tmp/v3-repacked
```

The standalone tests execute in a fresh unregistered grader namespace, deny Python-level runtime file/network access, observe a BytesIO TorchScript load, and check the structured path does not import Torch. Python 3.10 grammar is checked; CPU execution measurements use the pinned Python 3.13/Torch 2.7 environment.

Supplementary parts are distinct from the executable upload:

- `supplementary/code-models.tar.gz`: explicit source/config/checkpoint/run-history whitelist and per-file checksums.
- `checkpoints/task2-datasets-v2.tar`: actual training/validation datasets, grouped asset manifests and original generated corpora.
- `checkpoints/final-datasets-v4.tar.gz`: actual fresh final images/assets, manifest and generator recipe.
- Final qualification evidence and archive checksums will be added after the pending runs and review.
