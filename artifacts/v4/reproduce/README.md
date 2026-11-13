# V4 reproduction

Run from the repository root. The frozen source, selected model/policy, actual
datasets, all four runs and saved reports are checksum-bound in the supplementary
parts. Required unchanged V3 parts are listed separately; restore those at their
original repository-relative paths. Do not overwrite a saved evaluation directory.

## Restore the saved delivery

All archive paths, byte counts, SHA256s and per-member manifests are in
`artifacts/v4/supplementary/parts.json`. Every payload archive has been verified
locally, including all10,604 generated-data members. Keep that index and the
`SHA256SUMS` file beside the parts. Verify checksums before extraction.

Extract into a NEW directory, preserving repository-relative paths, in this order:

1. The three `required_parent_parts` V3 archives in `parts.json`.
2. `code-models-evidence-final.tar.gz`.
3. `generated-datasets.tar.gz`.
4. `source-completion.tar.gz` (supplies the plan/spec, lock files and starter).
5. `final-delivery.tar.gz` (latest handoff, receipts and ledger).

The earlier parts preserve historical in-progress records; the final-delivery
part supplies their completed status without rewriting the old archives. Do not
extract over concurrent changes in this live workspace. No training, final-data
unlock or evaluation is necessary to restore or reassemble the saved delivery.

## Software checks

```sh
MPLCONFIGDIR=/tmp/v4-mpl UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python -m unittest discover -s tests -v
PYTHONPATH=. UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python artifacts/v4/reproduce/assemble.py
```

Assembly reads saved outcomes only. It exits1 for an unqualified/stopped batch;
it never renders or unlocks fresh final outcomes. The CPU dependency lock is
unchanged. Ruff/ty commands, file lists and exits are in qualification/static-*.json.

Repackage the exact runtime with `tools.package_v4.package` using repository root,
`artifacts/v4/models/selected-export`, `artifacts/v4/models/policy.json`, and a NEW
output directory. Compare its SHA256 against models/candidate.json. Exported
loader bytes must remain unchanged. The source embeds exact frozen P3 once and
loads no files/assets/network at runtime.

## Recorded training

Exactly three scratch fold fits,20epochs each, seeds20261109/10/11; one scratch
production fit,14epochs, seed20261112. All use the original104526-parameter CNN,
AdamW, LR.001, weight decay.0001, batch448 and the recorded V3 augmentation.
Configs, compiler probes, initial tensor hashes, all checkpoints, logits and
histories are under runs/batch-1. No fifth fit is authorized by this recipe.

The host GPU interpreter was
`/home/enxi/coding/v3-vision-training-20261109/.venv/bin/python`,
Torch2.7.0+cu128/CUDA12.8, RTX5080Laptop16GB. GPU bootstrap imports Torch FIRST,
then appends read-only CPU project site-packages for public grid dependencies.
The precise hardware/environment probes and bootstrap receipts are archived.
Do not replace the repository CPU environment with CUDA packages.

Shared epoch14 used pre-calibration full-cell RAW CNN board accuracy, worst-class
recall, NLL, then earlier epoch. Calibration uses geomspace.25–8,41points plus1.
Policy selection scores the ACTUAL bounded parser, with public board metadata;
frozen truth labels score results afterwards. The historical raw-metric mistake,
its116/120 admissible transfer, and corrected120/120 receipt remain preserved.
`tools.v4_reselect` corrected saved logits without fitting; ordinary entrypoints
now require actual board contracts too. Model tensor weights/common epoch did
not change. Selected policy is full,T1.8340080864093418,confidence.90,margin.10.

Fresh training, if separately undertaken, must use a new output tree and the
archived corpus contracts. Recorded `train_v4.train_three_folds`,
`v4_selection.screen`, and `v4_production.fit` define the four-fit workflow.
Existing run directories are create-only; reproducing the exact delivered
executable uses the archived selected export and policy, not a new GPU fit.

## Target-host measurements

CPU interpreter:
`/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python`, CPU Torch2.7.0.
`run-development.sh` records the established Omarchy recipe: shared
`/tmp/grid-adventure-timing.lock`, CPU0, one thread, stable user-selected
Performance/EPP, fresh spawned workers and5-second parent watchdog beginning
before process creation. There is no power/profile mutation. Earlier warm policy
screens are development-only. Trace timings are never qualification timings.

The initial source0f9dc27a8bfe repaired all five former losses but lost another
case through repeated unchanged ambiguity. Its reports and the subsequent
tracking/actor-template diagnosis are preserved. Runtime repairs keep model and
policy fixed; all exact-source gates must be rerun for the repaired candidate.

Fresh final data may be unlocked ONCE by `python -m tools.v4_final` only after
all exact-source development gates pass. Never rerun it in the preserved tree.
Any later tuning retires that corpus. All333 public artwork groups were previously
exposed: fresh-layout results do not prove unseen-art generalization or platform
performance. No platform submission is part of this work.
