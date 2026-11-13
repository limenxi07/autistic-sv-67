# Reproduce training

Run commands from the repository root in Bash or Zsh. P1–P3 are search planners and have no training step. V3 and V4 train the same 104,526-parameter cell classifier; V4 uses a different data and selection workflow. Boss mode reuses V4 weights.

## 1. Environments

Install uv and Git LFS, then fetch the data:

```sh
git lfs install --local
git lfs pull
uv sync --locked
```

CPU is the default. GPU training uses a separate environment with the same lockfile:

```sh
# Linux x86-64 with an NVIDIA GPU and a compatible CUDA driver
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --locked --no-group cpu --group cu128
```

Both groups pin PyTorch 2.7.0 and torchvision 0.22.0. They are mutually exclusive within one environment. CUDA 12.8 does not run on macOS. Keep CPU export separate: the exporter explicitly requires CPU PyTorch 2.7.0.

Define these helpers in the shell used for the remaining commands:

```sh
cpu() { uv run --locked "$@"; }
gpu() { UV_PROJECT_ENVIRONMENT=.venv-cuda uv run --locked --no-group cpu --group cu128 "$@"; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/grid-mpl
mkdir -p "$MPLCONFIGDIR"
gpu python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

The recorded host was an RTX 5080 Laptop GPU, PyTorch 2.7.0+cu128, CUDA 12.8 and Python 3.13.18. H100 should support this stack, but has not been validated for this project. The lockfile fixes package versions, not GPU arithmetic: new runs may differ in weights, timings and selected epochs.

## 2. Recorded runs

All full fits use AdamW, learning rate 0.001, weight decay 0.0001, batch 448 and `torch.compile`. Use a new output directory for each reproduction.

| Run | Seed | Duration | Data / purpose |
|---|---:|---|---|
| V3 pilot | 20261109 | 600-second limit, up to 20 epochs | First 20,000 generated crops plus training-board crops; throughput diagnostic |
| V3 full | 20261109, 20261110, 20261111 | 20 epochs each | 200,000 augmented crops plus 72,600 training-board crops |
| V4 fold 0/1/2 | 20261109, 20261110, 20261111 | 20 epochs each | Three disjoint artwork-group folds |
| V4 production | 20261112 | Selected epoch; historically 14 | All exposed training artwork, with a separate transfer check |

Original configs and reports: [`artifacts/v3/runs/`](../artifacts/v3/runs/) and [`artifacts/v4/runs/batch-1/`](../artifacts/v4/runs/batch-1/). V3 selected seed 20261111. V4 selected temperature 1.8340080864093418, confidence 0.90, margin 0.10 and the full-cell view. These are recorded results, not values to force onto a new selection run.

## 3. Restore data and checkpoints

V3 input datasets are already in `artifacts/v3/data/` after fetching LFS. V4 corpora and full training histories are archived:

```sh
cpu python tools/restore_archives.py
cpu python tools/restore_training.py --output training/restored
```

The second command verifies archive and member hashes, then restores only datasets and saved training runs. It refuses an existing destination and does not restore old source code over this checkout. Successful completion writes `training/restored/restoration.json`.

```sh
export CORPORA="$PWD/training/restored/artifacts/v4/data/corpora"
export RUNS="$PWD/training/runs/reproduction-1"
```

The original checkpoint is restored under `training/restored/artifacts/v4/runs/batch-1/production/final.pt`. It must retain SHA256 `45a6768c467d0bacac481f70e033b4f1d71560c5db49c2553ae4ba1861207690`.

Dataset generation code is included in [`tools/vision_v3_data.py`](../tools/vision_v3_data.py) and [`tools/vision_v4_data.py`](../tools/vision_v4_data.py). V4 exposure, fold assignments and selection rules are in `artifacts/v4/data/`. Use the saved datasets above to reproduce the recorded inputs. To generate a separate V4 corpus from the saved rules:

```sh
cpu python -m tools.vision_v4_data \
  --exposure artifacts/v4/data/exposure.json \
  --folds artifacts/v4/data/folds.json \
  --output training/generated-v4
```

Regeneration writes a new corpus; verify its audit and manifests before training. Do not substitute new inputs into old reports.

## 4. V3 pilot and full fits

Define the shared input arguments:

```sh
v3_inputs=(
  --data artifacts/v3/data/train-200k-v2
  --validation artifacts/v3/data/validation-14k-v2
  --train-boards artifacts/v3/data/train-boards-600-v2
  --boards artifacts/v3/data/known-layout-unseen-art-v2
           artifacts/v3/data/new-layout-known-art-v2
           artifacts/v3/data/new-layout-unseen-art-v2
)
```

The pilot predates augmentation and W&B integration. Its exact trainer is preserved:

```sh
gpu python artifacts/v3/models/pilot-source/train_v3.py "${v3_inputs[@]}" \
  --seed 20261109 --epochs 20 --batch 448 --device cuda \
  --compile --compile-mode reduce-overhead --pilot-seconds 600 --limit-train 20000 \
  --output "$RUNS/v3-pilot"
```

Run all three full fits with the maintained trainer:

```sh
for seed in 20261109 20261110 20261111; do
  gpu python -m tools.train_vision_v3 "${v3_inputs[@]}" \
    --seed "$seed" --epochs 20 --batch 448 --device cuda \
    --compile --compile-mode reduce-overhead --augment --wandb-mode disabled \
    --output "$RUNS/v3-$seed"
done
```

W&B integration remains available; logging is disabled here so reproducing a run does not require an account. Compare validation metrics in each run's `summary.json`; keep final evaluation data out of selection.

## 5. V4 folds, selection and production

Train the three folds from scratch on the GPU:

```sh
gpu python -m tools.train_vision_v4 --corpora "$CORPORA" --output "$RUNS/v4"
```

Select the shared epoch, calibration and policy using CPU exports and downstream validation. Use `-c` here: the evaluator spawns workers and cannot restart a script read from standard input.

```sh
cpu python -c '
import os
from pathlib import Path
from tools.vision_v4_selection import screen
runs = Path(os.environ["RUNS"])
screen(corpora=Path(os.environ["CORPORA"]), batch=runs/"v4",
       output=runs/"v4-policy", collapse_margins=True)
'
```

Fit production from scratch using that run's selected epoch and policy:

```sh
gpu python - <<'PY'
import json, os
from pathlib import Path
from tools.train_vision_v4 import fit_production
runs = Path(os.environ['RUNS'])
selection = json.loads((runs/'v4-policy/selection.json').read_text())
config = json.loads(Path('artifacts/v4/runs/batch-1/production/config.json').read_text())
config.update(data=str(Path(os.environ['CORPORA'])/'production'),
              output=str(runs/'v4/production'), epochs=selection['epoch'])
result = fit_production(config, selection)
if not result['transfer_passed']:
    raise SystemExit('Production transfer check failed; inspect the report')
PY
```

This is one three-fold selection batch plus one production fit. The restored `final-300` corpus is excluded from training and selection. It is historical evaluation data, not a fresh blind holdout.

## 6. CPU export and evaluation

Export either a V3 `best.pt` or V4 `final.pt` with the CPU environment. This example uses the new V4 production model:

```sh
cpu python - <<'PY'
import json, os
from pathlib import Path
from tools.export_vision_v3 import export_model
from tools.export_vision_v4_numpy import export_numpy
runs = Path(os.environ['RUNS'])
report = export_model(runs/'v4/production/final.pt', runs/'torch-export')
export_numpy(runs/'torch-export/model.pt', runs/'numpy-export',
             temperature=report['temperature'])
PY
```

Both exporters check prediction parity. Stage the new model and selected policy with the maintained runtime, then create standalone Task 2 and Task 3 candidates:

```sh
cpu python - <<'PY'
import json, os, shutil
from pathlib import Path
from tools.vision_v4_selection import stage_runtime
from tools.package_vision_v4 import package
runs = Path(os.environ['RUNS'])
export = runs/'numpy-export'
selection = json.loads((runs/'v4-policy/selection.json').read_text())
report = json.loads((export/'report.json').read_text())
policy = {**selection['policy'], 'model_sha256': report['export_sha256']}
policy_path = runs/'policy.json'
policy_path.write_text(json.dumps(policy, indent=2) + '\n')
stage = runs/'runtime'
stage.mkdir(exist_ok=False)
stage_runtime(stage, export, policy)
shutil.copyfile('agents/boss_v4.py', stage/'agents/boss_v4.py')
for boss in (False, True):
    output = runs/'candidates'/('task3' if boss else 'task2')
    print(package(stage, export, policy_path, output, task3=boss))
PY
```

This uses the current runtime. Historical planner behavior requires the matching version commit and its original package. Keep newly built candidates separate from submitted releases.

Run public development checks on the new files:

```sh
cpu python -m tools.benchmark_vision_v3 --suite public --no-paired \
  --source "$RUNS/candidates/task2/task2_agent.py" --output "$RUNS/task2-public.json"
cpu python -m tools.benchmark_boss_v4 --mode image --seeds 42 123 211 \
  --source "$RUNS/candidates/task3/task3_agent.py" --output "$RUNS/task3-public.json"
```

Keep the source hash, command, hardware, data hashes and reports together. Public checks are not graded results. A new fit/export is a new candidate; old acceptance results do not transfer to it.

The current ready-to-paste files remain in [`submissions/`](../submissions/README.md). No new GPU training or H100 qualification was performed as part of repository cleanup.
