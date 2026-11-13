# Grid Adventure agents

Planning and image-based agents for the Grid Adventure capstone. Start with the version table, then read [development lessons](docs/development-history.md). The original [assignment](docs/assignment.md) is preserved separately.

The [documentation index](docs/README.md) links every stage plan and the [implementation/debugging chats](docs/chats/README.md).

## Versions

| Name | Purpose | Main module |
|---|---|---|
| Reference-state planner (P1) | Search using public game transitions | `agents.planner_p1` |
| Compact-state planner (P2) | Faster search with a compact simulator | `agents.compact_p2` |
| Boss-search planner (P2-next) | Improved ordering for difficult boards | `agents.boss_p2_next` |
| Hierarchical planner (P3) | Resource-aware subgoals and execution reserves | `agents.hierarchical_p3` |
| Image-tracking agent (V3) | Reconstruct and track boards from images | `agents.vision_v3` |
| Image-recovery agent (V4) | Recover from uncertain observations and slow setup | `agents.vision_v4` |
| Boss-mode agent (V4) | Explicit longer budget for Task 3, both observation formats | `agents.boss_v4` |

Vision agents reuse structured planners. Keep these modules together. Frozen submissions in `artifacts/` are independent historical releases; do not edit them when changing maintained code. Historical plans and reports use the old names: consult the [rename ledger](docs/renames.md).

## Submission files

The three current standalone agents are together in [`submissions/`](submissions/README.md). Copy the complete task file into its Agent cell. Older distinct packages live in `submissions/releases/`; old artifact paths are compatibility symlinks.

The original course material remains in `capstone-project.ipynb` and `tutorial.ipynb`. The benchmark starter is `agents/starter.py`; no Marimo runtime is required.

## Setup and tests

Install uv and Git LFS, then run:

```sh
git lfs install --local
git lfs pull
uv sync --locked
uv run --locked python -m unittest discover -s tests
```

The lockfile is authoritative. See [reproducibility](docs/reproducibility.md) for large archives, cache setup, environment recording and benchmark limits. W&B remains available through the existing optional training integration; no account is required for planner tests. Trackio is deferred, not installed by this cleanup.

## Reproduce training

Follow the [training README](training/README.md) for the V3 pilot, all three V3 full runs, and V4 fold selection and production training. It includes dataset restoration, seeds, CPU/CUDA environments, export and packaging commands. CPU PyTorch is the default; CUDA 12.8 training uses a separate `.venv-cuda` environment. Both retain PyTorch 2.7.0.

## Boss evaluation and packaging

```sh
uv run --locked python -m tools.benchmark_boss_v4 --source artifacts/v4-boss-budget-20261113/submission/task3_agent.py --output /tmp/boss-results.json --seeds 42 123 211
uv run --locked python -m tools.package_vision_v4 --help
```

The boss handoff records public development checks and a pending platform retest. These are not graded results. See [boss handoff](artifacts/v4-boss-budget-20261113/HANDOFF.md).

## Repository layout

- `agents/`: maintained planners, perception and tracking code.
- `tools/`: training, packaging and evaluation commands.
- `training/`: reproduction guide; local restored inputs and new runs are ignored by Git.
- `tests/`: regression tests and fixtures.
- `submissions/`: current standalone task files and older distinct releases.
- `artifacts/`: experiment reports and datasets; identical large data files share storage through relative symlinks.
- `docs/`: research, development history and rename mappings.

Version commits import existing snapshots; they do not reconstruct every historical edit. The final naming refactor is separate from those snapshots.
