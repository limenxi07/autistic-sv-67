# Files to submit

Copy the entire appropriate file into its task's Agent cell:

| Task | File |
|---|---|
| Task 1: structured planner | [task1_agent.py](task1_agent.py) |
| Task 2: image agent | [task2_agent.py](task2_agent.py) |
| Task 3: boss mode | [task3_agent.py](task3_agent.py) |

For the supporting source upload, include the repository's tracked files with Git LFS assets downloaded. This includes dataset generation scripts, training code, saved datasets, the [training guide](../training/README.md), and [implementation conversations](../docs/chats/README.md). Preserve relative symlinks and the archive chunks in `artifacts/`; omit local environments, caches and `.git/`.

Task 2 now contains runtime-repair candidate4 (`291207ab9396`). The user reported platform Debugging passes for B3 (3.69s) and B4 (4.58s), both performance1.0. These are two Debugging results, not graded-set certification. Task3 now contains the tested boss candidate (`ff7b47d1f2e6`), after the user reported Debugging passes with both GridState and ImageObservation. Boss scores and runtimes were not supplied. Task1 remains unchanged. See `manifest.json` for paths and hashes and `../artifacts/v4-runtime-repair-20261113/platform-debugging-results.json` for the reported results.

`releases/` preserves older distinct packages by source hash. Original artifact paths remain as relative symlinks so reports, tests and packaging tools can find them. Treat release files as immutable; regenerate into a new output directory when developing changes. Historical diagnostic scripts remain with their experiment reports.
