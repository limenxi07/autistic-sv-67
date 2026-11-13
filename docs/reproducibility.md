# Reproducing runs

Use `uv sync --locked` and `uv run --locked` to reject unintended lockfile changes. Commit the lockfile with dependency changes. The Python requirement is 3.13; record the exact patch version, uv version, OS, CPU, thread settings, source hash and input hashes with each result.

The [training README](../training/README.md) covers each recorded V3/V4 fit, dataset restoration and export. The lockfile provides mutually exclusive CPU and CUDA 12.8 groups, both on PyTorch 2.7.0; CPU is the default.

## Large assets

Git LFS holds large assets. After cloning, run `git lfs pull`. Two oversized archives are stored as ordered chunks; `docs/reproducibility/archive-parts.json` records their paths, sizes and SHA256 values. Reconstruct with:

```sh
uv run --locked python tools/restore_archives.py
```

Original archives are locally ignored, not deleted. The restoration command verifies existing archives and never overwrites mismatching files. `original-inventory.json` records pre-cleanup file hashes; source paths may now be found through the rename ledger. Frozen artifacts and reports retain original paths and bytes.

## Tests and timing

Use a writable, warmed Matplotlib cache for process-watchdog tests:

```sh
mkdir -p /tmp/grid-mpl /tmp/grid-cache
export MPLCONFIGDIR=/tmp/grid-mpl XDG_CACHE_HOME=/tmp/grid-cache
uv run --locked python -c 'import matplotlib.pyplot'
uv run --locked python -m unittest discover -s tests
```

Timing comparisons require the same CPU and one library thread. Record actual settings; dependency locking does not make timings identical across computers. Tracking must stay outside timed episodes and submitted agents. W&B integration remains unchanged; Trackio is deferred.

## Historical releases

Use frozen agents and their associated manifests to reproduce historical results. Renamed maintained modules can produce different package hashes because source comments and module names changed. Never attach old acceptance claims to a newly packaged source without checking it.

Historical receipts that name removed modules read their original bytes from the pre-rename Git snapshot recorded in `rename-map.json`. This verifies historical evidence, not current renamed code. Preserve Git history when cloning (avoid shallow clones for this check). New qualification still checks the actual current source against its new export; old receipts are not acceptance for changed source.

Cleanup environment: uv 0.13.8, CPython 3.13.15, macOS arm64. The existing environment passed `uv sync --locked --offline --dry-run` with no changes. This checks the lock/environment agreement, not a fresh network installation.

## Deduplicated files

Current submission packages and older distinct releases live in `submissions/`. Old artifact paths are relative symlinks. Byte-identical image/data assets share content-addressed files under `artifacts/shared/`; Git LFS manages those objects. Reports and logs remain independent. `deduplication.json` records every alias, target and content hash.

Clone with symlink support (Linux/macOS, or Windows with Git symlinks enabled), then run `git lfs pull`. Keep the repository layout intact when copying it; copying only a symlink directory omits its targets. Treat stored releases and shared data as immutable. Generate new experiments in new output directories, since writing through a symlink changes its shared target.

Verify the deduplicated tree after cloning with `uv run --locked python tools/verify_artifacts.py`. This checks each stored object once, validates relative symlinks, and verifies current submission hashes. Deduplication reduces working-tree copies; it does not rewrite existing Git history or unpack/rewrite historical archives.
