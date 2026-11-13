"""Acceptance gates for exact-source V3 evidence; never infer generated optima."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agents.vision_v3_bridge import CLASSES
from tools.benchmark_planner_p1 import THREAD_VARS
from tools.vision_v3_data import FAMILIES, canonical, validate_manifest


def key(row):
    return (row.get("label", row.get("level")), row.get("seed"))


def coverage(rows, expected):
    keys = [key(r) for r in rows]
    return len(keys) == len(set(keys)) and set(keys) == set(map(tuple, expected))


def resource_valid(row, *, image):
    meta = row.get("provenance") or row
    return (
        isinstance(row.get("evaluation_seconds"), (int, float))
        and 0 <= row["evaluation_seconds"] < 4.25
        and isinstance(row.get("cold_import_to_terminal_seconds"), (int, float))
        and 0 < row["cold_import_to_terminal_seconds"] < 5
        and isinstance(row.get("peak_rss_bytes"), int)
        and 0 < row["peak_rss_bytes"] < 1_000_000_000
        and len(meta.get("cpu_affinity") or []) == 1
        and all((meta.get("thread_limits") or {}).get(k) == "1" for k in THREAD_VARS)
        and (not image or row.get("torch_threads") == [1, 1])
        and isinstance(row.get("strict_cold_seconds"), (int, float))
        and 0 < row["strict_cold_seconds"] < 5
        and row.get("cold_watchdog_seconds") == 5.0
        and (row.get("power_state_before") or {}).get("epp") == "performance"
        and row.get("power_state_before") == row.get("power_state_after")
    )


def heldout_binding(rows, manifest):
    """Bind outcomes to frozen layouts, texture choices and decoded image hashes."""
    from collections import Counter

    try:
        dataset = manifest["dataset"]
        dataset_hash = hashlib.sha256(canonical(dataset)).hexdigest()
        records, audit = manifest["heldout_records"], manifest["heldout_audit"]
        raw = "".join(
            json.dumps(r, separators=(",", ":")) + "\n" for r in records
        ).encode()
        if (
            len(records) != 300
            or audit["verified_boards"] != 300
            or audit["violations"] != 0
            or audit["manifest_sha256"] != dataset_hash
            or hashlib.sha256(raw).hexdigest() != audit["records_sha256"]
        ):
            return False
        layouts = {r["sha256"]: r for r in dataset["layouts"] if r["split"] == "final"}
        assets = {a["path"]: a for a in dataset["assets"]}
        by_key = {key(row): row for row in rows}
        expected, appearances = [], {}
        classes, families = Counter(), Counter()
        for record in records:
            layout = layouts[record["layout_sha256"]]
            if (
                record["geometry_split"] != "final"
                or record["appearance_split"] != "final"
                or canonical(record["labels"]) != canonical(layout["layout"]["labels"])
            ):
                return False
            if any(
                assets[e["path"]] != e or e["split"] != "final"
                for e in record["constituents"]
            ):
                return False
            label = (
                record["family"]
                + "/"
                + layout["sha256"][:12]
                + "/"
                + str(record["appearance_seed"])
            )
            case = (label, layout["seed"])
            expected.append(case)
            row = by_key[case]
            job = row["job"]
            required = {
                "label": label,
                "seed": layout["seed"],
                "generated": layout["index"],
                "multiple_gems": layout.get("multiple_gems", False),
                "layout_split": "final",
                "expected_layout_sha256": layout["sha256"],
                "appearance_split": "final",
                "appearance_seed": record["appearance_seed"],
                "resolution": record["resolution"],
                "appearance_manifest": manifest["heldout_manifest_path"],
                "appearance_manifest_sha256": dataset_hash,
                "asset_root": manifest["heldout_asset_root"],
                "expected_image_sha256": record["image_sha256"],
                "expected_pixels_sha256": audit["pixels_sha256"][record["image"]],
            }
            if row["observation"] != "image" or any(
                job.get(k) != v for k, v in required.items()
            ):
                return False
            appearances.setdefault(layout["sha256"], []).append(
                record["appearance_seed"]
            )
            classes.update(
                k for line in record["labels"] for cell in line for k in cell
            )
            families[record["family"]] += 1
        return (
            coverage(rows, expected)
            and set(appearances) == set(layouts)
            and all(
                sorted(appearances[h]) == sorted(l["appearance_seeds"])
                for h, l in layouts.items()
            )
            and dict(classes)
            == audit["class_counts"]
            == manifest["heldout_class_counts"]
            and dict(families)
            == audit["family_counts"]
            == manifest["heldout_family_counts"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def qualify(
    public_report: dict, heldout_report: dict, regression_report: dict, manifest: dict
) -> dict:
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

    public = public_report.get("results", [])
    heldout = heldout_report.get("results", [])
    regression = regression_report.get("results", [])
    baseline = regression_report.get("baseline", [])
    oracle = regression_report.get("oracle", [])
    sha = manifest["source_sha256"]
    all_rows = public + heldout + regression + oracle
    baseline_by_key = {key(r): r for r in baseline}
    expected_public = [
        (label, seed)
        for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        for seed in (2, 43, 20261109)
    ]
    expected_oracle = [(label, 2) for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS]
    split_valid = True
    try:
        validate_manifest(manifest["dataset"])
    except (KeyError, ValueError):
        split_valid = False
    dataset_hash = hashlib.sha256(canonical(manifest["dataset"])).hexdigest()
    checks = {
        "source_under_decimal_2mb": 0 < manifest.get("bytes", 2_000_000) < 2_000_000,
        "model_export_parity": manifest.get("export_parity_passed") is True,
        "original_deadlines": manifest.get("configuration", {}).get("initial_seconds")
        == 3.0
        and manifest.get("configuration", {}).get("cutoff_seconds") == 3.5,
        "dataset_hash": dataset_hash == manifest.get("dataset_sha256"),
        "heldout_bound_to_frozen_corpus": heldout_binding(heldout, manifest),
        "declared_observation_modes": all(
            r.get("observation") == "image" for r in public + heldout
        )
        and all(r.get("observation") == "gridstate" for r in regression)
        and all(r.get("observation") == "oracle" for r in oracle),
        "disjoint_splits": split_valid,
        "100_frozen_final_layouts": sum(
            l.get("split") == "final" for l in manifest["dataset"].get("layouts", [])
        )
        == 100,
        "all_14_classes": set(manifest.get("heldout_class_counts", {})) == set(CLASSES)
        and all(v > 0 for v in manifest.get("heldout_class_counts", {}).values()),
        "ten_families_30_each": manifest.get("heldout_family_counts")
        == {k: 30 for k in FAMILIES},
        "public_66_coverage": coverage(public, expected_public),
        "public_66_wins": len(public) == 66
        and all(r.get("status") == "win" and r.get("win") is True for r in public),
        "public_performance": bool(public)
        and all(
            isinstance(r.get("effective_performance"), (int, float))
            and r["effective_performance"] >= 0.50
            for r in public
        ),
        "heldout_300_coverage": len(heldout) == 300
        and coverage(heldout, manifest.get("expected_heldout", [])),
        "heldout_initial_boards_99pct": len(heldout) == 300
        and sum(r.get("initial_board_exact") is True for r in heldout) >= 297,
        "heldout_wins_95pct": len(heldout) == 300
        and sum(r.get("status") == "win" and r.get("win") is True for r in heldout)
        >= 285,
        "generated_optima_unknown": all(
            r.get("effective_performance") is None
            if r.get("status") == "win"
            else r.get("effective_performance") == 0
            for r in heldout
        ),
        "source_hashes": bool(all_rows)
        and all(r.get("source_sha256") == sha for r in all_rows),
        "no_runtime_overrides": all(not r.get("runtime_overrides") for r in all_rows),
        "no_budget_reset": all(
            r.get("budget_reset") is False and r.get("budget_seconds") == [3.0, 3.5]
            for r in public + heldout + oracle
        ),
        "image_resources": all(resource_valid(r, image=True) for r in public + heldout),
        "structured_resources": all(
            resource_valid(r, image=False) for r in regression + oracle
        ),
        "regression_266_coverage": len(regression) == len(baseline) == 266
        and coverage(regression, manifest.get("expected_regression", []))
        and coverage(baseline, manifest.get("expected_regression", [])),
        "baseline_exact_backend": all(
            r.get("source_sha256") == manifest["backend_sha256"] for r in baseline
        ),
        "no_lost_regression_wins_or_rewards": bool(regression)
        and all(
            r.get("status") == "win"
            and r.get("win") is True
            and key(r) in baseline_by_key
            and r.get("reward", float("-inf"))
            >= baseline_by_key[key(r)].get("reward", float("inf"))
            for r in regression
        ),
        "oracle_22_paired": coverage(oracle, expected_oracle)
        and all(
            r.get("paired_reward_match") is True and r.get("status") == "win"
            for r in oracle
        ),
    }
    if manifest.get("artifact_path"):
        raw = Path(manifest["artifact_path"]).read_bytes()
        checks["artifact_bytes_and_hash"] = (
            len(raw) == manifest["bytes"] and hashlib.sha256(raw).hexdigest() == sha
        )
    return {
        "local_gates_passed": all(checks.values()),
        "checks": checks,
        "failures": [k for k, v in checks.items() if not v],
        "source_sha256": sha,
        "model_sha256": manifest["model_sha256"],
        "dataset_sha256": dataset_hash,
        "public_wins": sum(
            r.get("status") == "win" and r.get("win") is True for r in public
        ),
        "heldout_wins": sum(
            r.get("status") == "win" and r.get("win") is True for r in heldout
        ),
        "heldout_initial_boards_exact": sum(
            r.get("initial_board_exact") is True for r in heldout
        ),
        "max_evaluation_seconds": max(
            (r.get("evaluation_seconds", 0) for r in all_rows), default=None
        ),
        "max_cold_import_to_terminal_seconds": max(
            (r.get("cold_import_to_terminal_seconds", 0) for r in all_rows),
            default=None,
        ),
        "max_peak_rss_bytes": max(
            (r.get("peak_rss_bytes", 0) for r in all_rows), default=None
        ),
        "evidence_boundary": "Local public/synthetic qualification only. No private asset-distribution, Debugging or graded claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "heldout", "regression", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = qualify(
        *(
            json.loads(getattr(args, n).read_text())
            for n in ("public", "heldout", "regression", "manifest")
        )
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["local_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
