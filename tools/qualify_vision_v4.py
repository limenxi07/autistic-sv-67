"""Check V4 regressions and fresh layouts separately from unseen-art evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from agents.vision_v4_bridge import CLASSES
from tools.benchmark_vision_v4 import validate_timing_report
from tools.diagnose_vision_v4 import P3_SHA256
from tools.vision_v4_data import canonical, digest
from tools.vision_v3_data import FAMILIES
from tools.qualify_vision_v3 import key, resource_valid

V3_SHA256 = "71c0fd0e84d55dd76dc785ce8aba298766f6e1ca709f8f5c1d9a87e131f9ef1d"
CRITICAL = ("gem", "coin", "ghost", "box", "wall")


def _wins(row):
    return (
        row.get("status") == "win"
        and row.get("win") is True
        and row.get("lose") is False
    )


def _coverage(rows, jobs):
    expected = [(j["label"], j["seed"]) for j in jobs]
    actual = [key(r) for r in rows]
    return len(actual) == len(set(actual)) == len(expected) == len(
        set(expected)
    ) and set(actual) == set(expected)


def _bound(rows, jobs):
    if not _coverage(rows, jobs):
        return False
    expected = {(j["label"], j["seed"]): j for j in jobs}
    return all(r.get("job") == expected[key(r)] for r in rows)


def _timing(report):
    try:
        validate_timing_report(report)
        return True
    except ValueError:
        return False


def _temporal(rows):
    return all(
        r.get("budget_reset") is False
        and r.get("budget_seconds") == [3.0, 3.5]
        and not r.get("runtime_overrides")
        and not r.get("diagnostic_only")
        and not str(r.get("agent_info", {}).get("reason", "")).startswith(
            ("unsupported_observation", "emission_failure")
        )
        and int(r.get("agent_info", {}).get("stale_turns", 0)) == 0
        for r in rows
    )


def _frozen_corpus_binding(rows, manifest):
    """Derive jobs from hash-bound records, not another mutable job list."""
    try:
        records = manifest["fresh_records"]
        contract = manifest["corpus_contract"]
        frozen = manifest["corpus_final_manifest"]
        audit = manifest["data_audit"]
        if len(records) != 300:
            return False
        if (
            digest((json.dumps(contract, indent=2) + "\n").encode())
            != audit["contract_sha256"]
        ):
            return False
        if (
            digest((json.dumps(frozen, indent=2) + "\n").encode())
            != audit["final_manifest_sha256"]
        ):
            return False
        raw = "".join(
            json.dumps(r, separators=(",", ":")) + "\n" for r in records
        ).encode()
        if digest(raw) != frozen["final"]["records_sha256"]:
            return False
        appearance = manifest["fresh_appearance_manifest"]
        appearance_sha = digest(canonical(appearance))
        assets = {a["path"]: a for a in appearance["assets"]}
        layouts = {r["sha256"]: r for r in contract["layouts"]["final"]}
        appearances = {}
        expected = []
        counts = Counter()
        families = Counter()
        for record in records:
            layout = layouts[record["layout_sha256"]]
            payload = layout["layout"]
            if digest(canonical(payload)) != layout["sha256"] or canonical(
                record["labels"]
            ) != canonical(payload["labels"]):
                return False
            if (
                record["turn"] != 0
                or record["geometry_split"] != "final"
                or record["appearance_split"] != "final"
            ):
                return False
            if (
                record["layout_seed"] != layout["seed"]
                or record["layout_index"] != layout["index"]
                or record["family"] != layout["family"]
            ):
                return False
            if record["manifest_sha256"] != appearance_sha or any(
                assets[a["path"]] != a or a["split"] != "final"
                for a in record["constituents"]
            ):
                return False
            if (
                record["width"] != payload["width"]
                or record["height"] != payload["height"]
            ):
                return False
            if any(
                len(c) != 1 or c[0] not in CLASSES
                for row in record["labels"]
                for c in row
            ):
                return False
            visible = [[c[0] for c in row] for row in record["labels"]]
            if record["visible_labels"] != visible:
                return False
            labels = [k for row in visible for k in row]
            if labels.count("human") != 1 or labels.count("exit") < 1:
                return False
            counts.update(labels)
            families[record["family"]] += 1
            appearances.setdefault(layout["sha256"], []).append(
                record["appearance_seed"]
            )
            label = f"{record['family']}/{layout['sha256'][:12]}/{record['appearance_seed']}"
            expected.append(
                {
                    "label": label,
                    "seed": layout["seed"],
                    "generated": layout["index"],
                    "generator": "v4_optional_sides",
                    "multiple_gems": True,
                    "layout_split": "final",
                    "expected_layout_sha256": layout["sha256"],
                    "appearance_manifest": manifest["fresh_appearance_manifest_path"],
                    "appearance_manifest_sha256": appearance_sha,
                    "appearance_split": "final",
                    "appearance_seed": record["appearance_seed"],
                    "appearance_sampling": "class_group_variant",
                    "resolution": record["resolution"],
                    "asset_root": manifest["fresh_asset_root"],
                    "expected_pixels_sha256": record["pixels_sha256"],
                    "expected_image_sha256": record["image_sha256"],
                }
            )
        if len(layouts) != 100 or any(
            sorted(appearances.get(k, [])) != sorted(v["appearance_seeds"])
            for k, v in layouts.items()
        ):
            return False
        return (
            _bound(rows, expected)
            and manifest["fresh_jobs"] == expected
            and dict(counts) == manifest["fresh_class_counts"]
            and dict(families) == manifest["fresh_family_counts"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def assess_fresh(report, manifest):
    rows = report.get("results", [])
    jobs = manifest.get("fresh_jobs", [])
    support = Counter()
    correct = Counter()
    for row in rows:
        for actual, predicted, count in row.get("initial_confusion", ()):
            if actual in CLASSES and isinstance(count, int) and count >= 0:
                support[actual] += count
                if actual == predicted:
                    correct[actual] += count
    recalls = {
        k: {
            "correct": correct[k],
            "support": support[k],
            "recall": correct[k] / support[k] if support[k] else None,
        }
        for k in CLASSES
    }
    checks = {
        "fresh_jobs_bound": len(rows) == len(jobs) == 300
        and _bound(rows, jobs)
        and _frozen_corpus_binding(rows, manifest),
        "fresh_source_hash": report.get("source_sha256") == manifest["source_sha256"]
        and all(r.get("source_sha256") == manifest["source_sha256"] for r in rows),
        "fresh_exact_boards_99pct": len(rows) == 300
        and sum(r.get("initial_board_exact") is True for r in rows) >= 297,
        "fresh_wins_95pct": len(rows) == 300 and sum(_wins(r) for r in rows) >= 285,
        "critical_class_support_100": all(support[k] >= 100 for k in CRITICAL),
        "critical_class_recall_99pct": all(
            support[k] > 0 and correct[k] / support[k] >= 0.99 for k in CRITICAL
        ),
        "all14_class_denominators": all(support[k] > 0 for k in CLASSES),
        "class_counts_bound": dict(support) == manifest.get("fresh_class_counts"),
        "family_coverage_bound": Counter(r.get("label", "").split("/")[0] for r in rows)
        == Counter({k: 30 for k in FAMILIES})
        and manifest.get("fresh_family_counts") == {k: 30 for k in FAMILIES},
        "fresh100_layouts": len({j.get("expected_layout_sha256") for j in jobs}) == 100
        and all(
            n == 3
            for n in Counter(j.get("expected_layout_sha256") for j in jobs).values()
        ),
        "fresh_image_mode": all(
            r.get("observation") == "image" and r.get("entrypoint") == "Agent"
            for r in rows
        ),
        "fresh_uninstrumented_workers": _timing(report)
        and not report.get("diagnostic_only"),
        "fresh_resources": bool(rows)
        and all(resource_valid(r, image=True) for r in rows),
        "fresh_temporal": _temporal(rows),
        "generated_optima_unknown": all(
            "effective_performance" in r
            and (
                r["effective_performance"] is None
                if _wins(r)
                else r["effective_performance"] == 0
            )
            for r in rows
        ),
    }
    return {
        "checks": checks,
        "class_recall": recalls,
        "wins": sum(_wins(r) for r in rows),
        "exact_boards": sum(r.get("initial_board_exact") is True for r in rows),
    }


def _pairs(rows, baseline, *, equal):
    originals = {key(r): r for r in baseline}
    if len(originals) != len(baseline) or set(originals) != {key(r) for r in rows}:
        return False
    return all(
        _wins(r)
        and key(r) in originals
        and _wins(originals[key(r)])
        and (
            r.get("reward") == originals[key(r)].get("reward")
            if equal
            else r.get("reward", float("-inf"))
            >= originals[key(r)].get("reward", float("inf"))
        )
        for r in rows
    )


def qualify(
    public: dict,
    regression: dict,
    fresh_layout: dict,
    manifest: dict,
    *,
    untouched_art: dict | None = None,
) -> dict:
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

    p = public.get("results", [])
    s = regression.get("results", [])
    o = regression.get("oracle", [])
    retired_report = regression.get("retired_v5", {"results": []})
    v = retired_report.get("results", [])
    parent = regression.get("baseline", [])
    oracle_parent = regression.get("oracle_baseline", [])
    old_structured = regression.get("v3_structured", [])
    old_public = regression.get("v3_public", [])
    old_retired = regression.get("v3_retired", [])
    sha = manifest["source_sha256"]
    all_rows = p + s + o + v + fresh_layout.get("results", [])
    fresh = assess_fresh(fresh_layout, manifest)
    checks = dict(fresh["checks"])
    config = manifest.get("configuration", {})
    expected_public = {
        (label, seed)
        for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        for seed in (2, 43, 20261109)
    }
    expected_oracle = {(label, 2) for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS}
    old_winners = [r for r in old_retired if _wins(r)]
    repaired = {key(r): r for r in v}
    known_losses = {key(r) for r in old_retired if not _wins(r)}
    checks.update(
        {
            "source_under_decimal_2mb": 0 < manifest.get("bytes", 2000000) < 2000000,
            "exact_frozen_p3_backend": manifest.get("backend_sha256") == P3_SHA256,
            "model_and_policy_bound": bool(manifest.get("model_sha256"))
            and bool(manifest.get("policy_sha256"))
            and manifest.get("export_parity_passed") is True,
            "original_clock_and_entrypoint": config.get("initial_seconds") == 3
            and config.get("cutoff_seconds") == 3.5
            and config.get("reference_initial_seconds") == 2
            and config.get("default_structured") == "P3.Agent",
            "bounded_hypotheses": config.get("max_hypotheses") == 4
            and config.get("max_ambiguous_cells") == 2,
            "public66_exact_wins": len(p) == 66
            and {key(r) for r in p} == expected_public
            and all(_wins(r) and r.get("initial_board_exact") is True for r in p),
            "public66_preserved_rewards": len(old_public) == 66
            and all(r.get("source_sha256") == V3_SHA256 for r in old_public)
            and _pairs(p, old_public, equal=False),
            "public_jobs_bound": _bound(p, manifest.get("public_jobs", [])),
            "structured266_p3_equal": len(s) == len(parent) == 266
            and _bound(s, manifest.get("structured_jobs", []))
            and _pairs(s, parent, equal=True)
            and all(
                r.get("source_sha256") == P3_SHA256 and r.get("entrypoint") == "Agent"
                for r in parent
            ),
            "structured_preserved_v3": len(old_structured) == 266
            and _pairs(s, old_structured, equal=False)
            and all(r.get("source_sha256") == V3_SHA256 for r in old_structured),
            "oracle22_p3_adapter_equal": len(o) == len(oracle_parent) == 22
            and {key(r) for r in o} == expected_oracle
            and _pairs(o, oracle_parent, equal=True)
            and all(r.get("entrypoint") == "PlannerAdapter" for r in o)
            and all(
                r.get("source_sha256") == manifest.get("oracle_baseline_source_sha256")
                and r.get("budget_seconds") == [3.0, 3.5]
                for r in oracle_parent
            ),
            "retired_v5_300_exact_wins": len(v) == 300
            and all(_wins(r) and r.get("initial_board_exact") is True for r in v),
            "retired_v5_jobs_bound": len(manifest.get("retired_jobs", [])) == 300
            and _bound(v, manifest.get("retired_jobs", [])),
            "retired_v5_preserved295_rewards": len(old_winners) == 295
            and len(old_retired) == 300
            and all(r.get("source_sha256") == V3_SHA256 for r in old_retired)
            and all(
                key(r) in repaired
                and _wins(repaired[key(r)])
                and repaired[key(r)].get("reward", float("-inf")) >= r["reward"]
                for r in old_winners
            ),
            "five_known_losses_repaired_attributed": len(known_losses) == 5
            and all(k in repaired and _wins(repaired[k]) for k in known_losses)
            and len(manifest.get("known_failure_attribution", [])) == 5
            and all(
                a.get("first_causal_boundary")
                for a in manifest.get("known_failure_attribution", [])
            ),
            "all_candidate_source_hashes": bool(all_rows)
            and all(r.get("source_sha256") == sha for r in all_rows),
            "all_declared_modes": all(
                r.get("observation") == "image" and r.get("entrypoint") == "Agent"
                for r in p + v
            )
            and all(
                r.get("observation") == "gridstate" and r.get("entrypoint") == "Agent"
                for r in s
            )
            and all(r.get("observation") == "oracle" for r in o),
            "all_temporal_contracts": _temporal(all_rows),
            "all_resources": bool(all_rows)
            and all(
                resource_valid(r, image=r.get("observation") == "image")
                for r in all_rows
            ),
            "all_fresh_worker_reports": all(
                _timing(r)
                for r in (
                    public,
                    retired_report,
                    fresh_layout,
                    {"source_sha256": sha, "results": s + o},
                )
            ),
            "pre_fit_data_audit": manifest.get("data_audit", {}).get("passed") is True
            and manifest.get("data_audit", {}).get("violations") == [],
        }
    )
    artifact = manifest.get("artifact_path")
    checks["artifact_bytes_and_hash"] = (
        bool(artifact)
        and Path(artifact).is_file()
        and Path(artifact).stat().st_size == manifest.get("bytes")
        and hashlib.sha256(Path(artifact).read_bytes()).hexdigest() == sha
    )
    status = "unproven"
    untouched_checks = None
    if untouched_art is not None:
        groups = manifest.get("untouched_groups", [])
        exposure = manifest.get("exposure", {})
        provenance = (
            bool(groups)
            and all(g in exposure and exposure[g] is False for g in groups)
            and not set(groups) & set(manifest.get("training_groups", []))
        )
        independent = {
            **manifest,
            **manifest.get("untouched_corpus", {}),
            "fresh_jobs": manifest.get("untouched_jobs", []),
            "fresh_class_counts": manifest.get("untouched_class_counts", {}),
            "fresh_family_counts": manifest.get("untouched_family_counts", {}),
        }
        untouched_checks = assess_fresh(untouched_art, independent)["checks"]
        untouched_checks["untouched_group_lineage"] = provenance
        status = "passed" if all(untouched_checks.values()) else "failed"
    return {
        "regression_qualified": all(checks.values()),
        "unseen_art_status": status,
        "original_unseen_art_gate_passed": status == "passed",
        "checks": checks,
        "failures": [k for k, v in checks.items() if not v],
        "untouched_art_checks": untouched_checks,
        "corpus_contract_sha256": manifest.get("data_audit", {}).get("contract_sha256"),
        "final_manifest_sha256": manifest.get("data_audit", {}).get(
            "final_manifest_sha256"
        ),
        "source_sha256": sha,
        "model_sha256": manifest.get("model_sha256"),
        "policy_sha256": manifest.get("policy_sha256"),
        "fresh_layout_wins": fresh["wins"],
        "fresh_layout_exact_boards": fresh["exact_boards"],
        "fresh_class_recall": fresh["class_recall"],
        "public_wins": sum(_wins(r) for r in p),
        "retired_v5_wins": sum(_wins(r) for r in v),
        "retired_v5_exact_boards": sum(r.get("initial_board_exact") is True for r in v),
        "max_episode_seconds": max(
            (r.get("evaluation_seconds", 0) for r in all_rows), default=None
        ),
        "max_strict_cold_seconds": max(
            (r.get("strict_cold_seconds", 0) for r in all_rows), default=None
        ),
        "peak_rss_bytes": max(
            (r.get("peak_rss_bytes", 0) for r in all_rows), default=None
        ),
        "evidence_boundary": "Local public/synthetic evidence. Exposed public artwork fresh layouts do not prove unseen-art, Debugging, private, or graded performance.",
    }
