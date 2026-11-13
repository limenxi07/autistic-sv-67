"""Describe saved paired ablations without selecting new model/policy choices."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def run():
    root = Path("artifacts/v4/validation/development/matrix")
    summary = read(root / "matrix.json")
    reports = {name: read(root / (name + ".json"))
               for name in summary["candidate_sources"]}
    bycase = {name: {(r["label"], r["seed"]): r for r in report["results"]}
              for name, report in reports.items()}
    comparisons = {}
    for label, before, after in (
        ("backend_only", "historical_v3", "old_perception_old_tracking"),
        ("perception_with_old_tracking", "old_perception_old_tracking", "new_perception_old_tracking"),
        ("tracking_with_old_perception", "old_perception_old_tracking", "old_perception_new_tracking"),
        ("tracking_with_new_perception", "new_perception_old_tracking", "new_perception_new_tracking"),
        ("complete_v4_vs_v3", "historical_v3", "new_perception_new_tracking"),
    ):
        changes = []
        for key, old in bycase[before].items():
            new = bycase[after][key]
            changes.append({"label": key[0], "seed": key[1],
                            "before_status": old["status"], "after_status": new["status"],
                            "before_reward": old.get("reward"), "after_reward": new.get("reward"),
                            "reward_change": new["reward"] - old["reward"],
                            "before_exact": old["initial_board_exact"], "after_exact": new["initial_board_exact"]})
        comparisons[label] = {"before_source": summary["candidate_sources"][before],
                              "after_source": summary["candidate_sources"][after], "cases": changes}
    maxima = {}
    for name, report in reports.items():
        rows = report["results"]
        maxima[name] = {"episode_seconds": max(r["evaluation_seconds"] for r in rows),
                        "strict_cold_seconds": max(r["strict_cold_seconds"] for r in rows),
                        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rows)}
    result = {"matrix": summary, "comparisons": comparisons, "resources": maxima,
              "rationale": "P3 alone and recovery with old perception each remain8/13. New perception with old tracking reaches12/13 and13/13 exact boards. The complete fixed-policy V4 reaches13/13 wins/exact by retaining plans under unchanged ambiguity and independently resolving uncertain actors. Model/policy were fixed by fold selection before these episode outcomes; tracking repairs use retired development data only.",
              "diagnostic_stages": {"target_host": ["artifacts/v4/diagnostics/v3", "artifacts/v4/diagnostics/p3-control", "artifacts/v4/diagnostics/recovery-only"],
                                    "local_repair_traces": ["artifacts/v4/diagnostics/new-new-remaining-local", "artifacts/v4/diagnostics/ambiguity-repair-local", "artifacts/v4/diagnostics/actor-template-repair-local"],
                                    "qualification": False, "limitations": "Trace stages overlap through nested grid/planner calls. V4 first-frame classify_evidence time is in total elapsed but absent from named stages. Local Mac traces are causal development diagnostics, not Omarchy timing qualification."},
              "generated_optimality": "unknown", "unseen_art_status": "unproven"}
    output = Path("artifacts/v4/validation/attribution.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["rationale"])


if __name__ == "__main__":
    run()
