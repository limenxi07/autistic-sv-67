"""Audit V4 data and check overlap with historical regression layouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from agents.vision_v4_bridge import CLASSES
from tools.benchmark_vision_v3 import oracle_labels
from tools.vision_v4_data import canonical, digest, sha, visible_target


def audit_layout_contract(contract, historical):
    sets = {k: {r["sha256"] for r in rows} for k, rows in contract["layouts"].items()}
    final = sets["final"]
    if final & set(historical):
        raise ValueError("Final geometry overlaps historical regression/training")
    for key, values in sets.items():
        if key != "final" and final & values:
            raise ValueError("Final geometry overlaps V4 development")
    if len(set.union(*(v for k, v in sets.items() if k != "final"))) != sum(
        len(v) for k, v in sets.items() if k != "final"
    ):
        raise ValueError("V4 development geometry split overlap")
    return {
        "violations": [],
        "final_layouts": len(final),
        "historical_layouts": len(set(historical)),
    }


def historical_layouts():
    from public_levels.catalog import (
        PUBLIC_BOSS_LEVEL_BUILDERS,
        PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
    )
    from tests.boss_p2_next_fixtures import boss_like
    from tests.compact_p2_fixtures import heldout_case
    from tests.hierarchical_p3_fixtures import holdout_case

    states = [
        b()
        for b in list(PUBLIC_GAMEPLAY_LEVEL_BUILDERS.values())
        + list(PUBLIC_BOSS_LEVEL_BUILDERS.values())
    ]
    states.extend(heldout_case(i)[0] for i in range(200))
    states.extend(boss_like(i, split="heldout")[0] for i in range(36))
    states.extend(holdout_case(i)[0] for i in range(18))
    hashes = {
        digest(
            canonical(
                {"width": s.width, "height": s.height, "labels": oracle_labels(s)}
            )
        )
        for s in states
    }
    for path in Path("artifacts/v3/data").glob("manifest-v*.json"):
        hashes.update(r["sha256"] for r in json.loads(path.read_text())["layouts"])
    return hashes


def audit(corpora):
    corpora = Path(corpora)
    historical = historical_layouts()
    contract = json.loads((corpora / "contract.json").read_text())
    layout_audit = audit_layout_contract(contract, historical)
    files = {}
    boards = []
    for directory in sorted(p for p in corpora.iterdir() if p.is_dir()):
        if directory.name == "final-300":
            board_dirs = [directory]
        elif directory.name.startswith("fold-") or directory.name == "production":
            receipt = json.loads((directory / "receipt.json").read_text())
            if (
                receipt["initial"]["boards"] != 600
                or not 100 <= receipt["reachable"]["layout_count"]
                or receipt["reachable"]["boards"] > 600
            ):
                raise ValueError("Run corpus bounds violated")
            for name in ("train-crops", "validation-crops"):
                root = directory / name
                report = json.loads((root / "report.json").read_text())
                a = json.loads((root / "audit.json").read_text())
                if a["provenance_violations"] or a["data_report_sha256"] != sha(
                    root / "report.json"
                ):
                    raise ValueError("Crop audit not bound to data")
                if name == "train-crops" and report["crops"] != 200000:
                    raise ValueError("Wrong crop training count")
                for shard in report["shards"]:
                    path = root / shard["path"]
                    if sha(path) != shard["sha256"]:
                        raise ValueError("Changed crop shard")
                    files[str(path)] = {
                        "sha256": shard["sha256"],
                        "bytes": path.stat().st_size,
                    }
            board_dirs = [
                directory / name
                for name in ("train-initial", "train-temporal", "validation-boards")
            ]
        else:
            continue
        for root in board_dirs:
            report = json.loads((root / "report.json").read_text())
            if report["records_sha256"] != sha(root / "boards.jsonl"):
                raise ValueError("Changed board record set")
            records = [
                json.loads(line)
                for line in (root / "boards.jsonl").read_text().splitlines()
            ]
            for record in records:
                image = root / record["image"]
                if sha(image) != record["image_sha256"]:
                    raise ValueError("Changed board image")
                with Image.open(image) as opened:
                    pixels = np.asarray(opened)
                if digest(pixels.tobytes()) != record["pixels_sha256"]:
                    raise ValueError("Decoded board pixel mismatch")
                if [
                    [visible_target(c) for c in row] for row in record["labels"]
                ] != record["visible_labels"]:
                    raise ValueError("Wrong visible composite target")
                if not report.get("temporal") and any(
                    len(c) != 1 for row in record["labels"] for c in row
                ):
                    raise ValueError("Illegal initial overlap")
                files[str(image)] = {
                    "sha256": record["image_sha256"],
                    "bytes": image.stat().st_size,
                }
            if root.name == "final-300":
                counts = Counter(
                    k for r in records for row in r["visible_labels"] for k in row
                )
                families = Counter(r["family"] for r in records)
                if (
                    len(records) != 300
                    or len(families) != 10
                    or set(families.values()) != {30}
                ):
                    raise ValueError("Final family/count coverage failed")
                if any(
                    counts[k] < 100 for k in ("gem", "coin", "ghost", "box", "wall")
                ) or any(counts[k] == 0 for k in CLASSES):
                    raise ValueError("Final critical class coverage failed")
                layout_audit.update(
                    class_counts=dict(counts), family_counts=dict(families)
                )
            boards.append(
                {
                    "path": str(root),
                    "boards": len(records),
                    "report_sha256": sha(root / "report.json"),
                }
            )
    result = {
        "passed": True,
        "violations": [],
        "layout_audit": layout_audit,
        "boards": boards,
        "files": files,
        "source_sha256": sha(__file__),
        "contract_sha256": sha(corpora / "contract.json"),
        "final_manifest_sha256": sha(corpora / "final-manifest.json"),
        "final_outcomes_accessed": False,
    }
    (corpora / "audit-before-fit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.corpora)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
