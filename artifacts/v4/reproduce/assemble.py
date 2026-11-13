"""Reassemble saved evidence only; never rerun or unlock a final corpus."""

import json
from pathlib import Path

from tools.qualify_v4 import qualify
from tools.v4_development import regression_bundle
from tools.v4_final import group_recalls


def read(path):
    return json.loads(Path(path).read_text())


def run():
    development = Path("artifacts/v4/validation/development")
    final = Path("artifacts/v4/qualification/final-single-attempt")
    if (final / "qualification.json").is_file():
        manifest = read(development / "acceptance-manifest.json")
        fresh = read(final / "fresh-300.json")
        result = qualify(read(development / "public-66.json"),
                         regression_bundle(development), fresh, manifest)
        result.update(group_recalls(fresh["results"], manifest))
        saved = read(final / "qualification.json")
        if any(saved.get(k) != v for k, v in result.items()):
            raise ValueError("Saved qualification differs from reassembled evidence")
        passed = result["regression_qualified"]
    elif (development / "gates.json").is_file():
        result = {"stage": "development", "gates": read(development / "gates.json"),
                  "final_outcomes_unlocked": False,
                  "regression_qualified": False, "unseen_art_status": "unproven"}
        if (final / "unlock.json").exists():
            raise ValueError("An incomplete unlocked final requires explicit diagnosis")
        passed = False
    else:
        transfer = read("artifacts/v4/validation/transfer-admissible.json")
        result = {"stage": "production_transfer", "transfer": transfer,
                  "regression_qualified": False, "unseen_art_status": "unproven",
                  "final_outcomes_unlocked": False}
        passed = False
    output = Path("artifacts/v4/qualification/reassembled.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_class_artwork_group", "transfer")}, indent=2))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
