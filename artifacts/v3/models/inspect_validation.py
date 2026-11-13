"""Reproduce the pilot's validation errors by their public artwork constituent."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.train_v3 import CLASSES, CellCNN, load_crops, predict_logits

torch.set_num_threads(2)
root = ROOT / "artifacts/v3"
checkpoint = torch.load(root / "models/pilot-validation-snapshot.pt", map_location="cpu", weights_only=True)
model = CellCNN().eval()
model.load_state_dict(checkpoint["state_dict"])
images, labels, _ = load_crops(root / "data/validation-14k-v2", expected_split="validation")
logits = predict_logits(model, images, "cpu")
probabilities = torch.softmax(logits, 1).numpy()
prediction = probabilities.argmax(1)
records = [json.loads(line) for line in (root / "data/validation-14k-v2/samples.jsonl").read_text().splitlines()]
bad = np.flatnonzero(prediction != labels)
sources = Counter()
examples = []
for index in bad:
    row = records[int(index)]
    source = next(a["path"] for a in row["constituents"] if a["class"] == row["class"])
    sources[(source, CLASSES[int(prediction[index])])] += 1
    if len(examples) < 56:
        examples.append(int(index))
canvas = Image.new("RGB", (7 * 112, 8 * 130), "white")
draw = ImageDraw.Draw(canvas)
for i, index in enumerate(examples):
    x, y = i % 7 * 112, i // 7 * 130
    draw.text((x, y), f"{CLASSES[int(labels[index])]}>{CLASSES[int(prediction[index])]}", fill="black")
    draw.text((x, y + 13), f"{probabilities[index].max():.3f}", fill="black")
    canvas.paste(Image.fromarray(images[index]).resize((96, 96)), (x, y + 28))
canvas.save(root / "models/pilot-validation-errors.png")
report = {"epoch": checkpoint["epoch"], "errors": len(bad), "total": len(labels),
          "source_confusions": [{"asset": k[0], "predicted": k[1], "count": v} for k, v in sources.most_common()],
          "evidence_boundary": "Pilot snapshot, validation artwork only; no final outcome evaluated."}
(root / "models/pilot-validation-errors.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
