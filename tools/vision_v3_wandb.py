"""Development-only W&B tracking; never imported by the submitted runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def epoch_metrics(row, class_order):
    metrics = {"epoch": row["epoch"]}
    names = {
        "loss": "train/loss",
        "learning_rate": "train/learning_rate",
        "samples": "train/samples_seen",
        "global_step": "train/global_step",
        "elapsed_seconds": "perf/elapsed_seconds",
        "samples_per_second_including_validation": "perf/samples_per_second_including_validation",
        "peak_vram_bytes": "resource/gpu_peak_tensor_bytes",
        "peak_cpu_rss_bytes": "resource/cpu_peak_rss_bytes",
        "cell_accuracy": "validation/cell_accuracy",
        "macro_recall": "validation/macro_recall",
        "validation_nll": "validation/nll",
    }
    for source, destination in names.items():
        if row.get(source) is not None:
            metrics[destination] = row[source]
    confusion = row.get("confusion")
    if confusion is not None:
        for i, name in enumerate(class_order):
            support = sum(confusion[i])
            predicted = sum(r[i] for r in confusion)
            tp = confusion[i][i]
            precision, recall = tp / max(1, predicted), tp / max(1, support)
            metrics[f"validation/precision/{name}"] = precision
            metrics[f"validation/recall/{name}"] = recall
            metrics[f"validation/f1/{name}"] = (
                2 * precision * recall / max(1e-12, precision + recall)
            )
            metrics[f"validation/support/{name}"] = support
    for axis, result in row.get("axes", {}).items():
        for name, value in result.items():
            metrics[f"validation/{axis}/{name}"] = value
    return metrics


class TrainingLogger:
    def __init__(
        self,
        config,
        output,
        *,
        project="autistic-sv-67",
        entity=None,
        mode="online",
        training_pid=None,
        backfill=False,
    ):
        import wandb  # ty: ignore[unresolved-import] -- optional SDK verified in isolated CUDA environment

        self.wandb, self.output = wandb, Path(output)
        self.class_order = tuple(config["class_order"])
        self.last_row = None
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=self.output.name,
            config={**config, "logging_backfill": backfill},
            tags=[
                "v3",
                "cnn",
                "torch-compile" if config.get("compile") else "eager",
                "pilot" if config.get("pilot_seconds") else "full-training",
            ],
            job_type="pilot" if config.get("pilot_seconds") else "train",
            mode=mode,
            dir=str(self.output),
            settings=wandb.Settings(x_stats_pid=training_pid or os.getpid()),
        )
        self.run.define_metric("epoch")
        self.run.define_metric("train/global_step")
        self.run.define_metric("validation/*", step_metric="epoch")
        self.run.define_metric("train/loss_step", step_metric="train/global_step")
        self.run.define_metric("train/grad_norm", step_metric="train/global_step")
        self.run.define_metric("train/loss", step_metric="epoch")
        self.run.define_metric(
            "perf/train_step_seconds", step_metric="train/global_step"
        )
        identity = {
            "id": self.run.id,
            "url": self.run.url,
            "project": project,
            "entity": self.run.entity,
            "mode": mode,
            "backfill": backfill,
        }
        (self.output / "wandb-run.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        inputs = wandb.Artifact(
            "v3-input-manifests",
            type="dataset-manifest",
            metadata={"manifest_sha256": config.get("manifest_sha256")},
        )
        added = False
        for key in ("data", "validation", "train_boards"):
            if not config.get(key):
                continue
            directory = Path(config[key])
            for filename in ("report.json", "audit.json"):
                if (directory / filename).is_file():
                    inputs.add_file(str(directory / filename), name=f"{key}/{filename}")
                    added = True
        if added:
            self.run.log_artifact(inputs)

    def log_epoch(self, row):
        self.last_row = row
        self.run.log(epoch_metrics(row, self.class_order))

    def log_step(self, step, loss, grad_norm, learning_rate, step_seconds):
        self.run.log(
            {
                "train/global_step": step,
                "train/loss_step": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": learning_rate,
                "perf/train_step_seconds": step_seconds,
            }
        )

    def log_compile(self, report):
        self.run.log(
            {
                "epoch": 0,
                **{
                    f"compile/{key}": value
                    for key, value in report.items()
                    if isinstance(value, (int, float, bool))
                },
            }
        )

    def finish(self):
        if self.last_row and "confusion" in self.last_row:
            truth, prediction = [], []
            for i, row in enumerate(self.last_row["confusion"]):
                for j, count in enumerate(row):
                    truth.extend([i] * count)
                    prediction.extend([j] * count)
            self.run.log(
                {
                    "epoch": self.last_row["epoch"],
                    "validation/confusion_matrix": self.wandb.plot.confusion_matrix(
                        y_true=truth,
                        preds=prediction,
                        class_names=list(self.class_order),
                    ),
                }
            )
        for filename in ("summary.json", "validation-best.json", "compile-probe.json"):
            path = self.output / filename
            if path.exists():
                values = json.loads(path.read_text())
                for key, value in values.items():
                    if isinstance(value, (int, float, bool, str)):
                        self.run.summary[
                            f"final/{filename.removesuffix('.json')}/{key}"
                        ] = value
                for key, value in values.get("calibration", {}).items():
                    if isinstance(value, (int, float)):
                        self.run.summary[f"calibration/{key}"] = value
        checkpoint = self.output / "best.pt"
        if checkpoint.exists():
            artifact = self.wandb.Artifact(
                f"{self.output.name}-checkpoint",
                type="model",
                metadata={
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "class_order": list(self.class_order),
                },
            )
            artifact.add_file(str(checkpoint))
            for filename in (
                "config.json",
                "validation-best.json",
                "compile-probe.json",
                "summary.json",
            ):
                if (self.output / filename).exists():
                    artifact.add_file(str(self.output / filename))
            self.run.log_artifact(artifact, aliases=["best"])
        self.run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="autistic-sv-67")
    parser.add_argument("--entity")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--exit-file", type=Path)
    parser.add_argument("--training-pid", type=int)
    args = parser.parse_args()
    config = json.loads((args.output / "config.json").read_text())
    logger = TrainingLogger(
        config,
        args.output,
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        training_pid=args.training_pid,
        backfill=True,
    )
    if (args.output / "compile-probe.json").exists():
        logger.log_compile(json.loads((args.output / "compile-probe.json").read_text()))
    read_rows = 0
    while True:
        lines = (args.output / "history.jsonl").read_text().splitlines()
        for line in lines[read_rows:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # A writer may be in the middle of its last line.
            logger.log_epoch(row)
            read_rows += 1
        if not args.follow or args.exit_file is not None and args.exit_file.exists():
            break
        time.sleep(3)
    logger.finish()


if __name__ == "__main__":
    main()
