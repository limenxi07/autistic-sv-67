"""Small supervised V3 cell CNN. Deployment remains independent of this module."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.version
from PIL import Image
from torch import nn

CLASSES = (
    "boots",
    "box",
    "coin",
    "exit",
    "floor",
    "gem",
    "ghost",
    "human",
    "key",
    "lava",
    "locked",
    "opened",
    "shield",
    "wall",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CellCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 3
        for outgoing, stride in ((16, 1), (32, 2), (64, 2), (96, 2)):
            layers.extend(
                (nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1), nn.ReLU())
            )
            incoming = outgoing
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((2, 2)))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 14)
        )

    def forward(self, crops):
        return self.head(self.features(crops))


def augment_training_batch(images, *, grayscale_probability=0.2):
    """Preserve object geometry while reducing observed color/texture shortcuts.

    Reflection outside the cell, <=12% shrinking and ~one-pixel translation
    preserve foreground outlines. No hue swaps, flips or severe crops.
    """
    count = len(images)
    device, dtype = images.device, images.dtype
    theta = torch.zeros(count, 2, 3, device=device, dtype=dtype)
    scales = 1 + torch.rand(count, device=device) * 0.12
    theta[:, 0, 0] = scales
    theta[:, 1, 1] = scales
    theta[:, :, 2] = (torch.rand(count, 2, device=device) - 0.5) * 0.08
    grid = nn.functional.affine_grid(theta, images.shape, align_corners=False)
    values = nn.functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )
    gray = (values * values.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(
        1, keepdim=True
    )
    saturation = 0.6 + torch.rand(count, 1, 1, 1, device=device) * 0.8
    saturation[torch.rand(count, device=device) < grayscale_probability] = 0
    values = gray + saturation * (values - gray)
    contrast = 0.85 + torch.rand(count, 1, 1, 1, device=device) * 0.3
    mean = values.mean((1, 2, 3), keepdim=True)
    values = mean + contrast * (values - mean)
    brightness = 0.85 + torch.rand(count, 1, 1, 1, device=device) * 0.3
    return (values * brightness).clamp(0, 1)


def train_step(model, optimizer, images, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def prepare_training_model(
    model, images, labels, *, compile_mode="reduce-overhead", benchmark_steps=100
):
    """Measure eager/compiled updates and restore exact initial model weights.

    Keep the base module for validation and checkpoints. The optimized wrapper
    is used only for training; no _orig_mod prefixes enter deployment weights.
    """
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def synchronize():
        if images.is_cuda:
            torch.cuda.synchronize()

    def measure(candidate, steps, warmup=0):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        for _ in range(warmup):
            train_step(candidate, optimizer, images, labels)
        synchronize()
        began = perf_counter()
        for _ in range(steps):
            train_step(candidate, optimizer, images, labels)
        synchronize()
        return perf_counter() - began

    eager_seconds = measure(model, benchmark_steps, warmup=10)
    model.load_state_dict(initial)
    compiled = torch.compile(model, mode=compile_mode, dynamic=False)
    first_step_seconds = measure(compiled, 1)
    compiled_seconds = measure(compiled, benchmark_steps, warmup=10)
    model.load_state_dict(initial)
    for parameter in model.parameters():
        parameter.grad = None
    report = {
        "mode": compile_mode,
        "benchmark_steps": benchmark_steps,
        "batch": len(images),
        "first_compiled_step_seconds": first_step_seconds,
        "eager_step_seconds": eager_seconds / benchmark_steps,
        "compiled_step_seconds": compiled_seconds / benchmark_steps,
        "steady_speedup": eager_seconds / compiled_seconds,
        "initial_weights_restored": True,
    }
    return compiled, report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_crops(root, *, expected_split):
    root = Path(root)
    if (
        expected_split not in ("train", "validation")
        or not (root / "audit.json").is_file()
    ):
        raise ValueError("Training accepts only audited train/validation data")
    report = json.loads((root / "report.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    if (
        report["split"] != expected_split
        or audit["data_report_sha256"] != sha(root / "report.json")
        or audit["provenance_violations"] != 0
        or audit["verified_crops"] != report["crops"]
    ):
        raise ValueError("Dataset audit/split mismatch")
    images, labels = [], []
    for shard in report["shards"]:
        path = root / shard["path"]
        if sha(path) != shard["sha256"]:
            raise ValueError("Dataset shard hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["images"].shape != (shard["samples"], 48, 48, 3):
                raise ValueError("Dataset image shape mismatch")
            images.append(arrays["images"])
            labels.append(arrays["labels"])
    images, labels = np.concatenate(images), np.concatenate(labels)
    if (
        len(images) != report["crops"]
        or len(labels) != len(images)
        or labels.min() < 0
        or labels.max() >= 14
    ):
        raise ValueError("Dataset labels/count mismatch")
    return images, labels, report


def load_boards(root, *, geometry_split, appearance_split, manifest_sha256):
    root = Path(root)
    if "final" in (geometry_split, appearance_split):
        raise ValueError("Final holdout is excluded from model development")
    report = json.loads((root / "report.json").read_text())
    if (
        report["records_sha256"] != sha(root / "boards.jsonl")
        or report["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("Board manifest hash mismatch")
    boards = []
    for line in (root / "boards.jsonl").read_text().splitlines():
        record = json.loads(line)
        if (
            record["geometry_split"] != geometry_split
            or record["appearance_split"] != appearance_split
        ):
            raise ValueError("Board development split mismatch")
        if record["manifest_sha256"] != manifest_sha256 or any(
            a["split"] != appearance_split for a in record["constituents"]
        ):
            raise ValueError("Board constituent/manifest mismatch")
        path = root / record["image"]
        if sha(path) != record["image_sha256"]:
            raise ValueError("Board image bytes changed")
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        h, w = record["height"], record["width"]
        ih, iw = image.shape[:2]
        crops, labels = [], []
        for y in range(h):
            for x in range(w):
                layers = record["labels"][y][x]
                if len(layers) != 1:
                    raise ValueError("Initial training board has overlaps")
                crop = image[
                    y * ih // h : (y + 1) * ih // h, x * iw // w : (x + 1) * iw // w
                ]
                crops.append(
                    np.asarray(
                        Image.fromarray(crop).resize(
                            (48, 48), Image.Resampling.BILINEAR
                        )
                    )
                )
                labels.append(CLASSES.index(layers[0]))
        boards.append(
            (np.stack(crops), np.asarray(labels, dtype=np.int64), record["image"])
        )
    return boards


def tensor_batch(images, device):
    return (
        torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(
            device=device, dtype=torch.float32
        )
        / 255
    )


def predict_logits(model, images, device, batch=448):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            outputs.append(
                model(tensor_batch(images[start : start + batch], device)).cpu()
            )
    return torch.cat(outputs)


def validation_metrics(model, images, labels, boards, device):
    logits = predict_logits(model, images, device)
    predictions = logits.argmax(1).numpy()
    confusion = np.zeros((14, 14), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    recalls = np.diag(confusion) / np.maximum(1, confusion.sum(1))
    axes = {}
    for axis, items in boards.items():
        exact, cells, correct = 0, 0, 0
        for crops, truth, _ in items:
            guess = predict_logits(model, crops, device).argmax(1).numpy()
            exact += int(np.array_equal(guess, truth))
            cells += len(truth)
            correct += int((guess == truth).sum())
        axes[axis] = {
            "boards": len(items),
            "exact_boards": exact,
            "exact_board_accuracy": exact / max(1, len(items)),
            "cell_accuracy": correct / max(1, cells),
        }
    return {
        "cell_accuracy": float((predictions == labels).mean()),
        "macro_recall": float(recalls.mean()),
        "class_recall": dict(zip(CLASSES, recalls.tolist())),
        "confusion": confusion.tolist(),
        "axes": axes,
        "validation_nll": float(
            nn.functional.cross_entropy(logits, torch.from_numpy(labels).long())
        ),
    }, logits


def calibrate(logits, labels):
    """Select positive scalar temperature on validation NLL, retaining rankings."""
    logits = logits.detach().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    temperatures = sorted(set([1.0] + np.geomspace(0.25, 8.0, 41).tolist()))
    with torch.inference_mode():
        losses = [
            float(nn.functional.cross_entropy(logits / temperature, labels))
            for temperature in temperatures
        ]
        temperature = temperatures[int(np.argmin(losses))]
        probabilities = torch.softmax(logits / temperature, dim=1)
        confidence, prediction = probabilities.max(1)
        correct = prediction == labels
        ece = 0.0
        for lower in torch.arange(15) / 15:
            mask = (confidence > lower) & (confidence <= lower + 1 / 15)
            if mask.any():
                ece += float(
                    mask.float().mean()
                    * (confidence[mask].mean() - correct[mask].float().mean()).abs()
                )
    return {
        "temperature": temperature,
        "nll_before": float(nn.functional.cross_entropy(logits, labels)),
        "nll_after": min(losses),
        "expected_calibration_error_15_bins": ece,
        "selection": "validation-only scalar temperature grid, includes 1.0",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-boards", type=Path)
    parser.add_argument("--boards", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--batch", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot-seconds", type=float, default=0)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--wandb-project", default="autistic-sv-67")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    args = parser.parse_args()
    if args.batch % 14 or args.batch < 14:
        parser.error("Batch size must be a multiple of 14")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires a verified CUDA device")
        torch.cuda.manual_seed_all(args.seed)
    images, labels, data_report = load_crops(args.data, expected_split="train")
    validation_images, validation_labels, val_report = load_crops(
        args.validation, expected_split="validation"
    )
    manifest_hash = data_report["manifest_sha256"]
    if val_report["manifest_sha256"] != manifest_hash:
        raise ValueError("Train/validation manifest mismatch")
    if args.limit_train:
        images, labels = images[: args.limit_train], labels[: args.limit_train]
    if args.train_boards:
        extra = load_boards(
            args.train_boards,
            geometry_split="train",
            appearance_split="train",
            manifest_sha256=manifest_hash,
        )
        images = np.concatenate([images] + [b[0] for b in extra])
        labels = np.concatenate([labels] + [b[1] for b in extra])
    boards = {}
    for path in args.boards:
        meta = json.loads((path / "report.json").read_text())
        boards[path.name] = load_boards(
            path,
            geometry_split=meta["geometry_split"],
            appearance_split=meta["appearance_split"],
            manifest_sha256=manifest_hash,
        )
    config = {
        key: str(value)
        if isinstance(value, Path)
        else [str(p) for p in value]
        if key == "boards"
        else value
        for key, value in vars(args).items()
    }
    config.update(
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.0001,
        class_order=list(CLASSES),
        manifest_sha256=manifest_hash,
        train_report_sha256=sha(args.data / "report.json"),
        validation_report_sha256=sha(args.validation / "report.json"),
        source_sha256=sha(__file__),
        host=platform.node(),
        python=sys.version,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        train_samples=len(images),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger = None
    if args.wandb_mode != "disabled":
        from tools.vision_v3_wandb import TrainingLogger

        logger = TrainingLogger(
            config,
            args.output,
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
        )
    model = CellCNN().to(device)
    training_model = model
    if args.compile:
        probe_images = tensor_batch(images[: args.batch], device)
        probe_labels = torch.from_numpy(labels[: args.batch].astype(np.int64)).to(
            device
        )
        training_model, compile_report = prepare_training_model(
            model, probe_images, probe_labels, compile_mode=args.compile_mode
        )
        (args.output / "compile-probe.json").write_text(
            json.dumps(compile_report, indent=2) + "\n"
        )
        print(json.dumps({"compile_probe": compile_report}), flush=True)
        if logger:
            logger.log_compile(compile_report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    class_indices = [np.flatnonzero(labels == i) for i in range(14)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Training data omits a class")
    rng = np.random.default_rng(args.seed)
    best = (-1.0, -1.0)
    updates = samples = 0
    training_started = perf_counter()
    epoch = 0
    with (args.output / "history.jsonl").open("x") as history:
        while args.pilot_seconds or epoch < args.epochs:
            epoch += 1
            losses = []
            for _ in range(math.ceil(len(images) / args.batch)):
                step_started = perf_counter()
                chosen = np.concatenate(
                    [
                        rng.choice(indices, args.batch // 14, replace=True)
                        for indices in class_indices
                    ]
                )
                rng.shuffle(chosen)
                batch_images = tensor_batch(images[chosen], device)
                if args.augment:
                    batch_images = augment_training_batch(batch_images)
                batch_labels = torch.from_numpy(labels[chosen].astype(np.int64)).to(
                    device
                )
                losses.append(
                    train_step(training_model, optimizer, batch_images, batch_labels)
                )
                samples += len(chosen)
                updates += 1
                if logger and updates % 100 == 0:
                    grad_norm = float(
                        torch.linalg.vector_norm(
                            torch.stack(
                                [
                                    p.grad.detach().float().norm()
                                    for p in model.parameters()
                                    if p.grad is not None
                                ]
                            )
                        )
                    )
                    logger.log_step(
                        updates,
                        losses[-1],
                        grad_norm,
                        optimizer.param_groups[0]["lr"],
                        perf_counter() - step_started,
                    )
                if (
                    args.pilot_seconds
                    and perf_counter() - training_started >= args.pilot_seconds
                ):
                    break
            metrics, _ = validation_metrics(
                model, validation_images, validation_labels, boards, device
            )
            board_accuracy = (
                float(
                    np.mean(
                        [a["exact_board_accuracy"] for a in metrics["axes"].values()]
                    )
                )
                if boards
                else 0.0
            )
            rank = (board_accuracy, metrics["macro_recall"])
            elapsed = perf_counter() - training_started
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "elapsed_seconds": elapsed,
                "samples": samples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": updates,
                "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "samples_per_second_including_validation": samples / elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else None,
                **metrics,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            if logger:
                logger.log_epoch(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "epoch",
                            "loss",
                            "elapsed_seconds",
                            "cell_accuracy",
                            "macro_recall",
                            "samples_per_second_including_validation",
                            "peak_vram_bytes",
                        )
                    }
                ),
                flush=True,
            )
            if rank > best:
                best = rank
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "seed": args.seed,
                        "epoch": epoch,
                        "temperature": 1.0,
                        "config": config,
                        "metrics": metrics,
                    },
                    args.output / "best.pt",
                )
            if args.pilot_seconds and elapsed >= args.pilot_seconds:
                break
    best_checkpoint = torch.load(
        args.output / "best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    best_metrics, best_logits = validation_metrics(
        model, validation_images, validation_labels, boards, device
    )
    calibration = calibrate(best_logits, validation_labels)
    best_checkpoint["temperature"] = calibration["temperature"]
    best_checkpoint["calibration"] = calibration
    torch.save(best_checkpoint, args.output / "best.pt")
    (args.output / "validation-best.json").write_text(
        json.dumps({**best_metrics, "calibration": calibration}, indent=2) + "\n"
    )
    summary = {
        "elapsed_seconds": perf_counter() - started,
        "setup_seconds": training_started - started,
        "training_seconds": perf_counter() - training_started,
        "samples": samples,
        "updates": updates,
        "epochs": epoch,
        "checkpoint_sha256": sha(args.output / "best.pt"),
        "best_rank": best,
        "pilot": bool(args.pilot_seconds),
        "config": config,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
