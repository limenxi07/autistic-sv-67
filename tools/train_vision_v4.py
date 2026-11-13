"""Bounded, from-scratch V4 grouped CNN training and out-of-fold selection.

The four-run batch has three held-group fits and one fixed-epoch production fit.
Final qualification corpora never enter any loader in this module.
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.version
from PIL import Image

from agents.vision_v4_types import PerceptionPolicy
from tools.vision_v4_data import canonical, digest, sha, visible_target
from tools.train_vision_v3 import (
    CLASSES,
    CellCNN,
    augment_training_batch,
    calibrate,
    load_crops,
    predict_logits,
    prepare_training_model,
    tensor_batch,
    train_step,
)


class GroupSampler:
    def __init__(self, labels, groups, *, seed, class_count=14):
        self.rng = np.random.default_rng(seed)
        self.indices = []
        for kind in range(class_count):
            table = defaultdict(list)
            for index in np.flatnonzero(labels == kind):
                table[str(groups[index])].append(index)
            if not table:
                raise ValueError("Missing class support in sampler")
            self.indices.append(
                [np.asarray(table[k], dtype=np.int64) for k in sorted(table)]
            )

    def draw(self, count):
        if count % len(self.indices):
            raise ValueError("Batch must be divisible by class count")
        result = []
        for buckets in self.indices:
            selected = self.rng.integers(len(buckets), size=count // len(self.indices))
            result.extend(
                buckets[g][self.rng.integers(len(buckets[g]))] for g in selected
            )
        result = np.asarray(result, dtype=np.int64)
        self.rng.shuffle(result)
        return result


def validate_config(config):
    if list(config.get("class_order", CLASSES)) != list(CLASSES):
        raise ValueError("Fixed class order required")
    if (
        config.get("checkpoint") is not None
        or config.get("initial_checkpoint") is not None
    ):
        raise ValueError("Every fit starts from scratch")
    if not 1 <= config["epochs"] <= 20:
        raise ValueError("Each run is capped at20 epochs")
    if config["fold"] not in (0, 1, 2, "production"):
        raise ValueError("Only three folds and one production fit are allowed")
    for key in ("data", "train", "validation", "train_boards", "temporal"):
        if config.get(key) and any(
            "final" in p.lower() for p in Path(config[key]).parts
        ):
            raise ValueError("Final qualification data cannot enter training/selection")


def choose_shared_epoch(run_reports):
    if len(run_reports) != 3:
        raise ValueError("Exactly three fold runs required")
    histories = [{r["epoch"]: r for r in report["history"]} for report in run_reports]
    common = set.intersection(*(set(h) for h in histories))
    if not common:
        raise ValueError("No common held-fold epoch")
    ranks = {}
    for epoch in sorted(common):
        rows = [h[epoch] for h in histories]
        tp = np.sum([r["class_true_positive"] for r in rows], axis=0)
        support = np.sum([r["class_support"] for r in rows], axis=0)
        if (support == 0).any():
            raise ValueError("Missing out-of-fold class support")
        ranks[epoch] = (
            sum(r.get("raw_exact_boards", r["exact_boards"]) for r in rows)
            / sum(r["boards"] for r in rows),
            float(np.min(tp / support)),
            -sum(r["nll_sum"] for r in rows) / sum(r["cells"] for r in rows),
            -epoch,
        )
    return max(ranks, key=lambda epoch: ranks[epoch])


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=1, keepdims=True)


def policy_probabilities(full, inset, policy):
    probs = _softmax(full / policy.temperature)
    rivals = {CLASSES.index(k) for k in ("gem", "coin", "ghost", "shield")}
    protected = {
        CLASSES.index(k)
        for k in ("floor", "wall", "box", "exit", "locked", "opened", "lava")
    }
    winners = probs.argmax(1)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    selected = np.array(
        [
            w not in protected and any(int(k) in rivals for k in pair)
            for w, pair in zip(winners, top2)
        ]
    )
    disagreements = np.zeros(len(probs), dtype=bool)
    if policy.view == "inset_005":
        inner = _softmax(inset / policy.temperature)
        disagreements = selected & (winners != inner.argmax(1))
        probs[selected] = (probs[selected] + inner[selected]) / 2
    elif policy.view != "full":
        raise ValueError("Unknown view policy")
    return probs, disagreements, int(selected.sum()) if policy.view != "full" else 0


def score_raw(full, inset, labels, board_ids, policy):
    """CNN-only metrics for raw epoch selection and NLL calibration."""
    probs, disagreements, view_cells = policy_probabilities(full, inset, policy)
    predictions = probs.argmax(1)
    correct = predictions == labels
    support = np.bincount(labels, minlength=14)
    tp = np.bincount(labels[correct], minlength=14)
    confidence = probs.max(1)
    margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    uncertain = (
        (confidence < policy.min_confidence)
        | (margins < policy.min_margin)
        | disagreements
    )
    # Negative board ids are standalone crops, not artificial one-cell boards.
    boards = np.unique(board_ids[board_ids >= 0])
    raw_exact = sum(bool(correct[board_ids == b].all()) for b in boards)
    unresolved = sum(int(uncertain[board_ids == b].sum()) > 2 for b in boards)
    return {
        "errors": int((~correct).sum()),
        "exact_boards": raw_exact,
        "raw_exact_boards": raw_exact,
        "raw_only": True,
        "boards": len(boards),
        "class_true_positive": tp.tolist(),
        "class_support": support.tolist(),
        "worst_class_recall": float(np.min(tp[support > 0] / support[support > 0])),
        "high_confidence_errors": int(
            ((~correct) & (confidence >= policy.min_confidence)).sum()
        ),
        "nll_sum": float(
            -np.log(probs[np.arange(len(labels)), labels].clip(1e-300)).sum()
        ),
        "cells": len(labels),
        "disagreement_cells": int(disagreements.sum()),
        "extra_view_cells": view_cells,
        "boards_exceeding_ambiguity_cap": unresolved,
    }


def score_policy(full, inset, labels, board_ids, policy, *, board_records=None):
    """Policy admission always runs the actual bounded runtime parser."""
    if board_records is None:
        raise ValueError("Actual board contracts required for policy scoring")
    metrics = score_raw(full, inset, labels, board_ids, policy)
    probs, disagreements, _ = policy_probabilities(full, inset, policy)
    exact, rejected = _parsed_board_accuracy(
        full, inset, probs, disagreements, board_ids, policy, board_records
    )
    metrics.update(exact_boards=exact, parser_rejected_boards=rejected, raw_only=False)
    return metrics


def _parsed_board_accuracy(
    full, inset, probabilities, disagreements, board_ids, policy, records
):
    """Use the actual bounded parser with public metadata; labels only score it."""
    from agents.vision_v4_tracking import (
        _initial_hypotheses_policy,
        fatal_uncertainty,
        state_labels,
    )
    from agents.vision_v4_types import CellScores, PerceptionEvidence

    exact = rejected = 0
    for board in np.unique(board_ids[board_ids >= 0]):
        record = records[int(board)]
        selected = np.flatnonzero(board_ids == board)
        width, height = record["width"], record["height"]
        if len(selected) != width * height:
            raise ValueError("Board geometry does not match held-fold cells")
        details = []
        for position, index in enumerate(selected):
            if disagreements[index]:
                details.append(
                    (
                        position % width,
                        position // width,
                        CLASSES[int(full[index].argmax())],
                        CLASSES[int(inset[index].argmax())],
                    )
                )
        evidence = PerceptionEvidence(
            CellScores(
                probabilities[selected].astype(np.float32).reshape(height, width, 14),
                width,
                height,
            ),
            tuple(details),
            policy.view,
        )
        hypotheses = _initial_hypotheses_policy(evidence, record["info"], policy)
        valid = bool(hypotheses) and not any(fatal_uncertainty(h) for h in hypotheses)
        rejected += int(not valid)
        exact += int(
            valid
            and canonical(state_labels(hypotheses[0].state))
            == canonical(record["labels"])
        )
    return exact, rejected


def load_board_contracts(corpora, *, production=False):
    """Public constructors recover metadata; frozen labels stay evaluator-only."""
    from tests.vision_v3_fixtures import image_env
    from tools.vision_v4_data import new_layout

    records = {}
    cache = {}
    offset = 0
    names = ["production"] if production else [f"fold-{i}" for i in range(3)]
    for name in names:
        path = Path(corpora) / name / "validation-boards/boards.jsonl"
        for board, line in enumerate(path.read_text().splitlines()):
            record = json.loads(line)
            key = (
                record["layout_index"],
                record["geometry_split"],
                record["layout_seed"],
            )
            if key not in cache:
                state, _, _, _ = new_layout(
                    record["layout_index"],
                    split=record["geometry_split"],
                    multiple_gems=True,
                )
                state.seed = record["layout_seed"]
                state.turn_limit = 150
                observation, _ = image_env(state).reset()
                cache[key] = observation["info"]
            records[offset + board] = {**record, "info": cache[key]}
        offset += board + 1
    return records


def _load_boards(root, *, allow_temporal=False):
    root = Path(root)
    if "final" in root.name:
        raise ValueError("Final qualification data excluded")
    report = json.loads((root / "report.json").read_text())
    if report["records_sha256"] != sha(root / "boards.jsonl"):
        raise ValueError("Changed board records")
    all_full = []
    all_inset = []
    labels = []
    groups = []
    board_ids = []
    metadata = []
    for b, line in enumerate((root / "boards.jsonl").read_text().splitlines()):
        record = json.loads(line)
        if record["geometry_split"] == "final" or record["appearance_split"] == "final":
            raise ValueError("Final qualification record excluded")
        if record["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("Mixed board manifests")
        path = root / record["image"]
        if sha(path) != record["image_sha256"]:
            raise ValueError("Changed board pixels")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        h, w = record["height"], record["width"]
        ih, iw = pixels.shape[:2]
        assets = {a["class"]: a["group"] for a in record["constituents"]}
        for y in range(h):
            for x in range(w):
                layers = record["labels"][y][x]
                if not allow_temporal and len(layers) != 1:
                    raise ValueError("Illegal initial overlap")
                target = visible_target(layers)
                if record["visible_labels"][y][x] != target:
                    raise ValueError("Unestablished composite target")
                crop = pixels[
                    y * ih // h : (y + 1) * ih // h, x * iw // w : (x + 1) * iw // w
                ]
                ch, cw = crop.shape[:2]
                inner = crop[
                    int(ch * 0.05) : ch - int(ch * 0.05),
                    int(cw * 0.05) : cw - int(cw * 0.05),
                ]
                all_full.append(
                    np.asarray(
                        Image.fromarray(crop).resize(
                            (48, 48), Image.Resampling.BILINEAR
                        )
                    )
                )
                all_inset.append(
                    np.asarray(
                        Image.fromarray(inner).resize(
                            (48, 48), Image.Resampling.BILINEAR
                        )
                    )
                )
                labels.append(CLASSES.index(target))
                groups.append(assets[target])
                board_ids.append(b)
        metadata.append(
            {
                "board": b,
                "image": record["image"],
                "layout_sha256": record["layout_sha256"],
                "constituents": record["constituents"],
                "turn": record["turn"],
            }
        )
    return (
        np.stack(all_full),
        np.stack(all_inset),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        np.asarray(board_ids),
        metadata,
    )


def _crop_groups(root):
    root = Path(root)
    groups = []
    for line in (root / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        groups.append(
            next(a["group"] for a in row["constituents"] if a["class"] == row["class"])
        )
    return np.asarray(groups)


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def train_fold(config: dict) -> dict:
    validate_config(config)
    root = Path(config["data"])
    output = Path(config["output"])
    # Every intended run owns a create-only directory. No checkpoint restarts or
    # hidden extra sweeps can overwrite the four-run history.
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((root / "receipt.json").read_text())
    expected = (
        f"fold-{config['fold']}" if config["fold"] != "production" else "production"
    )
    if receipt["name"] != expected or receipt["checkpoint"] is not None:
        raise ValueError("Fold corpus or initialization mismatch")
    if sha(root / "manifest.json") != receipt["train_manifest_sha256"]:
        raise ValueError("Changed run manifest")
    torch.set_num_threads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Actual verified CUDA environment required for fit")
    torch.cuda.manual_seed_all(config["seed"])
    images, labels, _ = load_crops(root / "train-crops", expected_split="train")
    groups = _crop_groups(root / "train-crops")
    initial = _load_boards(root / "train-initial")
    temporal = _load_boards(root / "train-temporal", allow_temporal=True)
    images = np.concatenate((images, initial[0], temporal[0]))
    labels = np.concatenate((labels, initial[2], temporal[2]))
    groups = np.concatenate((groups, initial[3], temporal[3]))
    del initial, temporal
    val_images, val_labels, _ = load_crops(
        root / "validation-crops", expected_split="validation"
    )
    val_groups = _crop_groups(root / "validation-crops")
    boards = _load_boards(root / "validation-boards")
    val_full = np.concatenate((val_images, boards[0]))
    val_inner = np.stack(
        [
            np.asarray(
                Image.fromarray(c[2:46, 2:46]).resize(
                    (48, 48), Image.Resampling.BILINEAR
                )
            )
            for c in val_images
        ]
    )
    val_inner = np.concatenate((val_inner, boards[1]))
    val_labels = np.concatenate((val_labels, boards[2]))
    val_groups = np.concatenate((val_groups, boards[3]))
    board_ids = np.concatenate(
        (np.full(len(val_images), -1, dtype=np.int64), boards[4])
    )
    held = set(receipt["held_groups"])
    if held and (set(groups) & held or not set(val_groups) <= held):
        raise ValueError("Held artwork group leaked across fold data")
    config = {
        **config,
        "class_order": list(CLASSES),
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "augmentation": "unchanged V3 mild shape/grayscale recipe",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "host": platform.node(),
        "python": sys.version,
        "interpreter": sys.executable,
        "source_sha256": sha(__file__),
        "corpus_receipt_sha256": sha(root / "receipt.json"),
        "train_samples": len(images),
        "validation_samples": len(val_full),
        "checkpoint": None,
    }
    _save_json(output / "config.json", config)
    _save_json(output / "validation-boards.json", boards[5])
    model = CellCNN().to(device)
    initial_hash = digest(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    )
    _save_json(
        output / "initialization.json",
        {"random_initial_weights_sha256": initial_hash, "from_scratch": True},
    )
    batch = config.get("batch", 448)
    training = model
    if config.get("compile", True):
        training, probe = prepare_training_model(
            model,
            tensor_batch(images[:batch], device),
            torch.from_numpy(labels[:batch].astype(np.int64)).to(device),
        )
        _save_json(output / "compile-probe.json", probe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    sampler = GroupSampler(labels, groups, seed=config["seed"])
    history = []
    began = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        training.train()
        losses = []
        for _ in range(math.ceil(len(images) / batch)):
            chosen = sampler.draw(batch)
            x = augment_training_batch(tensor_batch(images[chosen], device))
            y = torch.from_numpy(labels[chosen].astype(np.int64)).to(device)
            losses.append(train_step(training, optimizer, x, y))
        full = predict_logits(model, val_full, device).numpy()
        inner = predict_logits(model, val_inner, device).numpy()
        metrics = score_raw(
            full, inner, val_labels, board_ids, PerceptionPolicy(1, 0.85, 0.2, "full")
        )
        metrics.update(
            epoch=epoch,
            loss=float(np.mean(losses)),
            elapsed_seconds=perf_counter() - began,
        )
        history.append(metrics)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "seed": config["seed"],
                "epoch": epoch,
                "temperature": 1.0,
                "class_order": list(CLASSES),
            },
            output / f"epoch-{epoch:02d}.pt",
        )
        np.savez_compressed(
            output / f"logits-{epoch:02d}.npz",
            full=full,
            inset=inner,
            labels=val_labels,
            groups=val_groups,
            board_ids=board_ids,
        )
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in (
                        "epoch",
                        "loss",
                        "exact_boards",
                        "boards",
                        "worst_class_recall",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )
    result = {
        "fold": config["fold"],
        "history": history,
        "output": str(output),
        "config_sha256": sha(output / "config.json"),
        "initialization_sha256": initial_hash,
    }
    _save_json(output / "report.json", result)
    return result


def collect_oof(run_reports: list[dict]) -> dict:
    epoch = choose_shared_epoch(run_reports)
    arrays = defaultdict(list)
    offset = 0
    receipts = []
    seen_groups = set()
    for report in run_reports:
        path = Path(report["output"]) / f"logits-{epoch:02d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            current = {k: saved[k].copy() for k in saved.files}
        groups = set(current["groups"])
        if groups & seen_groups:
            raise ValueError("Artwork group crosses out-of-fold predictions")
        seen_groups.update(groups)
        boards = current["board_ids"]
        current["board_ids"] = np.where(boards >= 0, boards + offset, -1)
        offset += int(boards.max()) + 1
        for key, value in current.items():
            arrays[key].append(value)
        receipts.append(
            {"path": str(path), "sha256": sha(path), "fold": report["fold"]}
        )
    return {
        "epoch": epoch,
        "arrays": {k: np.concatenate(v) for k, v in arrays.items()},
        "receipts": receipts,
        "historical_exposure": "public artwork development, not untouched artwork",
    }


def select_policy(oof: dict, development_episodes: dict) -> dict:
    if oof.get("board_records") is None:
        raise ValueError("Actual board contracts required for policy selection")
    arrays = oof["arrays"]
    calibration = calibrate(torch.from_numpy(arrays["full"]), arrays["labels"])
    candidates: list[dict[str, Any]] = []
    calibrations: dict[str, dict[str, Any]] = {"full": calibration}
    grid = sorted(set([1.0] + np.geomspace(0.25, 8.0, 41).tolist()))
    fused_losses = []
    for temperature in grid:
        metrics = score_raw(
            arrays["full"],
            arrays["inset"],
            arrays["labels"],
            arrays["board_ids"],
            PerceptionPolicy(temperature, 0.85, 0.2, "inset_005"),
        )
        fused_losses.append(metrics["nll_sum"] / metrics["cells"])
    calibrations["inset_005"] = {
        "temperature": grid[int(np.argmin(fused_losses))],
        "nll_after": min(fused_losses),
        "grid": grid,
    }
    for view in ("full", "inset_005"):
        temperature = float(calibrations[view]["temperature"])
        for confidence in (0.85, 0.90, 0.95):
            for margin in (0.10, 0.20, 0.30):
                policy = PerceptionPolicy(temperature, confidence, margin, view)
                metrics = score_policy(
                    arrays["full"],
                    arrays["inset"],
                    arrays["labels"],
                    arrays["board_ids"],
                    policy,
                    board_records=oof["board_records"],
                )
                key = f"{view}/{confidence:.2f}/{margin:.2f}"
                # Selection is incomplete until downstream development episodes
                # exist for each visual policy in contention. No invented wins.
                downstream = development_episodes.get(key)
                candidates.append(
                    {
                        "policy": policy._asdict(),
                        "metrics": metrics,
                        "downstream": downstream,
                        "key": key,
                    }
                )

    def rank(candidate):
        m = candidate["metrics"]
        d = candidate["downstream"] or {}
        return (
            m["exact_boards"],
            m["worst_class_recall"],
            -m["high_confidence_errors"],
            d.get("wins", -1),
            -d.get("seconds", float("inf")),
            -m["boards_exceeding_ambiguity_cap"],
            -m["extra_view_cells"],
        )

    candidates.sort(key=rank, reverse=True)
    selected = candidates[0]
    visual_rank = lambda c: (
        c["metrics"]["exact_boards"],
        c["metrics"]["worst_class_recall"],
        -c["metrics"]["high_confidence_errors"],
    )
    contenders = [c for c in candidates if visual_rank(c) == visual_rank(selected)]
    return {
        "epoch": oof["epoch"],
        "calibration": calibrations[selected["policy"]["view"]],
        "policy_calibrations": calibrations,
        "policy": selected["policy"],
        "selected_key": selected["key"],
        "candidates": candidates,
        "oof_receipts": oof["receipts"],
        "contender_keys": [c["key"] for c in contenders],
        "downstream_selection_complete": all(
            c["downstream"] is not None for c in contenders
        ),
        "selection_scope": oof["historical_exposure"],
    }


def fit_production(config: dict, selection: dict) -> dict:
    if config["fold"] != "production" or config["epochs"] != selection["epoch"]:
        raise ValueError(
            "Production must fit exactly the common selected epoch from scratch"
        )
    if not selection.get("downstream_selection_complete"):
        raise ValueError(
            "downstream development selection is required before production"
        )
    contracts = load_board_contracts(Path(config["data"]).parent, production=True)
    report = train_fold(config)
    output = Path(config["output"])
    final = output / f"epoch-{selection['epoch']:02d}.pt"
    weights = torch.load(final, map_location="cpu", weights_only=True)
    weights["temperature"] = selection["policy"]["temperature"]
    torch.save(weights, output / "final.pt")
    arrays = np.load(
        output / f"logits-{selection['epoch']:02d}.npz", allow_pickle=False
    )
    policy = PerceptionPolicy(**selection["policy"])
    transfer = score_policy(
        arrays["full"],
        arrays["inset"],
        arrays["labels"],
        arrays["board_ids"],
        policy,
        board_records=contracts,
    )
    # Fixed pre-final transfer gate. Failure ends this batch; it does not select
    # another production checkpoint, calibration scalar, or extra fit.
    support = np.array(transfer["class_support"])
    tp = np.array(transfer["class_true_positive"])
    passed = (
        transfer["exact_boards"] == transfer["boards"]
        and all(support > 0)
        and float(np.min(tp / support)) >= 0.99
    )
    result = {
        "checkpoint": str(output / "final.pt"),
        "checkpoint_sha256": sha(output / "final.pt"),
        "epoch": selection["epoch"],
        "policy": selection["policy"],
        "transfer": transfer,
        "transfer_passed": bool(passed),
        "training_report": report,
    }
    _save_json(output / "production-selection.json", result)
    return result


def train_three_folds(corpora: Path, output: Path) -> dict:
    corpora, output = Path(corpora), Path(output)
    if not (corpora / "final-manifest.json").is_file():
        raise ValueError(
            "All corpora and the final contract must finish before fitting"
        )
    receipt = json.loads((corpora / "final-manifest.json").read_text())
    if receipt["audit"]["violations"]:
        raise ValueError("Corpus exposure audit failed")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for fold in range(3):
        config = {
            "fold": fold,
            "epochs": 20,
            "seed": 20261109 + fold,
            "checkpoint": None,
            "class_order": list(CLASSES),
            "data": str(corpora / f"fold-{fold}"),
            "output": str(output / f"fold-{fold}"),
            "device": "cuda",
            "batch": 448,
            "compile": True,
        }
        reports.append(train_fold(config))
    oof = collect_oof(reports)
    oof["board_records"] = load_board_contracts(corpora)
    directory = output / "oof"
    directory.mkdir(exist_ok=False)
    np.savez_compressed(directory / "pooled.npz", **oof["arrays"])
    selection = select_policy(oof, {})
    _save_json(directory / "preselection.json", selection)
    _save_json(
        output / "folds-complete.json",
        {
            "reports": reports,
            "epoch": oof["epoch"],
            "pooled_sha256": sha(directory / "pooled.npz"),
            "preselection_sha256": sha(directory / "preselection.json"),
        },
    )
    return selection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train_three_folds(args.corpora, args.output), indent=2))
