"""Reliability-Aware Registration-Free Fusion V1.

This is an isolated experiment. It trains only a small reliability MLP on
frozen cached expert outputs and never modifies prior experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
SOURCE_CACHE = ROOT / "cache_features"
CURRENT_CV = ROOT / "experiments" / "scene_cv_6fold"
OUTPUT_ROOT = ROOT / "experiments" / "reliability_fusion_v1"
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
QUALITY_TEMPERATURE = 0.10
LAMBDA_VALUES = (0.0, 0.05, 0.10, 0.20)
SEED = 42

SCENES = [
    "GM01", "GM02", "GM03", "GM04", "GM05", "GM06",
    "GM07", "GM08", "GM09", "GM10", "GM11", "GM12",
    "GM13", "GM14", "GM15", "GM16", "GM17", "GM18",
]
SCENES_BY_SOURCE = {
    "train": ["GM01", "GM02", "GM06", "GM07", "GM08", "GM12", "GM13", "GM14", "GM18"],
    "validation": ["GM03", "GM09", "GM15"],
    "test": ["GM04", "GM05", "GM10", "GM11", "GM16", "GM17"],
}


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_source_caches() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for split in ("train", "validation", "test"):
        with np.load(SOURCE_CACHE / f"{split}_features.npz", allow_pickle=False) as data:
            result[split] = {key: np.asarray(data[key]) for key in data.files}
    return result


def split_by_scene(caches: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for source_split, scene_names in SCENES_BY_SOURCE.items():
        data = caches[source_split]
        for index, scene_name in enumerate(scene_names):
            mask = data["scene_indices"] == index
            result[scene_name] = {
                key: value[mask].copy()
                for key, value in data.items()
            }
    return result


def merge_scenes(scene_data: dict[str, dict[str, np.ndarray]], names: list[str]) -> dict[str, np.ndarray]:
    merged = {
        key: np.concatenate([scene_data[name][key] for name in names], axis=0)
        for key in scene_data[names[0]]
    }
    merged["scene_indices"] = np.concatenate(
        [
            np.full(
                len(scene_data[name]["labels"]),
                index,
                dtype=np.int64,
            )
            for index, name in enumerate(names)
        ],
        axis=0,
    )
    return merged


def make_folds() -> list[dict[str, Any]]:
    folds = []
    for fold in range(6):
        test = [SCENES[(3 * fold + offset) % 18] for offset in range(3)]
        validation = [SCENES[(3 * fold + 3 + offset) % 18] for offset in range(3)]
        train = [scene for scene in SCENES if scene not in test + validation]
        folds.append({"fold": fold + 1, "train": train, "validation": validation, "test": test})
    return folds


def model_inputs(data: dict[str, np.ndarray]) -> np.ndarray:
    p_sar = np.clip(data["sar_features"][:, 0], 0.0, 1.0)
    p_hsi = 1.0 / (1.0 + np.exp(-data["hsi_logits"].reshape(-1)))
    disagreement = np.abs(p_sar - p_hsi)
    hsi_confidence = np.clip(data["hsi_confidence"].reshape(-1), 0.0, 1.0)
    return np.column_stack([p_sar, p_hsi, disagreement, hsi_confidence]).astype(np.float32)


def expert_probabilities(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    p_sar = np.clip(data["sar_features"][:, 0], 0.0, 1.0).astype(np.float32)
    p_hsi = (1.0 / (1.0 + np.exp(-data["hsi_logits"].reshape(-1)))).astype(np.float32)
    return p_sar, p_hsi


class ReliabilityFusionV1(nn.Module):
    """Small two-head reliability estimator with normalized modality weights."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.sar_reliability = nn.Linear(hidden_dim, 1)
        self.hsi_reliability = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.sar_reliability.bias, -0.5)
        nn.init.constant_(self.hsi_reliability.bias, 0.5)

    def forward(self, inputs: torch.Tensor, p_sar: torch.Tensor, p_hsi: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context(inputs)
        r_sar = self.sar_reliability(context)
        r_hsi = self.hsi_reliability(context)
        weights = torch.softmax(torch.cat([r_sar, r_hsi], dim=1), dim=1)
        p_fusion = weights[:, 0:1] * p_sar + weights[:, 1:2] * p_hsi
        return {
            "p_fused": p_fusion,
            "p_sar": p_sar,
            "p_hsi": p_hsi,
            "r_sar": r_sar,
            "r_hsi": r_hsi,
            "w_sar": weights[:, 0:1],
            "w_hsi": weights[:, 1:2],
        }


def reliability_targets(p_sar: torch.Tensor, p_hsi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sar_quality = torch.exp(-(p_sar - labels).abs() / QUALITY_TEMPERATURE)
    hsi_quality = torch.exp(-(p_hsi - labels).abs() / QUALITY_TEMPERATURE)
    return torch.softmax(torch.cat([sar_quality, hsi_quality], dim=1), dim=1)


def dice_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    intersection = (probabilities * labels).sum()
    denominator = probabilities.sum() + labels.sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def total_loss(output: dict[str, torch.Tensor], labels: torch.Tensor, lambda_rel: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = output["p_fused"].float().clamp(1e-6, 1.0 - 1e-6)
    fusion_loss = F.binary_cross_entropy(probabilities, labels)
    fusion_loss = fusion_loss + dice_loss(probabilities, labels)
    targets = reliability_targets(output["p_sar"].detach(), output["p_hsi"].detach(), labels)
    predicted_weights = torch.cat([output["w_sar"], output["w_hsi"]], dim=1).float().clamp(1e-6, 1.0)
    reliability_loss = -(targets * predicted_weights.log()).sum(dim=1).mean()
    return fusion_loss + lambda_rel * reliability_loss, reliability_loss


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lambda_rel: float, optimizer: Any | None) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses = []
    reliability_losses = []
    all_probabilities = []
    all_labels = []
    all_sar_weights = []
    all_hsi_weights = []

    for inputs, p_sar, p_hsi, labels in loader:
        inputs = inputs.to(device)
        p_sar = p_sar.to(device)
        p_hsi = p_hsi.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(inputs, p_sar, p_hsi)
        loss, reliability_loss = total_loss(output, labels, lambda_rel)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        reliability_losses.append(float(reliability_loss.detach().cpu()))
        all_probabilities.append(output["p_fused"].detach().cpu().numpy().reshape(-1))
        all_labels.append(labels.detach().cpu().numpy().reshape(-1))
        all_sar_weights.append(output["w_sar"].detach().cpu().numpy().reshape(-1))
        all_hsi_weights.append(output["w_hsi"].detach().cpu().numpy().reshape(-1))

    return (
        float(np.mean(losses)),
        float(np.mean(reliability_losses)),
        np.concatenate(all_probabilities),
        np.concatenate(all_labels),
        np.concatenate(all_sar_weights),
        np.concatenate(all_hsi_weights),
    )


def make_loader(data: dict[str, np.ndarray], device: torch.device, shuffle: bool) -> DataLoader:
    inputs = torch.from_numpy(model_inputs(data))
    p_sar, p_hsi = expert_probabilities(data)
    labels = torch.from_numpy(data["labels"].astype(np.float32)).view(-1, 1)
    return DataLoader(
        TensorDataset(
            inputs,
            torch.from_numpy(p_sar).view(-1, 1),
            torch.from_numpy(p_hsi).view(-1, 1),
            labels,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    include_ranking: bool = True,
) -> dict[str, float]:
    if len(labels) == 0:
        raise ValueError("Cannot compute metrics for an empty scene subset.")

    predictions = (probabilities >= threshold).astype(np.int64)
    result = {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "iou": float(jaccard_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }

    if include_ranking:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc"] = float(average_precision_score(labels, probabilities))

    return result


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = [
        (
            classification_metrics(
                labels,
                probabilities,
                float(t),
                include_ranking=False,
            )["f1"],
            float(t),
        )
        for t in np.arange(0.01, 0.991, 0.005)
    ]
    return max(candidates)[1]


def train_variant(train: dict[str, np.ndarray], validation: dict[str, np.ndarray], lambda_rel: float, device: torch.device, checkpoint_path: Path) -> tuple[dict[str, Any], ReliabilityFusionV1]:
    print(
        f"\nSTART TRAINING | lambda_rel={lambda_rel:.2f} | "
        f"epochs={EPOCHS} | batch_size={BATCH_SIZE}"
    )

    model = ReliabilityFusionV1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train, device, True)
    validation_loader = make_loader(validation, device, False)
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, _, _, _, _, _ = run_epoch(model, train_loader, device, lambda_rel, optimizer)
        validation_loss, rel_loss, probabilities, labels, _, _ = run_epoch(model, validation_loader, device, lambda_rel, None)
        threshold = select_threshold(labels, probabilities)
        validation_f1 = classification_metrics(labels, probabilities, threshold)["f1"]
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        elapsed = time.time() - training_start
        epoch_seconds = time.time() - epoch_start
        progress = 100.0 * epoch / EPOCHS
        print(
            f"[{progress:6.2f}%] epoch {epoch:03d}/{EPOCHS} | "
            f"epoch_time={epoch_seconds:.2f}s | elapsed={elapsed / 60.0:.2f}m | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"rel_loss={rel_loss:.4f} | val_f1={validation_f1:.4f}"
        )
        if patience >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "lambda_rel": lambda_rel, "best_epoch": best_epoch, "best_validation_f1": best_f1}, checkpoint_path)
    return {"best_epoch": best_epoch, "best_validation_f1": best_f1, "epochs_completed": epoch}, model


def baseline_current_fusion(fold_number: int, data: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    checkpoint_path = CURRENT_CV / f"fold_{fold_number:02d}" / "weights" / "best_fusion_model.pth"
    if not checkpoint_path.exists():
        return np.full(len(data["labels"]), np.nan, dtype=np.float32)
    module = load_module(f"current_fusion_{fold_number}", ROOT / "04_fusion_model.py")
    model = module.build_fusion_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data["labels"]), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(data["labels"]))
            output = model(
                torch.from_numpy(data["sar_features"][start:end]).float().to(device),
                torch.from_numpy(data["hsi_features"][start:end]).float().to(device),
                torch.from_numpy(data["hsi_logits"][start:end]).float().to(device),
                torch.from_numpy(data["hsi_confidence"][start:end]).float().to(device),
            )
            predictions.append(output["p_fused"].cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def show_completion_animation(
    message: str,
) -> None:
    """Show a short terminal-only animation after a successful experiment."""

    frames = [
        "[          ]",
        "[==        ]",
        "[====      ]",
        "[======    ]",
        "[========  ]",
        "[==========]",
    ]

    for frame in frames:
        print(
            f"\r{message} {frame}",
            end="",
            flush=True,
        )
        time.sleep(0.08)

    print(
        f"\r{message} [==========] COMPLETE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 7), help="Run one fold only")
    parser.add_argument("--lambda-rel", type=float, choices=LAMBDA_VALUES, help="Run one reliability weight only")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caches = load_source_caches()
    scene_data = split_by_scene(caches)
    folds = make_folds()
    selected_folds = [fold for fold in folds if args.fold is None or fold["fold"] == args.fold]
    selected_lambdas = [args.lambda_rel] if args.lambda_rel is not None else list(LAMBDA_VALUES)
    all_rows = []
    all_scene_rows = []
    all_weight_rows = []
    all_significance_inputs = {"HSI Only": [], "Reliability Fusion": [], "Current Fusion": []}

    for fold in selected_folds:
        fold_dir = OUTPUT_ROOT / f"fold_{fold['fold']:02d}"
        train = merge_scenes(scene_data, fold["train"])
        validation = merge_scenes(scene_data, fold["validation"])
        test = merge_scenes(scene_data, fold["test"])
        test_labels = test["labels"]
        test_sar, test_hsi = expert_probabilities(test)
        validation_sar, validation_hsi = expert_probabilities(validation)
        sar_threshold = select_threshold(validation["labels"], validation_sar)
        hsi_threshold = select_threshold(validation["labels"], validation_hsi)
        current_fusion = baseline_current_fusion(fold["fold"], test, device)
        validation_current = baseline_current_fusion(fold["fold"], validation, device)
        current_threshold = select_threshold(validation["labels"], validation_current) if np.isfinite(validation_current).all() else 0.5

        for lambda_rel in selected_lambdas:
            variant_name = "Reliability Fusion (no supervision)" if lambda_rel == 0.0 else f"Reliability Fusion (lambda={lambda_rel:.2f})"
            checkpoint_path = fold_dir / f"lambda_{lambda_rel:.2f}" / "best_model.pth"
            train_info, model = train_variant(train, validation, lambda_rel, device, checkpoint_path)
            test_loader = make_loader(test, device, False)
            _, _, reliability_probabilities, _, sar_weights, hsi_weights = run_epoch(model, test_loader, device, lambda_rel, None)
            validation_loader = make_loader(validation, device, False)
            _, _, validation_probabilities, validation_labels, _, _ = run_epoch(model, validation_loader, device, lambda_rel, None)
            reliability_threshold = select_threshold(validation_labels, validation_probabilities)
            result = classification_metrics(test_labels, reliability_probabilities, reliability_threshold)
            result.update({"fold": fold["fold"], "model": variant_name, "lambda_rel": lambda_rel, "train_scenes": ";".join(fold["train"]), "validation_scenes": ";".join(fold["validation"]), "test_scenes": ";".join(fold["test"]), "best_epoch": train_info["best_epoch"]})
            all_rows.append(result)
            all_significance_inputs.setdefault(variant_name, []).append(result["f1"])
            all_weight_rows.append({"fold": fold["fold"], "model": variant_name, "mean_sar_weight": float(sar_weights.mean()), "std_sar_weight": float(sar_weights.std()), "mean_hsi_weight": float(hsi_weights.mean()), "high_disagreement_sar_weight": float(sar_weights[np.abs(test_sar - test_hsi) >= 0.40].mean()) if np.any(np.abs(test_sar - test_hsi) >= 0.40) else float("nan"), "high_disagreement_hsi_weight": float(hsi_weights[np.abs(test_sar - test_hsi) >= 0.40].mean()) if np.any(np.abs(test_sar - test_hsi) >= 0.40) else float("nan")})
            for scene_index, scene_name in enumerate(fold["test"]):
                mask = test["scene_indices"] == scene_index
                scene_result = classification_metrics(test_labels[mask], reliability_probabilities[mask], reliability_threshold)
                scene_result.update({"fold": fold["fold"], "scene": scene_name, "model": variant_name, "mean_sar_weight": float(sar_weights[mask].mean()), "mean_hsi_weight": float(hsi_weights[mask].mean())})
                all_scene_rows.append(scene_result)

        hsi_result = classification_metrics(test_labels, test_hsi, hsi_threshold)
        all_rows.append({**hsi_result, "fold": fold["fold"], "model": "HSI Only", "lambda_rel": "", "train_scenes": ";".join(fold["train"]), "validation_scenes": ";".join(fold["validation"]), "test_scenes": ";".join(fold["test"]), "best_epoch": ""})
        all_significance_inputs["HSI Only"].append(hsi_result["f1"])
        if np.isfinite(current_fusion).all():
            current_result = classification_metrics(test_labels, current_fusion, current_threshold)
            all_rows.append({**current_result, "fold": fold["fold"], "model": "Previous Current Fusion", "lambda_rel": "", "train_scenes": ";".join(fold["train"]), "validation_scenes": ";".join(fold["validation"]), "test_scenes": ";".join(fold["test"]), "best_epoch": ""})
            all_significance_inputs["Current Fusion"].append(current_result["f1"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "reliability_results.csv", all_rows)
    write_csv(OUTPUT_ROOT / "per_scene_metrics.csv", all_scene_rows)
    write_csv(OUTPUT_ROOT / "learned_weights.csv", all_weight_rows)

    hsi_values = np.array(all_significance_inputs["HSI Only"], dtype=float)
    significance = []
    for model_name, values in all_significance_inputs.items():
        if model_name == "HSI Only" or len(values) != len(hsi_values):
            continue
        values = np.array(values, dtype=float)
        difference = values - hsi_values
        significance.append({"model": model_name, "reference": "HSI Only", "mean_f1_difference": float(difference.mean()), "std_f1_difference": float(difference.std(ddof=1)), "paired_t_p_value": float(ttest_rel(values, hsi_values).pvalue), "wilcoxon_p_value": float(wilcoxon(difference).pvalue)})
    write_csv(OUTPUT_ROOT / "paired_statistics_vs_hsi.csv", significance)

    aggregate = {}
    for model_name in sorted({row["model"] for row in all_rows}):
        rows = [row for row in all_rows if row["model"] == model_name]
        aggregate[model_name] = {metric: {"mean": float(np.mean([row[metric] for row in rows])), "std": float(np.std([row[metric] for row in rows], ddof=1)), "values": [row[metric] for row in rows]} for metric in ("f1", "iou", "accuracy", "precision", "recall", "roc_auc", "pr_auc")}
    report = {"experiment": "Reliability-Aware Registration-Free Fusion V1", "protocol": {"folds": 6, "train_validation_test": "12/3/3 scenes", "test_labels_used_for_training_or_tuning": False, "quality_temperature": QUALITY_TEMPERATURE, "lambda_sweep": list(LAMBDA_VALUES), "inputs": ["p_sar", "p_hsi", "abs(p_sar-p_hsi)", "hsi_confidence"], "experts_frozen": True}, "aggregate_fold_results": aggregate, "paired_statistics": significance, "files": {"results": str(OUTPUT_ROOT / "reliability_results.csv"), "per_scene": str(OUTPUT_ROOT / "per_scene_metrics.csv"), "weights": str(OUTPUT_ROOT / "learned_weights.csv"), "statistics": str(OUTPUT_ROOT / "paired_statistics_vs_hsi.csv")}}
    (OUTPUT_ROOT / "reliability_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Reliability Fusion V1 complete")
    for name, values in aggregate.items():
        print(f"{name:<38} F1={values['f1']['mean']:.4f} +/- {values['f1']['std']:.4f} | IoU={values['iou']['mean']:.4f} +/- {values['iou']['std']:.4f}")
    print(f"Reports: {OUTPUT_ROOT}")
    show_completion_animation(
        "Reliability Fusion V1 training and analysis"
    )


if __name__ == "__main__":
    main()
