"""Audit the cached SAR/HSI fusion experiment without changing its protocol.

Validation data selects thresholds and baseline weights. Test data is used only
for final reporting after those choices are fixed. This script does not train.
"""

from __future__ import annotations

import csv
import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(
    os.environ.get(
        "FUSION_CACHE_DIR",
        ROOT / "cache_features",
    )
).resolve()
WEIGHTS_DIR = Path(
    os.environ.get(
        "FUSION_CHECKPOINT_DIR",
        ROOT / "weights",
    )
).resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "FUSION_AUDIT_DIR",
        ROOT / "audit",
    )
).resolve()
CHECKPOINT_PATH = WEIGHTS_DIR / "best_fusion_model.pth"
BATCH_SIZE = 1024
THRESHOLDS = np.arange(0.01, 0.991, 0.005)


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cache(name: str) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"{name}_features.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
        }
    except Exception as exc:
        return {"ranking_metrics_error": str(exc)}


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = [metrics(labels, probabilities, float(t)) for t in THRESHOLDS]
    selected = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return float(selected["threshold"]), selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def infer(model: torch.nn.Module, cache: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {}
    count = len(cache["labels"])
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        tensors = {
            key: torch.from_numpy(cache[key][start:end]).float().to(device)
            for key in ("sar_features", "hsi_features", "hsi_logits", "hsi_confidence")
        }
        result = model(**tensors)
        for key in ("p_fused", "p_sar", "p_hsi", "reliability", "gate", "bounded_delta"):
            if key in result:
                outputs.setdefault(key, []).append(result[key].detach().float().cpu().numpy().reshape(-1))
    return {key: np.concatenate(values) for key, values in outputs.items()}


def evaluate_models(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for name, values in probabilities.items():
        threshold = thresholds[name]
        row = {"model": name}
        row.update(metrics(labels, values, threshold))
        row.update(ranking_metrics(labels, values))
        rows.append(row)
    return rows


def scene_rows(
    labels: np.ndarray,
    scene_indices: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for scene in np.unique(scene_indices):
        mask = scene_indices == scene
        for name, values in probabilities.items():
            row = {"scene_index": int(scene), "model": name}
            row.update(metrics(labels[mask], values[mask], thresholds[name]))
            row.pop("threshold", None)
            rows.append(row)
    return rows


def confidence_rows(
    labels: np.ndarray,
    confidence: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    bins: int = 10,
) -> list[dict[str, Any]]:
    values = confidence.reshape(-1)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    rows = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (values >= lower) & ((values <= upper) if index == bins - 1 else (values < upper))
        if not mask.any():
            continue
        row = {
            "bin": index,
            "confidence_min": lower,
            "confidence_max": upper,
            "samples": int(mask.sum()),
            "mean_confidence": float(values[mask].mean()),
            "accuracy_at_threshold": metrics(labels[mask], probabilities[mask], threshold)["accuracy"],
            "f1_at_threshold": metrics(labels[mask], probabilities[mask], threshold)["f1"],
            "error_rate_at_threshold": float(((probabilities[mask] >= threshold).astype(int) != labels[mask]).mean()),
        }
        rows.append(row)
    return rows


def overlap_rows(labels: np.ndarray, hsi: np.ndarray, fusion: np.ndarray, threshold: float) -> dict[str, int]:
    actual = labels.astype(bool)
    hsi_error = (hsi >= threshold) != actual
    fusion_error = (fusion >= threshold) != actual
    return {
        "hsi_errors": int(hsi_error.sum()),
        "fusion_errors": int(fusion_error.sum()),
        "fusion_fixed_hsi": int((hsi_error & ~fusion_error).sum()),
        "fusion_introduced_errors": int((~hsi_error & fusion_error).sum()),
        "both_wrong": int((hsi_error & fusion_error).sum()),
        "both_correct": int((~hsi_error & ~fusion_error).sum()),
        "hsi_only_wrong": int((hsi_error & ~fusion_error).sum()),
    }


def bootstrap_significance(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_threshold: float,
    second_threshold: float,
    repeats: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate paired uncertainty for F1/IoU differences on test samples."""

    rng = np.random.default_rng(seed)
    sample_count = len(labels)
    f1_differences = np.empty(repeats, dtype=np.float64)
    iou_differences = np.empty(repeats, dtype=np.float64)

    for repeat in range(repeats):
        indices = rng.integers(0, sample_count, size=sample_count)
        first_metrics = metrics(labels[indices], first[indices], first_threshold)
        second_metrics = metrics(labels[indices], second[indices], second_threshold)
        f1_differences[repeat] = first_metrics["f1"] - second_metrics["f1"]
        iou_differences[repeat] = first_metrics["iou"] - second_metrics["iou"]

    f1_interval = np.quantile(f1_differences, [0.025, 0.975])
    iou_interval = np.quantile(iou_differences, [0.025, 0.975])
    return {
        "comparison": "fusion_minus_hsi",
        "first_threshold": float(first_threshold),
        "second_threshold": float(second_threshold),
        "bootstrap_repeats": repeats,
        "f1_difference": float(f1_differences.mean()),
        "f1_difference_ci_low": float(f1_interval[0]),
        "f1_difference_ci_high": float(f1_interval[1]),
        "iou_difference": float(iou_differences.mean()),
        "iou_difference_ci_low": float(iou_interval[0]),
        "iou_difference_ci_high": float(iou_interval[1]),
        "f1_probability_fusion_better": float(np.mean(f1_differences > 0.0)),
        "iou_probability_fusion_better": float(np.mean(iou_differences > 0.0)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_module = load_module("fusion_model_for_audit", ROOT / "04_fusion_model.py")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = model_module.build_fusion_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    validation = load_cache("validation")
    test = load_cache("test")
    val_output = infer(model, validation, device)
    test_output = infer(model, test, device)

    val_labels = validation["labels"]
    test_labels = test["labels"]
    val_fusion = val_output["p_fused"]
    test_fusion = test_output["p_fused"]
    val_hsi = 1.0 / (1.0 + np.exp(-validation["hsi_logits"].reshape(-1)))
    test_hsi = 1.0 / (1.0 + np.exp(-test["hsi_logits"].reshape(-1)))
    val_sar = validation["sar_features"][:, 0]
    test_sar = test["sar_features"][:, 0]

    fusion_threshold, val_fusion_threshold_metrics = select_threshold(val_labels, val_fusion)
    hsi_threshold, _ = select_threshold(val_labels, val_hsi)
    sar_threshold, _ = select_threshold(val_labels, val_sar)

    # Weighted probability fusion chooses its weight and threshold on validation only.
    weighted_candidates = []
    for weight in np.arange(0.0, 1.001, 0.05):
        val_weighted = weight * val_hsi + (1.0 - weight) * val_sar
        threshold, selected = select_threshold(val_labels, val_weighted)
        weighted_candidates.append((selected["f1"], float(weight), threshold))
    _, weighted_hsi_weight, weighted_threshold = max(weighted_candidates)
    test_weighted = weighted_hsi_weight * test_hsi + (1.0 - weighted_hsi_weight) * test_sar

    try:
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
        calibrator.fit(val_fusion.reshape(-1, 1), val_labels)
        test_platt = calibrator.predict_proba(test_fusion.reshape(-1, 1))[:, 1]
        platt_available = True
    except Exception as exc:
        test_platt = test_fusion
        platt_available = False
        platt_error = str(exc)

    probabilities = {
        "SAR Only": test_sar,
        "HSI Only": test_hsi,
        "Weighted SAR-HSI": test_weighted,
        "Registration-Free Fusion": test_fusion,
    }
    if platt_available:
        probabilities["Platt-Calibrated Fusion"] = test_platt
    thresholds = {
        "SAR Only": sar_threshold,
        "HSI Only": hsi_threshold,
        "Weighted SAR-HSI": weighted_threshold,
        "Registration-Free Fusion": fusion_threshold,
    }
    if platt_available:
        thresholds["Platt-Calibrated Fusion"] = 0.50
    final_rows = evaluate_models(test_labels, probabilities, thresholds)
    write_csv(OUTPUT_DIR / "model_comparison.csv", final_rows)

    scenes = scene_rows(test_labels, test["scene_indices"], probabilities, thresholds)
    write_csv(OUTPUT_DIR / "per_scene_metrics.csv", scenes)

    confidence = confidence_rows(
        test_labels,
        test["hsi_confidence"],
        test_fusion,
        fusion_threshold,
    )
    write_csv(OUTPUT_DIR / "confidence_bins.csv", confidence)

    overlap = overlap_rows(test_labels, test_hsi, test_fusion, fusion_threshold)
    significance = bootstrap_significance(
        test_labels,
        test_fusion,
        test_hsi,
        fusion_threshold,
        hsi_threshold,
    )
    write_csv(OUTPUT_DIR / "statistical_significance.csv", [significance])
    ablation_probabilities = {
        "Full fusion": test_fusion,
        "No SAR features (zero SAR vector)": infer(
            model,
            {**test, "sar_features": np.zeros_like(test["sar_features"])},
            device,
        )["p_fused"],
        "No HSI features (zero HSI vector)": infer(
            model,
            {**test, "hsi_features": np.zeros_like(test["hsi_features"])},
            device,
        )["p_fused"],
        "No HSI confidence (zero confidence)": infer(
            model,
            {**test, "hsi_confidence": np.zeros_like(test["hsi_confidence"])},
            device,
        )["p_fused"],
    }
    ablation_thresholds = {name: fusion_threshold for name in ablation_probabilities}
    ablations = evaluate_models(test_labels, ablation_probabilities, ablation_thresholds)
    write_csv(OUTPUT_DIR / "ablation_metrics.csv", ablations)

    reliability_values = test_output.get("gate")
    if reliability_values is None:
        reliability_analysis = {
            "available": False,
            "reason": "The active model does not expose gate or reliability output.",
        }
    else:
        reliability_values = reliability_values.reshape(-1)
        reliability_analysis = {
            "available": True,
            "source_output": "gate",
            "mean": float(reliability_values.mean()),
            "std": float(reliability_values.std()),
            "low_reliability_fraction_lt_0_25": float(np.mean(reliability_values < 0.25)),
            "high_reliability_fraction_gt_0_75": float(np.mean(reliability_values > 0.75)),
            "correlation_with_absolute_error": float(
                np.corrcoef(reliability_values, (test_fusion - test_labels) ** 2)[0, 1]
            ),
        }

    fixed_thresholds = {
        "SAR Only": 0.50,
        "HSI Only": 0.50,
        "Weighted SAR-HSI": 0.50,
        "Registration-Free Fusion": 0.50,
    }
    if platt_available:
        fixed_thresholds["Platt-Calibrated Fusion"] = 0.50
    fixed_rows = evaluate_models(test_labels, probabilities, fixed_thresholds)
    write_csv(OUTPUT_DIR / "model_comparison_fixed_0_50.csv", fixed_rows)

    missing_split_artifacts = [
        "No alternate split-specific feature caches were found.",
        "No split-specific fusion checkpoints were found.",
        "Therefore 2-3 additional scene-split retrains and mean +/- standard deviation cannot be claimed by this audit.",
    ]
    report = {
        "protocol": {
            "validation_selects_thresholds_and_weights": True,
            "test_used_for_final_reporting_only": True,
            "trained_model_changed": False,
            "test_samples": int(len(test_labels)),
            "test_scenes": int(len(np.unique(test["scene_indices"]))),
        },
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "epoch": checkpoint.get("epoch"),
            "stored_threshold": checkpoint.get("decision_threshold"),
        },
        "validation_selection": {
            "fusion_threshold": fusion_threshold,
            "fusion_metrics_at_selected_threshold": val_fusion_threshold_metrics,
            "hsi_threshold": hsi_threshold,
            "sar_threshold": sar_threshold,
            "weighted_hsi_weight": weighted_hsi_weight,
            "weighted_sar_weight": 1.0 - weighted_hsi_weight,
            "weighted_threshold": weighted_threshold,
        },
        "error_overlap_at_validation_selected_fusion_threshold": overlap,
        "reliability_analysis": reliability_analysis,
        "missing_requirements": {
            "additional_scene_splits": missing_split_artifacts,
            "scene_conditioned_reliability_justification": "Needs ablation and reliability tables plus alternate split retraining; this audit produces the first two but cannot fabricate the last.",
        },
        "files": {
            "model_comparison": str(OUTPUT_DIR / "model_comparison.csv"),
            "model_comparison_fixed_0_50": str(OUTPUT_DIR / "model_comparison_fixed_0_50.csv"),
            "per_scene_metrics": str(OUTPUT_DIR / "per_scene_metrics.csv"),
            "confidence_bins": str(OUTPUT_DIR / "confidence_bins.csv"),
            "ablation_metrics": str(OUTPUT_DIR / "ablation_metrics.csv"),
            "statistical_significance": str(OUTPUT_DIR / "statistical_significance.csv"),
        },
    }
    if not platt_available:
        report["platt_error"] = platt_error
    with (OUTPUT_DIR / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Fusion audit complete")
    print(f"Device: {device}")
    print(f"Validation-selected fusion threshold: {fusion_threshold:.3f}")
    print(f"Validation-selected HSI weight in weighted baseline: {weighted_hsi_weight:.2f}")
    print("\nTest model comparison:")
    for row in final_rows:
        print(f"  {row['model']:<28} F1={row['f1']:.4f} IoU={row['iou']:.4f} ROC-AUC={row.get('roc_auc', float('nan')):.4f}")
    print("\nHSI error overlap:")
    for key, value in overlap.items():
        print(f"  {key}: {value}")
    print(f"\nReports: {OUTPUT_DIR}")
    print("Additional scene-split retraining: NOT AVAILABLE from current artifacts")


if __name__ == "__main__":
    main()
