"""Oracle upper-bound analysis over the existing six-fold scene CV.

This script performs no training. It reconstructs p_sar and p_hsi from the
existing fold test caches, then uses labels only for post hoc oracle analysis.
"""

from __future__ import annotations

import csv
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
CV_ROOT = ROOT / "experiments" / "scene_cv_6fold"
OUTPUT_DIR = CV_ROOT / "oracle_fusion_analysis"
BATCH_SIZE = 2048
SCENE_NAMES = [
    "GM01", "GM02", "GM03", "GM04", "GM05", "GM06",
    "GM07", "GM08", "GM09", "GM10", "GM11", "GM12",
    "GM13", "GM14", "GM15", "GM16", "GM17", "GM18",
]


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    include_ranking: bool = True,
) -> dict[str, float]:
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


def choose_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = []
    for threshold in np.arange(0.01, 0.991, 0.005):
        row = metrics(
            labels,
            probabilities,
            float(threshold),
            include_ranking=False,
        )
        candidates.append(row)
    selected = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return float(selected["threshold"])


@torch.no_grad()
def model_probabilities(model: torch.nn.Module, cache: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    fused = []
    gates = []
    for start in range(0, len(cache["labels"]), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(cache["labels"]))
        sar = torch.from_numpy(cache["sar_features"][start:end]).float().to(device)
        hsi = torch.from_numpy(cache["hsi_features"][start:end]).float().to(device)
        logits = torch.from_numpy(cache["hsi_logits"][start:end]).float().to(device)
        confidence = torch.from_numpy(cache["hsi_confidence"][start:end]).float().to(device)
        output = model(sar, hsi, logits, confidence)
        fused.append(output["p_fused"].float().cpu().numpy().reshape(-1))
        gates.append(output["gate"].float().cpu().numpy().reshape(-1))
    return {
        "fusion": np.concatenate(fused),
        "gate": np.concatenate(gates),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def disagreement_rows(labels: np.ndarray, sar: np.ndarray, hsi: np.ndarray, average: np.ndarray, oracle: np.ndarray) -> list[dict[str, Any]]:
    disagreement = np.abs(sar - hsi)
    bins = [0.0, 0.05, 0.10, 0.20, 0.40, 1.000001]
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (disagreement >= lower) & (disagreement < upper)
        if not mask.any():
            continue
        row: dict[str, Any] = {
            "disagreement_bin": f"[{lower:.2f}, {min(upper, 1.0):.2f})",
            "disagreement_min": lower,
            "disagreement_max": min(upper, 1.0),
            "samples": int(mask.sum()),
            "mean_disagreement": float(disagreement[mask].mean()),
        }
        for name, values in (("sar", sar), ("hsi", hsi), ("simple_average", average), ("oracle", oracle)):
            result = metrics(labels[mask], values[mask], 0.5)
            row[f"{name}_f1"] = result["f1"]
            row[f"{name}_iou"] = result["iou"]
            row[f"{name}_accuracy"] = result["accuracy"]
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_module = load_module("oracle_fusion_model", ROOT / "04_fusion_model.py")
    all_fold_rows: list[dict[str, Any]] = []
    all_scene_rows: list[dict[str, Any]] = []
    all_disagreement_rows: list[dict[str, Any]] = []
    fold_summaries = []

    for fold_number in range(1, 7):
        fold_dir = CV_ROOT / f"fold_{fold_number:02d}"
        evaluation = json.loads((fold_dir / "evaluation.json").read_text(encoding="utf-8"))
        fold = evaluation["fold"]
        checkpoint = torch.load(
            fold_dir / "weights" / "best_fusion_model.pth",
            map_location=device,
            weights_only=False,
        )
        model = model_module.build_fusion_model().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        validation = load_cache(fold_dir / "cache_features" / "validation_features.npz")
        test = load_cache(fold_dir / "cache_features" / "test_features.npz")
        validation_hsi = 1.0 / (1.0 + np.exp(-validation["hsi_logits"].reshape(-1)))
        validation_sar = validation["sar_features"][:, 0]
        test_hsi = 1.0 / (1.0 + np.exp(-test["hsi_logits"].reshape(-1)))
        test_sar = test["sar_features"][:, 0]
        validation_fusion = model_probabilities(model, validation, device)["fusion"]
        test_fusion_output = model_probabilities(model, test, device)
        test_fusion = test_fusion_output["fusion"]
        labels = test["labels"]

        sar_threshold = choose_validation_threshold(validation["labels"], validation_sar)
        hsi_threshold = choose_validation_threshold(validation["labels"], validation_hsi)
        fusion_threshold = choose_validation_threshold(validation["labels"], validation_fusion)
        simple_average_validation = 0.5 * validation_sar + 0.5 * validation_hsi
        simple_average_test = 0.5 * test_sar + 0.5 * test_hsi
        average_threshold = choose_validation_threshold(validation["labels"], simple_average_validation)

        sar_error = np.abs(test_sar - labels)
        hsi_error = np.abs(test_hsi - labels)
        oracle = np.where(sar_error < hsi_error, test_sar, test_hsi)

        fold_values = {
            "SAR Only": (test_sar, sar_threshold),
            "HSI Only": (test_hsi, hsi_threshold),
            "Simple Average": (simple_average_test, average_threshold),
            "Oracle Fusion": (oracle, 0.5),
            "Registration-Free Fusion": (test_fusion, fusion_threshold),
        }
        fold_rows = []
        for name, (values, threshold) in fold_values.items():
            row = {
                "fold": fold_number,
                "train_scenes": ";".join(fold["train"]),
                "validation_scenes": ";".join(fold["validation"]),
                "test_scenes": ";".join(fold["test"]),
                "model": name,
            }
            row.update(metrics(labels, values, threshold))
            fold_rows.append(row)
        all_fold_rows.extend(fold_rows)

        for scene_index, scene_name in enumerate(fold["test"]):
            mask = test["scene_indices"] == scene_index
            sar_closer = sar_error[mask] < hsi_error[mask]
            hsi_closer = hsi_error[mask] < sar_error[mask]
            similar = np.abs(test_sar[mask] - test_hsi[mask]) <= 0.05
            scene_row = {
                "fold": fold_number,
                "scene": scene_name,
                "samples": int(mask.sum()),
                "sar_f1": metrics(labels[mask], test_sar[mask], sar_threshold)["f1"],
                "hsi_f1": metrics(labels[mask], test_hsi[mask], hsi_threshold)["f1"],
                "oracle_f1": metrics(labels[mask], oracle[mask], 0.5)["f1"],
                "sar_closer_percent": float(100.0 * sar_closer.mean()),
                "hsi_closer_percent": float(100.0 * hsi_closer.mean()),
                "similar_percent_abs_difference_le_0_05": float(100.0 * similar.mean()),
            }
            all_scene_rows.append(scene_row)

        fold_disagreement = disagreement_rows(labels, test_sar, test_hsi, simple_average_test, oracle)
        for row in fold_disagreement:
            row["fold"] = fold_number
        all_disagreement_rows.extend(fold_disagreement)
        fold_summaries.append({
            "fold": fold_number,
            "thresholds": {
                "sar": sar_threshold,
                "hsi": hsi_threshold,
                "simple_average": average_threshold,
                "fusion": fusion_threshold,
            },
            "test_scenes": fold["test"],
        })

    write_csv(OUTPUT_DIR / "oracle_results.csv", all_fold_rows)
    write_csv(OUTPUT_DIR / "oracle_per_scene.csv", all_scene_rows)
    write_csv(OUTPUT_DIR / "disagreement_analysis.csv", all_disagreement_rows)

    aggregate = {}
    for model_name in sorted({row["model"] for row in all_fold_rows}):
        rows = [row for row in all_fold_rows if row["model"] == model_name]
        aggregate[model_name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows], ddof=1)),
                "fold_values": [row[metric] for row in rows],
            }
            for metric in ("f1", "iou", "accuracy", "precision", "recall", "roc_auc", "pr_auc")
        }

    report = {
        "experiment": "Oracle Fusion Analysis",
        "protocol": {
            "fold_count": 6,
            "labels_used_for_training": False,
            "labels_used_for_threshold_tuning": False,
            "labels_used_for_oracle_analysis_only": True,
            "oracle_is_deployable": False,
            "oracle_tie_policy": "choose HSI when absolute errors are equal",
            "similar_prediction_definition": "abs(p_sar - p_hsi) <= 0.05",
        },
        "methodology": {
            "sar_error": "abs(p_sar - y)",
            "hsi_error": "abs(p_hsi - y)",
            "p_oracle": "p_sar if sar_error < hsi_error else p_hsi",
            "simple_average": "0.5 * p_sar + 0.5 * p_hsi",
            "thresholds": "SAR, HSI, simple average, and learned fusion thresholds selected on validation; oracle evaluated at 0.5",
        },
        "aggregate_fold_results": aggregate,
        "fold_thresholds": fold_summaries,
        "files": {
            "oracle_results": str(OUTPUT_DIR / "oracle_results.csv"),
            "oracle_per_scene": str(OUTPUT_DIR / "oracle_per_scene.csv"),
            "disagreement_analysis": str(OUTPUT_DIR / "disagreement_analysis.csv"),
            "summary": str(OUTPUT_DIR / "oracle_summary.json"),
        },
    }
    (OUTPUT_DIR / "oracle_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Oracle Fusion Analysis complete")
    for name, values in aggregate.items():
        print(f"{name:<28} F1={values['f1']['mean']:.4f} +/- {values['f1']['std']:.4f} | IoU={values['iou']['mean']:.4f} +/- {values['iou']['std']:.4f}")
    print(f"Reports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
