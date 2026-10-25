"""Run deterministic 18-scene cross-validation for cached fusion features.

Each fold uses 12 train, 3 validation, and 3 test scenes. Thresholds are
selected from validation predictions only. The test scenes are evaluated once
per fold after training and never affect optimization or model selection.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE_CACHE = ROOT / "cache_features"
CV_ROOT = ROOT / "experiments" / "scene_cv_6fold"
RESULTS_PATH = CV_ROOT / "cross_validation_results.csv"
SUMMARY_PATH = CV_ROOT / "cross_validation_summary.json"
BATCH_SIZE = 1024

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


def load_source_caches() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for split in ("train", "validation", "test"):
        with np.load(SOURCE_CACHE / f"{split}_features.npz", allow_pickle=False) as data:
            result[split] = {key: np.asarray(data[key]) for key in data.files}
    return result


def source_scene_data(
    caches: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
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


def save_split(
    output_dir: Path,
    split_name: str,
    scene_names: list[str],
    per_scene: dict[str, dict[str, np.ndarray]],
) -> None:
    parts = []
    for scene_index, scene_name in enumerate(scene_names):
        part = {
            key: value.copy()
            for key, value in per_scene[scene_name].items()
        }
        part["scene_indices"] = np.full(
            len(part["labels"]),
            scene_index,
            dtype=np.int64,
        )
        parts.append(part)

    merged = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{split_name}_features.npz", **merged)
    with (output_dir / f"{split_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split_name,
                "scenes": scene_names,
                "num_samples": int(len(merged["labels"])),
                "positive_samples": int(merged["labels"].sum()),
            },
            handle,
            indent=2,
        )


def make_folds() -> list[dict[str, list[str]]]:
    folds = []
    for fold in range(6):
        test_scenes = [SCENES[(3 * fold + offset) % len(SCENES)] for offset in range(3)]
        validation_scenes = [SCENES[(3 * fold + 3 + offset) % len(SCENES)] for offset in range(3)]
        train_scenes = [scene for scene in SCENES if scene not in test_scenes + validation_scenes]
        folds.append(
            {
                "fold": fold + 1,
                "train": train_scenes,
                "validation": validation_scenes,
                "test": test_scenes,
            }
        )
    return folds


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall}


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in np.arange(0.01, 0.991, 0.005):
        row = metrics(labels, probabilities, float(threshold))
        row["threshold"] = float(threshold)
        candidates.append(row)
    selected = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return selected["threshold"], selected


@torch.no_grad()
def infer(model: torch.nn.Module, data: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    count = len(data["labels"])
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        output = model(
            torch.from_numpy(data["sar_features"][start:end]).float().to(device),
            torch.from_numpy(data["hsi_features"][start:end]).float().to(device),
            torch.from_numpy(data["hsi_logits"][start:end]).float().to(device),
            torch.from_numpy(data["hsi_confidence"][start:end]).float().to(device),
        )
        predictions.append(output["p_fused"].detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def run_fold(
    fold: dict[str, list[str]],
    per_scene: dict[str, dict[str, np.ndarray]],
    model_module,
    device: torch.device,
) -> list[dict[str, object]]:
    fold_dir = CV_ROOT / f"fold_{fold['fold']:02d}"
    cache_dir = fold_dir / "cache_features"
    weights_dir = fold_dir / "weights"

    save_split(cache_dir, "train", fold["train"], per_scene)
    save_split(cache_dir, "validation", fold["validation"], per_scene)
    save_split(cache_dir, "test", fold["test"], per_scene)

    environment = os.environ.copy()
    environment["FUSION_CACHE_DIR"] = str(cache_dir)
    environment["FUSION_CHECKPOINT_DIR"] = str(weights_dir)
    subprocess.run(
        [sys.executable, str(ROOT / "05_train.py")],
        cwd=str(ROOT.parent),
        env=environment,
        check=True,
    )

    checkpoint = torch.load(weights_dir / "best_fusion_model.pth", map_location=device, weights_only=False)
    model = model_module.build_fusion_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    validation = {
        key: np.concatenate([per_scene[name][key] for name in fold["validation"]], axis=0)
        for key in per_scene[fold["validation"][0]]
    }
    test = {
        key: np.concatenate([per_scene[name][key] for name in fold["test"]], axis=0)
        for key in per_scene[fold["test"][0]]
    }
    validation_fusion = infer(model, validation, device)
    test_fusion = infer(model, test, device)
    validation_labels = validation["labels"]
    test_labels = test["labels"]
    validation_hsi = 1.0 / (1.0 + np.exp(-validation["hsi_logits"].reshape(-1)))
    test_hsi = 1.0 / (1.0 + np.exp(-test["hsi_logits"].reshape(-1)))
    validation_sar = validation["sar_features"][:, 0]
    test_sar = test["sar_features"][:, 0]

    fusion_threshold, _ = select_threshold(validation_labels, validation_fusion)
    hsi_threshold, _ = select_threshold(validation_labels, validation_hsi)
    sar_threshold, _ = select_threshold(validation_labels, validation_sar)

    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression()
    calibrator.fit(validation_fusion.reshape(-1, 1), validation_labels)
    test_platt = calibrator.predict_proba(test_fusion.reshape(-1, 1))[:, 1]

    rows = []
    model_values = {
        "SAR Only": (test_sar, sar_threshold),
        "HSI Only": (test_hsi, hsi_threshold),
        "Registration-Free Fusion": (test_fusion, fusion_threshold),
        "Platt-Calibrated Fusion": (test_platt, 0.50),
    }
    for name, (values, threshold) in model_values.items():
        row = {
            "fold": fold["fold"],
            "train_scenes": ";".join(fold["train"]),
            "validation_scenes": ";".join(fold["validation"]),
            "test_scenes": ";".join(fold["test"]),
            "model": name,
            "threshold": threshold,
        }
        row.update(metrics(test_labels, values, threshold))
        rows.append(row)

    with (fold_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold": fold,
                "thresholds": {
                    "sar": sar_threshold,
                    "hsi": hsi_threshold,
                    "fusion": fusion_threshold,
                    "platt": 0.5,
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )
    return rows


def main() -> None:
    CV_ROOT.mkdir(parents=True, exist_ok=True)
    caches = load_source_caches()
    per_scene = source_scene_data(caches)
    model_module = load_module("fusion_cv_model", ROOT / "04_fusion_model.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = make_folds()
    all_rows = []
    for fold in folds:
        print(f"\n=== CROSS-VALIDATION FOLD {fold['fold']}/6 ===")
        print("Train:", fold["train"])
        print("Validation:", fold["validation"])
        print("Test:", fold["test"])
        all_rows.extend(run_fold(fold, per_scene, model_module, device))

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {"folds": folds, "models": {}}
    for model_name in sorted({row["model"] for row in all_rows}):
        model_rows = [row for row in all_rows if row["model"] == model_name]
        summary["models"][model_name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in model_rows])),
                "std": float(np.std([row[metric] for row in model_rows], ddof=1)),
                "values": [row[metric] for row in model_rows],
            }
            for metric in ("f1", "iou", "precision", "recall")
        }

    summary["protocol"] = {
        "fold_count": 6,
        "train_scenes_per_fold": 12,
        "validation_scenes_per_fold": 3,
        "test_scenes_per_fold": 3,
        "thresholds_selected_from_validation_only": True,
        "gate_supervision_weight": 0.3,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nCross-validation complete")
    for model_name, values in summary["models"].items():
        print(
            f"{model_name:<28} "
            f"F1={values['f1']['mean']:.4f} +/- {values['f1']['std']:.4f} "
            f"IoU={values['iou']['mean']:.4f} +/- {values['iou']['std']:.4f}"
        )
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
