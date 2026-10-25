"""Diagnostic validation calibration analysis for the cached fusion model."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
import torch


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache_features"
WEIGHTS_DIR = ROOT / "weights"
CHECKPOINT_PATH = WEIGHTS_DIR / "best_fusion_model.pth"
BATCH_SIZE = 256


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(name: str) -> dict[str, np.ndarray]:
    with np.load(CACHE_DIR / f"{name}_features.npz") as data:
        return {key: data[key] for key in data.files}


@torch.no_grad()
def run_fusion(model, cache: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    count = len(cache["labels"])

    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        output = model(
            torch.from_numpy(cache["sar_features"][start:end]).float().to(device),
            torch.from_numpy(cache["hsi_features"][start:end]).float().to(device),
            torch.from_numpy(cache["hsi_logits"][start:end]).float().to(device),
            torch.from_numpy(cache["hsi_confidence"][start:end]).float().to(device),
        )
        predictions.append(output["p_fused"].detach().cpu().numpy().reshape(-1))

    return np.concatenate(predictions)


def f1_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return f1_score(labels, (probs >= threshold).astype(int), zero_division=0)


def best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.01,
    hi: float = 0.99,
    step: float = 0.005,
) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    threshold = lo

    while threshold <= hi + 1e-9:
        score = f1_at_threshold(probs, labels, threshold)
        if score > best_f1:
            best_t = threshold
            best_f1 = score
        threshold += step

    return best_t, best_f1


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_module = load_module("fusion_model", ROOT / "04_fusion_model.py")
    model = model_module.build_fusion_model().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    validation = load_split("validation")
    test = load_split("test")

    validation_probs = run_fusion(model, validation, device)
    test_probs = run_fusion(model, test, device)
    validation_labels = validation["labels"]
    test_labels = test["labels"]

    print("=" * 70)
    print("1) WIDE THRESHOLD SEARCH ON VALIDATION")
    print("=" * 70)
    wide_t, wide_f1 = best_threshold(validation_probs, validation_labels)
    wide_predictions = (test_probs >= wide_t).astype(int)
    print(f"  Best val threshold (0.01-0.99) : {wide_t:.3f} (val F1={wide_f1:.4f})")
    print(f"  Test F1 at this threshold      : {f1_score(test_labels, wide_predictions, zero_division=0):.4f}")
    print(f"  Test IoU at this threshold     : {jaccard_score(test_labels, wide_predictions, zero_division=0):.4f}")

    print("\n" + "=" * 70)
    print("2) PLATT SCALING FIT ON VALIDATION")
    print("=" * 70)
    calibrator = LogisticRegression()
    calibrator.fit(validation_probs.reshape(-1, 1), validation_labels)
    test_calibrated = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    calibrated_predictions = (test_calibrated >= 0.5).astype(int)
    print(f"  Test F1 at calibrated p>=0.5   : {f1_score(test_labels, calibrated_predictions, zero_division=0):.4f}")
    print(f"  Test IoU at calibrated p>=0.5  : {jaccard_score(test_labels, calibrated_predictions, zero_division=0):.4f}")

    print("\n" + "=" * 70)
    print("3) THRESHOLD-INDEPENDENT TEST METRICS")
    print("=" * 70)
    print(f"  Test ROC-AUC : {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"  Test PR-AUC  : {average_precision_score(test_labels, test_probs):.4f}")

    print("\n" + "=" * 70)
    print("4) ORIGINAL CHECKPOINT THRESHOLD")
    print("=" * 70)
    checkpoint_t = float(checkpoint.get("decision_threshold", 0.50))
    checkpoint_predictions = (test_probs >= checkpoint_t).astype(int)
    print(f"  Checkpoint threshold           : {checkpoint_t:.3f}")
    print(f"  Test F1 at checkpoint threshold: {f1_score(test_labels, checkpoint_predictions, zero_division=0):.4f}")
    print(f"  Test IoU at checkpoint threshold: {jaccard_score(test_labels, checkpoint_predictions, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
