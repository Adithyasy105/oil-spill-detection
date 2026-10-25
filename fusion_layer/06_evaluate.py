"""
STAGE 4 — FINAL TEST EVALUATION
06_evaluate.py

Purpose
-------
Evaluate the frozen cached experts and the trained
RegistrationFreeFusionNet on the untouched test cache.

IMPORTANT EXPERIMENTAL RULE
---------------------------

The test set is NEVER used for:

    - training
    - early stopping
    - checkpoint selection
    - learning-rate scheduling
    - architecture selection
    - hyperparameter tuning
    - threshold tuning

The best fusion checkpoint must already have been selected
using validation F1 only.

Cache contract
--------------

test_features.npz contains:

    sar_features       : (N, 42)
    hsi_features       : (N, 64)
    hsi_logits         : (N, 1)
    hsi_confidence     : (N, 1)
    labels             : (N,)
    scene_indices      : (N,)

SAR feature order
-----------------

The cache builder must define the following order:

    0 -> center probability
    1 -> local mean probability
    2 -> local standard deviation
    3 -> local minimum probability
    4 -> local maximum probability
    5 -> local contrast
    6 -> scene mean probability
    7 -> scene standard deviation probability
    8 -> scene positive fraction
    9 -> local-to-scene contrast
    10-41 -> projected SAR bottleneck embedding

The SAR-only baseline uses feature 0:

    sar_features[:, 0]

This is valid only because feature 0 is the SAR expert's
center probability.

HSI baseline
------------

The HSI expert baseline is obtained from:

    sigmoid(hsi_logits)

Fusion model
------------

Inputs:

    sar_features      : (N, 42)
    hsi_features      : (N, 64)
    hsi_logits        : (N, 1)
    hsi_confidence    : (N, 1)

Output dictionary must contain:

    p_fused

Outputs
-------

evaluation/
    evaluation_summary.json
    comparison.csv
    predictions.npz
    confusion_matrices.png
    roc_curves.png
    precision_recall_curves.png
    scene_metrics.csv

"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# PATH SETUP
# =============================================================================

THIS_FILE = Path(__file__).resolve()

FUSION_DIR = THIS_FILE.parent

PROJECT_ROOT = FUSION_DIR.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))


# =============================================================================
# DYNAMIC MODULE LOADING
# =============================================================================

from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


def load_python_module(
    name: str,
    path: Path,
):

    spec = spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module:\n{path}"
        )

    module = module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


CFG = load_python_module(
    "fusion_config",
    FUSION_DIR / "00_config.py",
)

FUSION_MODEL_MODULE = load_python_module(
    "fusion_model",
    FUSION_DIR / "04_fusion_model.py",
)


# =============================================================================
# CONFIGURATION
# =============================================================================


def get_config(
    names: List[str],
    default: Any = None,
) -> Any:

    for name in names:

        if hasattr(CFG, name):

            return getattr(
                CFG,
                name,
            )

    return default


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = int(
    get_config(
        [
            "SEED",
            "RANDOM_SEED",
        ],
        default=42,
    )
)

BATCH_SIZE = int(
    get_config(
        [
            "EVAL_BATCH_SIZE",
            "FUSION_BATCH_SIZE",
            "BATCH_SIZE",
        ],
        default=512,
    )
)

USE_AMP = bool(
    get_config(
        [
            "AMP",
            "USE_AMP",
        ],
        default=True,
    )
)

CACHE_DIR = Path(
    os.environ.get(
        "FUSION_CACHE_DIR",
        get_config(
            ["FUSION_CACHE_DIR"],
            default=FUSION_DIR / "cache_features",
        ),
    )
).resolve()

TEST_CACHE_PATH = (
    CACHE_DIR
    / "test_features.npz"
)

CHECKPOINT_DIR = Path(
    os.environ.get(
        "FUSION_CHECKPOINT_DIR",
        get_config(
            ["FUSION_CHECKPOINT_DIR"],
            default=FUSION_DIR / "weights",
        ),
    )
).resolve()

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best_fusion_model.pth"
)

OUTPUT_DIR = (
    Path(
        os.environ.get(
            "FUSION_EVALUATION_DIR",
            FUSION_DIR / "evaluation",
        )
    )
).resolve()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PATHS
# =============================================================================

SUMMARY_PATH = (
    OUTPUT_DIR
    / "evaluation_summary.json"
)

COMPARISON_CSV_PATH = (
    OUTPUT_DIR
    / "comparison.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "predictions.npz"
)

SCENE_METRICS_PATH = (
    OUTPUT_DIR
    / "scene_metrics.csv"
)

CONFUSION_FIGURE_PATH = (
    OUTPUT_DIR
    / "confusion_matrices.png"
)

ROC_FIGURE_PATH = (
    OUTPUT_DIR
    / "roc_curves.png"
)

PR_FIGURE_PATH = (
    OUTPUT_DIR
    / "precision_recall_curves.png"
)


# =============================================================================
# PRINT HELPERS
# =============================================================================


def banner(
    title: str,
) -> None:

    print()

    print("=" * 88)

    print(title)

    print("=" * 88)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =============================================================================
# LOAD TEST CACHE
# =============================================================================


def load_test_cache() -> Dict[str, np.ndarray]:

    banner(
        "LOADING UNTOUCHED TEST CACHE"
    )

    if not TEST_CACHE_PATH.exists():

        raise FileNotFoundError(
            "\nTest cache not found:\n"
            f"{TEST_CACHE_PATH}"
        )

    print(
        f"\nLoading:\n"
        f"  {TEST_CACHE_PATH}"
    )

    data = np.load(
        TEST_CACHE_PATH,
        allow_pickle=False,
    )

    required_keys = [
        "sar_features",
        "hsi_features",
        "hsi_logits",
        "hsi_confidence",
        "labels",
        "scene_indices",
    ]

    for key in required_keys:

        if key not in data.files:

            raise KeyError(
                "\nMissing required key:\n"
                f"{key}"
            )

    cache = {
        key: np.asarray(
            data[key]
        )
        for key in required_keys
    }

    sar_features = cache[
        "sar_features"
    ]

    hsi_features = cache[
        "hsi_features"
    ]

    hsi_logits = cache[
        "hsi_logits"
    ]

    hsi_confidence = cache[
        "hsi_confidence"
    ]

    labels = cache[
        "labels"
    ]

    scene_indices = cache[
        "scene_indices"
    ]

    # -------------------------------------------------------------
    # SHAPE VALIDATION
    # -------------------------------------------------------------

    if sar_features.ndim != 2:

        raise ValueError(
            f"sar_features must be 2-D, "
            f"got {sar_features.shape}"
        )

    if sar_features.shape[1] != 42:

        raise ValueError(
            "\nSAR feature dimension mismatch.\n"
            f"Expected 42.\n"
            f"Received {sar_features.shape[1]}"
        )

    if hsi_features.ndim != 2:

        raise ValueError(
            f"hsi_features must be 2-D, "
            f"got {hsi_features.shape}"
        )

    if hsi_features.shape[1] != 64:

        raise ValueError(
            "\nHSI feature dimension mismatch.\n"
            f"Expected 64.\n"
            f"Received {hsi_features.shape[1]}"
        )

    num_samples = (
        sar_features.shape[0]
    )

    arrays = [
        hsi_features,
        hsi_logits,
        hsi_confidence,
        labels,
        scene_indices,
    ]

    for array in arrays:

        if array.shape[0] != num_samples:

            raise ValueError(
                "\nCache arrays have inconsistent "
                "sample counts."
            )

    # -------------------------------------------------------------
    # RESHAPE
    # -------------------------------------------------------------

    hsi_logits = hsi_logits.reshape(
        num_samples,
        1,
    )

    hsi_confidence = hsi_confidence.reshape(
        num_samples,
        1,
    )

    labels = labels.reshape(
        -1
    ).astype(
        np.int64
    )

    scene_indices = scene_indices.reshape(
        -1
    ).astype(
        np.int64
    )

    # -------------------------------------------------------------
    # FINITE CHECK
    # -------------------------------------------------------------

    numeric_arrays = {
        "sar_features": sar_features,
        "hsi_features": hsi_features,
        "hsi_logits": hsi_logits,
        "hsi_confidence": hsi_confidence,
    }

    for name, array in numeric_arrays.items():

        if not np.isfinite(array).all():

            raise ValueError(
                f"\nNon-finite values found in:\n{name}"
            )

    if not np.isin(
        labels,
        [0, 1],
    ).all():

        raise ValueError(
            "\nLabels must contain only 0 and 1."
        )

    cache["hsi_logits"] = hsi_logits
    cache["hsi_confidence"] = hsi_confidence
    cache["labels"] = labels
    cache["scene_indices"] = scene_indices

    print(
        f"\nSamples         : {num_samples:,}"
    )

    print(
        f"SAR features    : {sar_features.shape}"
    )

    print(
        f"HSI features    : {hsi_features.shape}"
    )

    print(
        f"HSI logits      : {hsi_logits.shape}"
    )

    print(
        f"HSI confidence  : "
        f"{hsi_confidence.shape}"
    )

    positives = int(
        labels.sum()
    )

    print(
        f"Positives       : {positives:,} "
        f"({positives / len(labels):.4f})"
    )

    print(
        f"Unique scenes   : "
        f"{len(np.unique(scene_indices))}"
    )

    return cache


# =============================================================================
# NUMPY SIGMOID
# =============================================================================


def sigmoid_numpy(
    x: np.ndarray,
) -> np.ndarray:

    x = np.clip(
        x,
        -50.0,
        50.0,
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-x)
        )
    )


# =============================================================================
# BASELINE PROBABILITIES
# =============================================================================


def get_sar_probabilities(
    sar_features: np.ndarray,
) -> np.ndarray:
    """
    SAR-only baseline.

    Cached SAR feature order:

        0 -> center probability
        1 -> local mean probability
        2 -> local standard deviation
        3 -> local minimum probability
        4 -> local maximum probability
        5 -> local contrast

    Therefore:

        sar_features[:, 0]

    is used as the SAR expert's local center prediction.
    """

    probability = (
        sar_features[:, 0]
        .astype(np.float64)
    )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    return probability


def get_hsi_probabilities(
    hsi_logits: np.ndarray,
) -> np.ndarray:
    """
    HSI-only baseline.

    The cached HSI output is a logit.

    Convert:

        logit -> sigmoid(logit) -> probability
    """

    logits = hsi_logits.reshape(
        -1
    ).astype(
        np.float64
    )

    probability = sigmoid_numpy(
        logits
    )

    return probability


# =============================================================================
# LOAD FUSION MODEL
# =============================================================================


def build_fusion_model() -> Tuple[nn.Module, float]:

    banner(
        "LOADING BEST FUSION MODEL"
    )

    if not hasattr(
        FUSION_MODEL_MODULE,
        "RegistrationFreeFusionNet",
    ):

        raise AttributeError(
            "\n04_fusion_model.py does not contain "
            "RegistrationFreeFusionNet."
        )

    if not BEST_CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nBest checkpoint not found:\n"
            f"{BEST_CHECKPOINT_PATH}"
        )

    FusionNet = (
        FUSION_MODEL_MODULE
        .RegistrationFreeFusionNet
    )

    model = FusionNet()

    model = model.to(
        DEVICE
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "\nCheckpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    decision_threshold = float(
        checkpoint.get(
            "decision_threshold",
            0.50,
        )
    )

    model.eval()

    print(
        f"\nCheckpoint:\n"
        f"  {BEST_CHECKPOINT_PATH}"
    )

    print(
        f"\nSelected epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Decision threshold: {decision_threshold:.2f}"
    )

    metrics = checkpoint.get(
        "metrics",
        {},
    )

    if metrics:

        print(
            f"Validation F1: "
            f"{metrics.get('f1', 'unknown')}"
        )

        print(
            f"Validation IoU: "
            f"{metrics.get('iou', 'unknown')}"
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    return model, decision_threshold


# =============================================================================
# FUSION FORWARD
# =============================================================================


@torch.no_grad()
def get_fusion_probabilities(
    model: nn.Module,
    sar_features: np.ndarray,
    hsi_features: np.ndarray,
    hsi_logits: np.ndarray,
    hsi_confidence: np.ndarray,
) -> np.ndarray:

    banner(
        "RUNNING FUSION INFERENCE"
    )

    num_samples = (
        sar_features.shape[0]
    )

    predictions = []

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    for start in range(
        0,
        num_samples,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples,
        )

        sar_batch = torch.from_numpy(
            sar_features[start:end]
            .astype(np.float32)
        ).to(
            DEVICE,
            non_blocking=True,
        )

        hsi_batch = torch.from_numpy(
            hsi_features[start:end]
            .astype(np.float32)
        ).to(
            DEVICE,
            non_blocking=True,
        )

        logit_batch = torch.from_numpy(
            hsi_logits[start:end]
            .astype(np.float32)
        ).to(
            DEVICE,
            non_blocking=True,
        )

        confidence_batch = torch.from_numpy(
            hsi_confidence[start:end]
            .astype(np.float32)
        ).to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):

            output = model(
                sar_batch,
                hsi_batch,
                logit_batch,
                confidence_batch,
            )

            if not isinstance(
                output,
                dict,
            ):

                raise RuntimeError(
                    "\nFusion model must return "
                    "a dictionary."
                )

            if "p_fused" not in output:

                raise KeyError(
                    "\nFusion output does not contain "
                    "'p_fused'."
                )

            probability = output[
                "p_fused"
            ]

        probability = (
            probability
            .float()
            .view(-1)
            .cpu()
            .numpy()
        )

        predictions.append(
            probability
        )

    probabilities = np.concatenate(
        predictions,
        axis=0,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    print(
        f"\nFusion predictions: "
        f"{probabilities.shape}"
    )

    return probabilities


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:

    labels = labels.astype(
        np.int64
    )

    probabilities = np.clip(
        probabilities.astype(np.float64),
        0.0,
        1.0,
    )

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    iou = (
        tp
        /
        max(
            tp + fp + fn,
            1,
        )
    )

    try:

        roc_auc = roc_auc_score(
            labels,
            probabilities,
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            labels,
            probabilities,
        )

    except ValueError:

        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# SCENE-WISE METRICS
# =============================================================================


def compute_scene_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scene_indices: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:

    rows = []

    unique_scenes = np.unique(
        scene_indices
    )

    for scene_id in unique_scenes:

        mask = (
            scene_indices == scene_id
        )

        scene_labels = labels[mask]

        scene_probabilities = (
            probabilities[mask]
        )

        metrics = compute_metrics(
            scene_labels,
            scene_probabilities,
            threshold=threshold,
        )

        row = {
            "model": model_name,
            "scene_index": int(scene_id),
            "samples": int(mask.sum()),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return rows


# =============================================================================
# SAVE CSV
# =============================================================================


def save_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# CONFUSION MATRIX FIGURE
# =============================================================================


def plot_confusion_matrices(
    labels: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> None:

    model_names = list(
        model_probabilities.keys()
    )

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(
            5 * len(model_names),
            4,
        ),
    )

    if len(model_names) == 1:
        axes = [axes]

    for axis, name in zip(
        axes,
        model_names,
    ):

        probabilities = (
            model_probabilities[name]
        )

        predictions = (
            probabilities >= threshold
        ).astype(
            np.int64
        )

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        )

        image = axis.imshow(
            matrix
        )

        axis.set_title(
            name
        )

        axis.set_xlabel(
            "Predicted"
        )

        axis.set_ylabel(
            "True"
        )

        axis.set_xticks(
            [0, 1]
        )

        axis.set_yticks(
            [0, 1]
        )

        for row in range(2):

            for col in range(2):

                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# ROC CURVES
# =============================================================================


def plot_roc_curves(
    labels: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
) -> None:

    plt.figure(
        figsize=(7, 6)
    )

    for name, probabilities in (
        model_probabilities.items()
    ):

        fpr, tpr, _ = roc_curve(
            labels,
            probabilities,
        )

        auc = roc_auc_score(
            labels,
            probabilities,
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ROC_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# PRECISION-RECALL CURVES
# =============================================================================


def plot_pr_curves(
    labels: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
) -> None:

    plt.figure(
        figsize=(7, 6)
    )

    for name, probabilities in (
        model_probabilities.items()
    ):

        precision, recall, _ = (
            precision_recall_curve(
                labels,
                probabilities,
            )
        )

        ap = average_precision_score(
            labels,
            probabilities,
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.4f})",
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PR_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    set_seed(
        SEED
    )

    banner(
        "STAGE 4 — FINAL TEST EVALUATION"
    )

    print(
        f"\nDevice     : {DEVICE}"
    )

    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        f"Seed       : {SEED}"
    )

    print(
        "\nEvaluation contract:"
    )

    print(
        "  Training data       : NOT LOADED"
    )

    print(
        "  Validation data     : NOT USED"
    )

    print(
        "  Test cache          : EVALUATION ONLY"
    )

    print(
        "  Checkpoint          : BEST VALIDATION MODEL"
    )

    print(
        "  Decision threshold  : LOADED FROM VALIDATION CHECKPOINT"
    )

    # -------------------------------------------------------------
    # LOAD TEST CACHE
    # -------------------------------------------------------------

    cache = load_test_cache()

    sar_features = cache[
        "sar_features"
    ]

    hsi_features = cache[
        "hsi_features"
    ]

    hsi_logits = cache[
        "hsi_logits"
    ]

    hsi_confidence = cache[
        "hsi_confidence"
    ]

    labels = cache[
        "labels"
    ]

    scene_indices = cache[
        "scene_indices"
    ]

    # -------------------------------------------------------------
    # SAR BASELINE
    # -------------------------------------------------------------

    banner(
        "SAR-ONLY BASELINE"
    )

    sar_probabilities = (
        get_sar_probabilities(
            sar_features
        )
    )

    # -------------------------------------------------------------
    # HSI BASELINE
    # -------------------------------------------------------------

    banner(
        "HSI-ONLY BASELINE"
    )

    hsi_probabilities = (
        get_hsi_probabilities(
            hsi_logits
        )
    )

    # -------------------------------------------------------------
    # FUSION MODEL
    # -------------------------------------------------------------

    fusion_model, decision_threshold = (
        build_fusion_model()
    )

    fusion_probabilities = (
        get_fusion_probabilities(
            model=fusion_model,
            sar_features=sar_features,
            hsi_features=hsi_features,
            hsi_logits=hsi_logits,
            hsi_confidence=hsi_confidence,
        )
    )

    # -------------------------------------------------------------
    # MODEL COLLECTION
    # -------------------------------------------------------------

    model_probabilities = {
        "SAR Only":
            sar_probabilities,

        "HSI Only":
            hsi_probabilities,

        "Registration-Free Fusion":
            fusion_probabilities,
    }

    # -------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------

    banner(
        "FINAL TEST METRICS"
    )

    comparison_rows = []

    all_scene_rows = []

    all_metrics = {}

    for model_name, probabilities in (
        model_probabilities.items()
    ):

        metrics = compute_metrics(
            labels,
            probabilities,
            threshold=decision_threshold,
        )

        all_metrics[
            model_name
        ] = metrics

        row = {
            "model": model_name,
        }

        row.update(
            metrics
        )

        comparison_rows.append(
            row
        )

        scene_rows = (
            compute_scene_metrics(
                labels=labels,
                probabilities=probabilities,
                scene_indices=scene_indices,
                model_name=model_name,
                threshold=decision_threshold,
            )
        )

        all_scene_rows.extend(
            scene_rows
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"  Accuracy  : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"  F1        : "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"  IoU       : "
            f"{metrics['iou']:.6f}"
        )

        print(
            f"  ROC-AUC   : "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"  PR-AUC    : "
            f"{metrics['pr_auc']:.6f}"
        )

        print(
            f"  TN={metrics['tn']:,} "
            f"FP={metrics['fp']:,} "
            f"FN={metrics['fn']:,} "
            f"TP={metrics['tp']:,}"
        )

    # -------------------------------------------------------------
    # SAVE COMPARISON
    # -------------------------------------------------------------

    save_csv(
        COMPARISON_CSV_PATH,
        comparison_rows,
    )

    save_csv(
        SCENE_METRICS_PATH,
        all_scene_rows,
    )

    # -------------------------------------------------------------
    # SAVE PREDICTIONS
    # -------------------------------------------------------------

    np.savez_compressed(
        PREDICTIONS_PATH,

        labels=labels,

        scene_indices=scene_indices,

        sar_probabilities=sar_probabilities,

        hsi_probabilities=hsi_probabilities,

        fusion_probabilities=fusion_probabilities,
    )

    # -------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------

    banner(
        "GENERATING EVALUATION FIGURES"
    )

    plot_confusion_matrices(
        labels,
        model_probabilities,
        threshold=decision_threshold,
    )

    plot_roc_curves(
        labels,
        model_probabilities,
    )

    plot_pr_curves(
        labels,
        model_probabilities,
    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    summary = {

        "device":
            str(DEVICE),

        "seed":
            SEED,

        "test_cache":
            str(TEST_CACHE_PATH),

        "checkpoint":
            str(BEST_CHECKPOINT_PATH),

        "threshold":
            decision_threshold,

        "num_samples":
            int(len(labels)),

        "positive_samples":
            int(labels.sum()),

        "negative_samples":
            int(
                len(labels)
                - labels.sum()
            ),

        "unique_scenes":
            int(
                len(
                    np.unique(
                        scene_indices
                    )
                )
            ),

        "sar_feature_order": [

            "center_probability",

            "local_mean_probability",

            "local_standard_deviation",

            "local_minimum_probability",

            "local_maximum_probability",

            "local_contrast",

        ],

        "models":
            all_metrics,

        "files": {

            "comparison_csv":
                str(
                    COMPARISON_CSV_PATH
                ),

            "scene_metrics_csv":
                str(
                    SCENE_METRICS_PATH
                ),

            "predictions":
                str(
                    PREDICTIONS_PATH
                ),

            "confusion_matrices":
                str(
                    CONFUSION_FIGURE_PATH
                ),

            "roc_curves":
                str(
                    ROC_FIGURE_PATH
                ),

            "precision_recall_curves":
                str(
                    PR_FIGURE_PATH
                ),

        },

    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # FINAL COMPARISON
    # -------------------------------------------------------------

    banner(
        "FINAL COMPARISON"
    )

    print()

    print(
        f"{'MODEL':<32}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
        f"{'ROC-AUC':>12}"
        f"{'PR-AUC':>12}"
    )

    print(
        "-" * 76
    )

    for row in comparison_rows:

        print(
            f"{row['model']:<32}"
            f"{row['f1']:>10.4f}"
            f"{row['iou']:>10.4f}"
            f"{row['roc_auc']:>12.4f}"
            f"{row['pr_auc']:>12.4f}"
        )

    print()

    print(
        "[SUCCESS] Final test evaluation complete."
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":

    main()