"""CAFE Fusion v2 — Confidence-Aware Fusion Engine.

Registration-free SAR + HSI fusion with:
1. Logit-space mixing: z_fusion = (1 - g) * z_hsi + g * z_sar
2. g initialized near 0 (bias = -2.0 -> g ≈ 0.12, pure HSI protected default)
3. Hard masked reliability loss on disagreement in correctness:
   mask = (sar_correct ^ hsi_correct)
   target = 1 if (sar_correct & ~hsi_correct) else 0
4. Small HSI regularization term: L_hsi_reg = mean(g)
5. Threshold selected strictly on validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Paths
EXPERIMENT_ROOT = Path(__file__).resolve().parent
FUSION_LAYER_ROOT = EXPERIMENT_ROOT.parent.parent
CACHE_ROOT = FUSION_LAYER_ROOT / "cache_features"
CV_SPLIT_ROOT = FUSION_LAYER_ROOT / "experiments" / "scene_cv_6fold"

# Constants
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
EPS_HSI_REG = 0.01
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


def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_source_caches() -> Dict[str, Dict[str, np.ndarray]]:
    result = {}
    for split in ("train", "validation", "test"):
        npz_path = CACHE_ROOT / f"{split}_features.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing cache file: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            result[split] = {key: np.asarray(data[key]) for key in data.files}
    return result


def split_by_scene(caches: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
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


def merge_scenes(scene_data: Dict[str, Dict[str, np.ndarray]], names: List[str]) -> Dict[str, np.ndarray]:
    merged = {
        key: np.concatenate([scene_data[name][key] for name in names], axis=0)
        for key in scene_data[names[0]]
    }
    merged["scene_indices"] = np.concatenate(
        [
            np.full(len(scene_data[name]["labels"]), index, dtype=np.int64)
            for index, name in enumerate(names)
        ],
        axis=0,
    )
    return merged


def make_folds() -> List[Dict[str, Any]]:
    folds = []
    for fold in range(6):
        test = [SCENES[(3 * fold + offset) % 18] for offset in range(3)]
        validation = [SCENES[(3 * fold + 3 + offset) % 18] for offset in range(3)]
        train = [scene for scene in SCENES if scene not in test + validation]
        folds.append({"fold": fold + 1, "train": train, "validation": validation, "test": test})
    return folds


def build_gate_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Build 11 registration-free features for the gate network."""
    p_sar = np.clip(data["sar_features"][:, 0], 1e-6, 1.0 - 1e-6)
    z_sar = np.log(p_sar / (1.0 - p_sar))
    
    z_hsi = data["hsi_logits"].reshape(-1)
    p_hsi = np.clip(1.0 / (1.0 + np.exp(-z_hsi)), 1e-6, 1.0 - 1e-6)
    
    abs_diff = np.abs(p_sar - p_hsi)
    signed_diff = p_sar - p_hsi
    hsi_conf = np.clip(data["hsi_confidence"].reshape(-1), 0.0, 1.0)
    
    # Entropy: -p*log(p) - (1-p)*log(1-p)
    hsi_ent = -(p_hsi * np.log(p_hsi) + (1.0 - p_hsi) * np.log(1.0 - p_hsi))
    sar_ent = -(p_sar * np.log(p_sar) + (1.0 - p_sar) * np.log(1.0 - p_sar))
    
    sar_local_mean = np.clip(data["sar_features"][:, 1], 0.0, 1.0)
    sar_local_std = np.clip(data["sar_features"][:, 2], 0.0, 1.0)
    sar_contrast = np.clip(data["sar_features"][:, 5], 0.0, 1.0)
    
    features = np.column_stack([
        p_sar, z_sar, p_hsi, z_hsi, abs_diff, signed_diff,
        hsi_conf, hsi_ent, sar_ent, sar_local_mean, sar_local_std
    ]).astype(np.float32)
    return features


class TinyReliabilityGate(nn.Module):
    """Compact reliability gate producing g in [0, 1]. Initialized near 0 for HSI protection."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize bias to -2.0 -> sigmoid(-2.0) ≈ 0.119 (HSI default anchor)
        nn.init.constant_(self.net[-1].bias, -2.0)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class CAFEFusionModel(nn.Module):
    """Logit-space Confidence-Aware Fusion Model."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, fusion_mode: str = "logit"):
        super().__init__()
        self.gate = TinyReliabilityGate(input_dim=input_dim, hidden_dim=hidden_dim)
        self.fusion_mode = fusion_mode

    def forward(
        self,
        gate_inputs: torch.Tensor,
        z_sar: torch.Tensor,
        z_hsi: torch.Tensor,
        p_sar: torch.Tensor,
        p_hsi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        g = self.gate(gate_inputs)  # (B, 1) in [0, 1]
        
        if self.fusion_mode == "logit":
            z_fused = (1.0 - g) * z_hsi + g * z_sar
            p_fused = torch.sigmoid(z_fused)
        elif self.fusion_mode == "prob":
            p_fused = (1.0 - g) * p_hsi + g * p_sar
            p_fused = torch.clamp(p_fused, 1e-6, 1.0 - 1e-6)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
            
        return {
            "p_fused": p_fused,
            "g": g,
            "p_sar": p_sar,
            "p_hsi": p_hsi,
        }


def dice_loss(probabilities: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (probabilities * labels).sum()
    denominator = probabilities.sum() + labels.sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0 + eps)


def compute_masked_reliability_loss(
    g: torch.Tensor,
    p_sar: torch.Tensor,
    p_hsi: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sar_pred = (p_sar >= threshold)
    hsi_pred = (p_hsi >= threshold)
    y_bool = (labels >= 0.5)

    sar_correct = (sar_pred == y_bool)
    hsi_correct = (hsi_pred == y_bool)

    sar_wins = sar_correct & ~hsi_correct
    hsi_wins = hsi_correct & ~sar_correct
    mask = (sar_wins | hsi_wins).squeeze(-1)

    sar_wins_count = int(sar_wins.sum().item())
    hsi_wins_count = int(hsi_wins.sum().item())
    masked_count = int(mask.sum().item())

    stats = {
        "sar_wins": sar_wins_count,
        "hsi_wins": hsi_wins_count,
        "disagreement_samples": masked_count,
    }

    if masked_count == 0:
        return torch.tensor(0.0, device=g.device, requires_grad=True), stats

    targets = sar_wins.float().view(-1, 1)[mask]
    g_masked = g[mask]
    rel_loss = F.binary_cross_entropy(g_masked, targets)
    return rel_loss, stats


def compute_soft_reliability_loss(
    g: torch.Tensor,
    p_sar: torch.Tensor,
    p_hsi: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sar_err = (p_sar - labels).abs()
    hsi_err = (p_hsi - labels).abs()
    sar_quality = torch.exp(-sar_err / temperature)
    hsi_quality = torch.exp(-hsi_err / temperature)
    soft_weights = torch.softmax(torch.cat([sar_quality, hsi_quality], dim=1), dim=1)
    target_sar_weight = soft_weights[:, 0:1]
    rel_loss = F.binary_cross_entropy(g, target_sar_weight)
    stats = {
        "mean_soft_target": float(target_sar_weight.mean().item()),
    }
    return rel_loss, stats


def total_cafe_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_rel: float,
    eps_hsi_reg: float = EPS_HSI_REG,
    supervision_mode: str = "hard",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    p_fused = outputs["p_fused"].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(p_fused, labels)
    dice = dice_loss(p_fused, labels)
    seg_loss = bce + dice

    if supervision_mode == "hard":
        rel_loss, stats = compute_masked_reliability_loss(
            outputs["g"], outputs["p_sar"], outputs["p_hsi"], labels
        )
    elif supervision_mode == "soft":
        rel_loss, stats = compute_soft_reliability_loss(
            outputs["g"], outputs["p_sar"], outputs["p_hsi"], labels
        )
    else:
        raise ValueError(f"Unknown supervision_mode: {supervision_mode}")

    hsi_reg_loss = outputs["g"].mean()

    total = seg_loss + lambda_rel * rel_loss + eps_hsi_reg * hsi_reg_loss
    return total, seg_loss, rel_loss, hsi_reg_loss, stats


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    include_ranking: bool = True,
) -> Dict[str, float]:
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
        try:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities))
            result["pr_auc"] = float(average_precision_score(labels, probabilities))
        except Exception:
            result["roc_auc"] = float("nan")
            result["pr_auc"] = float("nan")
    return result


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_f1 = -1.0
    best_thresh = 0.5
    for t in np.arange(0.01, 0.991, 0.005):
        preds = (probabilities >= t).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh


def evaluate_gate_diagnostics(
    g_arr: np.ndarray,
    p_sar: np.ndarray,
    p_hsi: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    sar_correct = (p_sar >= 0.5) == (labels >= 0.5)
    hsi_correct = (p_hsi >= 0.5) == (labels >= 0.5)
    
    sar_wins = sar_correct & ~hsi_correct
    hsi_wins = hsi_correct & ~sar_correct
    disagreement_mask = sar_wins | hsi_wins
    
    high_diff_mask = np.abs(p_sar - p_hsi) >= 0.40
    
    brier_score = float(np.mean((g_arr - (sar_wins.astype(float))) ** 2))
    
    gate_roc_auc = float("nan")
    if disagreement_mask.sum() > 0 and sar_wins.sum() > 0 and hsi_wins.sum() > 0:
        try:
            gate_roc_auc = float(roc_auc_score(sar_wins[disagreement_mask].astype(int), g_arr[disagreement_mask]))
        except Exception:
            gate_roc_auc = float("nan")

    return {
        "mean_g": float(np.mean(g_arr)),
        "std_g": float(np.std(g_arr)),
        "mean_g_high_disagreement": float(np.mean(g_arr[high_diff_mask])) if high_diff_mask.sum() > 0 else float("nan"),
        "mean_g_low_disagreement": float(np.mean(g_arr[~high_diff_mask])) if (~high_diff_mask).sum() > 0 else float("nan"),
        "mean_g_sar_wins": float(np.mean(g_arr[sar_wins])) if sar_wins.sum() > 0 else float("nan"),
        "mean_g_hsi_wins": float(np.mean(g_arr[hsi_wins])) if hsi_wins.sum() > 0 else float("nan"),
        "gate_roc_auc": gate_roc_auc,
        "brier_score": brier_score,
    }


def make_tensor_loader(data: Dict[str, np.ndarray], device: torch.device, shuffle: bool) -> DataLoader:
    gate_feats = torch.from_numpy(build_gate_features(data))
    
    p_sar_raw = np.clip(data["sar_features"][:, 0], 1e-6, 1.0 - 1e-6).astype(np.float32)
    z_sar_raw = np.log(p_sar_raw / (1.0 - p_sar_raw)).astype(np.float32)
    
    z_hsi_raw = data["hsi_logits"].reshape(-1).astype(np.float32)
    p_hsi_raw = np.clip(1.0 / (1.0 + np.exp(-z_hsi_raw)), 1e-6, 1.0 - 1e-6).astype(np.float32)
    
    labels_raw = data["labels"].astype(np.float32).reshape(-1, 1)

    dataset = TensorDataset(
        gate_feats,
        torch.from_numpy(z_sar_raw).view(-1, 1),
        torch.from_numpy(z_hsi_raw).view(-1, 1),
        torch.from_numpy(p_sar_raw).view(-1, 1),
        torch.from_numpy(p_hsi_raw).view(-1, 1),
        torch.from_numpy(labels_raw),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_rel: float,
    optimizer: Any | None,
    eps_hsi_reg: float = EPS_HSI_REG,
    supervision_mode: str = "hard",
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    
    total_losses = []
    seg_losses = []
    rel_losses = []
    
    all_probs = []
    all_g = []
    all_labels = []

    for gate_f, z_sar, z_hsi, p_sar, p_hsi, labels in loader:
        gate_f = gate_f.to(device)
        z_sar = z_sar.to(device)
        z_hsi = z_hsi.to(device)
        p_sar = p_sar.to(device)
        p_hsi = p_hsi.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(gate_f, z_sar, z_hsi, p_sar, p_hsi)
        loss, seg_loss, rel_loss, hsi_reg, _ = total_cafe_loss(
            outputs, labels, lambda_rel, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode
        )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_losses.append(loss.item())
        seg_losses.append(seg_loss.item())
        rel_losses.append(rel_loss.item())

        all_probs.append(outputs["p_fused"].detach().cpu().numpy().reshape(-1))
        all_g.append(outputs["g"].detach().cpu().numpy().reshape(-1))
        all_labels.append(labels.detach().cpu().numpy().reshape(-1))

    return (
        float(np.mean(total_losses)),
        float(np.mean(seg_losses)),
        float(np.mean(rel_losses)),
        np.concatenate(all_probs),
        np.concatenate(all_g),
        np.concatenate(all_labels),
    )


def train_cafe_model(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    lambda_rel: float,
    device: torch.device,
    checkpoint_path: Path,
    fusion_mode: str = "logit",
    eps_hsi_reg: float = EPS_HSI_REG,
    supervision_mode: str = "hard",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], CAFEFusionModel]:
    model = CAFEFusionModel(input_dim=11, hidden_dim=64, fusion_mode=fusion_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    train_loader = make_tensor_loader(train_data, device, shuffle=True)
    val_loader = make_tensor_loader(val_data, device, shuffle=False)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_seg, tr_rel, _, _, _ = run_epoch(
            model, train_loader, device, lambda_rel, optimizer, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode
        )
        val_loss, val_seg, val_rel, val_probs, val_g, val_labels = run_epoch(
            model, val_loader, device, lambda_rel, None, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode
        )

        thresh = select_threshold(val_labels, val_probs)
        val_metrics = classification_metrics(val_labels, val_probs, thresh, include_ranking=False)
        val_f1 = val_metrics["f1"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or patience == 0):
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"  Epoch {epoch:03d}/{EPOCHS:03d} | elapsed={elapsed:.1f}m | "
                f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_rel={val_rel:.4f} | "
                f"val_f1={val_f1:.4f} (best={best_f1:.4f} @ ep{best_epoch}) | mean_g={val_g.mean():.3f}"
            )

        if patience >= PATIENCE:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "lambda_rel": lambda_rel,
            "fusion_mode": fusion_mode,
            "eps_hsi_reg": eps_hsi_reg,
            "supervision_mode": supervision_mode,
        },
        checkpoint_path,
    )
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "epochs_completed": epoch}, model


def verify_identity_reproduction(scene_data: Dict[str, Dict[str, np.ndarray]]) -> bool:
    """Step 2: Verify that setting g=0 reproduces HSI-only metrics exactly."""
    print("\n" + "=" * 60)
    print("STEP 2: HSI IDENTITY REPRODUCTION VERIFICATION")
    print("=" * 60)
    
    all_matched = True
    for scene_name in SCENES[:3]:
        data = scene_data[scene_name]
        labels = data["labels"]
        z_hsi = data["hsi_logits"].reshape(-1)
        p_hsi = 1.0 / (1.0 + np.exp(-z_hsi))
        
        # Simulated g = 0
        g = np.zeros_like(p_hsi)
        p_sar = np.clip(data["sar_features"][:, 0], 1e-6, 1.0 - 1e-6)
        z_sar = np.log(p_sar / (1.0 - p_sar))
        z_fused = (1.0 - g) * z_hsi + g * z_sar
        p_fused = 1.0 / (1.0 + np.exp(-z_fused))
        
        f1_hsi = f1_score(labels, (p_hsi >= 0.5).astype(int), zero_division=0)
        f1_fused = f1_score(labels, (p_fused >= 0.5).astype(int), zero_division=0)
        
        diff = abs(f1_hsi - f1_fused)
        status = "PASSED" if diff < 1e-5 else "FAILED"
        print(f"  Scene {scene_name}: HSI F1={f1_hsi:.6f} | g=0 F1={f1_fused:.6f} | Diff={diff:.1e} [{status}]")
        if diff >= 1e-5:
            all_matched = False
            
    print("=" * 60 + "\n")
    return all_matched


def run_fold(
    fold_info: Dict[str, Any],
    scene_data: Dict[str, Dict[str, np.ndarray]],
    lambda_rel: float,
    device: torch.device,
    output_dir: Path,
    fusion_mode: str = "logit",
    eps_hsi_reg: float = EPS_HSI_REG,
    supervision_mode: str = "hard",
    verbose: bool = True,
) -> Dict[str, Any]:
    fold_idx = fold_info["fold"]
    train_scenes = fold_info["train"]
    val_scenes = fold_info["validation"]
    test_scenes = fold_info["test"]

    train = merge_scenes(scene_data, train_scenes)
    val = merge_scenes(scene_data, val_scenes)
    test = merge_scenes(scene_data, test_scenes)

    checkpoint_path = output_dir / f"fold_{fold_idx:02d}" / f"lambda_{lambda_rel:.2f}" / "best_cafe_model.pth"
    train_info, model = train_cafe_model(
        train, val, lambda_rel, device, checkpoint_path,
        fusion_mode=fusion_mode, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode, verbose=verbose
    )

    # 1. Validation threshold selection
    val_loader = make_tensor_loader(val, device, shuffle=False)
    _, _, _, val_probs, val_g, val_labels = run_epoch(
        model, val_loader, device, lambda_rel, None, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode
    )
    best_thresh = select_threshold(val_labels, val_probs)

    # 2. Test evaluation
    test_loader = make_tensor_loader(test, device, shuffle=False)
    _, _, _, test_probs, test_g, test_labels = run_epoch(
        model, test_loader, device, lambda_rel, None, eps_hsi_reg=eps_hsi_reg, supervision_mode=supervision_mode
    )

    # Metrics
    test_metrics = classification_metrics(test_labels, test_probs, best_thresh)
    
    # Baselines for comparison
    test_sar_prob = np.clip(test["sar_features"][:, 0], 0.0, 1.0)
    test_hsi_prob = 1.0 / (1.0 + np.exp(-test["hsi_logits"].reshape(-1)))
    
    val_sar_prob = np.clip(val["sar_features"][:, 0], 0.0, 1.0)
    val_hsi_prob = 1.0 / (1.0 + np.exp(-val["hsi_logits"].reshape(-1)))
    
    sar_thresh = select_threshold(val["labels"], val_sar_prob)
    hsi_thresh = select_threshold(val["labels"], val_hsi_prob)
    
    sar_metrics = classification_metrics(test_labels, test_sar_prob, sar_thresh)
    hsi_metrics = classification_metrics(test_labels, test_hsi_prob, hsi_thresh)

    # Oracle
    err_sar = np.abs(test_sar_prob - test_labels)
    err_hsi = np.abs(test_hsi_prob - test_labels)
    test_oracle_prob = np.where(err_sar < err_hsi, test_sar_prob, test_hsi_prob)
    oracle_metrics = classification_metrics(test_labels, test_oracle_prob, 0.5)

    # Gate diagnostics
    gate_diag = evaluate_gate_diagnostics(test_g, test_sar_prob, test_hsi_prob, test_labels)

    # Per-scene metrics
    per_scene_results = []
    for s_idx, s_name in enumerate(test_scenes):
        s_mask = test["scene_indices"] == s_idx
        s_res = classification_metrics(test_labels[s_mask], test_probs[s_mask], best_thresh)
        s_sar = classification_metrics(test_labels[s_mask], test_sar_prob[s_mask], sar_thresh)
        s_hsi = classification_metrics(test_labels[s_mask], test_hsi_prob[s_mask], hsi_thresh)
        s_res.update({
            "fold": fold_idx,
            "scene": s_name,
            "f1_sar": s_sar["f1"],
            "f1_hsi": s_hsi["f1"],
            "mean_g": float(test_g[s_mask].mean()),
        })
        per_scene_results.append(s_res)

    return {
        "fold": fold_idx,
        "threshold": best_thresh,
        "fusion": test_metrics,
        "sar": sar_metrics,
        "hsi": hsi_metrics,
        "oracle": oracle_metrics,
        "gate_diagnostics": gate_diag,
        "per_scene": per_scene_results,
        "train_info": train_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CAFE Fusion v2 Experiment Runner")
    parser.add_argument("--fold", type=int, choices=range(1, 7), help="Run one fold only")
    parser.add_argument("--lambda-rel", type=float, default=0.10, help="Reliability supervision weight")
    parser.add_argument("--eps-hsi-reg", type=float, default=EPS_HSI_REG, help="HSI regularization weight")
    parser.add_argument("--supervision-mode", type=str, default="hard", choices=["hard", "soft"], help="Supervision mode")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check on fold 1 and verify diagnostics")
    parser.add_argument("--fusion-mode", type=str, default="logit", choices=["logit", "prob"], help="Fusion mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--identity-check", action="store_true", help="Run HSI identity check only")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    caches = load_source_caches()
    scene_data = split_by_scene(caches)
    folds = make_folds()

    # Step 2 check if requested
    if args.identity_check:
        passed = verify_identity_reproduction(scene_data)
        sys.exit(0 if passed else 1)

    verify_identity_reproduction(scene_data)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = EXPERIMENT_ROOT / f"mode_{args.fusion_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_folds = [f for f in folds if args.fold is None or f["fold"] == args.fold]
    if args.sanity:
        selected_folds = [folds[0]]

    all_fold_results = []
    all_scene_results = []
    all_diag_results = []

    for fold_info in selected_folds:
        print(f"\n>>> Running Fold {fold_info['fold']}/6 (lambda_rel={args.lambda_rel:.2f}, mode={args.fusion_mode}, sup={args.supervision_mode}, eps={args.eps_hsi_reg})")
        res = run_fold(
            fold_info, scene_data, args.lambda_rel, device, output_dir,
            fusion_mode=args.fusion_mode, eps_hsi_reg=args.eps_hsi_reg, supervision_mode=args.supervision_mode
        )
        all_fold_results.append(res)
        all_scene_results.extend(res["per_scene"])
        
        diag = {"fold": res["fold"], **res["gate_diagnostics"]}
        all_diag_results.append(diag)

        f_res = res["fusion"]
        h_res = res["hsi"]
        s_res = res["sar"]
        o_res = res["oracle"]
        print(f"  Fold {res['fold']} Results:")
        print(f"    SAR Only:  F1={s_res['f1']:.4f} | IoU={s_res['iou']:.4f}")
        print(f"    HSI Only:  F1={h_res['f1']:.4f} | IoU={h_res['iou']:.4f}")
        print(f"    CAFE v2:   F1={f_res['f1']:.4f} | IoU={f_res['iou']:.4f} (d_HSI = {f_res['f1'] - h_res['f1']:+.4f})")
        print(f"    Oracle:    F1={o_res['f1']:.4f} | IoU={o_res['iou']:.4f}")
        print(f"    Gate Diag: Mean g={res['gate_diagnostics']['mean_g']:.3f} | ROC-AUC={res['gate_diagnostics']['gate_roc_auc']:.3f}")

    # Summary
    if len(all_fold_results) > 1:
        f1_fusion = [r["fusion"]["f1"] for r in all_fold_results]
        f1_hsi = [r["hsi"]["f1"] for r in all_fold_results]
        f1_sar = [r["sar"]["f1"] for r in all_fold_results]
        f1_oracle = [r["oracle"]["f1"] for r in all_fold_results]

        diff_vs_hsi = np.array(f1_fusion) - np.array(f1_hsi)
        p_t = float(ttest_rel(f1_fusion, f1_hsi).pvalue)
        p_w = float(wilcoxon(diff_vs_hsi).pvalue) if len(diff_vs_hsi) >= 6 else float("nan")

        oracle_gap = np.mean(f1_oracle) - np.mean(f1_hsi)
        gap_recovered = (np.mean(f1_fusion) - np.mean(f1_hsi)) / (oracle_gap + 1e-8)

        print("\n" + "=" * 60)
        print(f"6-FOLD CROSS-VALIDATION SUMMARY (lambda_rel={args.lambda_rel:.2f})")
        print("=" * 60)
        print(f"SAR Only:  F1 = {np.mean(f1_sar):.4f} +/- {np.std(f1_sar, ddof=1):.4f}")
        print(f"HSI Only:  F1 = {np.mean(f1_hsi):.4f} +/- {np.std(f1_hsi, ddof=1):.4f}")
        print(f"CAFE v2:   F1 = {np.mean(f1_fusion):.4f} +/- {np.std(f1_fusion, ddof=1):.4f} (Diff = {np.mean(diff_vs_hsi):+.4f})")
        print(f"Oracle:    F1 = {np.mean(f1_oracle):.4f} +/- {np.std(f1_oracle, ddof=1):.4f}")
        print(f"Oracle Gap Recovered: {gap_recovered * 100:.2f}%")
        print(f"Paired t-test p-value: {p_t:.4f} | Wilcoxon p-value: {p_w:.4f}")
        print("=" * 60)

        # Save summary
        summary = {
            "experiment": "CAFE Fusion v2",
            "lambda_rel": args.lambda_rel,
            "fusion_mode": args.fusion_mode,
            "mean_f1_fusion": float(np.mean(f1_fusion)),
            "std_f1_fusion": float(np.std(f1_fusion, ddof=1)),
            "mean_f1_hsi": float(np.mean(f1_hsi)),
            "std_f1_hsi": float(np.std(f1_hsi, ddof=1)),
            "mean_f1_sar": float(np.mean(f1_sar)),
            "std_f1_sar": float(np.std(f1_sar, ddof=1)),
            "oracle_gap_recovered_pct": float(gap_recovered * 100),
            "paired_t_p_value": p_t,
            "wilcoxon_p_value": p_w,
            "folds": [
                {
                    "fold": r["fold"],
                    "f1_fusion": r["fusion"]["f1"],
                    "f1_hsi": r["hsi"]["f1"],
                    "f1_sar": r["sar"]["f1"],
                    "f1_oracle": r["oracle"]["f1"],
                    "gate_diagnostics": r["gate_diagnostics"],
                }
                for r in all_fold_results
            ],
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Save per-scene CSV
    if all_scene_results:
        with open(output_dir / "per_scene_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_scene_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_scene_results)

    # Save diagnostics CSV
    if all_diag_results:
        with open(output_dir / "gate_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_diag_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_diag_results)


if __name__ == "__main__":
    main()
