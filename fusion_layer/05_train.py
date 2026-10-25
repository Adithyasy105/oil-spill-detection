# """
# STAGE 3 — FUSION TRAINING
# 05_train.py

# Purpose
# -------
# Train ONLY the registration-free fusion network.

# Data flow
# ---------
# 02_fusion_dataset.py
#         |
#         |-- HSI patch: (B, 5, 5, 224)
#         |
#         |-- SAR patch: (B, 5, 5)
#         |
#         +-- label:     (B,)
#                     |
#                     v
#           Frozen HSI preprocessing
#                     |
#                     v
#              PCA / spectral processor
#                     |
#                     v
#               (B, 1, 25, 5, 5)
#                     |
#                     v
#              Frozen HMOSDNet
#                     |
#                     +---- 64-D HSI feature
#                     |
#                     +---- HSI confidence
#                     |
#                     v

#           Frozen SAR expert
#                     |
#                     v
#              SAR probability map
#                     |
#                     v
#               6-D SAR evidence
#                     |
#                     v

#         RegistrationFreeFusionNet
#                     |
#                     v
#              Fused probability
#                     |
#                     v
#                  BCE loss

# IMPORTANT
# ---------
# SAR and HSI are NOT assumed to be pixel registered.

# The dataset independently samples:
#     - an HSI patch from one location in a scene
#     - a SAR patch independently from the same scene

# The label belongs only to the HSI sample.

# The SAR expert and HSI expert are frozen.

# ONLY RegistrationFreeFusionNet is optimized.
# """

# from __future__ import annotations

# import csv
# import json
# import random
# import sys
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# from torch.amp import GradScaler
# from tqdm import tqdm


# # =============================================================================
# # PATH SETUP
# # =============================================================================

# THIS_FILE = Path(__file__).resolve()
# FUSION_DIR = THIS_FILE.parent

# if str(FUSION_DIR) not in sys.path:
#     sys.path.insert(0, str(FUSION_DIR))


# from importlib.util import module_from_spec, spec_from_file_location


# def load_python_module(name: str, path: Path):
#     spec = spec_from_file_location(name, str(path))

#     if spec is None or spec.loader is None:
#         raise ImportError(
#             f"Could not import module:\n{path}"
#         )

#     module = module_from_spec(spec)
#     spec.loader.exec_module(module)

#     return module


# CFG = load_python_module(
#     "fusion_config",
#     FUSION_DIR / "00_config.py",
# )

# DATASET_MODULE = load_python_module(
#     "fusion_dataset",
#     FUSION_DIR / "02_fusion_dataset.py",
# )

# EXPERTS = load_python_module(
#     "fusion_experts",
#     FUSION_DIR / "03_load_experts.py",
# )

# FUSION_MODEL_MODULE = load_python_module(
#     "fusion_model",
#     FUSION_DIR / "04_fusion_model.py",
# )


# # =============================================================================
# # CONFIGURATION HELPERS
# # =============================================================================

# def get_config(
#     names: List[str],
#     default: Any = None,
#     required: bool = False,
# ) -> Any:

#     for name in names:
#         if hasattr(CFG, name):
#             return getattr(CFG, name)

#     if required:
#         raise AttributeError(
#             "\nMissing required configuration.\n"
#             f"Expected one of:\n{names}"
#         )

#     return default


# # =============================================================================
# # TRAINING CONFIGURATION
# # =============================================================================

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# BATCH_SIZE = int(
#     get_config(
#         ["BATCH_SIZE", "FUSION_BATCH_SIZE"],
#         default=8,
#     )
# )

# NUM_EPOCHS = int(
#     get_config(
#         ["EPOCHS", "FUSION_EPOCHS"],
#         default=60,
#     )
# )

# LEARNING_RATE = float(
#     get_config(
#         ["LEARNING_RATE", "FUSION_LEARNING_RATE"],
#         default=1e-4,
#     )
# )

# WEIGHT_DECAY = float(
#     get_config(
#         ["WEIGHT_DECAY", "FUSION_WEIGHT_DECAY"],
#         default=1e-4,
#     )
# )

# USE_AMP = bool(
#     get_config(
#         ["AMP", "USE_AMP"],
#         default=True,
#     )
# )

# NUM_WORKERS = int(
#     get_config(
#         ["NUM_WORKERS", "FUSION_NUM_WORKERS"],
#         default=0,
#     )
# )

# SEED = int(
#     get_config(
#         ["SEED", "RANDOM_SEED"],
#         default=42,
#     )
# )

# CHECKPOINT_DIR = Path(
#     get_config(
#         ["FUSION_CHECKPOINT_DIR"],
#         default=FUSION_DIR / "weights",
#     )
# ).resolve()

# CHECKPOINT_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )

# BEST_CHECKPOINT_PATH = (
#     CHECKPOINT_DIR / "best_fusion_model.pth"
# )

# LAST_CHECKPOINT_PATH = (
#     CHECKPOINT_DIR / "last_fusion_model.pth"
# )

# HISTORY_JSON_PATH = (
#     CHECKPOINT_DIR / "fusion_training_history.json"
# )

# HISTORY_CSV_PATH = (
#     CHECKPOINT_DIR / "fusion_training_history.csv"
# )

# SUMMARY_PATH = (
#     CHECKPOINT_DIR / "fusion_training_summary.json"
# )


# # =============================================================================
# # PRINT HELPERS
# # =============================================================================

# def banner(title: str) -> None:

#     print()
#     print("=" * 88)
#     print(title)
#     print("=" * 88)


# # =============================================================================
# # REPRODUCIBILITY
# # =============================================================================

# def set_seed(seed: int) -> None:

#     random.seed(seed)

#     np.random.seed(seed)

#     torch.manual_seed(seed)

#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True


# # =============================================================================
# # HSI PREPROCESSING
# # =============================================================================

# def preprocess_hsi_batch(
#     hsi_patches: torch.Tensor,
#     processor: Any,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Convert:

#         (B, 5, 5, 224)

#     into the exact HMOSDNet input:

#         (B, 1, 25, 5, 5)

#     The spectral processor is applied independently to every
#     HSI pixel spectrum.

#     No SAR information is used here.
#     """

#     if hsi_patches.ndim != 4:
#         raise ValueError(
#             "\nExpected HSI batch shape:\n"
#             "(B, 5, 5, 224)\n"
#             f"Received:\n{tuple(hsi_patches.shape)}"
#         )

#     batch_size, patch_h, patch_w, num_bands = (
#         hsi_patches.shape
#     )

#     if patch_h != 5 or patch_w != 5:
#         raise ValueError(
#             "\nHSI patch spatial shape must be 5x5.\n"
#             f"Received: {patch_h}x{patch_w}"
#         )

#     # Move to CPU because sklearn/joblib processor works with NumPy.
#     hsi_np = (
#         hsi_patches
#         .detach()
#         .cpu()
#         .numpy()
#         .astype(np.float32)
#     )

#     # The corrected SpectralProcessor expects an HxWxC cube, so transform
#     # each local 5x5 patch without flattening away its spatial dimensions.
#     transformed = np.stack(
#         [processor.transform(patch) for patch in hsi_np],
#         axis=0,
#     )

#     transformed = np.asarray(
#         transformed,
#         dtype=np.float32,
#     )

#     if transformed.ndim != 4:
#         raise RuntimeError(
#             "\nSpectral processor returned invalid shape:\n"
#             f"{transformed.shape}"
#         )

#     processed_bands = transformed.shape[3]

#     if processed_bands != 25:
#         raise RuntimeError(
#             "\nHSI spectral processor output mismatch.\n"
#             "HMOSDNet expects 25 spectral bands.\n"
#             f"Processor returned: {processed_bands}"
#         )

#     # (B, 5, 5, 25)
#     #       ->
#     # (B, 25, 5, 5)
#     transformed = np.transpose(
#         transformed,
#         (0, 3, 1, 2),
#     )

#     hsi_tensor = torch.from_numpy(
#         np.ascontiguousarray(transformed)
#     )

#     # (B, 25, 5, 5)
#     #       ->
#     # (B, 1, 25, 5, 5)
#     hsi_tensor = hsi_tensor.unsqueeze(1)

#     return hsi_tensor.to(
#         device,
#         non_blocking=True,
#     )


# # =============================================================================
# # HSI FEATURE EXTRACTION
# # =============================================================================

# @torch.no_grad()
# def extract_hsi_features(
#     hsi_model: nn.Module,
#     hsi_input: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     HMOSDNet input:

#         (B, 1, 25, 5, 5)

#     Model returns:

#         logits
#         features
#         None
#         confidence

#     The model's feature tensor is:

#         (B, 64, 25)

#     We aggregate the 25 internal spatial positions using mean pooling:

#         (B, 64)

#     This gives the exact 64-D HSI feature expected by
#     RegistrationFreeFusionNet.

#     No positional relationship with SAR is assumed.
#     """

#     hsi_model.eval()

#     logits, features, _, confidence = hsi_model(
#         hsi_input,
#         return_features=True,
#     )

#     if features.ndim == 3:
#         # (B, 64, 25)
#         #       ->
#         # (B, 64)
#         hsi_features = features.mean(dim=-1)

#     elif features.ndim == 2:
#         hsi_features = features

#     else:
#         raise RuntimeError(
#             "\nUnexpected HSI feature shape:\n"
#             f"{tuple(features.shape)}"
#         )

#     if hsi_features.shape[1] != 64:
#         raise RuntimeError(
#             "\nHSI feature dimension mismatch.\n"
#             "Fusion model expects 64-D HSI features.\n"
#             f"Received: {hsi_features.shape[1]}"
#         )

#     confidence = confidence.reshape(
#         confidence.shape[0],
#         -1,
#     )

#     confidence = confidence.mean(
#         dim=1,
#         keepdim=True,
#     )

#     return hsi_features, logits.reshape(logits.shape[0], -1).mean(dim=1, keepdim=True), confidence


# # =============================================================================
# # SAR INPUT PREPARATION
# # =============================================================================

# def prepare_sar_input(
#     sar_patches: torch.Tensor,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Dataset SAR patch:

#         (B, 5, 5)

#     Convert to the SAR expert's original input contract:

#         (B, 3, 256, 256)

#     The grayscale patch is dB/z-score normalized and replicated to three
#     channels, matching sar_layer/02_dataset.py.
#     """

#     if sar_patches.ndim != 3:
#         raise ValueError(
#             "\nExpected SAR patch shape:\n"
#             "(B, 5, 5)\n"
#             f"Received:\n{tuple(sar_patches.shape)}"
#         )

#     sar_np = sar_patches.detach().cpu().numpy().astype(np.float32)
#     sar_inputs = []
#     for patch in sar_np:
#         patch_db = 10.0 * np.log10(np.abs(patch) + 1e-6)
#         patch_norm = (patch_db - patch_db.mean()) / (patch_db.std() + 1e-6)
#         patch_rgb = np.stack([patch_norm, patch_norm, patch_norm], axis=0)
#         sar_inputs.append(patch_rgb)

#     sar_input = torch.from_numpy(np.stack(sar_inputs).astype(np.float32)).to(
#         device,
#         non_blocking=True,
#     )

#     sar_input = torch.nn.functional.interpolate(
#         sar_input,
#         size=(256, 256),
#         mode="bilinear",
#         align_corners=False,
#     )

#     return sar_input


# # =============================================================================
# # SAR OUTPUT EXTRACTION
# # =============================================================================

# def extract_tensor_from_output(
#     output: Any,
# ) -> torch.Tensor:
#     """
#     Safely extract the main tensor from common PyTorch model outputs.
#     """

#     if torch.is_tensor(output):
#         return output

#     if isinstance(output, (list, tuple)):

#         tensors = [
#             item
#             for item in output
#             if torch.is_tensor(item)
#         ]

#         if not tensors:
#             raise RuntimeError(
#                 "SAR expert returned a tuple/list without tensors."
#             )

#         return tensors[0]

#     if isinstance(output, dict):

#         preferred_keys = [
#             "logits",
#             "logit",
#             "out",
#             "output",
#             "prediction",
#             "pred",
#         ]

#         for key in preferred_keys:

#             if key in output and torch.is_tensor(output[key]):
#                 return output[key]

#         for value in output.values():

#             if torch.is_tensor(value):
#                 return value

#     raise RuntimeError(
#         "\nCould not extract tensor from SAR model output.\n"
#         f"Output type: {type(output)}"
#     )


# @torch.no_grad()
# def run_sar_expert(
#     sar_model: nn.Module,
#     sar_input: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Run frozen SAR expert and return a probability map.

#     Output shape:

#         (B, 1, H, W)
#     """

#     sar_model.eval()

#     output = sar_model(sar_input)

#     output = extract_tensor_from_output(output)

#     if output.ndim == 3:
#         output = output.unsqueeze(1)

#     if output.ndim != 4:
#         raise RuntimeError(
#             "\nUnexpected SAR expert output shape:\n"
#             f"{tuple(output.shape)}"
#         )

#     # Detect whether the output already looks like probabilities.
#     output_min = float(output.detach().min())
#     output_max = float(output.detach().max())

#     if output_min >= 0.0 and output_max <= 1.0:
#         probability = output
#     else:
#         probability = torch.sigmoid(output)

#     return probability


# # =============================================================================
# # SAR EVIDENCE FEATURES
# # =============================================================================

# def extract_sar_features(
#     sar_probability: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Convert the SAR expert probability map into a 6-D evidence vector.

#     Features:

#         1. mean probability
#         2. maximum probability
#         3. minimum probability
#         4. standard deviation
#         5. positive-area fraction at threshold 0.50
#         6. positive-area fraction at threshold 0.75

#     Output:

#         (B, 6)

#     This is scene/sample-level SAR evidence.
#     It is NOT concatenated pixel-wise with HSI.
#     """

#     if sar_probability.ndim != 4:
#         raise ValueError(
#             "\nExpected SAR probability shape:\n"
#             "(B, 1, H, W)\n"
#             f"Received:\n{tuple(sar_probability.shape)}"
#         )

#     flat = sar_probability.flatten(
#         start_dim=1,
#     )

#     mean = flat.mean(
#         dim=1,
#         keepdim=True,
#     )

#     maximum = flat.max(
#         dim=1,
#         keepdim=True,
#     ).values

#     minimum = flat.min(
#         dim=1,
#         keepdim=True,
#     ).values

#     std = flat.std(
#         dim=1,
#         unbiased=False,
#         keepdim=True,
#     )

#     area_50 = (
#         flat >= 0.50
#     ).float().mean(
#         dim=1,
#         keepdim=True,
#     )

#     area_75 = (
#         flat >= 0.75
#     ).float().mean(
#         dim=1,
#         keepdim=True,
#     )

#     features = torch.cat(
#         [
#             mean,
#             maximum,
#             minimum,
#             std,
#             area_50,
#             area_75,
#         ],
#         dim=1,
#     )

#     if features.shape[1] != 6:
#         raise RuntimeError(
#             f"SAR feature dimension must be 6, got {features.shape[1]}"
#         )

#     return features


# # =============================================================================
# # FUSION MODEL FORWARD
# # =============================================================================

# def run_fusion_model(
#     fusion_model: nn.Module,
#     sar_features: torch.Tensor,
#     hsi_features: torch.Tensor,
#     hsi_logits: torch.Tensor,
#     hsi_confidence: torch.Tensor,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Run RegistrationFreeFusionNet.

#     The current fusion model was designed around:
#         SAR features: (B, 6)
#         HSI features: (B, 64)

#     HSI confidence is not passed as a separate spatial signal.
#     It is only used as an optional reliability modulation if the
#     fusion model supports it.
#     """

#     output = fusion_model(
#         sar_features,
#         hsi_features,
#         hsi_logits,
#         hsi_confidence,
#     )

#     if isinstance(output, dict):
#         return output

#     if torch.is_tensor(output):

#         return {
#             "p_fused": output,
#         }

#     if isinstance(output, (tuple, list)):

#         names = [
#             "p_fused",
#             "p_sar",
#             "p_hsi",
#             "reliability",
#             "correction",
#             "signed_correction",
#             "bounded_delta",
#             "fusion_context",
#             "sar_embedding",
#             "hsi_embedding",
#             "evidence_embedding",
#         ]

#         result = {}

#         for name, value in zip(names, output):

#             if torch.is_tensor(value):
#                 result[name] = value

#         if "p_fused" not in result:
#             raise RuntimeError(
#                 "Fusion model did not return p_fused."
#             )

#         return result

#     raise RuntimeError(
#         "\nUnexpected fusion model output type:\n"
#         f"{type(output)}"
#     )


# # =============================================================================
# # METRICS
# # =============================================================================

# def compute_binary_metrics(
#     probabilities: torch.Tensor,
#     labels: torch.Tensor,
#     threshold: float = 0.5,
# ) -> Dict[str, float]:

#     probabilities = probabilities.detach()

#     labels = labels.detach()

#     predictions = (
#         probabilities >= threshold
#     ).long()

#     labels = labels.long()

#     tp = int(
#         ((predictions == 1) & (labels == 1))
#         .sum()
#         .item()
#     )

#     tn = int(
#         ((predictions == 0) & (labels == 0))
#         .sum()
#         .item()
#     )

#     fp = int(
#         ((predictions == 1) & (labels == 0))
#         .sum()
#         .item()
#     )

#     fn = int(
#         ((predictions == 0) & (labels == 1))
#         .sum()
#         .item()
#     )

#     total = max(tp + tn + fp + fn, 1)

#     accuracy = (tp + tn) / total

#     precision = tp / max(tp + fp, 1)

#     recall = tp / max(tp + fn, 1)

#     f1 = (
#         2.0 * precision * recall
#         / max(precision + recall, 1e-8)
#     )

#     iou = tp / max(tp + fp + fn, 1)

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "iou": iou,
#     }


# # =============================================================================
# # ONE EPOCH
# # =============================================================================

# def run_epoch(
#     loader,
#     sar_model: nn.Module,
#     hsi_model: nn.Module,
#     processor: Any,
#     fusion_model: nn.Module,
#     optimizer,
#     criterion,
#     scaler,
#     training: bool,
# ) -> Dict[str, float]:

#     if training:
#         fusion_model.train()
#     else:
#         fusion_model.eval()

#     sar_model.eval()
#     hsi_model.eval()

#     total_loss = 0.0
#     total_samples = 0

#     all_probabilities = []
#     all_labels = []

#     amp_enabled = (
#         USE_AMP
#         and DEVICE.type == "cuda"
#     )

#     batch_iterator = tqdm(
#         loader,
#         desc="train" if training else "val",
#         leave=False,
#         unit="batch",
#     )

#     for batch in batch_iterator:

#         hsi_patch = batch["hsi_patch"]

#         sar_patch = batch["sar_patch"]

#         labels = batch["label"].to(
#             DEVICE,
#             non_blocking=True,
#         )

#         labels_float = labels.float().view(
#             -1,
#             1,
#         )

#         # ---------------------------------------------------------
#         # HSI preprocessing and frozen feature extraction
#         # ---------------------------------------------------------

#         hsi_input = preprocess_hsi_batch(
#             hsi_patch,
#             processor,
#             DEVICE,
#         )

#         with torch.no_grad(), torch.autocast(
#             device_type="cuda",
#             enabled=amp_enabled,
#         ):
#                 hsi_features, hsi_logits, hsi_confidence = (
#                 extract_hsi_features(
#                     hsi_model,
#                     hsi_input,
#                 )
#             )

#         # ---------------------------------------------------------
#         # SAR frozen expert and evidence extraction
#         # ---------------------------------------------------------

#         sar_input = prepare_sar_input(
#             sar_patch,
#             DEVICE,
#         )

#         with torch.no_grad(), torch.autocast(
#             device_type="cuda",
#             enabled=amp_enabled,
#         ):
#             sar_probability = run_sar_expert(
#                 sar_model,
#                 sar_input,
#             )

#             sar_features = extract_sar_features(
#                 sar_probability
#             )

#         # ---------------------------------------------------------
#         # Fusion optimization
#         # ---------------------------------------------------------

#         if training:

#             optimizer.zero_grad(
#                 set_to_none=True
#             )

#         with torch.set_grad_enabled(training):

#             with torch.autocast(
#                 device_type="cuda",
#                 enabled=amp_enabled,
#             ):

#                 outputs = run_fusion_model(
#                     fusion_model,
#                     sar_features,
#                     hsi_features,
#                     hsi_logits,
#                     hsi_confidence,
#                 )

#                 p_fused = outputs["p_fused"]

#                 if p_fused.ndim == 1:
#                     p_fused = p_fused.unsqueeze(1)

#                 p_fused = p_fused.clamp(
#                     min=1e-6,
#                     max=1.0 - 1e-6,
#                 )

#                 # BCELoss is not supported inside CUDA autocast. The fusion
#                 # model returns probabilities, so compute this scalar loss
#                 # outside autocast in float32.
#                 with torch.autocast(device_type="cuda", enabled=False):
#                     loss = criterion(
#                         p_fused.float(),
#                         labels_float.float(),
#                     )

#             if training:

#                 if amp_enabled:

#                     scaler.scale(loss).backward()

#                     scaler.unscale_(optimizer)

#                     torch.nn.utils.clip_grad_norm_(
#                         fusion_model.parameters(),
#                         max_norm=5.0,
#                     )

#                     scaler.step(optimizer)

#                     scaler.update()

#                 else:

#                     loss.backward()

#                     torch.nn.utils.clip_grad_norm_(
#                         fusion_model.parameters(),
#                         max_norm=5.0,
#                     )

#                     optimizer.step()

#         batch_size = labels.shape[0]

#         total_loss += (
#             float(loss.detach().item())
#             * batch_size
#         )

#         total_samples += batch_size

#         all_probabilities.append(
#             p_fused.detach().view(-1).cpu()
#         )

#         all_labels.append(
#             labels.detach().view(-1).cpu()
#         )

#     if total_samples == 0:
#         raise RuntimeError(
#             "Epoch received zero samples."
#         )

#     probabilities = torch.cat(
#         all_probabilities,
#         dim=0,
#     )

#     labels = torch.cat(
#         all_labels,
#         dim=0,
#     )

#     metrics = compute_binary_metrics(
#         probabilities,
#         labels,
#     )

#     metrics["loss"] = (
#         total_loss / total_samples
#     )

#     return metrics


# # =============================================================================
# # SAVE HISTORY
# # =============================================================================

# def save_history(
#     history: List[Dict[str, Any]],
# ) -> None:

#     with open(
#         HISTORY_JSON_PATH,
#         "w",
#         encoding="utf-8",
#     ) as file:

#         json.dump(
#             history,
#             file,
#             indent=2,
#         )

#     if history:

#         fieldnames = list(
#             history[0].keys()
#         )

#         with open(
#             HISTORY_CSV_PATH,
#             "w",
#             newline="",
#             encoding="utf-8",
#         ) as file:

#             writer = csv.DictWriter(
#                 file,
#                 fieldnames=fieldnames,
#             )

#             writer.writeheader()

#             writer.writerows(history)


# # =============================================================================
# # CHECKPOINT
# # =============================================================================

# def save_checkpoint(
#     path: Path,
#     epoch: int,
#     fusion_model: nn.Module,
#     optimizer,
#     metrics: Dict[str, float],
#     history: List[Dict[str, Any]],
# ) -> None:

#     checkpoint = {
#         "epoch": epoch,
#         "model_state_dict": (
#             fusion_model.state_dict()
#         ),
#         "optimizer_state_dict": (
#             optimizer.state_dict()
#         ),
#         "metrics": metrics,
#         "history": history,
#         "config": {
#             "batch_size": BATCH_SIZE,
#             "epochs": NUM_EPOCHS,
#             "learning_rate": LEARNING_RATE,
#             "weight_decay": WEIGHT_DECAY,
#             "sar_feature_dim": 6,
#             "hsi_feature_dim": 64,
#             "registration_free": True,
#         },
#     }

#     torch.save(
#         checkpoint,
#         path,
#     )


# # =============================================================================
# # DATASET BUILDING
# # =============================================================================

# def build_loaders():

#     banner(
#         "BUILDING FUSION DATALOADERS"
#     )

#     build_dataloader = getattr(
#         DATASET_MODULE,
#         "build_dataloader",
#     )

#     print(
#         f"Batch size : {BATCH_SIZE}"
#     )

#     print("\nBuilding training loader...")

#     train_dataset, train_loader = (
#         build_dataloader(
#             split="train",
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#         )
#     )

#     print(
#         f"[OK] Train samples: {len(train_dataset)}"
#     )

#     print(
#         "\nBuilding validation loader..."
#     )

#     try:

#         val_dataset, val_loader = (
#             build_dataloader(
#                 split="validation",
#                 batch_size=BATCH_SIZE,
#                 shuffle=False,
#                 num_workers=NUM_WORKERS,
#             )
#         )

#     except Exception:

#         val_dataset, val_loader = (
#             build_dataloader(
#                 split="val",
#                 batch_size=BATCH_SIZE,
#                 shuffle=False,
#                 num_workers=NUM_WORKERS,
#             )
#         )

#     print(
#         f"[OK] Validation samples: {len(val_dataset)}"
#     )

#     return (
#         train_dataset,
#         train_loader,
#         val_dataset,
#         val_loader,
#     )


# # =============================================================================
# # MAIN TRAINING
# # =============================================================================

# def main() -> None:

#     set_seed(SEED)

#     banner(
#         "STAGE 3 — REGISTRATION-FREE FUSION TRAINING"
#     )

#     print(f"\nDevice        : {DEVICE}")
#     print(f"Batch size    : {BATCH_SIZE}")
#     print(f"Epochs        : {NUM_EPOCHS}")
#     print(f"Learning rate : {LEARNING_RATE}")
#     print(f"Weight decay  : {WEIGHT_DECAY}")
#     print(f"AMP           : {USE_AMP}")
#     print(f"Seed          : {SEED}")

#     print(
#         "\nRegistration contract:"
#     )

#     print(
#         "  SAR/HSI pixel registration : NOT ASSUMED"
#     )

#     print(
#         "  Pixel-wise concatenation   : FORBIDDEN"
#     )

#     print(
#         "  Experts                    : FROZEN"
#     )

#     print(
#         "  Trainable component         : FUSION ONLY"
#     )

#     # -----------------------------------------------------------------
#     # DATA
#     # -----------------------------------------------------------------

#     (
#         train_dataset,
#         train_loader,
#         val_dataset,
#         val_loader,
#     ) = build_loaders()

#     # -----------------------------------------------------------------
#     # EXPERTS
#     # -----------------------------------------------------------------

#     banner(
#         "LOADING FROZEN EXPERTS"
#     )

#     sar_model = EXPERTS.load_sar_expert(
#         DEVICE
#     )

#     hsi_model, processor = (
#         EXPERTS.load_hsi_expert(
#             DEVICE
#         )
#     )

#     sar_model.eval()
#     hsi_model.eval()

#     for parameter in sar_model.parameters():
#         parameter.requires_grad = False

#     for parameter in hsi_model.parameters():
#         parameter.requires_grad = False

#     print(
#         "\n[OK] SAR expert frozen"
#     )

#     print(
#         "[OK] HSI expert frozen"
#     )

#     # -----------------------------------------------------------------
#     # FUSION MODEL
#     # -----------------------------------------------------------------

#     banner(
#         "BUILDING TRAINABLE FUSION MODEL"
#     )

#     if not hasattr(
#         FUSION_MODEL_MODULE,
#         "RegistrationFreeFusionNet",
#     ):
#         raise AttributeError(
#             "\n04_fusion_model.py does not contain "
#             "RegistrationFreeFusionNet."
#         )

#     FusionNet = (
#         FUSION_MODEL_MODULE
#         .RegistrationFreeFusionNet
#     )

#     fusion_model = FusionNet()

#     fusion_model = fusion_model.to(
#         DEVICE
#     )

#     total_parameters = sum(
#         parameter.numel()
#         for parameter in fusion_model.parameters()
#     )

#     trainable_parameters = sum(
#         parameter.numel()
#         for parameter in fusion_model.parameters()
#         if parameter.requires_grad
#     )

#     print(
#         f"Total parameters     : {total_parameters:,}"
#     )

#     print(
#         f"Trainable parameters : {trainable_parameters:,}"
#     )

#     # -----------------------------------------------------------------
#     # OPTIMIZATION
#     # -----------------------------------------------------------------

#     optimizer = torch.optim.AdamW(
#         fusion_model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#     )

#     criterion = nn.BCELoss()

#     amp_enabled = (
#         USE_AMP
#         and DEVICE.type == "cuda"
#     )

#     scaler = GradScaler(
#         "cuda",
#         enabled=amp_enabled,
#     )

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         factor=0.5,
#         patience=8,
#         min_lr=1e-6,
#     )

#     # -----------------------------------------------------------------
#     # TRAINING LOOP
#     # -----------------------------------------------------------------

#     banner(
#         "STARTING FUSION TRAINING"
#     )

#     history = []

#     best_val_f1 = float("-inf")
#     best_epoch = 0

#     training_start = time.time()

#     for epoch in range(
#         1,
#         NUM_EPOCHS + 1,
#     ):

#         epoch_start = time.time()

#         print(
#             f"\nEpoch {epoch}/{NUM_EPOCHS}"
#         )

#         # -------------------------------------------------------------
#         # TRAIN
#         # -------------------------------------------------------------

#         train_metrics = run_epoch(
#             loader=train_loader,
#             sar_model=sar_model,
#             hsi_model=hsi_model,
#             processor=processor,
#             fusion_model=fusion_model,
#             optimizer=optimizer,
#             criterion=criterion,
#             scaler=scaler,
#             training=True,
#         )

#         # -------------------------------------------------------------
#         # VALIDATION
#         # -------------------------------------------------------------

#         val_metrics = run_epoch(
#             loader=val_loader,
#             sar_model=sar_model,
#             hsi_model=hsi_model,
#             processor=processor,
#             fusion_model=fusion_model,
#             optimizer=optimizer,
#             criterion=criterion,
#             scaler=scaler,
#             training=False,
#         )

#         scheduler.step(
#             val_metrics["f1"]
#         )

#         current_lr = (
#             optimizer.param_groups[0]["lr"]
#         )

#         epoch_seconds = (
#             time.time()
#             - epoch_start
#         )

#         row = {
#             "epoch": epoch,
#             "learning_rate": current_lr,
#             "train_loss": train_metrics["loss"],
#             "train_accuracy": train_metrics["accuracy"],
#             "train_precision": train_metrics["precision"],
#             "train_recall": train_metrics["recall"],
#             "train_f1": train_metrics["f1"],
#             "train_iou": train_metrics["iou"],
#             "val_loss": val_metrics["loss"],
#             "val_accuracy": val_metrics["accuracy"],
#             "val_precision": val_metrics["precision"],
#             "val_recall": val_metrics["recall"],
#             "val_f1": val_metrics["f1"],
#             "val_iou": val_metrics["iou"],
#             "epoch_seconds": epoch_seconds,
#         }

#         history.append(
#             row
#         )

#         save_history(
#             history
#         )

#         print(
#             f"  Train | "
#             f"Loss={train_metrics['loss']:.6f} | "
#             f"F1={train_metrics['f1']:.4f} | "
#             f"IoU={train_metrics['iou']:.4f}"
#         )

#         print(
#             f"  Val   | "
#             f"Loss={val_metrics['loss']:.6f} | "
#             f"F1={val_metrics['f1']:.4f} | "
#             f"IoU={val_metrics['iou']:.4f}"
#         )

#         print(
#             f"  LR    | {current_lr:.8f}"
#         )

#         # -------------------------------------------------------------
#         # SAVE LAST
#         # -------------------------------------------------------------

#         save_checkpoint(
#             path=LAST_CHECKPOINT_PATH,
#             epoch=epoch,
#             fusion_model=fusion_model,
#             optimizer=optimizer,
#             metrics=val_metrics,
#             history=history,
#         )

#         # -------------------------------------------------------------
#         # SAVE BEST
#         # -------------------------------------------------------------

#         if val_metrics["f1"] > best_val_f1:

#             best_val_f1 = (
#                 val_metrics["f1"]
#             )

#             best_epoch = epoch

#             save_checkpoint(
#                 path=BEST_CHECKPOINT_PATH,
#                 epoch=epoch,
#                 fusion_model=fusion_model,
#                 optimizer=optimizer,
#                 metrics=val_metrics,
#                 history=history,
#             )

#             print(
#                 "\n  [BEST] New best validation F1"
#             )

#             print(
#                 f"         Epoch : {best_epoch}"
#             )

#             print(
#                 f"         F1    : {best_val_f1:.6f}"
#             )

#     # -----------------------------------------------------------------
#     # FINAL SUMMARY
#     # -----------------------------------------------------------------

#     total_training_seconds = (
#         time.time()
#         - training_start
#     )

#     summary = {
#         "device": str(DEVICE),
#         "epochs_completed": NUM_EPOCHS,
#         "best_epoch": best_epoch,
#         "best_validation_f1": best_val_f1,
#         "total_training_seconds": total_training_seconds,
#         "best_checkpoint": str(
#             BEST_CHECKPOINT_PATH
#         ),
#         "last_checkpoint": str(
#             LAST_CHECKPOINT_PATH
#         ),
#         "registration_free": True,
#         "sar_expert_frozen": True,
#         "hsi_expert_frozen": True,
#         "fusion_only_trainable": True,
#         "train_samples": len(train_dataset),
#         "validation_samples": len(val_dataset),
#     }

#     with open(
#         SUMMARY_PATH,
#         "w",
#         encoding="utf-8",
#     ) as file:

#         json.dump(
#             summary,
#             file,
#             indent=2,
#         )

#     banner(
#         "FUSION TRAINING COMPLETE"
#     )

#     print(
#         f"Best epoch          : {best_epoch}"
#     )

#     print(
#         f"Best validation F1  : {best_val_f1:.6f}"
#     )

#     print(
#         f"Training time       : "
#         f"{total_training_seconds / 60.0:.2f} minutes"
#     )

#     print(
#         f"\nBest checkpoint:"
#     )

#     print(
#         f"  {BEST_CHECKPOINT_PATH}"
#     )

#     print(
#         "\nLast checkpoint:"
#     )

#     print(
#         f"  {LAST_CHECKPOINT_PATH}"
#     )

#     print(
#         "\nTraining history:"
#     )

#     print(
#         f"  {HISTORY_JSON_PATH}"
#     )

#     print(
#         f"  {HISTORY_CSV_PATH}"
#     )

#     print(
#         "\n[SUCCESS] Training complete."
#     )

#     print(
#         "[SUCCESS] SAR expert remained frozen."
#     )

#     print(
#         "[SUCCESS] HSI expert remained frozen."
#     )

#     print(
#         "[SUCCESS] Only fusion layers were optimized."
#     )

#     print(
#         "[SUCCESS] No pixel-level SAR/HSI registration was assumed."
#     )


# # =============================================================================
# # ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     main()


"""
STAGE 3 — REGISTRATION-FREE FUSION TRAINING
============================================

05_train.py

Purpose
-------
Train ONLY RegistrationFreeFusionNet using precomputed cached features.

The frozen experts have already been executed during the cache-generation
stage. Therefore this training script MUST NOT:

    - load raw HSI/SAR patches
    - import 02_fusion_dataset.py
    - import 03_load_experts.py
    - run the SAR expert
    - run HMOSDNet
    - run the HSI spectral processor

Data flow
---------

train_features.npz
validation_features.npz
        |
        v
Cached tensors:

    sar_features    : (N, 42)
    hsi_features    : (N, 64)
    hsi_logits      : (N, 1)
    hsi_confidence  : (N, 1)
    labels          : (N,)
    scene_indices   : (N,)
        |
        v
RegistrationFreeFusionNet
        |
        v
p_fused
        |
        +--------------------+
        |                    |
        v                    v
     BCE Loss            Dice Loss
        |
        +--------+
                 |
                 v

       Residual regularization
       using bounded_delta
                 |
                 v
            Total Loss


IMPORTANT
---------

SAR and HSI are NOT assumed to be pixel registered.

The fusion network operates only on scene/sample-level cached evidence.

No pixel-wise concatenation occurs.

The frozen SAR and HSI experts are NOT loaded in this script.

Only RegistrationFreeFusionNet is optimized.


CACHE CONTRACT
==============

Each .npz cache contains:

    sar_features    float32   (N, 42)
    hsi_features    float32   (N, 64)
    hsi_logits      float32   (N, 1)
    hsi_confidence  float32   (N, 1)
    labels          int64     (N,)
    scene_indices   int64     (N,)


CORRECT SAR FEATURE ORDER
=========================

The actual cached SAR feature order is:

    0 : center probability
    1 : local mean probability
    2 : local standard deviation
    3 : local minimum probability
    4 : local maximum probability
    5 : local contrast

This script does NOT reorder the SAR features.

The model receives the cached features exactly as generated.


TRAINING SPLITS
===============

Used for optimization:

    train_features.npz

Used for validation / checkpoint selection:

    validation_features.npz

NOT used here:

    test_features.npz

Test evaluation must happen only after training is complete.
"""


from __future__ import annotations


# =============================================================================
# STANDARD LIBRARY
# =============================================================================

import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# THIRD PARTY
# =============================================================================

import numpy as np

import torch
import torch.nn as nn

from torch.amp import GradScaler

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from tqdm import tqdm


# =============================================================================
# PATH SETUP
# =============================================================================

THIS_FILE = Path(__file__).resolve()

FUSION_DIR = THIS_FILE.parent


if str(FUSION_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(FUSION_DIR),
    )


# =============================================================================
# DYNAMIC MODULE LOADER
# =============================================================================

from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


def load_python_module(
    name: str,
    path: Path,
):
    """
    Load a Python module directly from a file path.
    """

    spec = spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "\nCould not import module:\n"
            f"{path}"
        )

    module = module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# LOAD CONFIG
# =============================================================================

CFG = load_python_module(
    "fusion_config",
    FUSION_DIR / "00_config.py",
)


FUSION_MODEL_MODULE = load_python_module(
    "fusion_model",
    FUSION_DIR / "04_fusion_model.py",
)


# =============================================================================
# CONFIG HELPER
# =============================================================================

def get_config(
    names: List[str],
    default: Any = None,
    required: bool = False,
) -> Any:
    """
    Return the first configuration attribute that exists.
    """

    for name in names:

        if hasattr(CFG, name):

            return getattr(
                CFG,
                name,
            )

    if required:

        raise AttributeError(
            "\nMissing required configuration.\n"
            f"Expected one of:\n{names}"
        )

    return default


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BATCH_SIZE = int(
    get_config(
        [
            "FUSION_BATCH_SIZE",
            "BATCH_SIZE",
        ],
        default=256,
    )
)


NUM_EPOCHS = int(
    get_config(
        [
            "FUSION_EPOCHS",
            "EPOCHS",
        ],
        default=100,
    )
)


LEARNING_RATE = float(
    get_config(
        [
            "FUSION_LEARNING_RATE",
            "LEARNING_RATE",
        ],
        default=1e-3,
    )
)


WEIGHT_DECAY = float(
    get_config(
        [
            "FUSION_WEIGHT_DECAY",
            "WEIGHT_DECAY",
        ],
        default=1e-4,
    )
)


NUM_WORKERS = int(
    get_config(
        [
            "FUSION_NUM_WORKERS",
            "NUM_WORKERS",
        ],
        default=0,
    )
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


USE_AMP = bool(
    get_config(
        [
            "USE_AMP",
            "AMP",
        ],
        default=True,
    )
)


# -----------------------------------------------------------------------------
# LOSS CONFIGURATION
# -----------------------------------------------------------------------------

BCE_WEIGHT = float(
    get_config(
        [
            "FUSION_BCE_WEIGHT",
        ],
        default=1.0,
    )
)


DICE_WEIGHT = float(
    get_config(
        [
            "FUSION_DICE_WEIGHT",
        ],
        default=0.20,
    )
)


RESIDUAL_REG_WEIGHT = float(
    get_config(
        [
            "FUSION_RESIDUAL_REG_WEIGHT",
        ],
        default=1e-3,
    )
)


GATE_SUPERVISION_WEIGHT = float(
    get_config(
        [
            "FUSION_GATE_SUPERVISION_WEIGHT",
            "GATE_SUPERVISION_WEIGHT",
        ],
        default=0.0,
    )
)


# -----------------------------------------------------------------------------
# OPTIMIZATION
# -----------------------------------------------------------------------------

MAX_GRAD_NORM = float(
    get_config(
        [
            "FUSION_MAX_GRAD_NORM",
        ],
        default=5.0,
    )
)


# -----------------------------------------------------------------------------
# EARLY STOPPING
# -----------------------------------------------------------------------------

EARLY_STOPPING_PATIENCE = int(
    get_config(
        [
            "FUSION_EARLY_STOPPING_PATIENCE",
        ],
        default=20,
    )
)


EARLY_STOPPING_MIN_DELTA = float(
    get_config(
        [
            "FUSION_EARLY_STOPPING_MIN_DELTA",
        ],
        default=1e-5,
    )
)


# -----------------------------------------------------------------------------
# CACHE PATHS
# -----------------------------------------------------------------------------

CACHE_DIR = Path(
    os.environ.get(
        "FUSION_CACHE_DIR",
        get_config(
            [
                "FUSION_CACHE_DIR",
            ],
            default=FUSION_DIR / "cache_features",
        ),
    )
).resolve()


TRAIN_CACHE_PATH = (
    CACHE_DIR
    / "train_features.npz"
)


VALIDATION_CACHE_PATH = (
    CACHE_DIR
    / "validation_features.npz"
)


TEST_CACHE_PATH = (
    CACHE_DIR
    / "test_features.npz"
)


# -----------------------------------------------------------------------------
# OUTPUT PATHS
# -----------------------------------------------------------------------------

CHECKPOINT_DIR = Path(
    os.environ.get(
        "FUSION_CHECKPOINT_DIR",
        get_config(
            [
                "FUSION_CHECKPOINT_DIR",
            ],
            default=FUSION_DIR / "weights",
        ),
    )
).resolve()


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best_fusion_model.pth"
)


LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "last_fusion_model.pth"
)


HISTORY_JSON_PATH = (
    CHECKPOINT_DIR
    / "fusion_training_history.json"
)


HISTORY_CSV_PATH = (
    CHECKPOINT_DIR
    / "fusion_training_history.csv"
)


SUMMARY_PATH = (
    CHECKPOINT_DIR
    / "fusion_training_summary.json"
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
    """
    Set random seeds.

    benchmark=True is used because the cached input sizes remain fixed.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


# =============================================================================
# CACHE DATA STRUCTURE
# =============================================================================

@dataclass
class CachedFeatures:

    sar_features: np.ndarray

    hsi_features: np.ndarray

    hsi_logits: np.ndarray

    hsi_confidence: np.ndarray

    labels: np.ndarray

    scene_indices: np.ndarray


# =============================================================================
# CACHE VALIDATION
# =============================================================================

REQUIRED_CACHE_KEYS = (
    "sar_features",
    "hsi_features",
    "hsi_logits",
    "hsi_confidence",
    "labels",
    "scene_indices",
)


def validate_array_finite(
    name: str,
    array: np.ndarray,
) -> None:
    """
    Ensure numeric arrays contain no NaN or Inf.
    """

    if np.issubdtype(
        array.dtype,
        np.number,
    ):

        if not np.isfinite(
            array
        ).all():

            raise ValueError(
                "\nCache contains NaN or Inf values.\n"
                f"Array: {name}"
            )


def load_cached_features(
    cache_path: Path,
) -> CachedFeatures:
    """
    Load and validate a cached feature file.

    Expected contract:

        sar_features    (N, 6)
        hsi_features    (N, 64)
        hsi_logits      (N, 1)
        hsi_confidence  (N, 1)
        labels          (N,)
        scene_indices   (N,)
    """

    if not cache_path.exists():

        raise FileNotFoundError(
            "\nCache file not found:\n"
            f"{cache_path}"
        )

    print(
        f"\nLoading cache:\n"
        f"  {cache_path}"
    )

    with np.load(
        cache_path,
        allow_pickle=False,
    ) as data:

        available_keys = set(
            data.files
        )

        missing_keys = [
            key
            for key in REQUIRED_CACHE_KEYS
            if key not in available_keys
        ]

        if missing_keys:

            raise KeyError(
                "\nCache is missing required arrays.\n"
                f"Missing: {missing_keys}\n"
                f"Available: {sorted(available_keys)}"
            )

        sar_features = np.asarray(
            data["sar_features"],
            dtype=np.float32,
        )

        hsi_features = np.asarray(
            data["hsi_features"],
            dtype=np.float32,
        )

        hsi_logits = np.asarray(
            data["hsi_logits"],
            dtype=np.float32,
        )

        hsi_confidence = np.asarray(
            data["hsi_confidence"],
            dtype=np.float32,
        )

        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        )

        scene_indices = np.asarray(
            data["scene_indices"],
            dtype=np.int64,
        )

    # -------------------------------------------------------------------------
    # SHAPE VALIDATION
    # -------------------------------------------------------------------------

    if sar_features.ndim != 2:

        raise ValueError(
            f"sar_features must be 2D, got "
            f"{sar_features.shape}"
        )

    if hsi_features.ndim != 2:

        raise ValueError(
            f"hsi_features must be 2D, got "
            f"{hsi_features.shape}"
        )

    if hsi_logits.ndim != 2:

        raise ValueError(
            f"hsi_logits must be 2D, got "
            f"{hsi_logits.shape}"
        )

    if hsi_confidence.ndim != 2:

        raise ValueError(
            f"hsi_confidence must be 2D, got "
            f"{hsi_confidence.shape}"
        )

    if labels.ndim != 1:

        raise ValueError(
            f"labels must be 1D, got "
            f"{labels.shape}"
        )

    if scene_indices.ndim != 1:

        raise ValueError(
            f"scene_indices must be 1D, got "
            f"{scene_indices.shape}"
        )

    # -------------------------------------------------------------------------
    # FEATURE DIMENSIONS
    # -------------------------------------------------------------------------

    if sar_features.shape[1] != 42:

        raise ValueError(
            "\nSAR feature dimension mismatch.\n"
            f"Expected: 42\n"
            f"Received: {sar_features.shape[1]}"
        )

    if hsi_features.shape[1] != 64:

        raise ValueError(
            "\nHSI feature dimension mismatch.\n"
            f"Expected: 64\n"
            f"Received: {hsi_features.shape[1]}"
        )

    if hsi_logits.shape[1] != 1:

        raise ValueError(
            "\nHSI logits dimension mismatch.\n"
            f"Expected: 1\n"
            f"Received: {hsi_logits.shape[1]}"
        )

    if hsi_confidence.shape[1] != 1:

        raise ValueError(
            "\nHSI confidence dimension mismatch.\n"
            f"Expected: 1\n"
            f"Received: {hsi_confidence.shape[1]}"
        )

    # -------------------------------------------------------------------------
    # SAMPLE COUNTS
    # -------------------------------------------------------------------------

    n = len(
        labels
    )

    arrays_to_check = {
        "sar_features": sar_features,
        "hsi_features": hsi_features,
        "hsi_logits": hsi_logits,
        "hsi_confidence": hsi_confidence,
        "scene_indices": scene_indices,
    }

    for name, array in arrays_to_check.items():

        if len(array) != n:

            raise ValueError(
                "\nCache sample count mismatch.\n"
                f"labels: {n}\n"
                f"{name}: {len(array)}"
            )

    if n == 0:

        raise ValueError(
            "\nCache contains zero samples."
        )

    # -------------------------------------------------------------------------
    # FINITE VALUE CHECK
    # -------------------------------------------------------------------------

    validate_array_finite(
        "sar_features",
        sar_features,
    )

    validate_array_finite(
        "hsi_features",
        hsi_features,
    )

    validate_array_finite(
        "hsi_logits",
        hsi_logits,
    )

    validate_array_finite(
        "hsi_confidence",
        hsi_confidence,
    )

    # -------------------------------------------------------------------------
    # LABEL VALIDATION
    # -------------------------------------------------------------------------

    unique_labels = np.unique(
        labels
    )

    if not np.all(
        np.isin(
            unique_labels,
            [0, 1],
        )
    ):

        raise ValueError(
            "\nLabels must contain only binary values 0 and 1.\n"
            f"Found: {unique_labels.tolist()}"
        )

    # -------------------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------------------

    positives = int(
        labels.sum()
    )

    positive_fraction = (
        positives / n
    )

    print(
        f"  Samples         : {n:,}"
    )

    print(
        f"  SAR features    : {sar_features.shape}"
    )

    print(
        f"  HSI features    : {hsi_features.shape}"
    )

    print(
        f"  HSI logits      : {hsi_logits.shape}"
    )

    print(
        f"  HSI confidence  : {hsi_confidence.shape}"
    )

    print(
        f"  Positives       : "
        f"{positives:,} "
        f"({positive_fraction:.4f})"
    )

    print(
        f"  Unique scenes   : "
        f"{len(np.unique(scene_indices)):,}"
    )

    return CachedFeatures(

        sar_features=sar_features,

        hsi_features=hsi_features,

        hsi_logits=hsi_logits,

        hsi_confidence=hsi_confidence,

        labels=labels,

        scene_indices=scene_indices,
    )


# =============================================================================
# DATASET
# =============================================================================

class FusionCacheDataset(
    Dataset,
):
    """
    Dataset backed entirely by precomputed NumPy feature arrays.

    No expert inference occurs here.
    """

    def __init__(
        self,
        cached: CachedFeatures,
    ) -> None:

        self.sar_features = torch.from_numpy(
            np.ascontiguousarray(
                cached.sar_features
            )
        )

        self.hsi_features = torch.from_numpy(
            np.ascontiguousarray(
                cached.hsi_features
            )
        )

        self.hsi_logits = torch.from_numpy(
            np.ascontiguousarray(
                cached.hsi_logits
            )
        )

        self.hsi_confidence = torch.from_numpy(
            np.ascontiguousarray(
                cached.hsi_confidence
            )
        )

        self.labels = torch.from_numpy(
            np.ascontiguousarray(
                cached.labels
            )
        )

        self.scene_indices = torch.from_numpy(
            np.ascontiguousarray(
                cached.scene_indices
            )
        )

    def __len__(
        self,
    ) -> int:

        return int(
            self.labels.shape[0]
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        return {

            "sar_features":
                self.sar_features[index],

            "hsi_features":
                self.hsi_features[index],

            "hsi_logits":
                self.hsi_logits[index],

            "hsi_confidence":
                self.hsi_confidence[index],

            "label":
                self.labels[index],

            "scene_index":
                self.scene_indices[index],
        }


# =============================================================================
# BUILD DATALOADERS
# =============================================================================

def build_loaders() -> Tuple[
    FusionCacheDataset,
    DataLoader,
    FusionCacheDataset,
    DataLoader,
]:

    banner(
        "LOADING CACHED FUSION FEATURES"
    )

    # IMPORTANT:
    # Only train and validation are loaded here.
    #
    # test_features.npz must not participate in training,
    # model selection, early stopping, LR scheduling,
    # or threshold selection.

    train_cached = load_cached_features(
        TRAIN_CACHE_PATH
    )

    validation_cached = load_cached_features(
        VALIDATION_CACHE_PATH
    )

    train_dataset = FusionCacheDataset(
        train_cached
    )

    validation_dataset = FusionCacheDataset(
        validation_cached
    )

    pin_memory = (
        DEVICE.type == "cuda"
    )

    persistent_workers = (
        NUM_WORKERS > 0
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=pin_memory,

        persistent_workers=persistent_workers,

        drop_last=False,
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=pin_memory,

        persistent_workers=persistent_workers,

        drop_last=False,
    )

    return (

        train_dataset,

        train_loader,

        validation_dataset,

        validation_loader,
    )


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def dice_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Binary Dice loss.

    Suitable because the fusion model returns probabilities.

    Dice is computed over the current batch.
    """

    probabilities = probabilities.reshape(
        probabilities.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        probabilities
        * targets
    ).sum(
        dim=1
    )

    denominator = (
        probabilities.sum(
            dim=1
        )
        +
        targets.sum(
            dim=1
        )
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return (
        1.0
        - dice.mean()
    )


def residual_regularization(
    outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Penalize unnecessarily large residual corrections.

    The fusion architecture may expose:

        outputs["bounded_delta"]

    If it is unavailable, return zero.

    This keeps the training script compatible with a simpler fusion model
    while using the regularization when the architecture supports it.
    """

    bounded_delta = outputs.get(
        "bounded_delta"
    )

    if bounded_delta is None:

        # Find a valid tensor/device without creating CPU/GPU mismatch.
        p_fused = outputs.get(
            "p_fused"
        )

        if p_fused is None:

            raise RuntimeError(
                "Fusion output does not contain p_fused."
            )

        return torch.zeros(
            (),
            device=p_fused.device,
            dtype=p_fused.dtype,
        )

    return (
        bounded_delta.float()
        .pow(2)
        .mean()
    )


def gate_supervision_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Teach gate=1 to prefer HSI and gate=0 to prefer SAR per sample."""

    gate = outputs["gate"].float().clamp(
        eps,
        1.0 - eps,
    )

    p_sar = outputs["p_sar"].float()
    p_hsi = outputs["p_hsi"].float()
    targets = targets.float()

    err_sar = (p_sar - targets).abs()
    err_hsi = (p_hsi - targets).abs()

    target_gate = torch.sigmoid(
        8.0 * (err_sar - err_hsi)
    ).detach()

    return nn.functional.binary_cross_entropy(
        gate,
        target_gate,
    )


def compute_total_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    bce_criterion: nn.Module,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
]:
    """
    Total objective:

        BCE_WEIGHT * BCE
        +
        DICE_WEIGHT * Dice
        +
        RESIDUAL_REG_WEIGHT * residual penalty
    """

    probabilities = probabilities.clamp(
        min=1e-6,
        max=1.0 - 1e-6,
    )

    bce = bce_criterion(
        probabilities.float(),
        targets.float(),
    )

    dice = dice_loss(
        probabilities.float(),
        targets.float(),
    )

    residual = residual_regularization(
        outputs
    ).float()

    gate_supervision = gate_supervision_loss(
        outputs,
        targets,
    ).float()

    total = (

        BCE_WEIGHT
        * bce

        +

        DICE_WEIGHT
        * dice

        +

        RESIDUAL_REG_WEIGHT
        * residual

        +

        GATE_SUPERVISION_WEIGHT
        * gate_supervision
    )

    components = {

        "bce_loss": bce.detach(),

        "dice_loss": dice.detach(),

        "residual_loss": residual.detach(),

        "gate_supervision_loss": gate_supervision.detach(),
    }

    return (
        total,
        components,
    )


# =============================================================================
# FUSION MODEL OUTPUT
# =============================================================================

def run_fusion_model(
    fusion_model: nn.Module,
    sar_features: torch.Tensor,
    hsi_features: torch.Tensor,
    hsi_logits: torch.Tensor,
    hsi_confidence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Expected input contract:

        sar_features    : (B, 6)
        hsi_features    : (B, 64)
        hsi_logits      : (B, 1)
        hsi_confidence  : (B, 1)

    Expected output:

        dictionary containing at least:

            p_fused

    Optionally:

            bounded_delta
            p_sar
            p_hsi
            reliability
            correction
            ...
    """

    outputs = fusion_model(

        sar_features,

        hsi_features,

        hsi_logits,

        hsi_confidence,
    )

    if isinstance(
        outputs,
        dict,
    ):

        if "p_fused" not in outputs:

            raise RuntimeError(
                "\nFusion model output dictionary does not contain:\n"
                "p_fused"
            )

        return outputs

    if torch.is_tensor(
        outputs
    ):

        return {
            "p_fused": outputs,
        }

    raise RuntimeError(
        "\nUnexpected fusion model output type:\n"
        f"{type(outputs)}"
    )


# =============================================================================
# METRICS
# =============================================================================

def compute_binary_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.50,
) -> Dict[str, float]:

    probabilities = probabilities.detach()

    labels = labels.detach()

    predictions = (
        probabilities >= threshold
    ).long()

    labels = labels.long()

    tp = int(
        (
            (predictions == 1)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )

    tn = int(
        (
            (predictions == 0)
            &
            (labels == 0)
        )
        .sum()
        .item()
    )

    fp = int(
        (
            (predictions == 1)
            &
            (labels == 0)
        )
        .sum()
        .item()
    )

    fn = int(
        (
            (predictions == 0)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )

    total = max(
        tp + tn + fp + fn,
        1,
    )

    accuracy = (
        tp + tn
    ) / total

    precision = tp / max(
        tp + fp,
        1,
    )

    recall = tp / max(
        tp + fn,
        1,
    )

    f1 = (
        2.0
        * precision
        * recall
    ) / max(
        precision + recall,
        1e-8,
    )

    iou = tp / max(
        tp + fp + fn,
        1,
    )

    specificity = tn / max(
        tn + fp,
        1,
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "iou": iou,

        "specificity": specificity,

        "balanced_accuracy":
            balanced_accuracy,
    }


# =============================================================================
# ONE EPOCH
# =============================================================================

def run_epoch(
    loader: DataLoader,
    fusion_model: nn.Module,
    optimizer,
    bce_criterion: nn.Module,
    scaler: GradScaler,
    training: bool,
) -> Dict[str, float]:

    if training:

        fusion_model.train()

    else:

        fusion_model.eval()

    total_loss = 0.0

    total_bce_loss = 0.0

    total_dice_loss = 0.0

    total_residual_loss = 0.0

    total_gate_supervision_loss = 0.0

    total_samples = 0

    all_probabilities = []

    all_labels = []

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    iterator = tqdm(

        loader,

        desc=(
            "train"
            if training
            else "validation"
        ),

        leave=False,

        unit="batch",
    )

    for batch in iterator:

        # ---------------------------------------------------------------------
        # MOVE CACHED FEATURES TO DEVICE
        # ---------------------------------------------------------------------

        sar_features = batch[
            "sar_features"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        hsi_features = batch[
            "hsi_features"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        hsi_logits = batch[
            "hsi_logits"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        hsi_confidence = batch[
            "hsi_confidence"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels_float = (
            labels
            .float()
            .view(
                -1,
                1,
            )
        )

        # ---------------------------------------------------------------------
        # ZERO GRADIENTS
        # ---------------------------------------------------------------------

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        # ---------------------------------------------------------------------
        # FORWARD / LOSS
        # ---------------------------------------------------------------------

        grad_context = (
            torch.enable_grad()
            if training
            else torch.no_grad()
        )

        with grad_context:

            with torch.autocast(

                device_type=DEVICE.type,

                enabled=amp_enabled,
            ):

                outputs = run_fusion_model(

                    fusion_model,

                    sar_features,

                    hsi_features,

                    hsi_logits,

                    hsi_confidence,
                )

                p_fused = outputs[
                    "p_fused"
                ]

                if p_fused.ndim == 1:

                    p_fused = p_fused.unsqueeze(
                        1
                    )

                if (
                    p_fused.ndim != 2
                    or p_fused.shape[1] != 1
                ):

                    raise RuntimeError(
                        "\nInvalid p_fused shape.\n"
                        f"Expected: (B, 1)\n"
                        f"Received: {tuple(p_fused.shape)}"
                    )

                # BCE is calculated in float32 for numerical stability.
                with torch.autocast(

                    device_type=DEVICE.type,

                    enabled=False,
                ):

                    loss, loss_components = (
                        compute_total_loss(

                            p_fused.float(),

                            labels_float.float(),

                            outputs,

                            bce_criterion,
                        )
                    )

        # ---------------------------------------------------------------------
        # BACKPROPAGATION
        # ---------------------------------------------------------------------

        if training:

            if amp_enabled:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(

                    fusion_model.parameters(),

                    max_norm=MAX_GRAD_NORM,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    fusion_model.parameters(),

                    max_norm=MAX_GRAD_NORM,
                )

                optimizer.step()

        # ---------------------------------------------------------------------
        # ACCUMULATE
        # ---------------------------------------------------------------------

        batch_size = labels.shape[0]

        total_loss += (

            float(
                loss.detach().item()
            )
            * batch_size
        )

        total_bce_loss += (

            float(
                loss_components[
                    "bce_loss"
                ].item()
            )
            * batch_size
        )

        total_dice_loss += (

            float(
                loss_components[
                    "dice_loss"
                ].item()
            )
            * batch_size
        )

        total_residual_loss += (

            float(
                loss_components[
                    "residual_loss"
                ].item()
            )
            * batch_size
        )

        total_gate_supervision_loss += (
            float(
                loss_components[
                    "gate_supervision_loss"
                ].item()
            )
            * batch_size
        )

        total_samples += batch_size

        all_probabilities.append(

            p_fused
            .detach()
            .float()
            .view(-1)
            .cpu()
        )

        all_labels.append(

            labels
            .detach()
            .view(-1)
            .cpu()
        )

        iterator.set_postfix(

            loss=f"{loss.item():.4f}"
        )

    # -------------------------------------------------------------------------
    # FINAL METRICS
    # -------------------------------------------------------------------------

    if total_samples == 0:

        raise RuntimeError(
            "\nEpoch received zero samples."
        )

    probabilities = torch.cat(

        all_probabilities,

        dim=0,
    )

    labels = torch.cat(

        all_labels,

        dim=0,
    )

    metrics = compute_binary_metrics(

        probabilities,

        labels,

        threshold=0.50,
    )

    metrics["loss"] = (

        total_loss
        / total_samples
    )

    metrics["bce_loss"] = (

        total_bce_loss
        / total_samples
    )

    metrics["dice_loss"] = (

        total_dice_loss
        / total_samples
    )

    metrics["residual_loss"] = (

        total_residual_loss
        / total_samples
    )

    metrics["gate_supervision_loss"] = (
        total_gate_supervision_loss
        / total_samples
    )

    # Keep raw validation outputs available for one final threshold search.
    # These tensors are removed from history rows and are used only in memory.
    metrics["probabilities"] = probabilities

    metrics["labels"] = labels

    return metrics


# =============================================================================
# HISTORY
# =============================================================================

def save_history(
    history: List[Dict[str, Any]],
) -> None:

    with open(

        HISTORY_JSON_PATH,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            history,

            file,

            indent=2,
        )

    if history:

        fieldnames = list(
            history[0].keys()
        )

        with open(

            HISTORY_CSV_PATH,

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
                history
            )


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    fusion_model: nn.Module,
    optimizer,
    scheduler,
    metrics: Dict[str, float],
    history: List[Dict[str, Any]],
    best_val_f1: float,
    decision_threshold: float | None = None,
) -> None:
    """
    Save complete training state.
    """

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            fusion_model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "metrics":
            metrics,

        "best_val_f1":
            best_val_f1,

        "decision_threshold":
            decision_threshold,

        "history":
            history,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "bce_weight":
                BCE_WEIGHT,

            "dice_weight":
                DICE_WEIGHT,

            "residual_reg_weight":
                RESIDUAL_REG_WEIGHT,

            "gate_supervision_weight":
                GATE_SUPERVISION_WEIGHT,

            "sar_feature_dim":
                42,

            "hsi_feature_dim":
                64,

            "registration_free":
                True,

            "cached_features":
                True,

            "sar_feature_order": [

                "center_probability",

                "local_mean_probability",

                "local_standard_deviation",

                "local_minimum_probability",

                "local_maximum_probability",

                "local_contrast",
            ],
        },
    }

    torch.save(

        checkpoint,

        path,
    )


def find_best_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    t_min: float = 0.20,
    t_max: float = 0.80,
    t_step: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Select a decision threshold using validation predictions only."""

    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    threshold = t_min

    while threshold <= t_max + 1e-9:

        metrics = compute_binary_metrics(
            probabilities,
            labels,
            threshold=threshold,
        )

        if metrics["f1"] > best_f1:

            best_threshold = threshold
            best_f1 = metrics["f1"]
            best_metrics = metrics

        threshold += t_step

    return best_threshold, best_metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # SEED
    # -------------------------------------------------------------------------

    set_seed(
        SEED
    )

    # -------------------------------------------------------------------------
    # START
    # -------------------------------------------------------------------------

    banner(
        "STAGE 3 — CACHE-ONLY REGISTRATION-FREE FUSION TRAINING"
    )

    print(
        f"\nDevice        : {DEVICE}"
    )

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Epochs        : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    print(
        f"AMP           : {USE_AMP}"
    )

    print(
        f"Seed          : {SEED}"
    )

    print(
        "\nCache directory:"
    )

    print(
        f"  {CACHE_DIR}"
    )

    print(
        "\nTraining contract:"
    )

    print(
        "  Raw dataset loading        : DISABLED"
    )

    print(
        "  SAR expert inference       : DISABLED"
    )

    print(
        "  HSI expert inference       : DISABLED"
    )

    print(
        "  HSI preprocessing          : DISABLED"
    )

    print(
        "  Pixel registration         : NOT ASSUMED"
    )

    print(
        "  Pixel-wise concatenation   : FORBIDDEN"
    )

    print(
        "  Trainable component        : FUSION ONLY"
    )

    print(
        "  Test cache during training : FORBIDDEN"
    )

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    (

        train_dataset,

        train_loader,

        validation_dataset,

        validation_loader,

    ) = build_loaders()

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    banner(
        "BUILDING REGISTRATION-FREE FUSION MODEL"
    )

    if not hasattr(

        FUSION_MODEL_MODULE,

        "RegistrationFreeFusionNet",
    ):

        raise AttributeError(
            "\n04_fusion_model.py does not contain:\n"
            "RegistrationFreeFusionNet"
        )

    FusionNet = (

        FUSION_MODEL_MODULE
        .RegistrationFreeFusionNet
    )

    fusion_model = FusionNet()

    fusion_model = fusion_model.to(
        DEVICE
    )

    total_parameters = sum(

        parameter.numel()

        for parameter in fusion_model.parameters()
    )

    trainable_parameters = sum(

        parameter.numel()

        for parameter in fusion_model.parameters()

        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    if trainable_parameters == 0:

        raise RuntimeError(
            "\nFusion model has zero trainable parameters."
        )

    # -------------------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(

        fusion_model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # Model returns probabilities, so use BCELoss.

    bce_criterion = nn.BCELoss()

    # -------------------------------------------------------------------------
    # AMP
    # -------------------------------------------------------------------------

    amp_enabled = (

        USE_AMP

        and DEVICE.type == "cuda"
    )

    scaler = GradScaler(

        "cuda",

        enabled=amp_enabled,
    )

    # -------------------------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # -------------------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=8,

        min_lr=1e-6,
    )

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    banner(
        "STARTING FUSION TRAINING"
    )

    history: List[
        Dict[str, Any]
    ] = []

    best_val_f1 = float(
        "-inf"
    )

    best_epoch = 0

    best_validation_probabilities = None

    best_validation_labels = None

    calibrated_threshold = 0.50

    epochs_without_improvement = 0

    training_start = time.time()

    for epoch in range(

        1,

        NUM_EPOCHS + 1,
    ):

        epoch_start = time.time()

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        train_metrics = run_epoch(

            loader=train_loader,

            fusion_model=fusion_model,

            optimizer=optimizer,

            bce_criterion=bce_criterion,

            scaler=scaler,

            training=True,
        )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        validation_metrics = run_epoch(

            loader=validation_loader,

            fusion_model=fusion_model,

            optimizer=optimizer,

            bce_criterion=bce_criterion,

            scaler=scaler,

            training=False,
        )

        # ---------------------------------------------------------------------
        # LR SCHEDULER
        # ---------------------------------------------------------------------

        scheduler.step(

            validation_metrics["f1"]
        )

        current_lr = (

            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (

            time.time()
            - epoch_start
        )

        # ---------------------------------------------------------------------
        # HISTORY
        # ---------------------------------------------------------------------

        row = {

            "epoch":
                epoch,

            "learning_rate":
                current_lr,

            "train_loss":
                train_metrics["loss"],

            "train_bce_loss":
                train_metrics["bce_loss"],

            "train_dice_loss":
                train_metrics["dice_loss"],

            "train_residual_loss":
                train_metrics["residual_loss"],

            "train_gate_supervision_loss":
                train_metrics["gate_supervision_loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "train_iou":
                train_metrics["iou"],

            "train_specificity":
                train_metrics["specificity"],

            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

            "val_loss":
                validation_metrics["loss"],

            "val_bce_loss":
                validation_metrics["bce_loss"],

            "val_dice_loss":
                validation_metrics["dice_loss"],

            "val_residual_loss":
                validation_metrics[
                    "residual_loss"
                ],

            "val_gate_supervision_loss":
                validation_metrics[
                    "gate_supervision_loss"
                ],

            "val_accuracy":
                validation_metrics["accuracy"],

            "val_precision":
                validation_metrics["precision"],

            "val_recall":
                validation_metrics["recall"],

            "val_f1":
                validation_metrics["f1"],

            "val_iou":
                validation_metrics["iou"],

            "val_specificity":
                validation_metrics["specificity"],

            "val_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            row
        )

        save_history(
            history
        )

        # ---------------------------------------------------------------------
        # PRINT METRICS
        # ---------------------------------------------------------------------

        print(

            f"  Train | "

            f"Loss={train_metrics['loss']:.6f} | "

            f"BCE={train_metrics['bce_loss']:.6f} | "

            f"Gate={train_metrics['gate_supervision_loss']:.6f} | "

            f"F1={train_metrics['f1']:.4f} | "

            f"IoU={train_metrics['iou']:.4f}"
        )

        print(

            f"  Val   | "

            f"Loss={validation_metrics['loss']:.6f} | "

            f"BCE={validation_metrics['bce_loss']:.6f} | "

            f"Gate={validation_metrics['gate_supervision_loss']:.6f} | "

            f"F1={validation_metrics['f1']:.4f} | "

            f"IoU={validation_metrics['iou']:.4f}"
        )

        print(
            f"  LR    | "
            f"{current_lr:.8f}"
        )

        # ---------------------------------------------------------------------
        # SAVE LAST CHECKPOINT
        # ---------------------------------------------------------------------

        save_checkpoint(

            path=LAST_CHECKPOINT_PATH,

            epoch=epoch,

            fusion_model=fusion_model,

            optimizer=optimizer,

            scheduler=scheduler,

            metrics=validation_metrics,

            history=history,

            best_val_f1=best_val_f1,

            decision_threshold=calibrated_threshold,
        )

        # ---------------------------------------------------------------------
        # BEST CHECKPOINT / EARLY STOPPING
        # ---------------------------------------------------------------------

        improved = (

            validation_metrics["f1"]

            >

            best_val_f1
            + EARLY_STOPPING_MIN_DELTA
        )

        if improved:

            best_val_f1 = (

                validation_metrics["f1"]
            )

            best_epoch = epoch

            best_validation_probabilities = validation_metrics[
                "probabilities"
            ].clone()

            best_validation_labels = validation_metrics[
                "labels"
            ].clone()

            epochs_without_improvement = 0

            save_checkpoint(

                path=BEST_CHECKPOINT_PATH,

                epoch=epoch,

                fusion_model=fusion_model,

                optimizer=optimizer,

                scheduler=scheduler,

                metrics=validation_metrics,

                history=history,

                best_val_f1=best_val_f1,

                decision_threshold=calibrated_threshold,
            )

            print(
                "\n  [BEST] New best validation F1"
            )

            print(
                f"         Epoch : "
                f"{best_epoch}"
            )

            print(
                f"         F1    : "
                f"{best_val_f1:.6f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                "\n  No validation F1 improvement."
            )

            print(
                f"  Early stopping counter: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ---------------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------------

        if (

            epochs_without_improvement

            >=

            EARLY_STOPPING_PATIENCE
        ):

            print(
                "\n[EARLY STOPPING]"
            )

            print(
                f"No validation F1 improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    # -------------------------------------------------------------------------
    # TRAINING COMPLETE
    # -------------------------------------------------------------------------

    total_training_seconds = (

        time.time()
        - training_start
    )

    if (
        best_validation_probabilities is None
        or best_validation_labels is None
    ):

        raise RuntimeError(
            "No best validation predictions were captured."
        )

    calibrated_threshold, calibrated_metrics = find_best_threshold(
        best_validation_probabilities,
        best_validation_labels,
    )

    best_checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    best_checkpoint["decision_threshold"] = calibrated_threshold

    best_checkpoint[
        "calibrated_validation_metrics"
    ] = calibrated_metrics

    torch.save(
        best_checkpoint,
        BEST_CHECKPOINT_PATH,
    )

    epochs_completed = len(
        history
    )

    summary = {

        "device":
            str(DEVICE),

        "epochs_requested":
            NUM_EPOCHS,

        "epochs_completed":
            epochs_completed,

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            best_val_f1,

        "decision_threshold":
            calibrated_threshold,

        "total_training_seconds":
            total_training_seconds,

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(validation_dataset),

        "test_samples_used":
            0,

        "cache_training":
            True,

        "registration_free":
            True,

        "sar_expert_loaded":
            False,

        "hsi_expert_loaded":
            False,

        "fusion_only_trainable":
            True,

        "best_checkpoint":
            str(BEST_CHECKPOINT_PATH),

        "last_checkpoint":
            str(LAST_CHECKPOINT_PATH),

        "train_cache":
            str(TRAIN_CACHE_PATH),

        "validation_cache":
            str(VALIDATION_CACHE_PATH),

        "test_cache_not_used":
            str(TEST_CACHE_PATH),

        "sar_feature_order": [

            "center_probability",

            "local_mean_probability",

            "local_standard_deviation",

            "local_minimum_probability",

            "local_maximum_probability",

            "local_contrast",
        ],
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

    banner(
        "FUSION TRAINING COMPLETE"
    )

    print(
        f"\nEpochs completed     : "
        f"{epochs_completed}"
    )

    print(
        f"Best epoch            : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1    : "
        f"{best_val_f1:.6f}"
    )

    print(
        f"Training time         : "
        f"{total_training_seconds / 60.0:.2f} minutes"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        f"  {BEST_CHECKPOINT_PATH}"
    )

    print(
        "\nLast checkpoint:"
    )

    print(
        f"  {LAST_CHECKPOINT_PATH}"
    )

    print(
        "\nTraining history:"
    )

    print(
        f"  {HISTORY_JSON_PATH}"
    )

    print(
        f"  {HISTORY_CSV_PATH}"
    )

    print(
        "\nSummary:"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        "\n[SUCCESS] Cache-only training completed."
    )

    print(
        "[SUCCESS] SAR expert was not loaded."
    )

    print(
        "[SUCCESS] HSI expert was not loaded."
    )

    print(
        "[SUCCESS] No HSI preprocessing was performed."
    )

    print(
        "[SUCCESS] Only RegistrationFreeFusionNet was optimized."
    )

    print(
        "[SUCCESS] Test cache was not used during training."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()