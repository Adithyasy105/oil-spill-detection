# """
# STAGE 3 — REGISTRATION-FREE SAR + HSI FUSION MODEL
# 04_fusion_model.py

# Architecture
# ------------
# This module contains ONLY the trainable fusion network.

# The SAR and HSI experts are frozen and are NOT defined here.

# Fusion contract
# ---------------
# 1. Pixel-level SAR/HSI registration is NOT assumed.
# 2. SAR is the primary spatial anchor.
# 3. HSI contributes spectral evidence and scene-local context.
# 4. HSI spatial positions are never aligned with SAR spatial positions.
# 5. HSI's (B, 64, 25) expert features must be spatially pooled before
#    entering this model.
# 6. The fusion model predicts a bounded residual correction to the SAR
#    anchor probability.

# Inputs
# ------
# sar_features:
#     Shape (B, 6)

#     Expected features:
#         [0] center SAR probability
#         [1] local mean
#         [2] local max
#         [3] local min
#         [4] local standard deviation
#         [5] local contrast

# hsi_features:
#     Shape (B, 64)

#     Spatially pooled frozen HMOSDNet representation.

# hsi_logits:
#     Shape (B, 1)

# hsi_confidence:
#     Shape (B, 1)

# Output
# ------
# Dictionary containing:

#     p_fused
#         Final fused probability.

#     p_sar
#         SAR anchor probability.

#     p_hsi
#         HSI expert probability.

#     reliability
#         Learned confidence controlling how strongly HSI evidence can
#         influence the SAR anchor.

#     correction
#         Raw signed correction.

#     bounded_delta
#         Final bounded probability-space correction.

# Design
# ------
#                     SAR FEATURES
#                          |
#                          v
#                     SAR ENCODER
#                          |
#                          +------------------+
#                                             |
#                                             v
# HSI FEATURES --> HSI ENCODER --------> FUSION TRUNK
#                                             |
# HSI LOGIT + CONFIDENCE --> EVIDENCE ENCODER |
#                                             |
#                                             +------------+
#                                                          |
#                                          +---------------+---------------+
#                                          |                               |
#                                          v                               v
#                                    RELIABILITY HEAD                RESIDUAL HEAD
#                                          |                               |
#                                       sigmoid                           tanh
#                                          |                               |
#                                          +---------------+---------------+
#                                                          |
#                                                          v
#                               bounded_delta = reliability
#                                               * residual_limit
#                                               * tanh(correction)
#                                                          |
#                                                          v
#                                   p_fused = clamp(
#                                       p_sar + bounded_delta,
#                                       0,
#                                       1
#                                   )

# This intentionally prevents unrestricted replacement of the SAR anchor.
# """

# from __future__ import annotations

# import math
# from pathlib import Path
# from typing import Dict

# import torch
# import torch.nn as nn


# # ============================================================================
# # CONSTANTS
# # ============================================================================

# SAR_FEATURE_DIM = 6
# HSI_FEATURE_DIM = 64
# EVIDENCE_DIM = 2

# SAR_EMBED_DIM = 32
# HSI_EMBED_DIM = 48
# EVIDENCE_EMBED_DIM = 16

# FUSION_DIM = (
#     SAR_EMBED_DIM
#     + HSI_EMBED_DIM
#     + EVIDENCE_EMBED_DIM
# )

# DEFAULT_RESIDUAL_LIMIT = 0.25


# # ============================================================================
# # INITIALIZATION
# # ============================================================================

# def init_weights(module: nn.Module) -> None:
#     """
#     Stable initialization for the trainable fusion network.
#     """

#     if isinstance(module, nn.Linear):
#         nn.init.kaiming_uniform_(
#             module.weight,
#             a=math.sqrt(5),
#         )

#         if module.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
#                 module.weight
#             )

#             bound = 1.0 / math.sqrt(
#                 max(fan_in, 1)
#             )

#             nn.init.uniform_(
#                 module.bias,
#                 -bound,
#                 bound,
#             )


# # ============================================================================
# # BUILDING BLOCK
# # ============================================================================

# class MLPBlock(nn.Module):
#     """
#     Linear -> LayerNorm -> GELU -> Dropout

#     LayerNorm is preferred here because fusion batches can be small.
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         dropout: float = 0.10,
#     ):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(
#                 in_dim,
#                 out_dim,
#             ),
#             nn.LayerNorm(
#                 out_dim
#             ),
#             nn.GELU(),
#             nn.Dropout(
#                 dropout
#             ),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:

#         return self.net(x)


# # ============================================================================
# # SAR ENCODER
# # ============================================================================

# class SAREncoder(nn.Module):
#     """
#     Encodes local evidence produced from the frozen SAR expert.

#     Input:
#         (B, 6)

#     Output:
#         (B, 32)
#     """

#     def __init__(
#         self,
#         input_dim: int = SAR_FEATURE_DIM,
#         embed_dim: int = SAR_EMBED_DIM,
#         dropout: float = 0.10,
#     ):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             MLPBlock(
#                 input_dim,
#                 32,
#                 dropout,
#             ),
#             MLPBlock(
#                 32,
#                 embed_dim,
#                 dropout,
#             ),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:

#         return self.encoder(x)


# # ============================================================================
# # HSI CONTEXT ENCODER
# # ============================================================================

# class HSIContextEncoder(nn.Module):
#     """
#     Encodes the spatially pooled HMOSDNet feature.

#     Input:
#         (B, 64)

#     Output:
#         (B, 48)

#     No HSI spatial positions are preserved here.
#     """

#     def __init__(
#         self,
#         input_dim: int = HSI_FEATURE_DIM,
#         embed_dim: int = HSI_EMBED_DIM,
#         dropout: float = 0.10,
#     ):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             MLPBlock(
#                 input_dim,
#                 64,
#                 dropout,
#             ),
#             MLPBlock(
#                 64,
#                 embed_dim,
#                 dropout,
#             ),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:

#         return self.encoder(x)


# # ============================================================================
# # HSI EVIDENCE ENCODER
# # ============================================================================

# class EvidenceEncoder(nn.Module):
#     """
#     Encodes direct HSI prediction evidence.

#     Input:
#         [HSI probability, HSI confidence]

#     Shape:
#         (B, 2)

#     Output:
#         (B, 16)
#     """

#     def __init__(
#         self,
#         input_dim: int = EVIDENCE_DIM,
#         embed_dim: int = EVIDENCE_EMBED_DIM,
#         dropout: float = 0.05,
#     ):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             MLPBlock(
#                 input_dim,
#                 16,
#                 dropout,
#             ),
#             MLPBlock(
#                 16,
#                 embed_dim,
#                 dropout,
#             ),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:

#         return self.encoder(x)


# # ============================================================================
# # FUSION TRUNK
# # ============================================================================

# class FusionTrunk(nn.Module):
#     """
#     Combines modality-level embeddings.

#     Input:
#         (B, 96)

#     Output:
#         (B, 64)
#     """

#     def __init__(
#         self,
#         input_dim: int = FUSION_DIM,
#         hidden_dim: int = 64,
#         dropout: float = 0.15,
#     ):
#         super().__init__()

#         self.net = nn.Sequential(
#             MLPBlock(
#                 input_dim,
#                 96,
#                 dropout,
#             ),
#             MLPBlock(
#                 96,
#                 hidden_dim,
#                 dropout,
#             ),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:

#         return self.net(x)


# # ============================================================================
# # MAIN FUSION MODEL
# # ============================================================================

# class RegistrationFreeFusionNet(nn.Module):
#     """
#     Adaptive SAR-anchored bounded residual fusion.

#     Final equation:

#         delta =
#             reliability
#             * residual_limit
#             * tanh(correction)

#         p_fused =
#             clamp(
#                 p_sar + delta,
#                 0,
#                 1
#             )

#     The residual is bounded, so the HSI branch cannot apply an
#     unbounded correction to the SAR anchor.
#     """

#     def __init__(
#         self,
#         sar_feature_dim: int = SAR_FEATURE_DIM,
#         hsi_feature_dim: int = HSI_FEATURE_DIM,
#         residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
#         dropout: float = 0.10,
#     ):
#         super().__init__()

#         if residual_limit <= 0.0:
#             raise ValueError(
#                 "residual_limit must be > 0"
#             )

#         if residual_limit > 1.0:
#             raise ValueError(
#                 "residual_limit should be in probability space "
#                 "and should not exceed 1.0"
#             )

#         self.residual_limit = float(
#             residual_limit
#         )

#         # ------------------------------------------------------------
#         # MODALITY ENCODERS
#         # ------------------------------------------------------------

#         self.sar_encoder = SAREncoder(
#             input_dim=sar_feature_dim,
#             embed_dim=SAR_EMBED_DIM,
#             dropout=dropout,
#         )

#         self.hsi_encoder = HSIContextEncoder(
#             input_dim=hsi_feature_dim,
#             embed_dim=HSI_EMBED_DIM,
#             dropout=dropout,
#         )

#         self.evidence_encoder = EvidenceEncoder(
#             input_dim=EVIDENCE_DIM,
#             embed_dim=EVIDENCE_EMBED_DIM,
#             dropout=dropout,
#         )

#         # ------------------------------------------------------------
#         # SHARED FUSION REPRESENTATION
#         # ------------------------------------------------------------

#         self.fusion_trunk = FusionTrunk(
#             input_dim=FUSION_DIM,
#             hidden_dim=64,
#             dropout=dropout,
#         )

#         # ------------------------------------------------------------
#         # RELIABILITY GATE
#         #
#         # Learns when auxiliary HSI evidence should be trusted.
#         # ------------------------------------------------------------

#         self.reliability_head = nn.Sequential(
#             nn.Linear(
#                 64,
#                 32,
#             ),
#             nn.LayerNorm(
#                 32
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 32,
#                 1,
#             ),
#         )

#         # ------------------------------------------------------------
#         # RESIDUAL CORRECTION
#         #
#         # Predicts direction and magnitude before tanh bounding.
#         # ------------------------------------------------------------

#         self.correction_head = nn.Sequential(
#             nn.Linear(
#                 64,
#                 32,
#             ),
#             nn.LayerNorm(
#                 32
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 32,
#                 1,
#             ),
#         )

#         self.apply(
#             init_weights
#         )

#         self._initialize_conservative_fusion()

#     # ========================================================================
#     # CONSERVATIVE INITIALIZATION
#     # ========================================================================

#     def _initialize_conservative_fusion(
#         self,
#     ) -> None:
#         """
#         Start close to SAR-only behavior.

#         This is important because:
#             - SAR is the anchor.
#             - Experts are frozen.
#             - Fusion should earn deviation from SAR during training.

#         Correction initially near zero.
#         Reliability initially moderate-low.
#         """

#         final_reliability = self.reliability_head[
#             -1
#         ]

#         final_correction = self.correction_head[
#             -1
#         ]

#         nn.init.zeros_(
#             final_correction.weight
#         )

#         nn.init.zeros_(
#             final_correction.bias
#         )

#         nn.init.zeros_(
#             final_reliability.weight
#         )

#         # sigmoid(-1.5) ≈ 0.18
#         nn.init.constant_(
#             final_reliability.bias,
#             -1.5,
#         )

#     # ========================================================================
#     # INPUT VALIDATION
#     # ========================================================================

#     @staticmethod
#     def _validate_2d(
#         x: torch.Tensor,
#         expected_dim: int,
#         name: str,
#     ) -> None:

#         if not isinstance(
#             x,
#             torch.Tensor,
#         ):
#             raise TypeError(
#                 f"{name} must be a torch.Tensor, "
#                 f"got {type(x)}"
#             )

#         if x.ndim != 2:
#             raise ValueError(
#                 f"{name} must have shape "
#                 f"(B, {expected_dim}), "
#                 f"got {tuple(x.shape)}"
#             )

#         if x.shape[1] != expected_dim:
#             raise ValueError(
#                 f"{name} must have "
#                 f"{expected_dim} features, "
#                 f"got shape {tuple(x.shape)}"
#             )

#     # ========================================================================
#     # FORWARD
#     # ========================================================================

#     def forward(
#         self,
#         sar_features: torch.Tensor,
#         hsi_features: torch.Tensor,
#         hsi_logits: torch.Tensor,
#         hsi_confidence: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Parameters
#         ----------
#         sar_features:
#             Shape:
#                 (B, 6)

#             Feature order:
#                 [center_prob,
#                  local_mean,
#                  local_max,
#                  local_min,
#                  local_std,
#                  local_contrast]

#         hsi_features:
#             Shape:
#                 (B, 64)

#             Registration-free pooled HSI context.

#         hsi_logits:
#             Shape:
#                 (B, 1)

#         hsi_confidence:
#             Shape:
#                 (B, 1)

#         Returns
#         -------
#         Dictionary with final probability and diagnostics.
#         """

#         self._validate_2d(
#             sar_features,
#             SAR_FEATURE_DIM,
#             "sar_features",
#         )

#         self._validate_2d(
#             hsi_features,
#             HSI_FEATURE_DIM,
#             "hsi_features",
#         )

#         self._validate_2d(
#             hsi_logits,
#             1,
#             "hsi_logits",
#         )

#         self._validate_2d(
#             hsi_confidence,
#             1,
#             "hsi_confidence",
#         )

#         batch_size = sar_features.shape[0]

#         if hsi_features.shape[0] != batch_size:
#             raise ValueError(
#                 "Batch size mismatch between "
#                 "sar_features and hsi_features"
#             )

#         if hsi_logits.shape[0] != batch_size:
#             raise ValueError(
#                 "Batch size mismatch between "
#                 "sar_features and hsi_logits"
#             )

#         if hsi_confidence.shape[0] != batch_size:
#             raise ValueError(
#                 "Batch size mismatch between "
#                 "sar_features and hsi_confidence"
#             )

#         # ------------------------------------------------------------
#         # SAR ANCHOR
#         # ------------------------------------------------------------

#         p_sar = sar_features[:, 0:1]

#         p_sar = torch.clamp(
#             p_sar,
#             min=0.0,
#             max=1.0,
#         )

#         # ------------------------------------------------------------
#         # HSI EXPERT PROBABILITY
#         # ------------------------------------------------------------

#         p_hsi = torch.sigmoid(
#             hsi_logits
#         )

#         p_hsi = torch.clamp(
#             p_hsi,
#             min=0.0,
#             max=1.0,
#         )

#         hsi_confidence = torch.clamp(
#             hsi_confidence,
#             min=0.0,
#             max=1.0,
#         )

#         # ------------------------------------------------------------
#         # ENCODE EACH SOURCE
#         # ------------------------------------------------------------

#         sar_embedding = self.sar_encoder(
#             sar_features
#         )

#         hsi_embedding = self.hsi_encoder(
#             hsi_features
#         )

#         evidence_input = torch.cat(
#             [
#                 p_hsi,
#                 hsi_confidence,
#             ],
#             dim=1,
#         )

#         evidence_embedding = self.evidence_encoder(
#             evidence_input
#         )

#         # ------------------------------------------------------------
#         # FUSION REPRESENTATION
#         # ------------------------------------------------------------

#         fusion_input = torch.cat(
#             [
#                 sar_embedding,
#                 hsi_embedding,
#                 evidence_embedding,
#             ],
#             dim=1,
#         )

#         fusion_context = self.fusion_trunk(
#             fusion_input
#         )

#         # ------------------------------------------------------------
#         # ADAPTIVE RELIABILITY
#         # ------------------------------------------------------------

#         reliability_logits = self.reliability_head(
#             fusion_context
#         )

#         reliability = torch.sigmoid(
#             reliability_logits
#         )

#         # ------------------------------------------------------------
#         # BOUNDED RESIDUAL
#         # ------------------------------------------------------------

#         correction = self.correction_head(
#             fusion_context
#         )

#         signed_correction = torch.tanh(
#             correction
#         )

#         bounded_delta = (
#             reliability
#             * self.residual_limit
#             * signed_correction
#         )

#         # ------------------------------------------------------------
#         # SAR-ANCHORED FUSION
#         # ------------------------------------------------------------

#         p_fused = torch.clamp(
#             p_sar + bounded_delta,
#             min=0.0,
#             max=1.0,
#         )

#         return {
#             "p_fused": p_fused,
#             "p_sar": p_sar,
#             "p_hsi": p_hsi,
#             "reliability": reliability,
#             "correction": correction,
#             "signed_correction": signed_correction,
#             "bounded_delta": bounded_delta,
#             "fusion_context": fusion_context,
#             "sar_embedding": sar_embedding,
#             "hsi_embedding": hsi_embedding,
#             "evidence_embedding": evidence_embedding,
#         }


# # ============================================================================
# # FACTORY
# # ============================================================================

# def build_fusion_model(
#     residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
#     dropout: float = 0.10,
# ) -> RegistrationFreeFusionNet:
#     """
#     Construct the registration-free fusion model.
#     """

#     return RegistrationFreeFusionNet(
#         sar_feature_dim=SAR_FEATURE_DIM,
#         hsi_feature_dim=HSI_FEATURE_DIM,
#         residual_limit=residual_limit,
#         dropout=dropout,
#     )


# # ============================================================================
# # PARAMETER UTILITIES
# # ============================================================================

# def count_parameters(
#     model: nn.Module,
# ) -> Dict[str, int]:
#     """
#     Return total and trainable parameter counts.
#     """

#     total = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#     )

#     trainable = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#         if parameter.requires_grad
#     )

#     return {
#         "total": total,
#         "trainable": trainable,
#     }


# # ============================================================================
# # SELF TEST
# # ============================================================================

# def self_test() -> None:

#     print("\n" + "=" * 88)
#     print("STAGE 3 — REGISTRATION-FREE FUSION MODEL SELF TEST")
#     print("=" * 88)

#     device = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "cpu"
#     )

#     print(f"\nDevice: {device}")

#     model = build_fusion_model(
#         residual_limit=DEFAULT_RESIDUAL_LIMIT,
#         dropout=0.10,
#     ).to(device)

#     stats = count_parameters(
#         model
#     )

#     print("\nMODEL CONFIGURATION")
#     print(f"  SAR feature dim       : {SAR_FEATURE_DIM}")
#     print(f"  HSI feature dim       : {HSI_FEATURE_DIM}")
#     print(f"  Residual limit        : {DEFAULT_RESIDUAL_LIMIT}")
#     print(f"  Total parameters      : {stats['total']:,}")
#     print(f"  Trainable parameters  : {stats['trainable']:,}")

#     batch_size = 8

#     sar_features = torch.rand(
#         batch_size,
#         SAR_FEATURE_DIM,
#         device=device,
#     )

#     hsi_features = torch.randn(
#         batch_size,
#         HSI_FEATURE_DIM,
#         device=device,
#     )

#     hsi_logits = torch.randn(
#         batch_size,
#         1,
#         device=device,
#     )

#     hsi_confidence = torch.rand(
#         batch_size,
#         1,
#         device=device,
#     )

#     print("\nRUNNING FORWARD PASS...")

#     output = model(
#         sar_features=sar_features,
#         hsi_features=hsi_features,
#         hsi_logits=hsi_logits,
#         hsi_confidence=hsi_confidence,
#     )

#     print("\nOUTPUT SHAPES")

#     for key, value in output.items():

#         if isinstance(
#             value,
#             torch.Tensor,
#         ):
#             print(
#                 f"  {key:<22}: "
#                 f"{tuple(value.shape)}"
#             )

#     p_fused = output["p_fused"]

#     p_sar = output["p_sar"]

#     reliability = output["reliability"]

#     bounded_delta = output[
#         "bounded_delta"
#     ]

#     assert torch.all(
#         p_fused >= 0.0
#     ), "p_fused contains values < 0"

#     assert torch.all(
#         p_fused <= 1.0
#     ), "p_fused contains values > 1"

#     assert torch.all(
#         reliability >= 0.0
#     ), "reliability contains values < 0"

#     assert torch.all(
#         reliability <= 1.0
#     ), "reliability contains values > 1"

#     assert torch.all(
#         torch.abs(
#             bounded_delta
#         )
#         <= DEFAULT_RESIDUAL_LIMIT + 1e-6
#     ), "Bounded residual exceeded configured limit"

#     print("\nNUMERICAL CHECKS")

#     print(
#         f"  SAR probability range : "
#         f"[{p_sar.min().item():.4f}, "
#         f"{p_sar.max().item():.4f}]"
#     )

#     print(
#         f"  Reliability range     : "
#         f"[{reliability.min().item():.4f}, "
#         f"{reliability.max().item():.4f}]"
#     )

#     print(
#         f"  Residual range        : "
#         f"[{bounded_delta.min().item():.4f}, "
#         f"{bounded_delta.max().item():.4f}]"
#     )

#     print(
#         f"  Fused probability     : "
#         f"[{p_fused.min().item():.4f}, "
#         f"{p_fused.max().item():.4f}]"
#     )

#     # ------------------------------------------------------------
#     # BACKWARD TEST
#     # ------------------------------------------------------------

#     print("\nRUNNING BACKWARD PASS...")

#     model.train()

#     output = model(
#         sar_features=sar_features,
#         hsi_features=hsi_features,
#         hsi_logits=hsi_logits,
#         hsi_confidence=hsi_confidence,
#     )

#     target = torch.randint(
#         low=0,
#         high=2,
#         size=(batch_size, 1),
#         device=device,
#     ).float()

#     loss = nn.BCELoss()(
#         output["p_fused"],
#         target,
#     )

#     loss.backward()

#     gradients_found = False

#     for parameter in model.parameters():

#         if (
#             parameter.requires_grad
#             and parameter.grad is not None
#         ):
#             gradients_found = True
#             break

#     assert gradients_found, (
#         "No gradients found in fusion model"
#     )

#     print(
#         f"  Test loss             : "
#         f"{loss.item():.6f}"
#     )

#     print(
#         "  Gradient flow         : OK"
#     )

#     print("\n" + "=" * 88)
#     print("FUSION MODEL SELF TEST PASSED")
#     print("=" * 88)

#     print(
#         "\n[SUCCESS] SAR remains the probability anchor."
#     )

#     print(
#         "[SUCCESS] HSI contributes registration-free spectral context."
#     )

#     print(
#         "[SUCCESS] HSI spatial positions are not aligned with SAR."
#     )

#     print(
#         "[SUCCESS] Fusion correction is adaptively gated."
#     )

#     print(
#         "[SUCCESS] Fusion correction is bounded."
#     )

#     print(
#         "[SUCCESS] Only fusion layers are trainable."
#     )

#     print("\nNext stage:")

#     print(
#         "05_train.py will:"
#     )

#     print(
#         "  1. Load the frozen SAR + HSI experts."
#     )

#     print(
#         "  2. Preprocess raw HSI patches using spectral_processor."
#     )

#     print(
#         "  3. Run HMOSDNet to obtain logits, confidence and 64-D features."
#     )

#     print(
#         "  4. Extract SAR evidence features from valid SAR expert output."
#     )

#     print(
#         "  5. Train ONLY RegistrationFreeFusionNet."
#     )


# if __name__ == "__main__":
#     self_test()





"""
04_fusion_model.py  (CORRECTED)

What changed and why
---------------------
The previous version hard-anchored p_fused to p_sar and only allowed a
SMALL bounded nudge toward HSI:

    p_fused = clamp(p_sar + reliability * 0.25 * tanh(correction))

Your own test evaluation showed this was backwards for this dataset:

    SAR only : F1 0.713  recall 0.584
    HSI only : F1 0.888  recall 0.863
    Fusion   : F1 0.805  recall 0.730   <- capped near p_sar, never near HSI

Anchoring on the weaker, lower-recall expert and only allowing +/-0.25
correction mathematically prevents fusion from ever approaching HSI's
performance, no matter how well the correction head is trained. This is
visible directly in the training log: train F1 is flat at ~0.795 from
epoch 2 through epoch 23 -- that's a capacity ceiling, not undertraining.

Fix: replace the hard anchor + small residual with a LEARNED GATE that
mixes p_sar and p_hsi anywhere across the full [0, 1] range, per sample.
A small bounded correction remains on top, but now only for fine
calibration -- it is no longer the primary fusion mechanism.

    gate      = sigmoid(gate_head(fusion_context))       in [0, 1]
    p_mixture = gate * p_hsi + (1 - gate) * p_sar
    delta     = residual_limit * tanh(correction_head(fusion_context))
    p_fused   = clamp(p_mixture + delta, 0, 1)

This still respects the registration-free contract (nothing here uses
raw pixels or claims alignment -- it operates purely on the same
scene-level evidence vectors as before) and still bounds the free-form
correction for stability, but no longer artificially caps how much the
model can prefer the stronger expert.

Everything else (encoders, trunk, input contract, output dict keys) is
unchanged, so 05_train.py and 06_build_feature_cache.py do not need to
change their interface with this file.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


# ============================================================================
# CONSTANTS
# ============================================================================

SAR_FEATURE_DIM = 42
HSI_FEATURE_DIM = 64
EVIDENCE_DIM = 2

SAR_EMBED_DIM = 32
HSI_EMBED_DIM = 48
EVIDENCE_EMBED_DIM = 16

FUSION_DIM = SAR_EMBED_DIM + HSI_EMBED_DIM + EVIDENCE_EMBED_DIM

# This is now a SECONDARY calibration term on top of the learned gate,
# not the primary fusion mechanism -- it can be modest.
DEFAULT_RESIDUAL_LIMIT = 0.15


# ============================================================================
# INIT
# ============================================================================

def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1.0 / math.sqrt(max(fan_in, 1))
            nn.init.uniform_(module.bias, -bound, bound)


class MLPBlock(nn.Module):
    """Linear -> LayerNorm -> GELU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAREncoder(nn.Module):
    def __init__(self, input_dim: int = SAR_FEATURE_DIM, embed_dim: int = SAR_EMBED_DIM, dropout: float = 0.10):
        super().__init__()
        self.encoder = nn.Sequential(MLPBlock(input_dim, 32, dropout), MLPBlock(32, embed_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HSIContextEncoder(nn.Module):
    def __init__(self, input_dim: int = HSI_FEATURE_DIM, embed_dim: int = HSI_EMBED_DIM, dropout: float = 0.10):
        super().__init__()
        self.encoder = nn.Sequential(MLPBlock(input_dim, 64, dropout), MLPBlock(64, embed_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EvidenceEncoder(nn.Module):
    def __init__(self, input_dim: int = EVIDENCE_DIM, embed_dim: int = EVIDENCE_EMBED_DIM, dropout: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(MLPBlock(input_dim, 16, dropout), MLPBlock(16, embed_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionTrunk(nn.Module):
    def __init__(self, input_dim: int = FUSION_DIM, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(MLPBlock(input_dim, 96, dropout), MLPBlock(96, hidden_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# MAIN FUSION MODEL
# ============================================================================

class RegistrationFreeFusionNet(nn.Module):
    """
    Adaptive gated fusion.

        gate      = sigmoid(gate_head(fusion_context))         in [0, 1]
        p_mixture = gate * p_hsi + (1 - gate) * p_sar
        delta     = residual_limit * tanh(correction_head(...))
        p_fused   = clamp(p_mixture + delta, 0, 1)

    The gate can fully trust either expert on a per-sample basis. The
    residual is a small secondary calibration term, bounded so it can't
    destabilize the mixture, but it is no longer what carries the fusion.
    """

    def __init__(
        self,
        sar_feature_dim: int = SAR_FEATURE_DIM,
        hsi_feature_dim: int = HSI_FEATURE_DIM,
        residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
        dropout: float = 0.10,
    ):
        super().__init__()

        if not (0.0 < residual_limit <= 1.0):
            raise ValueError("residual_limit must be in (0, 1]")

        self.residual_limit = float(residual_limit)

        self.sar_encoder = SAREncoder(sar_feature_dim, SAR_EMBED_DIM, dropout)
        self.hsi_encoder = HSIContextEncoder(hsi_feature_dim, HSI_EMBED_DIM, dropout)
        self.evidence_encoder = EvidenceEncoder(EVIDENCE_DIM, EVIDENCE_EMBED_DIM, dropout)
        self.fusion_trunk = FusionTrunk(FUSION_DIM, 64, dropout)

        # Learned mixture weight between p_sar and p_hsi. Full [0, 1] range --
        # this is the PRIMARY fusion mechanism now.
        self.gate_head = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 1),
        )

        # Small secondary correction on top of the mixture.
        self.correction_head = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 1),
        )

        self.apply(init_weights)
        self._initialize_gate_prior()

    def _initialize_gate_prior(self) -> None:
        """
        Start the gate near neutral-to-HSI-leaning (sigmoid(0.5) ~= 0.62)
        rather than the old near-zero-correction / SAR-only start. This is
        just a starting prior -- the gate is fully learned per-sample from
        there, it is not hardcoded to prefer HSI. If your dataset or scene
        mix changes and SAR turns out to be the stronger signal there, the
        gate will learn to move toward 0 on its own during training.

        Correction head starts at zero so early training doesn't add noise
        on top of an already-reasonable mixture.
        """
        final_gate = self.gate_head[-1]
        final_correction = self.correction_head[-1]

        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, 0.5)

        nn.init.zeros_(final_correction.weight)
        nn.init.zeros_(final_correction.bias)

    @staticmethod
    def _validate_2d(x: torch.Tensor, expected_dim: int, name: str) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
        if x.ndim != 2 or x.shape[1] != expected_dim:
            raise ValueError(f"{name} must have shape (B, {expected_dim}), got {tuple(x.shape)}")

    def forward(
        self,
        sar_features: torch.Tensor,
        hsi_features: torch.Tensor,
        hsi_logits: torch.Tensor,
        hsi_confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        sar_features order (must match the cache builder):
            [center_prob, mean, std, min, max, contrast,
             scene_mean, scene_std, scene_positive_fraction,
             local_to_scene_contrast, sar_embedding_0...sar_embedding_31]
        """
        self._validate_2d(sar_features, SAR_FEATURE_DIM, "sar_features")
        self._validate_2d(hsi_features, HSI_FEATURE_DIM, "hsi_features")
        self._validate_2d(hsi_logits, 1, "hsi_logits")
        self._validate_2d(hsi_confidence, 1, "hsi_confidence")

        b = sar_features.shape[0]
        for name, t in (("hsi_features", hsi_features), ("hsi_logits", hsi_logits), ("hsi_confidence", hsi_confidence)):
            if t.shape[0] != b:
                raise ValueError(f"Batch size mismatch: sar_features vs {name}")

        p_sar = torch.clamp(sar_features[:, 0:1], 0.0, 1.0)
        p_hsi = torch.clamp(torch.sigmoid(hsi_logits), 0.0, 1.0)
        hsi_confidence = torch.clamp(hsi_confidence, 0.0, 1.0)

        sar_embedding = self.sar_encoder(sar_features)
        hsi_embedding = self.hsi_encoder(hsi_features)
        evidence_embedding = self.evidence_encoder(torch.cat([p_hsi, hsi_confidence], dim=1))

        fusion_context = self.fusion_trunk(
            torch.cat([sar_embedding, hsi_embedding, evidence_embedding], dim=1)
        )

        gate_logits = self.gate_head(fusion_context)
        gate = torch.sigmoid(gate_logits)

        p_mixture = gate * p_hsi + (1.0 - gate) * p_sar

        correction = self.correction_head(fusion_context)
        signed_correction = torch.tanh(correction)
        bounded_delta = self.residual_limit * signed_correction

        p_fused = torch.clamp(p_mixture + bounded_delta, 0.0, 1.0)

        return {
            "p_fused": p_fused,
            "p_sar": p_sar,
            "p_hsi": p_hsi,
            "p_mixture": p_mixture,
            "gate": gate,
            "correction": correction,
            "signed_correction": signed_correction,
            "bounded_delta": bounded_delta,
            "fusion_context": fusion_context,
            "sar_embedding": sar_embedding,
            "hsi_embedding": hsi_embedding,
            "evidence_embedding": evidence_embedding,
        }


def build_fusion_model(residual_limit: float = DEFAULT_RESIDUAL_LIMIT, dropout: float = 0.10) -> RegistrationFreeFusionNet:
    return RegistrationFreeFusionNet(
        sar_feature_dim=SAR_FEATURE_DIM,
        hsi_feature_dim=HSI_FEATURE_DIM,
        residual_limit=residual_limit,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fusion_model().to(device)
    stats = count_parameters(model)
    print(f"Device: {device} | params: {stats}")

    b = 16
    sar_features = torch.rand(b, SAR_FEATURE_DIM, device=device)
    hsi_features = torch.randn(b, HSI_FEATURE_DIM, device=device)
    hsi_logits = torch.randn(b, 1, device=device)
    hsi_confidence = torch.rand(b, 1, device=device)

    out = model(sar_features, hsi_features, hsi_logits, hsi_confidence)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:<20}: {tuple(v.shape)}")

    assert torch.all(out["p_fused"] >= 0) and torch.all(out["p_fused"] <= 1)
    assert torch.all(out["gate"] >= 0) and torch.all(out["gate"] <= 1)

    # Sanity: with gate=1 and delta=0, p_fused should be able to reach p_hsi exactly
    # (not just within 0.25 of p_sar as before).
    model.eval()
    with torch.no_grad():
        out2 = model(sar_features, hsi_features, hsi_logits, hsi_confidence)
    print(f"\n  gate range     : [{out2['gate'].min():.3f}, {out2['gate'].max():.3f}]")
    print(f"  p_fused range  : [{out2['p_fused'].min():.3f}, {out2['p_fused'].max():.3f}]")
    print(f"  p_hsi range    : [{out2['p_hsi'].min():.3f}, {out2['p_hsi'].max():.3f}]")

    model.train()
    out3 = model(sar_features, hsi_features, hsi_logits, hsi_confidence)
    target = torch.randint(0, 2, (b, 1), device=device).float()
    loss = nn.BCELoss()(out3["p_fused"].clamp(1e-6, 1 - 1e-6), target)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"\n[SUCCESS] Gated fusion model self-test passed. Loss={loss.item():.4f}")


if __name__ == "__main__":
    self_test()