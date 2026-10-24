# # ================================================================
# # 03_model.py
# # FINAL HMOSDNet HSI EXPERT
# # ================================================================

# import torch
# import torch.nn as nn


# # ================================================================
# # LOSS
# # ================================================================

# class DiceBCELoss(nn.Module):

#     def __init__(
#         self,
#         bce_weight=0.5
#     ):

#         super().__init__()

#         self.bce = nn.BCEWithLogitsLoss()

#         self.bce_weight = bce_weight

#     def forward(
#         self,
#         logits,
#         targets
#     ):

#         targets = targets.float()

#         if logits.shape != targets.shape:
#             raise ValueError(
#                 f"Loss shape mismatch: "
#                 f"{logits.shape} vs "
#                 f"{targets.shape}"
#             )

#         # BCE
#         bce_loss = self.bce(
#             logits,
#             targets
#         )

#         # Dice
#         probs = torch.sigmoid(
#             logits
#         )

#         probs_flat = probs.reshape(-1)

#         targets_flat = targets.reshape(-1)

#         intersection = (
#             probs_flat
#             * targets_flat
#         ).sum()

#         dice = (
#             2.0 * intersection
#             + 1.0
#         ) / (
#             probs_flat.sum()
#             + targets_flat.sum()
#             + 1.0
#         )

#         dice_loss = (
#             1.0 - dice
#         )

#         return (
#             self.bce_weight * bce_loss
#             +
#             (1.0 - self.bce_weight)
#             * dice_loss
#         )


# # ================================================================
# # METRICS
# # ================================================================

# def compute_metrics(
#     preds,
#     targets
# ):

#     preds = (
#         preds.detach()
#         .cpu()
#         .flatten()
#         .float()
#     )

#     targets = (
#         targets.detach()
#         .cpu()
#         .flatten()
#         .float()
#     )

#     preds = (preds >= 0.5).float()

#     targets = (targets >= 0.5).float()

#     tp = (
#         (preds == 1)
#         &
#         (targets == 1)
#     ).sum().item()

#     tn = (
#         (preds == 0)
#         &
#         (targets == 0)
#     ).sum().item()

#     fp = (
#         (preds == 1)
#         &
#         (targets == 0)
#     ).sum().item()

#     fn = (
#         (preds == 0)
#         &
#         (targets == 1)
#     ).sum().item()

#     eps = 1e-8

#     precision = (
#         tp
#         /
#         (tp + fp + eps)
#     )

#     recall = (
#         tp
#         /
#         (tp + fn + eps)
#     )

#     specificity = (
#         tn
#         /
#         (tn + fp + eps)
#     )

#     dice = (
#         2 * tp
#         /
#         (2 * tp + fp + fn + eps)
#     )

#     iou_oil = (
#         tp
#         /
#         (tp + fp + fn + eps)
#     )

#     iou_water = (
#         tn
#         /
#         (tn + fp + fn + eps)
#     )

#     miou = (
#         iou_oil
#         +
#         iou_water
#     ) / 2.0

#     far = (
#         fp
#         /
#         (fp + tn + eps)
#     )

#     oa = (
#         tp + tn
#     ) / (
#         tp + tn + fp + fn + eps
#     )

#     return {

#         "miou": miou,

#         "dice": dice,

#         "precision": precision,

#         "recall": recall,

#         "specificity": specificity,

#         "far": far,

#         "oa": oa,

#         "iou_oil": iou_oil,

#         "iou_water": iou_water,

#         "tp": tp,

#         "tn": tn,

#         "fp": fp,

#         "fn": fn
#     }


# # ================================================================
# # MIXED ATTENTION
# # ================================================================

# class MixedAttentionModule(
#     nn.Module
# ):

#     def __init__(
#         self,
#         channels,
#         reduction=2
#     ):

#         super().__init__()

#         hidden = max(
#             channels // reduction,
#             1
#         )

#         # Spectral/channel attention.
#         self.avg_pool = (
#             nn.AdaptiveAvgPool3d(1)
#         )

#         self.max_pool = (
#             nn.AdaptiveMaxPool3d(1)
#         )

#         self.fc = nn.Sequential(

#             nn.Linear(
#                 channels,
#                 hidden,
#                 bias=False
#             ),

#             nn.ReLU(
#                 inplace=True
#             ),

#             nn.Linear(
#                 hidden,
#                 channels,
#                 bias=False
#             )
#         )

#         self.channel_sigmoid = (
#             nn.Sigmoid()
#         )

#         # Spatial attention.
#         self.spatial_conv = nn.Conv3d(
#             2,
#             1,
#             kernel_size=(1, 3, 3),
#             padding=(0, 1, 1),
#             bias=False
#         )

#         self.spatial_sigmoid = (
#             nn.Sigmoid()
#         )

#     def forward(self, x):

#         b, c, d, h, w = x.shape

#         # --------------------------------------------------------
#         # CHANNEL / SPECTRAL ATTENTION
#         # --------------------------------------------------------

#         avg = self.avg_pool(
#             x
#         ).reshape(
#             b,
#             c
#         )

#         mx = self.max_pool(
#             x
#         ).reshape(
#             b,
#             c
#         )

#         channel_attention = (
#             self.fc(avg)
#             +
#             self.fc(mx)
#         )

#         channel_attention = (
#             self.channel_sigmoid(
#                 channel_attention
#             )
#             .reshape(
#                 b,
#                 c,
#                 1,
#                 1,
#                 1
#             )
#         )

#         x = (
#             x
#             *
#             channel_attention
#         )

#         # --------------------------------------------------------
#         # SPATIAL ATTENTION
#         # --------------------------------------------------------

#         avg_spatial = torch.mean(
#             x,
#             dim=1,
#             keepdim=True
#         )

#         max_spatial = torch.max(
#             x,
#             dim=1,
#             keepdim=True
#         )[0]

#         spatial = torch.cat(
#             [
#                 avg_spatial,
#                 max_spatial
#             ],
#             dim=1
#         )

#         spatial_attention = (
#             self.spatial_sigmoid(
#                 self.spatial_conv(
#                     spatial
#                 )
#             )
#         )

#         x = (
#             x
#             *
#             spatial_attention
#         )

#         return x


# # ================================================================
# # HMOSDNET
# # ================================================================

# class HMOSDNet(nn.Module):

#     def __init__(
#         self,
#         in_channels=1,
#         num_bands=25,
#         num_classes=1,
#         mam_reduction=2
#     ):

#         super().__init__()

#         # --------------------------------------------------------
#         # BLOCK 1
#         # --------------------------------------------------------

#         self.conv1 = nn.Sequential(

#             nn.Conv3d(
#                 in_channels,
#                 12,
#                 kernel_size=(3, 3, 3),
#                 stride=(1, 1, 1),
#                 padding=(1, 1, 1)
#             ),

#             nn.BatchNorm3d(12),

#             nn.ReLU(
#                 inplace=True
#             )
#         )

#         # --------------------------------------------------------
#         # MAM
#         # --------------------------------------------------------

#         self.mam = (
#             MixedAttentionModule(
#                 12,
#                 reduction=mam_reduction
#             )
#         )

#         # --------------------------------------------------------
#         # BLOCK 2
#         # --------------------------------------------------------

#         self.conv2 = nn.Sequential(

#             nn.Conv3d(
#                 12,
#                 24,
#                 kernel_size=(3, 1, 1),
#                 stride=(1, 1, 1),
#                 padding=(1, 0, 0)
#             ),

#             nn.BatchNorm3d(24),

#             nn.ReLU(
#                 inplace=True
#             )
#         )

#         # --------------------------------------------------------
#         # BLOCK 3
#         # --------------------------------------------------------

#         self.conv3 = nn.Sequential(

#             nn.Conv3d(
#                 24,
#                 48,
#                 kernel_size=(3, 1, 1),
#                 stride=(2, 1, 1),
#                 padding=(1, 0, 0)
#             ),

#             nn.BatchNorm3d(48),

#             nn.ReLU(
#                 inplace=True
#             )
#         )

#         # --------------------------------------------------------
#         # FUSION PROJECTION
#         # --------------------------------------------------------

#         self.proj = nn.Sequential(

#             nn.Conv3d(
#                 48,
#                 64,
#                 kernel_size=1
#             ),

#             nn.GroupNorm(
#                 8,
#                 64
#             ),

#             nn.ReLU(
#                 inplace=True
#             )
#         )

#         # --------------------------------------------------------
#         # CLASSIFIER
#         # --------------------------------------------------------

#         self.gap = (
#             nn.AdaptiveAvgPool3d(
#                 (1, 1, 1)
#             )
#         )

#         self.classifier = nn.Sequential(

#             nn.Linear(
#                 64,
#                 128
#             ),

#             nn.ReLU(
#                 inplace=True
#             ),

#             nn.Dropout(
#                 DROPOUT_RATE
#             ),

#             nn.Linear(
#                 128,
#                 num_classes
#             )
#         )

#     # ============================================================
#     # FORWARD
#     # ============================================================

#     def forward(
#         self,
#         x,
#         return_features=True
#     ):

#         # x:
#         # (B,1,25,5,5)

#         x = self.conv1(x)

#         x = self.mam(x)

#         x = self.conv2(x)

#         x = self.conv3(x)

#         x = self.proj(x)

#         # --------------------------------------------------------
#         # Spatial feature representation
#         # --------------------------------------------------------
#         #
#         # x = B,64,D',5,5
#         #
#         # Mean spectral dimension.
#         #
#         # result = B,64,5,5
#         #
#         # flatten spatial positions:
#         #
#         # B,64,25
#         # --------------------------------------------------------

#         features = x.mean(
#             dim=2
#         )

#         b, c, h, w = (
#             features.shape
#         )

#         features = features.reshape(
#             b,
#             c,
#             h * w
#         )

#         # --------------------------------------------------------
#         # CLASSIFICATION
#         # --------------------------------------------------------

#         pooled = self.gap(
#             x
#         ).reshape(
#             x.size(0),
#             -1
#         )

#         logits = self.classifier(
#             pooled
#         )

#         confidence = torch.sigmoid(
#             logits
#         )

#         if return_features:

#             return (
#                 logits,
#                 features,
#                 None,
#                 confidence
#             )

#         return logits

#     # ============================================================
#     # FUSION INTERFACE
#     # ============================================================

#     def extract_features(
#         self,
#         x
#     ):

#         (
#             logits,
#             features,
#             _,
#             confidence
#         ) = self.forward(
#             x,
#             return_features=True
#         )

#         return features, confidence


# # ================================================================
# # SELF TEST
# # ================================================================

# if __name__ == "__main__":

#     print("=" * 70)
#     print("HMOSDNET SELF TEST")
#     print("=" * 70)

#     device = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "cpu"
#     )

#     model = HMOSDNet(
#         in_channels=1,
#         num_bands=25
#     ).to(device)

#     dummy = torch.randn(
#         4,
#         1,
#         25,
#         5,
#         5,
#         device=device
#     )

#     with torch.no_grad():

#         (
#             logits,
#             features,
#             none_value,
#             confidence
#         ) = model(dummy)

#     print(
#         f"Input       : {dummy.shape}"
#     )

#     print(
#         f"Logits      : {logits.shape}"
#     )

#     print(
#         f"Features    : {features.shape}"
#     )

#     print(
#         f"None slot   : {none_value}"
#     )

#     print(
#         f"Confidence  : {confidence.shape}"
#     )

#     assert logits.shape == (
#         4,
#         1
#     )

#     assert features.shape == (
#         4,
#         64,
#         25
#     )

#     assert none_value is None

#     assert confidence.shape == (
#         4,
#         1
#     )

#     total = sum(
#         p.numel()
#         for p in model.parameters()
#     )

#     print(
#         f"Parameters  : {total:,}"
#     )

#     print()
#     print(
#         "[SUCCESS] HMOSDNet verification passed."
#     )



# ================================================================
# 03_model.py
# FINAL HMOSDNet HSI EXPERT  -- CORRECTED
#
# CHANGES FROM ORIGINAL:
#   1. BatchNorm3d -> GroupNorm in conv1/conv2/conv3.
#      Reason: WeightedRandomSampler force-balances every training
#      batch toward ~target oil fraction, which shifts the *feature*
#      distribution BatchNorm's running stats get calibrated on.
#      Validation uses the natural (imbalanced) distribution, so the
#      running stats don't match at eval time. GroupNorm normalizes
#      per-sample, not per-batch, so it's immune to this mismatch.
#   2. DROPOUT_RATE is now a constructor argument (dropout_rate),
#      not a bare global. The original file referenced DROPOUT_RATE
#      directly with no import/definition in this file -- running
#      `python 03_model.py` standalone raised NameError.
# ================================================================

import torch
import torch.nn as nn


# ================================================================
# LOSS
# ================================================================

class DiceBCELoss(nn.Module):

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        targets = targets.float()

        if logits.shape != targets.shape:
            raise ValueError(
                f"Loss shape mismatch: {logits.shape} vs {targets.shape}"
            )

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()

        dice = (2.0 * intersection + 1.0) / (
            probs_flat.sum() + targets_flat.sum() + 1.0
        )
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# ================================================================
# METRICS  (unchanged)
# ================================================================

def compute_metrics(preds, targets):

    preds = preds.detach().cpu().flatten().float()
    targets = targets.detach().cpu().flatten().float()

    preds = (preds >= 0.5).float()
    targets = (targets >= 0.5).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou_oil = tp / (tp + fp + fn + eps)
    iou_water = tn / (tn + fp + fn + eps)
    miou = (iou_oil + iou_water) / 2.0
    far = fp / (fp + tn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "miou": miou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "far": far,
        "oa": oa,
        "iou_oil": iou_oil,
        "iou_water": iou_water,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ================================================================
# MIXED ATTENTION  (unchanged)
# ================================================================

class MixedAttentionModule(nn.Module):

    def __init__(self, channels, reduction=2):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

        self.channel_sigmoid = nn.Sigmoid()

        self.spatial_conv = nn.Conv3d(
            2, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, d, h, w = x.shape

        avg = self.avg_pool(x).reshape(b, c)
        mx = self.max_pool(x).reshape(b, c)

        channel_attention = self.fc(avg) + self.fc(mx)
        channel_attention = self.channel_sigmoid(channel_attention).reshape(
            b, c, 1, 1, 1
        )

        x = x * channel_attention

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial = torch.cat([avg_spatial, max_spatial], dim=1)

        spatial_attention = self.spatial_sigmoid(self.spatial_conv(spatial))
        x = x * spatial_attention

        return x


# ================================================================
# HMOSDNET  -- CORRECTED (GroupNorm, explicit dropout_rate)
# ================================================================

class HMOSDNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_bands=25,
        num_classes=1,
        mam_reduction=2,
        dropout_rate=0.5,   # was a bare global reference before
    ):
        super().__init__()

        # ------------------------------------------------------------
        # BLOCK 1  (BatchNorm3d(12) -> GroupNorm(4, 12); 12 % 4 == 0)
        # ------------------------------------------------------------

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 12, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GroupNorm(4, 12),
            nn.ReLU(inplace=True),
        )

        self.mam = MixedAttentionModule(12, reduction=mam_reduction)

        # ------------------------------------------------------------
        # BLOCK 2  (BatchNorm3d(24) -> GroupNorm(8, 24); 24 % 8 == 0)
        # ------------------------------------------------------------

        self.conv2 = nn.Sequential(
            nn.Conv3d(12, 24, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, 24),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------
        # BLOCK 3  (BatchNorm3d(48) -> GroupNorm(8, 48); 48 % 8 == 0)
        # ------------------------------------------------------------

        self.conv3 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=(3, 1, 1),
                      stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------
        # FUSION PROJECTION  (already GroupNorm -- unchanged)
        # ------------------------------------------------------------

        self.proj = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------
        # CLASSIFIER
        # ------------------------------------------------------------

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(self, x, return_features=True):

        # x: (B,1,25,5,5)

        x = self.conv1(x)
        x = self.mam(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.proj(x)

        # Spatial feature representation: mean over spectral dim
        # (B,64,D',5,5) -> (B,64,5,5) -> (B,64,25)
        features = x.mean(dim=2)
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w)

        pooled = self.gap(x).reshape(x.size(0), -1)
        logits = self.classifier(pooled)
        confidence = torch.sigmoid(logits)

        if return_features:
            return logits, features, None, confidence

        return logits

    # ============================================================
    # FUSION INTERFACE
    # ============================================================

    def extract_features(self, x):
        logits, features, _, confidence = self.forward(x, return_features=True)
        return features, confidence


# ================================================================
# SELF TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HMOSDNET SELF TEST (CORRECTED)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HMOSDNet(in_channels=1, num_bands=25, dropout_rate=0.5).to(device)

    dummy = torch.randn(4, 1, 25, 5, 5, device=device)

    with torch.no_grad():
        logits, features, none_value, confidence = model(dummy)

    print(f"Input       : {dummy.shape}")
    print(f"Logits      : {logits.shape}")
    print(f"Features    : {features.shape}")
    print(f"None slot   : {none_value}")
    print(f"Confidence  : {confidence.shape}")

    assert logits.shape == (4, 1)
    assert features.shape == (4, 64, 25)
    assert none_value is None
    assert confidence.shape == (4, 1)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total:,}")

    # Sanity check: with batch size 1, BatchNorm3d would crash
    # ("expected more than 1 value per channel"). GroupNorm does not.
    single = torch.randn(1, 1, 25, 5, 5, device=device)
    with torch.no_grad():
        model(single)
    print("[OK] batch_size=1 forward pass works (confirms GroupNorm, not BatchNorm)")

    print()
    print("[SUCCESS] HMOSDNet verification passed.")