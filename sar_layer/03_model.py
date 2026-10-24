# %% [markdown]
# # 📡 Stage 1: SAR Expert — CBD-Net Model & Loss Functions
# **Contextual & Boundary-Supervised Detection Network**
#
# Architecture per CBD-Net paper + our README_ARCHITECTURE.md:
# - **Encoder:** ResNet-18 (ImageNet pretrained) with scSE attention
# - **Decoder:** UNet decoder with skip connections
# - **Dual Output:**
#   - Path A: Segmentation mask (1, 256, 256) — pixel-level oil/water
#   - Path B: Encoder features (512, 8, 8) — sent to SQCA fusion in Stage 3
# - **Loss:** L_SAR = BCE + λ × MBL (boundary-supervised)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2
import numpy as np

# %% [markdown]
# ## 1. Mean Boundary Loss (MBL)
#
# From CBD-Net paper: Multi-scale boundary supervision using Canny edges.
#
# ```
# MBL = Σ_s w_s × BCE(B_pred^s, B_gt^s)
# ```
#
# Where:
# - s ∈ {fine, medium, coarse} — 3 boundary scales
# - w_s = [0.5, 0.3, 0.2] — fine edges matter most
# - B_pred^s — boundary predicted at scale s (from decoder intermediate features)
# - B_gt^s — ground truth boundary at scale s (Canny on GT mask)

# %%
class MultiScaleBoundaryHead(nn.Module):
    """
    Multi-scale boundary prediction head.
    From CBD-Net: Extracts edges at multiple scales to supervise object boundaries.
    
    Fix 1: Non-circular input. Takes decoder feature maps (16-ch) instead of final logits (1-ch).
    """
    def __init__(self, in_channels=16):
        super().__init__()
        
        # Scale 1: Fine boundaries (full resolution)
        self.boundary_conv_fine = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        self.boundary_conv_medium = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        self.boundary_conv_coarse = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, features):
        """
        Args:
            features: Decoder feature maps (B, 16, H, W)
        
        Returns:
            List of boundary logits at 3 scales [(B,1,H,W), (B,1,H,W), (B,1,H,W)]
        """
        h, w = features.shape[2:]
        
        # Scale 1: Fine boundaries
        b_fine = self.boundary_conv_fine(features)
        
        # Scale 2: Medium boundaries (downscale, process, upscale)
        down_2x = F.interpolate(features, scale_factor=0.5, mode='bilinear', align_corners=False)
        b_medium = self.boundary_conv_medium(down_2x)
        b_medium = F.interpolate(b_medium, size=(h, w), mode='bilinear', align_corners=False)
        
        # Scale 3: Coarse boundaries (further downscale)
        down_4x = F.interpolate(features, scale_factor=0.25, mode='bilinear', align_corners=False)
        b_coarse = self.boundary_conv_coarse(down_4x)
        b_coarse = F.interpolate(b_coarse, size=(h, w), mode='bilinear', align_corners=False)
        
        return [b_fine, b_medium, b_coarse]


class MeanBoundaryLoss(nn.Module):
    """
    Mean Boundary Loss (MBL) from CBD-Net.
    
    MBL = Σ_s w_s × BCE(B_pred^s, B_gt^s)
    
    Multi-scale boundary supervision ensures sharp oil-water edges.
    """
    def __init__(self, scale_weights=[0.5, 0.3, 0.2]):
        super().__init__()
        self.scale_weights = scale_weights
    
    def forward(self, boundary_preds, boundary_gts):
        """
        Args:
            boundary_preds: list of (B, 1, H, W) logits — 3 scales from boundary head
            boundary_gts: list of (B, 1, H, W) binary boundary maps — 3 scales from Canny

        Returns:
            MBL scalar loss
        """
        total_loss = 0.0
        for w, pred, gt in zip(self.scale_weights, boundary_preds, boundary_gts):
            # Use BCE with logits for numerical stability
            scale_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')
            total_loss += w * scale_loss
        return total_loss

# %% [markdown]
# ## 2. CBD-Net Combined Loss
#
# From our architecture:
# ```
# L_SAR = BCE(mask_pred, mask_gt) + λ_boundary × MBL
# ```
# Where λ_boundary = 0.3 (with ablation planned at {0.1, 0.2, 0.3, 0.5})

# %%
class CBDNetLoss(nn.Module):
    """
    CBD-Net combined loss: BCE + λ × MBL
    
    BCE: Standard binary cross-entropy for segmentation (with pos_weight for class imbalance)
    MBL: Multi-scale boundary loss for sharp edges
    
    WARNING 3 fix: Oil spill datasets are heavily imbalanced (5-15% oil pixels).
    Without pos_weight, the model learns to predict "all water" and gets 90%+ OA
    while mIoU stays near 0. pos_weight upweights the oil class in BCE.
    """
    def __init__(self, lambda_boundary=0.3, scale_weights=[0.5, 0.3, 0.2], pos_weight=None):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.mbl = MeanBoundaryLoss(scale_weights)
        # pos_weight: scalar tensor, typically (1-pos_ratio)/pos_ratio ~ 6-19
        # This upweights false negatives (missed oil) relative to false positives
        self.pos_weight = pos_weight  # torch.tensor([value]) or None
    
    def forward(self, seg_logits, mask_gt, boundary_preds, boundary_gts):
        """
        Args:
            seg_logits: (B, 1, H, W) segmentation logits
            mask_gt: (B, 1, H, W) binary ground truth mask
            boundary_preds: list of boundary logits at 3 scales
            boundary_gts: list of boundary maps at 3 scales

        Returns:
            total_loss: L_SAR = BCE + λ × MBL
            bce_loss: BCE component (for logging)
            mbl_loss: MBL component (for logging)
        """
        # 1. Binary Cross-Entropy loss (with class imbalance correction)
        bce_loss = F.binary_cross_entropy_with_logits(
            seg_logits, mask_gt, reduction='mean',
            pos_weight=self.pos_weight
        )
        
        # 2. Mean Boundary Loss
        mbl_loss = self.mbl(boundary_preds, boundary_gts)
        
        # 3. Combined loss
        total_loss = bce_loss + self.lambda_boundary * mbl_loss
        
        return total_loss, bce_loss, mbl_loss

# %% [markdown]
# ## 3. CBD-Net Model (Dual-Path Output)
#
# The model wraps `smp.Unet` to provide two outputs:
# - **Path A:** Segmentation mask (for oil spill detection)
# - **Path B:** Encoder bottleneck features (512, 8, 8) — sent to SQCA in Stage 3
#
# The boundary head derives multi-scale edge predictions from the segmentation output.

# %%
class CBDNetSAR(nn.Module):
    """
    CBD-Net SAR Expert — Contextual & Boundary-Supervised Detection Network.
    
    Architecture:
        Encoder: ResNet-18 + scSE attention (ImageNet pretrained)
        Decoder: UNet decoder with skip connections
        Boundary Head: Multi-scale boundary predictor
        
    ALWAYS returns 4 values (Bug 1 fix — consistent return signature):
        1. seg_logits: (B, 1, 256, 256) — segmentation logits
        2. boundary_logits: list of 3 × (B, 1, 256, 256) — boundary predictions
        3. encoder_features: (B, 512, 8, 8) — bottleneck features for SQCA
        4. mask_conf: (B, 1) — Mask_conf = mean(sigmoid(seg_logits))
    """
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet",
                 in_channels=3, classes=1, decoder_attention="scse"):
        super().__init__()
        
        # Core UNet model from segmentation_models_pytorch
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention
        )
        
        # Multi-scale boundary prediction head
        self.boundary_head = MultiScaleBoundaryHead()
        
        # Store encoder/decoder references for single-pass forward
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.seg_head = self.unet.segmentation_head
        
    def forward(self, x):
        """
        Forward pass — ALWAYS returns 4 values (Bug 1 fix).
        
        Uses single encoder pass (no double computation):
          1. encoder(x) → feature pyramid
          2. decoder(features) → decoded features
          3. seg_head(decoded) → segmentation logits
          4. boundary_head(logits) → multi-scale boundaries
        
        Args:
            x: (B, 3, 256, 256) input SAR image
        
        Returns:
            seg_logits: (B, 1, 256, 256) segmentation logits
            boundary_logits: list of 3 × (B, 1, 256, 256) boundary maps
            encoder_features: (B, 512, 8, 8) bottleneck features for SQCA
            mask_conf: (B, 1) confidence scalar = mean(sigmoid(seg_logits))
        """
        # Step 1: Single encoder pass -> captures feature pyramid
        # ResNet-18 stages: [3->64, 64->64, 64->128, 128->256, 256->512]
        features = self.encoder(x)
        encoder_features = features[-1]  # (B, 512, 8, 8) -- bottleneck for SQCA
        
        # Step 2: Decoder pass -> upsamples with skip connections
        # smp decoder expects features as a List[Tensor]
        decoder_output = self.decoder(features)
        
        # Step 3: Segmentation head -> final logits (1-channel)
        seg_logits = self.seg_head(decoder_output)
        
        # Step 4: Boundary head -> multi-scale edge predictions
        # Fix 1: Pass decoder_output (16-ch) instead of seg_logits (1-ch) to break circularity
        boundary_logits = self.boundary_head(decoder_output)
        
        # Step 5: Mask confidence for GSFF gate (Stage 3)
        # Detach to ensure it doesn't affect the training graph
        mask_conf = torch.sigmoid(seg_logits).mean(dim=[1, 2, 3], keepdim=False).detach().unsqueeze(1)
        # mask_conf shape: (B, 1)
        
        return seg_logits, boundary_logits, encoder_features, mask_conf
    
    def extract_features(self, x):
        """
        Extract only encoder features for Stage 3 frozen inference.
        Single encoder pass — no decoder computation needed.
        
        Args:
            x: (B, 3, 256, 256) input SAR image
            
        Returns:
            encoder_features: (B, 512, 8, 8) — bottleneck features
            mask_conf: (B, 1) — SAR segmentation confidence
        """
        with torch.no_grad():
            # Full forward to get both features and mask_conf
            seg_logits, _, encoder_features, mask_conf = self.forward(x)
        return encoder_features, mask_conf

# %% [markdown]
# ## 4. Model Verification
#
# Quick check that all shapes are correct and dual-path output works.

# %%
if __name__ == "__main__" and 'SKIP_SELF_TEST' not in dir():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")
    
    # Create model
    model = CBDNetSAR().to(device)
    
    # Dummy input (batch of 4, 3-channel, 256x256)
    dummy = torch.randn(4, 3, 256, 256).to(device)
    
    # Test forward pass — ALWAYS returns 4 values (Bug 1 fix)
    seg_logits, boundary_logits, features, mask_conf = model(dummy)
    print("  Forward Pass (always 4 values):")
    print(f"    Seg logits:       {seg_logits.shape}")
    print(f"    Boundary scales:  {len(boundary_logits)}")
    for i, b in enumerate(boundary_logits):
        print(f"      Scale {i}:        {b.shape}")
    print(f"    Encoder features: {features.shape}")
    print(f"    Mask confidence:  {mask_conf.shape} (values: {mask_conf[:2, 0].tolist()})")
    print(f"    -> Ready for SQCA: (B, 512, 8, 8) -> projected to (B, 128, 64)")
    
    # Test frozen extraction (for Stage 3)
    features, conf = model.extract_features(dummy)
    print(f"\n  Frozen Extraction (Stage 3):")
    print(f"    Features:  {features.shape}")
    print(f"    Mask_conf: {conf.shape}")
    
    # Test loss with pos_weight
    pw = torch.tensor([10.0]).to(device)  # Example pos_weight
    criterion = CBDNetLoss(lambda_boundary=0.3, pos_weight=pw)
    mask_gt = (torch.randn(4, 1, 256, 256) > 0).float().to(device)
    boundary_gts = [
        (torch.randn(4, 1, 256, 256) > 0.8).float().to(device)
        for _ in range(3)
    ]
    total, bce, mbl = criterion(seg_logits, mask_gt, boundary_logits, boundary_gts)
    print(f"\n  Loss Computation (pos_weight=10.0):")
    print(f"    BCE:   {bce.item():.4f}")
    print(f"    MBL:   {mbl.item():.4f}")
    print(f"    Total: {total.item():.4f} (BCE + 0.3*MBL)")
    
    # Model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Stats:")
    print(f"    Total params:     {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    print(f"    Size:             ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
