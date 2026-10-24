# %% [markdown]
# # 📡 Stage 1: SAR Expert Configuration
# **CBD-Net — Contextual & Boundary-Supervised Oil Spill Detection**
#
# All hyperparameters, paths, and training configuration in one place.
# Aligned with: README_ARCHITECTURE.md and CBD-Net (IEEE TGRS, 2022)

# %%
import os

# ═══════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() or '__file__' in globals() else os.getcwd()

# SOS Dataset paths
SOS_TRAIN_DIR = os.path.join(PROJECT_ROOT, "SOS_dataset", "train", "sentinel")
SOS_TEST_DIR  = os.path.join(PROJECT_ROOT, "SOS_dataset", "test", "sentinel")

# Output paths
SAR_LAYER_DIR = os.path.join(PROJECT_ROOT, "sar_layer")
WEIGHTS_DIR   = os.path.join(SAR_LAYER_DIR, "weights")
LOGS_DIR      = os.path.join(SAR_LAYER_DIR, "logs")
VIS_DIR       = os.path.join(SAR_LAYER_DIR, "visualizations")

try:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (CBD-Net, aligned with README_ARCHITECTURE.md)
# ═══════════════════════════════════════════════════════════════════
ENCODER_NAME    = "resnet18"       # ResNet-18 backbone (CBD-Net paper)
ENCODER_WEIGHTS = "imagenet"       # ImageNet pretrained initialization
IN_CHANNELS     = 3                # 3-channel (grayscale replicated to RGB for pretrained weights)
NUM_CLASSES     = 1                # Binary: oil (1) vs water (0)
DECODER_ATTENTION = "scse"         # scSE attention blocks in decoder (CBD-Net)
IMG_SIZE        = 256              # 256×256 patches

# Encoder output channels (ResNet-18 bottleneck)
# This is the feature map sent to SQCA fusion in Stage 3
ENCODER_FEATURE_DIM  = 512         # ResNet-18 last layer: 512 channels
ENCODER_FEATURE_SIZE = 8           # 256/32 = 8 (5 downsampling stages)
# → Output feature map: (512, 8, 8) for SQCA fusion

# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTION (Aligned with Architecture)
# ═══════════════════════════════════════════════════════════════════
# L_SAR = BCE + λ_boundary × MBL
# MBL = Σ_s w_s × BCE(B_pred^s, B_gt^s)
LAMBDA_BOUNDARY = 0.3              # λ_boundary weight for MBL
MBL_SCALE_WEIGHTS = [0.5, 0.3, 0.2]  # Multi-scale boundary weights [fine, medium, coarse]
CANNY_LOW_THRESHOLD  = 50          # Canny edge detection lower threshold
CANNY_HIGH_THRESHOLD = 150         # Canny edge detection upper threshold

# ═══════════════════════════════════════════════════════════════════
#  TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════
BATCH_SIZE    = 8                  # Fits RTX 4050 6GB VRAM
NUM_EPOCHS    = 50                 # Full training (CBD-Net used 50-100)
LEARNING_RATE = 1e-4               # AdamW initial LR (more stable than 2e-4)
WEIGHT_DECAY  = 1e-4               # L2 regularization
GRAD_CLIP     = 1.0                # Gradient clipping max norm

# Scheduler
SCHEDULER_TYPE = "cosine"          # CosineAnnealingLR
T_MAX          = NUM_EPOCHS        # Full cosine cycle

# Data split
VAL_SPLIT     = 0.15               # 15% of train for validation
RANDOM_SEED   = 42                 # Reproducibility

# Mixed Precision
USE_AMP = True                     # FP16 training for speed

# Early stopping
PATIENCE = 15                      # Stop if no improvement for 15 epochs

# ═══════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ═══════════════════════════════════════════════════════════════════
AUG_HFLIP_PROB   = 0.5             # Horizontal flip probability
AUG_VFLIP_PROB   = 0.5             # Vertical flip probability
AUG_ROT90_PROB   = 0.5             # 90° rotation probability
AUG_NOISE_PROB   = 0.3             # Gaussian noise probability
AUG_NOISE_STD    = 0.02            # Noise standard deviation (simulates speckle)

# ═══════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════
THRESHOLD = 0.5                    # Binary classification threshold
MC_DROPOUT_PASSES = 20             # T=20 for Bayesian inference (Stage 3)

# ═══════════════════════════════════════════════════════════════════
#  QUICK SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  SAR Expert (CBD-Net) Configuration")
print("=" * 60)
print(f"  Encoder:       {ENCODER_NAME} + {DECODER_ATTENTION}")
print(f"  Image Size:    {IMG_SIZE}×{IMG_SIZE}")
print(f"  Feature Out:   ({ENCODER_FEATURE_DIM}, {ENCODER_FEATURE_SIZE}, {ENCODER_FEATURE_SIZE})")
print(f"  Loss:          BCE + {LAMBDA_BOUNDARY}×MBL")
print(f"  MBL Scales:    {MBL_SCALE_WEIGHTS}")
print(f"  Batch Size:    {BATCH_SIZE}")
print(f"  Epochs:        {NUM_EPOCHS}")
print(f"  LR:            {LEARNING_RATE}")
print(f"  AMP:           {USE_AMP}")
print(f"  Train Dir:     {SOS_TRAIN_DIR}")
print(f"  Test Dir:      {SOS_TEST_DIR}")
print("=" * 60)
