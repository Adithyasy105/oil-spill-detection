# # ================================================================
# # 01_config.py
# # HSI EXPERT — FINAL LOCKED CONFIGURATION
# # HMOSDN-style HSI oil-spill expert
# # ================================================================

# from pathlib import Path
# import os
# import random
# import numpy as np
# import torch


# # ================================================================
# # PROJECT PATHS
# # ================================================================

# PROJECT_ROOT = Path(
#     r"C:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL"
# )

# HSI_ROOT = PROJECT_ROOT / "hsi_layer"

# DATASET_ROOT = PROJECT_ROOT / "new_dataset" / "OSD"

# TRAIN_HSI_DIR = (
#     DATASET_ROOT
#     / "TrainCopy-20260405T092743Z-1-001"
#     / "TrainCopy"
# )

# VAL_HSI_DIR = (
#     DATASET_ROOT
#     / "ValCopy-20260405T092906Z-1-001"
#     / "ValCopy"
# )

# TEST_HSI_DIR = (
#     DATASET_ROOT
#     / "TestCopy-20260405T092640Z-1-001"
#     / "TestCopy"
# )


# # ================================================================
# # OUTPUT DIRECTORIES
# # ================================================================

# HSI_LOGS_DIR = HSI_ROOT / "logs"
# HSI_WEIGHTS_DIR = HSI_ROOT / "weights"
# HSI_RESULTS_DIR = HSI_ROOT / "results"

# HSI_LOGS_DIR.mkdir(parents=True, exist_ok=True)
# HSI_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
# HSI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# # ================================================================
# # REPRODUCIBILITY
# # ================================================================

# SEED = 42


# def seed_everything(seed=SEED):

#     random.seed(seed)
#     np.random.seed(seed)

#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# seed_everything()


# # ================================================================
# # DEVICE
# # ================================================================

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # ================================================================
# # HSI SPECTRAL PREPROCESSING
# # ================================================================

# ORIGINAL_BANDS = 224

# BANDS_TO_REMOVE = (
#     list(range(103, 113))
#     + list(range(149, 170))
# )

# EXPECTED_REMAINING_BANDS = (
#     ORIGINAL_BANDS - len(BANDS_TO_REMOVE)
# )

# # 224 - 31 = 193

# PCA_COMPONENTS = 25


# # ================================================================
# # SPATIAL PATCHING
# # ================================================================

# PATCH_SIZE = 5

# PATCH_RADIUS = PATCH_SIZE // 2

# # Training sampling stride.
# # This controls number of training samples, NOT patch size.
# TRAIN_STRIDE = 3

# # Validation uses the same sampling geometry.
# VAL_STRIDE = 3

# # Final dense evaluation.
# # Every pixel gets a prediction.
# TEST_STRIDE = 1


# # ================================================================
# # MODEL
# # ================================================================

# HSI_FILTERS = [12, 24, 48]

# HSI_EMBED_DIM = 64

# MAM_REDUCTION = 2

# DROPOUT_RATE = 0.5


# # ================================================================
# # TRAINING
# # ================================================================

# BATCH_SIZE = 256

# NUM_EPOCHS = 60

# LEARNING_RATE = 5e-4

# WEIGHT_DECAY = 1e-4

# GRAD_CLIP = 1.0

# EARLY_STOPPING_PATIENCE = 12

# MIN_DELTA = 1e-4


# # ================================================================
# # LOSS
# # ================================================================

# BCE_WEIGHT = 0.5

# DICE_WEIGHT = 0.5

# # We deliberately do NOT use pos_weight because the
# # WeightedRandomSampler balances the training samples.


# # ================================================================
# # DATA LOADING
# # ================================================================

# NUM_WORKERS = 0 if os.name == "nt" else 4

# PIN_MEMORY = torch.cuda.is_available()


# # ================================================================
# # EVALUATION
# # ================================================================

# DEFAULT_THRESHOLD = 0.50

# THRESHOLD_GRID = np.round(
#     np.arange(0.05, 0.951, 0.01),
#     2
# )


# # ================================================================
# # ARTIFACT PATHS
# # ================================================================

# BEST_CHECKPOINT = (
#     HSI_WEIGHTS_DIR / "best_hsi_expert.pth"
# )

# FINAL_CHECKPOINT = (
#     HSI_WEIGHTS_DIR / "frozen_hsi_expert.pth"
# )

# SPECTRAL_PROCESSOR_PATH = (
#     HSI_WEIGHTS_DIR / "spectral_processor.pkl"
# )

# SPLIT_FILE = (
#     HSI_WEIGHTS_DIR / "hsi_split.txt"
# )

# METRICS_FILE = (
#     HSI_RESULTS_DIR / "hsi_test_metrics.txt"
# )


# # ================================================================
# # CONFIG PRINT
# # ================================================================

# print("=" * 70)
# print("HSI FINAL CONFIGURATION")
# print("=" * 70)

# print(f"Project root       : {PROJECT_ROOT}")
# print(f"Train HSI          : {TRAIN_HSI_DIR}")
# print(f"Val HSI            : {VAL_HSI_DIR}")
# print(f"Test HSI           : {TEST_HSI_DIR}")

# print(f"Device             : {DEVICE}")

# if torch.cuda.is_available():
#     print(f"GPU                : {torch.cuda.get_device_name(0)}")

# print()
# print(f"Original bands     : {ORIGINAL_BANDS}")
# print(f"Removed bands      : {len(BANDS_TO_REMOVE)}")
# print(f"Remaining bands    : {EXPECTED_REMAINING_BANDS}")
# print(f"PCA components     : {PCA_COMPONENTS}")

# print()
# print(f"Patch size         : {PATCH_SIZE}x{PATCH_SIZE}")
# print(f"Train stride       : {TRAIN_STRIDE}")
# print(f"Validation stride  : {VAL_STRIDE}")
# print(f"Test stride        : {TEST_STRIDE}")

# print()
# print(f"Batch size         : {BATCH_SIZE}")
# print(f"Epochs             : {NUM_EPOCHS}")
# print(f"Learning rate      : {LEARNING_RATE}")
# print(f"Weight decay       : {WEIGHT_DECAY}")

# print("=" * 70)


# ================================================================
# 01_config.py
# HSI EXPERT -- FINAL LOCKED CONFIGURATION  -- CORRECTED
#
# CHANGES FROM ORIGINAL:
#   1. LEARNING_RATE lowered 5e-4 -> 1e-4. The model has only ~16.8K
#      parameters; 5e-4 with batch size 256 was taking large steps
#      relative to model capacity.
#   2. Added WARMUP_EPOCHS (linear LR warmup) to avoid a large first
#      update before GroupNorm statistics have stabilized.
#   3. Added TARGET_OIL_FRACTION. The old sampler used raw inverse
#      class frequency, which forces ~50% oil per epoch and causes
#      heavy repetition of a small, spatially-correlated oil-patch
#      pool (45,515 samples from 9 scenes with 5x5 patches at
#      stride 3 -> lots of overlap). 0.20 is a much softer target
#      (~4x oversampling vs the natural 5.27%, not ~18x).
#   4. EARLY_STOPPING_PATIENCE tightened 12 -> 8, since the corrected
#      setup should show a real (not one-epoch-spike) learning curve.
#      Raise it back if you find it's stopping before val curves
#      have leveled off.
#   5. Scheduler is now ReduceLROnPlateau (see 04_train.py), not
#      CosineAnnealingLR -- cosine annealing over T_max=60 barely
#      decays during the first ~15-25 epochs, which is all the model
#      was actually getting before early stopping fired.
# ================================================================

from pathlib import Path
import os
import random
import numpy as np
import torch


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if '__file__' in locals() or '__file__' in globals()
    else Path.cwd()
)

HSI_ROOT = PROJECT_ROOT / "hsi_layer_corrected"

DATASET_ROOT = PROJECT_ROOT / "new_dataset" / "OSD"

TRAIN_HSI_DIR = (
    DATASET_ROOT / "TrainCopy-20260405T092743Z-1-001" / "TrainCopy"
)

VAL_HSI_DIR = (
    DATASET_ROOT / "ValCopy-20260405T092906Z-1-001" / "ValCopy"
)

TEST_HSI_DIR = (
    DATASET_ROOT / "TestCopy-20260405T092640Z-1-001" / "TestCopy"
)


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

HSI_LOGS_DIR = HSI_ROOT / "logs"
HSI_WEIGHTS_DIR = HSI_ROOT / "weights"
HSI_RESULTS_DIR = HSI_ROOT / "results"

try:
    HSI_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    HSI_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    HSI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


# ================================================================
# REPRODUCIBILITY
# ================================================================

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# HSI SPECTRAL PREPROCESSING
# ================================================================

ORIGINAL_BANDS = 224

BANDS_TO_REMOVE = list(range(103, 113)) + list(range(149, 170))

EXPECTED_REMAINING_BANDS = ORIGINAL_BANDS - len(BANDS_TO_REMOVE)  # 193

PCA_COMPONENTS = 25


# ================================================================
# SPATIAL PATCHING
# ================================================================

PATCH_SIZE = 5
PATCH_RADIUS = PATCH_SIZE // 2

TRAIN_STRIDE = 3
VAL_STRIDE = 3
TEST_STRIDE = 1


# ================================================================
# MODEL
# ================================================================

HSI_FILTERS = [12, 24, 48]
HSI_EMBED_DIM = 64
MAM_REDUCTION = 2
DROPOUT_RATE = 0.5


# ================================================================
# TRAINING  -- CORRECTED
# ================================================================

BATCH_SIZE = 256
NUM_EPOCHS = 60

LEARNING_RATE = 1e-4          # was 5e-4
WARMUP_EPOCHS = 2             # new: linear warmup over first 2 epochs

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 8   # was 12
MIN_DELTA = 1e-4

# ReduceLROnPlateau settings (used in 04_train.py)
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 4
LR_MIN = 1e-6


# ================================================================
# LOSS
# ================================================================

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# We do NOT use pos_weight because the WeightedRandomSampler
# already rebalances the training distribution (see
# TARGET_OIL_FRACTION below). Using both at once double-counts
# the imbalance correction.


# ================================================================
# SAMPLING / CLASS BALANCE  -- NEW
# ================================================================

# Old behavior: raw inverse-frequency weights forced ~50% oil per
# epoch, causing ~18x effective oversampling of a small, spatially
# correlated oil-patch pool (fast memorization / overfitting).
# New behavior: target a softer, still-meaningful oversampling ratio.
TARGET_OIL_FRACTION = 0.20


# ================================================================
# DATA LOADING
# ================================================================

NUM_WORKERS = 0 if os.name == "nt" else 4
PIN_MEMORY = torch.cuda.is_available()


# ================================================================
# EVALUATION
# ================================================================

DEFAULT_THRESHOLD = 0.50

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ================================================================
# ARTIFACT PATHS
# ================================================================

BEST_CHECKPOINT = HSI_WEIGHTS_DIR / "best_hsi_expert.pth"
FINAL_CHECKPOINT = HSI_WEIGHTS_DIR / "frozen_hsi_expert.pth"
SPECTRAL_PROCESSOR_PATH = HSI_WEIGHTS_DIR / "spectral_processor.pkl"
SPLIT_FILE = HSI_WEIGHTS_DIR / "hsi_split.txt"
METRICS_FILE = HSI_RESULTS_DIR / "hsi_test_metrics.txt"


# ================================================================
# CONFIG PRINT
# ================================================================

print("=" * 70)
print("HSI FINAL CONFIGURATION (CORRECTED)")
print("=" * 70)

print(f"Project root       : {PROJECT_ROOT}")
print(f"Train HSI          : {TRAIN_HSI_DIR}")
print(f"Val HSI            : {VAL_HSI_DIR}")
print(f"Test HSI           : {TEST_HSI_DIR}")

print(f"Device             : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU                : {torch.cuda.get_device_name(0)}")

print()
print(f"Original bands     : {ORIGINAL_BANDS}")
print(f"Removed bands      : {len(BANDS_TO_REMOVE)}")
print(f"Remaining bands    : {EXPECTED_REMAINING_BANDS}")
print(f"PCA components     : {PCA_COMPONENTS}")

print()
print(f"Patch size         : {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Train stride       : {TRAIN_STRIDE}")
print(f"Validation stride  : {VAL_STRIDE}")
print(f"Test stride        : {TEST_STRIDE}")

print()
print(f"Batch size         : {BATCH_SIZE}")
print(f"Epochs             : {NUM_EPOCHS}")
print(f"Learning rate      : {LEARNING_RATE}  (was 5e-4)")
print(f"Warmup epochs      : {WARMUP_EPOCHS}")
print(f"Weight decay       : {WEIGHT_DECAY}")
print(f"Target oil frac.   : {TARGET_OIL_FRACTION}  (was ~0.50 implicit)")
print(f"Early stop patience: {EARLY_STOPPING_PATIENCE}  (was 12)")

print("=" * 70)