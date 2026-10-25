"""
STAGE 3 — REGISTRATION-FREE SAR + HSI FUSION
Central configuration.

Scientific contract
-------------------
1. SAR and HSI experts are frozen.
2. No pixel-level SAR/HSI registration is assumed.
3. HSI is used as spectral / scene context.
4. SAR remains the spatial segmentation anchor.
5. Only fusion modules are trainable.
6. Thresholds and calibration are selected using validation only.
7. Test scenes are never used during fusion training or model selection.
"""

from pathlib import Path
import random
import numpy as np
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL"
).resolve()

FUSION_DIR = PROJECT_ROOT / "fusion_layer"

SAR_DIR = PROJECT_ROOT / "sar_layer"
HSI_DIR = PROJECT_ROOT / "hsi_layer_corrected"

OSD_ROOT = PROJECT_ROOT / "new_dataset" / "OSD"


# ============================================================================
# EXPERT CHECKPOINTS
# ============================================================================

SAR_MODEL_FILE = SAR_DIR / "03_model.py"
SAR_CHECKPOINT = SAR_DIR / "weights" / "best_sar_expert.pth"

HSI_MODEL_FILE = HSI_DIR / "03_model.py"
HSI_DATASET_FILE = HSI_DIR / "02_dataset.py"

HSI_CHECKPOINT = (
    HSI_DIR / "weights" / "frozen_hsi_expert.pth"
)

HSI_PROCESSOR = (
    HSI_DIR / "weights" / "spectral_processor.pkl"
)

# Compatibility aliases used by 03_load_experts.py and later fusion stages.
SAR_MODEL_PATH = SAR_MODEL_FILE
SAR_CHECKPOINT_PATH = SAR_CHECKPOINT
HSI_MODEL_PATH = HSI_MODEL_FILE
HSI_CHECKPOINT_PATH = HSI_CHECKPOINT
HSI_PROCESSOR_PATH = HSI_PROCESSOR


# ============================================================================
# OSD DATASET
# ============================================================================

TRAIN_ROOT = (
    OSD_ROOT
    / "TrainCopy-20260405T092743Z-1-001"
    / "TrainCopy"
)

VAL_ROOT = (
    OSD_ROOT
    / "ValCopy-20260405T092906Z-1-001"
    / "ValCopy"
)

TEST_ROOT = (
    OSD_ROOT
    / "TestCopy-20260405T092640Z-1-001"
    / "TestCopy"
)

SAR_TRAIN_ROOT = (
    OSD_ROOT
    / "TrainSAR-20260405T092903Z-1-001"
    / "TrainSAR"
)

SAR_VAL_ROOT = (
    OSD_ROOT
    / "ValSAR-20260405T092922Z-1-001"
    / "ValSAR"
)

SAR_TEST_ROOT = (
    OSD_ROOT
    / "TestSAR-20260405T092736Z-1-001"
    / "TestSAR"
)


# ============================================================================
# LOCKED SCENE SPLIT
# ============================================================================

TRAIN_SCENES = [
    "GM01",
    "GM02",
    "GM06",
    "GM07",
    "GM08",
    "GM12",
    "GM13",
    "GM14",
    "GM18",
]

VAL_SCENES = [
    "GM03",
    "GM09",
    "GM15",
]

TEST_SCENES = [
    "GM04",
    "GM05",
    "GM10",
    "GM11",
    "GM16",
    "GM17",
]


# ============================================================================
# DATA PARAMETERS
# ============================================================================

SAR_IMAGE_SIZE = 256

HSI_ORIGINAL_BANDS = 224
HSI_REMAINING_BANDS = 193
HSI_PCA_COMPONENTS = 25
HSI_PATCH_SIZE = 5


# ============================================================================
# FUSION ARCHITECTURE
# ============================================================================

# Common latent dimension.
FUSION_DIM = 128

# Number of attention heads.
NUM_HEADS = 4

# Dropout used only in trainable fusion layers.
DROPOUT = 0.10

# Hidden dimension for gating/context MLP.
GATE_HIDDEN_DIM = 256

# Maximum magnitude of residual correction added to SAR logits.
#
# The actual correction is:
#
# correction = MAX_RESIDUAL_SCALE * tanh(raw_correction)
#
# This prevents the fusion head from arbitrarily destroying the frozen
# SAR expert's spatial segmentation.
MAX_RESIDUAL_SCALE = 1.50


# ============================================================================
# TRAINING
# ============================================================================

SEED = 42

# Cached feature vectors do not require image-sized micro-batches.
BATCH_SIZE = 256

NUM_EPOCHS = 60

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = True

AMP = True

GRAD_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 10


# ============================================================================
# LOSS
# ============================================================================

# Segmentation BCE component.
BCE_WEIGHT = 1.0

# Dice component.
DICE_WEIGHT = 1.0

# Auxiliary residual regularization.
#
# Prevents the fusion head from making unnecessarily large corrections.
RESIDUAL_REG_WEIGHT = 0.01

# Auxiliary supervision for the per-sample HSI-versus-SAR gate.
GATE_SUPERVISION_WEIGHT = 0.3


# ============================================================================
# VALIDATION-ONLY CALIBRATION / THRESHOLDS
# ============================================================================

# Do not tune these on test.

SAR_BASE_THRESHOLD = 0.66

HSI_BASE_THRESHOLD = 0.71

FUSION_THRESHOLD_SEARCH_MIN = 0.20
FUSION_THRESHOLD_SEARCH_MAX = 0.80
FUSION_THRESHOLD_SEARCH_STEP = 0.01


# ============================================================================
# OUTPUT PATHS
# ============================================================================

WEIGHTS_DIR = FUSION_DIR / "weights"
RESULTS_DIR = FUSION_DIR / "results"
LOGS_DIR = FUSION_DIR / "logs"
VIS_DIR = FUSION_DIR / "visualizations"

BEST_FUSION_CHECKPOINT = (
    WEIGHTS_DIR / "best_registration_free_fusion.pth"
)

LAST_FUSION_CHECKPOINT = (
    WEIGHTS_DIR / "last_registration_free_fusion.pth"
)

TRAINING_HISTORY_FILE = (
    LOGS_DIR / "fusion_training_history.csv"
)

FUSION_SPLIT_MANIFEST = (
    RESULTS_DIR / "fusion_split_manifest.txt"
)

DATASET_AUDIT_FILE = (
    RESULTS_DIR / "dataset_exploration.json"
)


# ============================================================================
# DEVICE
# ============================================================================

def get_device():
    """
    Return CUDA when available.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int = SEED):
    """
    Make the experiment reproducible as far as practical.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories():

    for path in [
        FUSION_DIR,
        WEIGHTS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        VIS_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================================
# PROJECT VALIDATION
# ============================================================================

def validate_required_paths():

    required = {
        "SAR model": SAR_MODEL_FILE,
        "SAR checkpoint": SAR_CHECKPOINT,
        "HSI model": HSI_MODEL_FILE,
        "HSI dataset": HSI_DATASET_FILE,
        "HSI checkpoint": HSI_CHECKPOINT,
        "HSI processor": HSI_PROCESSOR,
        "SAR train root": SAR_TRAIN_ROOT,
        "SAR validation root": SAR_VAL_ROOT,
        "SAR test root": SAR_TEST_ROOT,
        "OSD train root": TRAIN_ROOT,
        "OSD validation root": VAL_ROOT,
        "OSD test root": TEST_ROOT,
    }

    missing = []

    print("=" * 88)
    print("STAGE 3 — FUSION CONFIGURATION CHECK")
    print("=" * 88)

    for name, path in required.items():

        if path.exists():

            print(f"[OK]   {name}")
            print(f"       {path}")

        else:

            print(f"[MISSING] {name}")
            print(f"          {path}")

            missing.append(str(path))

    print("=" * 88)

    if missing:

        raise FileNotFoundError(
            "Required paths are missing:\n"
            + "\n".join(missing)
        )


# ============================================================================
# SPLIT INTEGRITY
# ============================================================================

def validate_scene_splits():

    train_set = set(TRAIN_SCENES)
    val_set = set(VAL_SCENES)
    test_set = set(TEST_SCENES)

    assert train_set.isdisjoint(val_set), (
        "TRAIN and VAL scene leakage detected."
    )

    assert train_set.isdisjoint(test_set), (
        "TRAIN and TEST scene leakage detected."
    )

    assert val_set.isdisjoint(test_set), (
        "VAL and TEST scene leakage detected."
    )


# ============================================================================
# CONFIG DISPLAY
# ============================================================================

def print_config():

    device = get_device()

    print()
    print("=" * 88)
    print("STAGE 3 — REGISTRATION-FREE SAR + HSI FUSION")
    print("=" * 88)

    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Fusion directory    : {FUSION_DIR}")

    print()

    print(f"Device              : {device}")

    if device.type == "cuda":

        print(
            "GPU                 : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    print("EXPERTS")

    print(f"SAR checkpoint      : {SAR_CHECKPOINT}")
    print(f"HSI checkpoint      : {HSI_CHECKPOINT}")
    print(f"HSI processor       : {HSI_PROCESSOR}")

    print()

    print("LOCKED SCENES")

    print(f"Train ({len(TRAIN_SCENES)}): {TRAIN_SCENES}")
    print(f"Val   ({len(VAL_SCENES)}): {VAL_SCENES}")
    print(f"Test  ({len(TEST_SCENES)}): {TEST_SCENES}")

    print()

    print("FUSION CONTRACT")

    print("Pixel registration  : NOT ASSUMED")
    print("SAR role            : Spatial anchor")
    print("HSI role            : Spectral / scene context")
    print("Experts             : Frozen")
    print("Trainable modules   : Fusion only")
    print(
        "Correction          : "
        f"Bounded residual, max scale={MAX_RESIDUAL_SCALE}"
    )

    print()

    print("TRAINING")

    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Epochs              : {NUM_EPOCHS}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"AMP                 : {AMP}")

    print("=" * 88)


# ============================================================================
# MAIN SELF TEST
# ============================================================================

if __name__ == "__main__":

    ensure_directories()

    validate_required_paths()

    validate_scene_splits()

    seed_everything()

    print_config()

    print()

    print("[SUCCESS] Fusion configuration is valid.")