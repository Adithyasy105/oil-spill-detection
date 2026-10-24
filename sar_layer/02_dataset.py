# %% [markdown]
# # 📡 Stage 1: SAR Expert — Dataset & Preprocessing
# **SOS Dataset Loader with Boundary Targets (Canny Edge)**
#
# This module handles:
# - Loading SOS dataset (both flat and nested structures)
# - SAR-specific preprocessing (dB scaling + Z-score normalization)
# - Canny edge extraction for boundary supervision (MBL)
# - Data augmentation (flips, rotations, Gaussian noise)
# - Train/Val/Test split management

# %%
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import config — handles multiple execution contexts
import sys
_sar_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() or '__file__' in globals() else os.getcwd()
if _sar_dir not in sys.path:
    sys.path.insert(0, _sar_dir)

# Only load config if not already loaded (avoids re-exec when chained)
if 'IMG_SIZE' not in dir():
    _cfg_path = os.path.join(_sar_dir, '01_config.py')
    if os.path.exists(_cfg_path):
        exec(open(_cfg_path, encoding='utf-8').read())
    else:
        # Fallback defaults for cloud deployment
        IMG_SIZE = 256
        IN_CHANNELS = 3
        NUM_CLASSES = 1
        ENCODER_NAME = "resnet18"
        ENCODER_WEIGHTS = "imagenet"
        DECODER_ATTENTION = "scse"
        CANNY_LOW_THRESHOLD = 50
        CANNY_HIGH_THRESHOLD = 150
        MBL_SCALE_WEIGHTS = [0.5, 0.3, 0.2]
        AUG_HFLIP_PROB = 0.5
        AUG_VFLIP_PROB = 0.5
        AUG_ROT90_PROB = 0.5
        AUG_NOISE_PROB = 0.2
        AUG_NOISE_STD = 0.05
        VAL_SPLIT = 0.15
        BATCH_SIZE = 8
        RANDOM_SEED = 42
        SOS_TRAIN_DIR = ""
        SOS_TEST_DIR = ""

# %% [markdown]
# ## 1. Canny Edge Extraction (Boundary Targets for MBL)
#
# The MBL loss requires ground truth boundary maps. We extract these from
# the segmentation masks using Canny edge detection at multiple scales.

# %%
def extract_boundary_multiscale(mask, scales=[1, 3, 5],
                                 low_thresh=50, high_thresh=150):
    """
    Extract multi-scale boundary maps from a binary mask using Canny edge detection.

    For MBL = Σ_s w_s × BCE(B_pred^s, B_gt^s):
      - Scale 1 (kernel=1): Fine edges — exact boundary pixels
      - Scale 3 (kernel=3): Medium edges — 3-pixel neighborhood
      - Scale 5 (kernel=5): Coarse edges — 5-pixel neighborhood (smoothed)

    Args:
        mask: Binary mask (H, W), values 0 or 1 (float32 or uint8)
        scales: List of Gaussian blur kernel sizes before Canny
        low_thresh: Canny lower threshold
        high_thresh: Canny upper threshold

    Returns:
        List of boundary maps, one per scale, each (H, W) float32 in [0, 1]
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

    boundaries = []
    for k in scales:
        if k > 1:
            # Gaussian blur at this scale 
            blurred = cv2.GaussianBlur(mask_uint8, (k, k), 0)
        else:
            blurred = mask_uint8

        # Canny edge detection
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Dilate edges slightly for better supervision signal
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Normalize to [0, 1]
        edges = edges.astype(np.float32) / 255.0
        boundaries.append(edges)

    return boundaries

# %% [markdown]
# ## 2. SAR Preprocessing Pipeline
#
# Per CBD-Net paper and our architecture:
# 1. Load grayscale SAR image
# 2. Apply dB scaling: `10 × log10(|pixel| + ε)` — compresses dynamic range
# 3. Z-score normalization: `(x - μ) / σ` — centers distribution at 0
# 4. Replicate to 3 channels (for pretrained ResNet-18 ImageNet weights)

# %%
def preprocess_sar(image):
    """
    SAR-specific preprocessing pipeline.

    Args:
        image: Grayscale image (H, W) uint8

    Returns:
        Preprocessed image (H, W) float32, z-score normalized
    """
    image = image.astype(np.float32)

    # Step 1: dB scaling — compresses the huge dynamic range of SAR backscatter
    # 10 * log10(|pixel| + ε) transforms multiplicative noise to additive
    image = 10.0 * np.log10(np.abs(image) + 1e-6)

    # Step 2: Z-score normalization — centers at 0 with unit variance
    mean = image.mean()
    std = image.std()
    image = (image - mean) / (std + 1e-6)

    return image

# %% [markdown]
# ## 3. SOS Dataset Class
#
# Handles two directory structures in the SOS dataset:
# - **Flat (train):** `XXXXX_sat.jpg` + `XXXXX_mask.png` in same folder
# - **Nested (test):** `sat/XXXXX_sat.jpg` + `gt/XXXXX_mask.png` in subfolders

# %%
class SOSDataset(Dataset):
    """
    SOS (Deep-SAR Oil Spill) Dataset for CBD-Net Training.

    Returns:
        image: (3, 256, 256) float32 — preprocessed SAR image (3-ch for ResNet)
        mask: (1, 256, 256) float32 — binary segmentation mask
        boundaries: list of (1, 256, 256) float32 — multi-scale boundary maps for MBL
    """
    def __init__(self, root_dir, augment=False, return_boundaries=True):
        self.root_dir = root_dir
        self.augment = augment
        self.return_boundaries = return_boundaries

        # Detect directory structure
        self.pairs = self._discover_pairs(root_dir)
        print(f"  SOSDataset: Loaded {len(self.pairs)} SAR-mask pairs from {root_dir}")

    def _discover_pairs(self, root_dir):
        """Find all (image_path, mask_path) pairs in the dataset."""
        pairs = []

        # Check for flat structure: XXXXX_sat.jpg + XXXXX_mask.png
        flat_images = sorted([f for f in os.listdir(root_dir) if f.endswith('_sat.jpg')])
        if flat_images:
            for img_name in flat_images:
                base = img_name.replace('_sat.jpg', '')
                mask_name = base + '_mask.png'
                img_path = os.path.join(root_dir, img_name)
                mask_path = os.path.join(root_dir, mask_name)
                if os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
            return pairs

        # Check for nested structure: sat/ + gt/
        sat_dir = os.path.join(root_dir, 'sat')
        gt_dir = os.path.join(root_dir, 'gt')
        if os.path.exists(sat_dir) and os.path.exists(gt_dir):
            sat_files = sorted([f for f in os.listdir(sat_dir) if f.endswith(('.jpg', '.png'))])
            for sat_name in sat_files:
                base = os.path.splitext(sat_name)[0].replace('_sat', '')
                # Try multiple mask naming conventions
                for mask_ext in ['_mask.png', '.png', '_mask.jpg', '.jpg']:
                    mask_path = os.path.join(gt_dir, base + mask_ext)
                    if os.path.exists(mask_path):
                        pairs.append((os.path.join(sat_dir, sat_name), mask_path))
                        break
            return pairs

        raise FileNotFoundError(f"Could not find SOS images in {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 1. Load SAR image (grayscale — SAR is single-channel radar)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # 2. Load ground truth mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # 3. Resize to standard size (256×256)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # 4. SAR preprocessing (dB scaling + Z-score normalization)
        image = preprocess_sar(image)

        # 5. Binary mask: oil (white, >127) → 1, water (black) → 0
        mask = (mask > 127).astype(np.float32)

        # 6. Extract boundary maps for MBL loss (before augmentation to maintain consistency)
        if self.return_boundaries:
            boundaries = extract_boundary_multiscale(
                mask,
                scales=[1, 3, 5],
                low_thresh=CANNY_LOW_THRESHOLD,
                high_thresh=CANNY_HIGH_THRESHOLD
            )

        # 7. Data augmentation (applied consistently to image, mask, and boundaries)
        if self.augment:
            image, mask, boundaries = self._augment(
                image, mask,
                boundaries if self.return_boundaries else None
            )

        # 8. Convert to PyTorch tensors
        # Image: replicate grayscale to 3 channels for pretrained ResNet-18
        image_3ch = np.stack([image.copy()] * 3, axis=0)  # (3, H, W)
        image_tensor = torch.from_numpy(image_3ch).float()
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()  # (1, H, W)

        if self.return_boundaries:
            boundary_tensors = [
                torch.from_numpy(b.copy()).unsqueeze(0).float()
                for b in boundaries
            ]  # List of (1, H, W) tensors
            return image_tensor, mask_tensor, boundary_tensors
        else:
            return image_tensor, mask_tensor

    def _augment(self, image, mask, boundaries=None):
        """Apply consistent augmentation to image, mask, and boundary maps."""

        # Horizontal flip
        if np.random.rand() < AUG_HFLIP_PROB:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            if boundaries:
                boundaries = [cv2.flip(b, 1) for b in boundaries]

        # Vertical flip
        if np.random.rand() < AUG_VFLIP_PROB:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            if boundaries:
                boundaries = [cv2.flip(b, 0) for b in boundaries]

        # Random 90° rotation
        if np.random.rand() < AUG_ROT90_PROB:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
            if boundaries:
                boundaries = [np.rot90(b, k).copy() for b in boundaries]

        # Gaussian noise (simulates radar speckle variation)
        if np.random.rand() < AUG_NOISE_PROB:
            noise = np.random.normal(0, AUG_NOISE_STD, image.shape).astype(np.float32)
            image = image + noise

        return image, mask, boundaries if boundaries else []

# %% [markdown]
# ## 4. Data Splitting Utilities
#
# Creates train/val splits from the training set, keeping test as holdout.
# Uses stratified splitting based on oil pixel ratio to ensure balanced splits.

# %%
def compute_oil_ratio(mask_path):
    """Compute the fraction of oil pixels in a mask for stratified splitting."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    return (mask > 127).sum() / mask.size

def create_data_loaders(val_split=VAL_SPLIT, batch_size=BATCH_SIZE, seed=RANDOM_SEED):
    """
    Create train, validation, and test data loaders.

    Strategy:
        - Train set (SOS train): split into 85% train / 15% val (stratified by oil ratio)
        - Test set (SOS test): kept as holdout, no augmentation

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n  Loading SOS Dataset...")
    print("  " + "=" * 50)

    # Full train dataset (with augmentation disabled for splitting)
    full_train_ds = SOSDataset(SOS_TRAIN_DIR, augment=False)

    # Compute oil ratios for stratified splitting
    print("  Computing oil ratios for stratified split...")
    oil_ratios = []
    for _, mask_path in full_train_ds.pairs:
        ratio = compute_oil_ratio(mask_path)
        oil_ratios.append(ratio)

    # Stratify: bin oil ratios into categories (empty, low, medium, high)
    oil_bins = np.digitize(oil_ratios, bins=[0.001, 0.05, 0.15, 0.30])

    # Split indices
    indices = list(range(len(full_train_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed,
        stratify=oil_bins
    )

    # Create train dataset WITH augmentation
    train_ds_aug = SOSDataset(SOS_TRAIN_DIR, augment=True)
    # Create val dataset WITHOUT augmentation
    val_ds_noaug = SOSDataset(SOS_TRAIN_DIR, augment=False)

    train_subset = Subset(train_ds_aug, train_idx)
    val_subset = Subset(val_ds_noaug, val_idx)

    # Test dataset
    test_ds = SOSDataset(SOS_TEST_DIR, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"\n  Train: {len(train_subset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_subset)} samples ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_ds)} samples ({len(test_loader)} batches)")
    print("  " + "=" * 50)

    return train_loader, val_loader, test_loader

# %% [markdown]
# ## 5. Quick Verification
#
# Test the dataset loading and boundary extraction.

# %%
if __name__ == "__main__" and 'SKIP_SELF_TEST' not in dir():
    # Verify dataset works
    ds = SOSDataset(SOS_TRAIN_DIR, augment=True)
    img, mask, bounds = ds[0]
    print(f"\n  Sample shapes:")
    print(f"    Image:    {img.shape} (min={img.min():.3f}, max={img.max():.3f})")
    print(f"    Mask:     {mask.shape} (unique={torch.unique(mask).tolist()})")
    print(f"    Boundary scales: {len(bounds)}")
    for i, b in enumerate(bounds):
        print(f"      Scale {i}: {b.shape} (edge pixels: {(b > 0).sum().item()})")

    # Test full data loader creation
    train_loader, val_loader, test_loader = create_data_loaders()

    # Verify a batch
    for batch in train_loader:
        images, masks, boundaries = batch
        print(f"\n  Batch shapes:")
        print(f"    Images:     {images.shape}")
        print(f"    Masks:      {masks.shape}")
        print(f"    Boundaries: {len(boundaries)} scales × {boundaries[0].shape}")
        break
