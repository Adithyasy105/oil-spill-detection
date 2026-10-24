# # ================================================================
# # 02_dataset.py
# # FINAL HSI DATASET + SPECTRAL PREPROCESSING
# # ================================================================

# import os
# import numpy as np
# import scipy.io as sio
# import joblib

# import torch
# from torch.utils.data import (
#     Dataset,
#     DataLoader,
#     WeightedRandomSampler
# )

# from sklearn.decomposition import PCA


# # ================================================================
# # CONFIG
# # ================================================================

# try:
#     _CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     _CONFIG_DIR = os.getcwd()

# _CONFIG_PATH = os.path.join(
#     _CONFIG_DIR,
#     "01_config.py"
# )

# if os.path.exists(_CONFIG_PATH):

#     if "PCA_COMPONENTS" not in globals():
#         exec(
#             open(
#                 _CONFIG_PATH,
#                 encoding="utf-8"
#             ).read()
#         )


# # ================================================================
# # DATA LOADING
# # ================================================================

# def load_hsi_scene(path):

#     path = str(path)

#     if not os.path.exists(path):
#         raise FileNotFoundError(path)

#     data = sio.loadmat(path)

#     if "img" not in data:
#         raise KeyError(
#             f"{path}: missing 'img'"
#         )

#     if "map" in data:
#         gt = data["map"]

#     elif "gt" in data:
#         gt = data["gt"]

#     else:
#         raise KeyError(
#             f"{path}: missing 'map' or 'gt'"
#         )

#     img = data["img"].astype(
#         np.float32,
#         copy=False
#     )

#     gt = gt.astype(
#         np.float32,
#         copy=False
#     )

#     if img.ndim != 3:
#         raise ValueError(
#             f"{path}: expected HxWxC image, "
#             f"got {img.shape}"
#         )

#     if gt.ndim != 2:
#         gt = np.squeeze(gt)

#     if gt.ndim != 2:
#         raise ValueError(
#             f"{path}: invalid GT shape {gt.shape}"
#         )

#     if img.shape[:2] != gt.shape:
#         raise ValueError(
#             f"{path}: image/GT mismatch: "
#             f"{img.shape[:2]} vs {gt.shape}"
#         )

#     # Binary oil-spill target.
#     gt = (gt > 0).astype(np.float32)

#     return img, gt


# # ================================================================
# # SCENE DISCOVERY
# # ================================================================

# def discover_scene_files(directory):

#     directory = os.path.abspath(str(directory))

#     if not os.path.isdir(directory):
#         raise FileNotFoundError(
#             f"HSI directory does not exist:\n{directory}"
#         )

#     files = sorted(
#         [
#             os.path.join(directory, f)
#             for f in os.listdir(directory)
#             if f.lower().endswith(".mat")
#         ]
#     )

#     if not files:
#         raise RuntimeError(
#             f"No .mat files found in:\n{directory}"
#         )

#     return files


# def scene_name(path):

#     return os.path.splitext(
#         os.path.basename(path)
#     )[0]


# # ================================================================
# # SPECTRAL PROCESSOR
# # ================================================================

# class SpectralProcessor:

#     """
#     Correct preprocessing contract:

#         RAW HSI
#           ↓
#         remove absorption bands
#           ↓
#         training-derived per-band min/max
#           ↓
#         normalize
#           ↓
#         PCA-25
#           ↓
#         model

#     IMPORTANT:
#         No normalization is applied after PCA.
#     """

#     def __init__(
#         self,
#         n_components=25,
#         bands_to_remove=None
#     ):

#         self.n_components = n_components

#         self.bands_to_remove = (
#             list(bands_to_remove)
#             if bands_to_remove is not None
#             else list(BANDS_TO_REMOVE)
#         )

#         self.pca = PCA(
#             n_components=n_components,
#             random_state=SEED
#         )

#         self.band_min = None
#         self.band_max = None

#         self.is_fitted = False

#     # ------------------------------------------------------------
#     # BAND REMOVAL
#     # ------------------------------------------------------------

#     def remove_absorption_bands(self, img):

#         if img.ndim != 3:
#             raise ValueError(
#                 f"Expected HxWxC image, got {img.shape}"
#             )

#         bands = img.shape[2]

#         # Already reduced dataset.
#         if bands == EXPECTED_REMAINING_BANDS:
#             return img

#         if bands != ORIGINAL_BANDS:
#             raise ValueError(
#                 f"Expected either "
#                 f"{ORIGINAL_BANDS} raw bands or "
#                 f"{EXPECTED_REMAINING_BANDS} reduced bands, "
#                 f"got {bands}"
#             )

#         keep = [
#             i
#             for i in range(bands)
#             if i not in self.bands_to_remove
#         ]

#         out = img[:, :, keep]

#         if out.shape[2] != EXPECTED_REMAINING_BANDS:
#             raise RuntimeError(
#                 "Incorrect number of remaining bands: "
#                 f"{out.shape[2]}"
#             )

#         return out

#     # ------------------------------------------------------------
#     # FIT
#     # ------------------------------------------------------------

#     def fit(self, train_images):

#         print()
#         print("=" * 70)
#         print("FITTING HSI SPECTRAL PROCESSOR")
#         print("=" * 70)

#         if not train_images:
#             raise ValueError(
#                 "No training images supplied."
#             )

#         samples = []

#         rng = np.random.default_rng(SEED)

#         for idx, img in enumerate(train_images):

#             img = img.astype(
#                 np.float32,
#                 copy=False
#             )

#             img = self.remove_absorption_bands(img)

#             h, w, c = img.shape

#             pixels = img.reshape(
#                 -1,
#                 c
#             )

#             # Deterministic bounded sampling.
#             max_samples = min(
#                 100_000,
#                 pixels.shape[0]
#             )

#             if pixels.shape[0] > max_samples:

#                 indices = rng.choice(
#                     pixels.shape[0],
#                     size=max_samples,
#                     replace=False
#                 )

#                 pixels = pixels[indices]

#             samples.append(pixels)

#             print(
#                 f"  Train scene {idx + 1:02d}: "
#                 f"{h}x{w}x{c} | "
#                 f"samples={pixels.shape[0]:,}"
#             )

#         all_pixels = np.concatenate(
#             samples,
#             axis=0
#         )

#         del samples

#         print(
#             f"\nTotal PCA fitting samples: "
#             f"{all_pixels.shape[0]:,}"
#         )

#         # --------------------------------------------------------
#         # TRAIN-ONLY NORMALIZATION
#         # --------------------------------------------------------

#         self.band_min = np.nanmin(
#             all_pixels,
#             axis=0
#         )

#         self.band_max = np.nanmax(
#             all_pixels,
#             axis=0
#         )

#         # Prevent invalid ranges.
#         invalid = (
#             ~np.isfinite(self.band_min)
#             |
#             ~np.isfinite(self.band_max)
#         )

#         if np.any(invalid):
#             raise ValueError(
#                 "Non-finite band statistics detected."
#             )

#         ranges = (
#             self.band_max
#             - self.band_min
#         )

#         if np.any(ranges < 1e-12):

#             print(
#                 "WARNING: constant spectral bands detected."
#             )

#             ranges = np.maximum(
#                 ranges,
#                 1e-8
#             )

#         # Normalize before PCA.
#         normalized = (
#             all_pixels
#             - self.band_min
#         ) / (
#             ranges + 1e-8
#         )

#         normalized = np.nan_to_num(
#             normalized,
#             nan=0.0,
#             posinf=1.0,
#             neginf=0.0
#         )

#         # --------------------------------------------------------
#         # PCA
#         # --------------------------------------------------------

#         print(
#             f"Fitting PCA with "
#             f"{self.n_components} components..."
#         )

#         self.pca.fit(normalized)

#         self.is_fitted = True

#         explained = (
#             np.sum(
#                 self.pca.explained_variance_ratio_
#             )
#             * 100
#         )

#         print(
#             f"PCA explained variance: "
#             f"{explained:.3f}%"
#         )

#         print("=" * 70)

#     # ------------------------------------------------------------
#     # TRANSFORM
#     # ------------------------------------------------------------

#     def transform(self, img):

#         if not self.is_fitted:
#             raise RuntimeError(
#                 "SpectralProcessor has not been fitted."
#             )

#         img = img.astype(
#             np.float32,
#             copy=False
#         )

#         img = self.remove_absorption_bands(img)

#         h, w, c = img.shape

#         if c != EXPECTED_REMAINING_BANDS:
#             raise ValueError(
#                 f"Expected {EXPECTED_REMAINING_BANDS} "
#                 f"bands before PCA, got {c}"
#             )

#         pixels = img.reshape(
#             -1,
#             c
#         )

#         ranges = (
#             self.band_max
#             - self.band_min
#         )

#         pixels = (
#             pixels
#             - self.band_min
#         ) / (
#             ranges + 1e-8
#         )

#         pixels = np.nan_to_num(
#             pixels,
#             nan=0.0,
#             posinf=1.0,
#             neginf=0.0
#         )

#         # IMPORTANT:
#         # No min-max normalization after PCA.
#         pixels_pca = self.pca.transform(
#             pixels
#         )

#         pixels_pca = np.asarray(
#             pixels_pca,
#             dtype=np.float32
#         )

#         return pixels_pca.reshape(
#             h,
#             w,
#             self.n_components
#         )

#     # ------------------------------------------------------------
#     # SAVE
#     # ------------------------------------------------------------

#     def save(self, path):

#         if not self.is_fitted:
#             raise RuntimeError(
#                 "Cannot save unfitted processor."
#             )

#         payload = {

#             "pca": self.pca,

#             "band_min":
#                 self.band_min,

#             "band_max":
#                 self.band_max,

#             "bands_to_remove":
#                 self.bands_to_remove,

#             "n_components":
#                 self.n_components,

#             "original_bands":
#                 ORIGINAL_BANDS,

#             "remaining_bands":
#                 EXPECTED_REMAINING_BANDS,

#             "seed":
#                 SEED,

#             "contract":
#                 "train_only_fit; no_post_pca_normalization"
#         }

#         joblib.dump(
#             payload,
#             path
#         )

#         print(
#             f"[OK] Processor saved:\n{path}"
#         )

#     # ------------------------------------------------------------
#     # LOAD
#     # ------------------------------------------------------------

#     def load(self, path):

#         if not os.path.exists(path):
#             raise FileNotFoundError(path)

#         payload = joblib.load(path)

#         self.pca = payload["pca"]

#         self.band_min = payload["band_min"]

#         self.band_max = payload["band_max"]

#         self.bands_to_remove = payload[
#             "bands_to_remove"
#         ]

#         self.n_components = payload[
#             "n_components"
#         ]

#         self.is_fitted = True

#         return self


# # ================================================================
# # PATCH DATASET
# # ================================================================

# class HOSDDataset(Dataset):

#     def __init__(
#         self,
#         scene_paths,
#         processor,
#         patch_size=5,
#         stride=3,
#         augment=False
#     ):

#         self.scene_paths = list(scene_paths)

#         self.processor = processor

#         self.patch_size = patch_size

#         self.radius = patch_size // 2

#         self.stride = stride

#         self.augment = augment

#         self.scenes = []

#         self.index = []

#         self.labels = []

#         print()
#         print(
#             f"Building HSI dataset "
#             f"({len(self.scene_paths)} scenes)..."
#         )

#         for scene_idx, path in enumerate(
#             self.scene_paths
#         ):

#             img, mask = load_hsi_scene(
#                 path
#             )

#             pca_img = processor.transform(
#                 img
#             )

#             padded = np.pad(
#                 pca_img,
#                 (
#                     (self.radius, self.radius),
#                     (self.radius, self.radius),
#                     (0, 0)
#                 ),
#                 mode="reflect"
#             )

#             self.scenes.append(
#                 (
#                     padded,
#                     mask
#                 )
#             )

#             h, w = mask.shape

#             scene_indices = []

#             for row in range(
#                 0,
#                 h,
#                 self.stride
#             ):

#                 for col in range(
#                     0,
#                     w,
#                     self.stride
#                 ):

#                     self.index.append(
#                         (
#                             scene_idx,
#                             row,
#                             col
#                         )
#                     )

#                     self.labels.append(
#                         float(mask[row, col])
#                     )

#                     scene_indices.append(
#                         float(mask[row, col])
#                     )

#             oil_fraction = (
#                 np.mean(mask)
#                 * 100.0
#             )

#             print(
#                 f"  {scene_name(path):<8} | "
#                 f"{h}x{w} | "
#                 f"Oil={oil_fraction:.2f}% | "
#                 f"Samples={len(scene_indices):,}"
#             )

#     def __len__(self):

#         return len(self.index)

#     def __getitem__(self, idx):

#         scene_idx, row, col = (
#             self.index[idx]
#         )

#         img_pad, mask = (
#             self.scenes[scene_idx]
#         )

#         p = self.patch_size

#         patch = img_pad[
#             row:row + p,
#             col:col + p,
#             :
#         ]

#         label = mask[
#             row,
#             col
#         ]

#         # H,W,C -> C,H,W
#         patch = patch.transpose(
#             2,
#             0,
#             1
#         )

#         # Add input channel dimension.
#         # C,H,W -> 1,C,H,W
#         patch = np.expand_dims(
#             patch,
#             axis=0
#         )

#         # Spatial augmentation only.
#         if self.augment:

#             if np.random.rand() < 0.5:
#                 patch = np.flip(
#                     patch,
#                     axis=2
#                 ).copy()

#             if np.random.rand() < 0.5:
#                 patch = np.flip(
#                     patch,
#                     axis=3
#                 ).copy()

#         x = torch.from_numpy(
#             patch
#         ).float()

#         y = torch.tensor(
#             label,
#             dtype=torch.float32
#         )

#         return x, y


# # ================================================================
# # BUILD TRAIN/VAL
# # ================================================================

# def build_train_val_loaders(
#     processor,
#     batch_size=BATCH_SIZE
# ):

#     train_files = discover_scene_files(
#         TRAIN_HSI_DIR
#     )

#     val_files = discover_scene_files(
#         VAL_HSI_DIR
#     )

#     print()
#     print("=" * 70)
#     print("LOCKED HSI SPLIT")
#     print("=" * 70)

#     print("\nTRAIN:")

#     for f in train_files:
#         print(
#             f"  {scene_name(f)}"
#         )

#     print("\nVALIDATION:")

#     for f in val_files:
#         print(
#             f"  {scene_name(f)}"
#         )

#     print("=" * 70)

#     # ------------------------------------------------------------
#     # FIT PCA ONLY ON TRAIN
#     # ------------------------------------------------------------

#     train_images = []

#     for path in train_files:

#         img, _ = load_hsi_scene(
#             path
#         )

#         train_images.append(
#             img
#         )

#     processor.fit(
#         train_images
#     )

#     del train_images

#     # ------------------------------------------------------------
#     # SAVE PROCESSOR
#     # ------------------------------------------------------------

#     processor.save(
#         SPECTRAL_PROCESSOR_PATH
#     )

#     # ------------------------------------------------------------
#     # DATASETS
#     # ------------------------------------------------------------

#     train_ds = HOSDDataset(
#         train_files,
#         processor,
#         patch_size=PATCH_SIZE,
#         stride=TRAIN_STRIDE,
#         augment=True
#     )

#     val_ds = HOSDDataset(
#         val_files,
#         processor,
#         patch_size=PATCH_SIZE,
#         stride=VAL_STRIDE,
#         augment=False
#     )

#     # ------------------------------------------------------------
#     # CLASS BALANCING
#     # ------------------------------------------------------------

#     labels = np.asarray(
#         train_ds.labels,
#         dtype=np.int64
#     )

#     counts = np.bincount(
#         labels,
#         minlength=2
#     )

#     print()
#     print(
#         f"Training water samples: "
#         f"{counts[0]:,}"
#     )

#     print(
#         f"Training oil samples:   "
#         f"{counts[1]:,}"
#     )

#     oil_fraction = (
#         counts[1]
#         /
#         max(
#             counts.sum(),
#             1
#         )
#     )

#     print(
#         f"Training oil ratio: "
#         f"{oil_fraction * 100:.3f}%"
#     )

#     # Inverse class frequency.
#     class_weights = (
#         1.0
#         /
#         np.maximum(
#             counts,
#             1
#         )
#     )

#     sample_weights = (
#         class_weights[
#             labels
#         ]
#     )

#     sampler = WeightedRandomSampler(
#         weights=torch.as_tensor(
#             sample_weights,
#             dtype=torch.double
#         ),
#         num_samples=len(
#             sample_weights
#         ),
#         replacement=True
#     )

#     # ------------------------------------------------------------
#     # LOADERS
#     # ------------------------------------------------------------

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY
#     )

#     print()
#     print(
#         f"Train samples: "
#         f"{len(train_ds):,}"
#     )

#     print(
#         f"Val samples:   "
#         f"{len(val_ds):,}"
#     )

#     return (
#         train_loader,
#         val_loader,
#         train_files,
#         val_files
#     )


# # ================================================================
# # SELF TEST
# # ================================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 70)
#     print("HSI DATASET SELF TEST")
#     print("=" * 70)

#     train_files = discover_scene_files(
#         TRAIN_HSI_DIR
#     )

#     print(
#         f"Found {len(train_files)} training scenes."
#     )

#     processor = SpectralProcessor(
#         n_components=PCA_COMPONENTS,
#         bands_to_remove=BANDS_TO_REMOVE
#     )

#     images = []

#     for path in train_files[:1]:

#         img, gt = load_hsi_scene(
#             path
#         )

#         print(
#             f"Raw image: {img.shape}"
#         )

#         print(
#             f"GT: {gt.shape}"
#         )

#         images.append(img)

#     processor.fit(images)

#     transformed = processor.transform(
#         images[0]
#     )

#     print(
#         f"PCA image: {transformed.shape}"
#     )

#     assert transformed.shape[2] == PCA_COMPONENTS

#     ds = HOSDDataset(
#         train_files[:1],
#         processor,
#         patch_size=PATCH_SIZE,
#         stride=20,
#         augment=False
#     )

#     x, y = ds[0]

#     print(
#         f"Patch: {x.shape}"
#     )

#     print(
#         f"Label: {y.item()}"
#     )

#     assert x.shape == (
#         1,
#         PCA_COMPONENTS,
#         PATCH_SIZE,
#         PATCH_SIZE
#     )

#     print()
#     print("[SUCCESS] Dataset self-test passed.")


# ================================================================
# 02_dataset.py
# FINAL HSI DATASET + SPECTRAL PREPROCESSING  -- CORRECTED
#
# CHANGE FROM ORIGINAL:
#   build_train_val_loaders() now computes sampler weights to hit
#   TARGET_OIL_FRACTION (default 0.20) instead of raw inverse-
#   frequency weighting (which forced ~50% oil per epoch and caused
#   heavy repetition of a small, spatially correlated oil-patch
#   pool). Everything else is unchanged from the original file.
# ================================================================

import os
import numpy as np
import scipy.io as sio
import joblib

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.decomposition import PCA


# ================================================================
# CONFIG
# ================================================================

try:
    _CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _CONFIG_DIR = os.getcwd()

_CONFIG_PATH = os.path.join(_CONFIG_DIR, "01_config.py")

if os.path.exists(_CONFIG_PATH):
    if "PCA_COMPONENTS" not in globals():
        exec(open(_CONFIG_PATH, encoding="utf-8").read())


# ================================================================
# DATA LOADING
# ================================================================

def load_hsi_scene(path):

    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = sio.loadmat(path)

    if "img" not in data:
        raise KeyError(f"{path}: missing 'img'")

    if "map" in data:
        gt = data["map"]
    elif "gt" in data:
        gt = data["gt"]
    else:
        raise KeyError(f"{path}: missing 'map' or 'gt'")

    img = data["img"].astype(np.float32, copy=False)
    gt = gt.astype(np.float32, copy=False)

    if img.ndim != 3:
        raise ValueError(f"{path}: expected HxWxC image, got {img.shape}")

    if gt.ndim != 2:
        gt = np.squeeze(gt)

    if gt.ndim != 2:
        raise ValueError(f"{path}: invalid GT shape {gt.shape}")

    if img.shape[:2] != gt.shape:
        raise ValueError(
            f"{path}: image/GT mismatch: {img.shape[:2]} vs {gt.shape}"
        )

    gt = (gt > 0).astype(np.float32)

    return img, gt


# ================================================================
# SCENE DISCOVERY
# ================================================================

def discover_scene_files(directory):

    directory = os.path.abspath(str(directory))

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"HSI directory does not exist:\n{directory}")

    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".mat")
    )

    if not files:
        raise RuntimeError(f"No .mat files found in:\n{directory}")

    return files


def scene_name(path):
    return os.path.splitext(os.path.basename(path))[0]


# ================================================================
# SPECTRAL PROCESSOR  (unchanged from original)
# ================================================================

class SpectralProcessor:
    """
    RAW HSI -> remove absorption bands -> train-derived min/max
    -> normalize -> PCA-25 -> model. No normalization after PCA.
    """

    def __init__(self, n_components=25, bands_to_remove=None):
        self.n_components = n_components
        self.bands_to_remove = (
            list(bands_to_remove) if bands_to_remove is not None
            else list(BANDS_TO_REMOVE)
        )
        self.pca = PCA(n_components=n_components, random_state=SEED)
        self.band_min = None
        self.band_max = None
        self.is_fitted = False

    def remove_absorption_bands(self, img):
        if img.ndim != 3:
            raise ValueError(f"Expected HxWxC image, got {img.shape}")

        bands = img.shape[2]

        if bands == EXPECTED_REMAINING_BANDS:
            return img

        if bands != ORIGINAL_BANDS:
            raise ValueError(
                f"Expected either {ORIGINAL_BANDS} raw bands or "
                f"{EXPECTED_REMAINING_BANDS} reduced bands, got {bands}"
            )

        keep = [i for i in range(bands) if i not in self.bands_to_remove]
        out = img[:, :, keep]

        if out.shape[2] != EXPECTED_REMAINING_BANDS:
            raise RuntimeError(
                f"Incorrect number of remaining bands: {out.shape[2]}"
            )

        return out

    def fit(self, train_images):

        print()
        print("=" * 70)
        print("FITTING HSI SPECTRAL PROCESSOR")
        print("=" * 70)

        if not train_images:
            raise ValueError("No training images supplied.")

        samples = []
        rng = np.random.default_rng(SEED)

        for idx, img in enumerate(train_images):
            img = img.astype(np.float32, copy=False)
            img = self.remove_absorption_bands(img)

            h, w, c = img.shape
            pixels = img.reshape(-1, c)

            max_samples = min(100_000, pixels.shape[0])
            if pixels.shape[0] > max_samples:
                indices = rng.choice(pixels.shape[0], size=max_samples, replace=False)
                pixels = pixels[indices]

            samples.append(pixels)

            print(f"  Train scene {idx + 1:02d}: {h}x{w}x{c} | samples={pixels.shape[0]:,}")

        all_pixels = np.concatenate(samples, axis=0)
        del samples

        print(f"\nTotal PCA fitting samples: {all_pixels.shape[0]:,}")

        self.band_min = np.nanmin(all_pixels, axis=0)
        self.band_max = np.nanmax(all_pixels, axis=0)

        invalid = ~np.isfinite(self.band_min) | ~np.isfinite(self.band_max)
        if np.any(invalid):
            raise ValueError("Non-finite band statistics detected.")

        ranges = self.band_max - self.band_min
        if np.any(ranges < 1e-12):
            print("WARNING: constant spectral bands detected.")
            ranges = np.maximum(ranges, 1e-8)

        normalized = (all_pixels - self.band_min) / (ranges + 1e-8)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        print(f"Fitting PCA with {self.n_components} components...")
        self.pca.fit(normalized)
        self.is_fitted = True

        explained = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained:.3f}%")
        print("=" * 70)

    def transform(self, img):

        if not self.is_fitted:
            raise RuntimeError("SpectralProcessor has not been fitted.")

        img = img.astype(np.float32, copy=False)
        img = self.remove_absorption_bands(img)

        h, w, c = img.shape

        if c != EXPECTED_REMAINING_BANDS:
            raise ValueError(
                f"Expected {EXPECTED_REMAINING_BANDS} bands before PCA, got {c}"
            )

        pixels = img.reshape(-1, c)

        ranges = self.band_max - self.band_min
        pixels = (pixels - self.band_min) / (ranges + 1e-8)
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)

        pixels_pca = self.pca.transform(pixels)
        pixels_pca = np.asarray(pixels_pca, dtype=np.float32)

        return pixels_pca.reshape(h, w, self.n_components)

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted processor.")

        payload = {
            "pca": self.pca,
            "band_min": self.band_min,
            "band_max": self.band_max,
            "bands_to_remove": self.bands_to_remove,
            "n_components": self.n_components,
            "original_bands": ORIGINAL_BANDS,
            "remaining_bands": EXPECTED_REMAINING_BANDS,
            "seed": SEED,
            "contract": "train_only_fit; no_post_pca_normalization",
        }

        joblib.dump(payload, path)
        print(f"[OK] Processor saved:\n{path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        payload = joblib.load(path)

        self.pca = payload["pca"]
        self.band_min = payload["band_min"]
        self.band_max = payload["band_max"]
        self.bands_to_remove = payload["bands_to_remove"]
        self.n_components = payload["n_components"]
        self.is_fitted = True

        return self


# ================================================================
# PATCH DATASET  (unchanged from original)
# ================================================================

class HOSDDataset(Dataset):

    def __init__(self, scene_paths, processor, patch_size=5, stride=3, augment=False):

        self.scene_paths = list(scene_paths)
        self.processor = processor
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.stride = stride
        self.augment = augment

        self.scenes = []
        self.index = []
        self.labels = []

        print()
        print(f"Building HSI dataset ({len(self.scene_paths)} scenes)...")

        for scene_idx, path in enumerate(self.scene_paths):

            img, mask = load_hsi_scene(path)
            pca_img = processor.transform(img)

            padded = np.pad(
                pca_img,
                ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
                mode="reflect",
            )

            self.scenes.append((padded, mask))

            h, w = mask.shape
            scene_indices = []

            for row in range(0, h, self.stride):
                for col in range(0, w, self.stride):
                    self.index.append((scene_idx, row, col))
                    self.labels.append(float(mask[row, col]))
                    scene_indices.append(float(mask[row, col]))

            oil_fraction = np.mean(mask) * 100.0

            print(
                f"  {scene_name(path):<8} | {h}x{w} | "
                f"Oil={oil_fraction:.2f}% | Samples={len(scene_indices):,}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        scene_idx, row, col = self.index[idx]
        img_pad, mask = self.scenes[scene_idx]

        p = self.patch_size

        patch = img_pad[row:row + p, col:col + p, :]
        label = mask[row, col]

        patch = patch.transpose(2, 0, 1)          # H,W,C -> C,H,W
        patch = np.expand_dims(patch, axis=0)      # -> 1,C,H,W

        if self.augment:
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=2).copy()
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=3).copy()

        x = torch.from_numpy(patch).float()
        y = torch.tensor(label, dtype=torch.float32)

        return x, y


# ================================================================
# SAMPLER WEIGHTS  -- NEW, replaces raw inverse-frequency
# ================================================================

def compute_target_fraction_weights(labels, target_fraction):
    """
    Compute per-sample weights so that, in expectation, a batch drawn
    with WeightedRandomSampler(replacement=True) contains roughly
    `target_fraction` positive (oil) samples.

    Raw inverse-frequency weighting (weight = 1 / count per class)
    implicitly targets a 50/50 split. When the positive class has
    very few unique, spatially-correlated samples (as here: 45,515
    oil patches from only 9 scenes, heavily overlapping at stride 3),
    a 50/50 target forces ~9-10x repetition of that small pool per
    epoch, which is a fast route to memorization/overfitting rather
    than learning a general oil spectral signature.

    Solving for per-class weight w_pos, w_neg such that
        (n_pos * w_pos) / (n_pos * w_pos + n_neg * w_neg) = target_fraction
    with w_neg fixed at 1 / n_neg gives:
        w_pos = (target_fraction / (1 - target_fraction)) * (n_neg / n_pos) * w_neg
    """

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)

    n_neg, n_pos = counts[0], counts[1]

    if n_pos == 0:
        raise ValueError("No positive (oil) samples found in training set.")

    w_neg = 1.0 / max(n_neg, 1)
    ratio = target_fraction / (1.0 - target_fraction)
    w_pos = ratio * (n_neg / n_pos) * w_neg

    class_weights = np.array([w_neg, w_pos], dtype=np.float64)
    sample_weights = class_weights[labels]

    return sample_weights, counts


# ================================================================
# BUILD TRAIN/VAL  -- sampler section corrected
# ================================================================

def build_train_val_loaders(processor, batch_size=BATCH_SIZE):

    train_files = discover_scene_files(TRAIN_HSI_DIR)
    val_files = discover_scene_files(VAL_HSI_DIR)

    print()
    print("=" * 70)
    print("LOCKED HSI SPLIT")
    print("=" * 70)

    print("\nTRAIN:")
    for f in train_files:
        print(f"  {scene_name(f)}")

    print("\nVALIDATION:")
    for f in val_files:
        print(f"  {scene_name(f)}")

    print("=" * 70)

    # ------------------------------------------------------------
    # FIT PCA ONLY ON TRAIN
    # ------------------------------------------------------------

    train_images = [load_hsi_scene(path)[0] for path in train_files]

    processor.fit(train_images)
    del train_images

    processor.save(SPECTRAL_PROCESSOR_PATH)

    # ------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------

    train_ds = HOSDDataset(
        train_files, processor, patch_size=PATCH_SIZE,
        stride=TRAIN_STRIDE, augment=True,
    )

    val_ds = HOSDDataset(
        val_files, processor, patch_size=PATCH_SIZE,
        stride=VAL_STRIDE, augment=False,
    )

    # ------------------------------------------------------------
    # CLASS BALANCING -- CORRECTED
    # ------------------------------------------------------------

    target_fraction = globals().get("TARGET_OIL_FRACTION", 0.20)

    sample_weights, counts = compute_target_fraction_weights(
        train_ds.labels, target_fraction
    )

    print()
    print(f"Training water samples: {counts[0]:,}")
    print(f"Training oil samples:   {counts[1]:,}")
    print(f"Natural oil ratio:      {counts[1] / max(counts.sum(), 1) * 100:.3f}%")
    print(f"Sampler target ratio:   {target_fraction * 100:.1f}%  (was ~50% implicit)")

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    print()
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples:   {len(val_ds):,}")

    return train_loader, val_loader, train_files, val_files


# ================================================================
# SELF TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HSI DATASET SELF TEST (CORRECTED)")
    print("=" * 70)

    train_files = discover_scene_files(TRAIN_HSI_DIR)
    print(f"Found {len(train_files)} training scenes.")

    processor = SpectralProcessor(
        n_components=PCA_COMPONENTS, bands_to_remove=BANDS_TO_REMOVE
    )

    images = []
    for path in train_files[:1]:
        img, gt = load_hsi_scene(path)
        print(f"Raw image: {img.shape}")
        print(f"GT: {gt.shape}")
        images.append(img)

    processor.fit(images)

    transformed = processor.transform(images[0])
    print(f"PCA image: {transformed.shape}")
    assert transformed.shape[2] == PCA_COMPONENTS

    ds = HOSDDataset(
        train_files[:1], processor, patch_size=PATCH_SIZE, stride=20, augment=False
    )

    x, y = ds[0]
    print(f"Patch: {x.shape}")
    print(f"Label: {y.item()}")

    assert x.shape == (1, PCA_COMPONENTS, PATCH_SIZE, PATCH_SIZE)

    # Quick check of the new weighting function
    weights, counts = compute_target_fraction_weights(ds.labels, 0.20)
    print(f"Sample weight check -- unique weight values: {np.unique(weights)}")

    print()
    print("[SUCCESS] Dataset self-test passed.")