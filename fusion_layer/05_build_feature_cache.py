"""
STAGE 6 — BUILD FROZEN EXPERT FEATURE CACHE
============================================

Purpose
-------
Run the frozen SAR and HSI experts ONCE and save the outputs.

After this script completes, fusion training must NOT repeatedly run:

    - SAR preprocessing
    - CBDNetSAR
    - HSI spectral processor
    - HMOSDNet

on every epoch.

Instead:

    registration-free dataset
                |
                +----------------------+
                |                      |
                v                      v
          frozen SAR expert      frozen HSI expert
                |                      |
                v                      v
           SAR features          HSI features
                |                      |
                +----------+-----------+
                           |
                           v
                    .npz feature cache
                           |
                           v
                   FAST fusion training


REGISTRATION CONTRACT
---------------------
SAR and HSI pixels are NOT assumed to be registered.

The existing FusionPatchDataset determines:

    HSI:
        (y_hsi, x_hsi)

    SAR:
        (y_sar, x_sar)

The coordinates are independently sampled according to the registration-
free dataset design.

The label is determined ONLY from:

    gt_mask[y_hsi, x_hsi]

SAR is used only as scene-level complementary evidence.


CACHE CONTENT
-------------

sar_features:
    (N, 6)

    Feature order:

        0 -> center probability
        1 -> local mean probability
        2 -> local standard deviation
        3 -> local minimum probability
        4 -> local maximum probability
        5 -> local contrast = max - min

IMPORTANT:

    sar_features[:, 0]

is the probability anchor used by:

    RegistrationFreeFusionNet


hsi_features:
    (N, 64)

hsi_logits:
    (N, 1)

hsi_confidence:
    (N, 1)

labels:
    (N,)

scene_indices:
    (N,)


OUTPUT
------

fusion_layer/cache_features/

    train_features.npz
    validation_features.npz
    test_features.npz


IMPORTANT PERFORMANCE DESIGN
----------------------------

SAR:

    CBDNetSAR is run ONCE PER SCENE.

    The complete SAR image is converted into the same 3-channel 256x256
    representation expected by the SAR expert.

    The resulting segmentation probability map is cached in memory while
    building each split.

HSI:

    HMOSDNet is run once for every HSI patch.

    Because the HSI expert is frozen, all outputs are permanently cached.

After this script:

    05_train.py only reads cached arrays.

"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA


# =============================================================================
# PATH SETUP
# =============================================================================

THIS_FILE = Path(__file__).resolve()
FUSION_DIR = THIS_FILE.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))


from importlib.util import module_from_spec, spec_from_file_location


def load_python_module(
    module_name: str,
    module_path: Path,
):
    """
    Import a Python module from an explicit path.
    """

    spec = spec_from_file_location(
        module_name,
        str(module_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"\nCould not import module:\n{module_path}"
        )

    module = module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# =============================================================================
# LOAD PROJECT MODULES
# =============================================================================

CFG = load_python_module(
    "fusion_cache_config",
    FUSION_DIR / "00_config.py",
)

DATASET_MODULE = load_python_module(
    "fusion_cache_dataset",
    FUSION_DIR / "02_fusion_dataset.py",
)

EXPERTS = load_python_module(
    "fusion_cache_experts",
    FUSION_DIR / "03_load_experts.py",
)


# =============================================================================
# CONFIG HELPER
# =============================================================================

def get_cfg(
    names: List[str],
    default: Any = None,
):

    for name in names:

        if hasattr(CFG, name):
            return getattr(CFG, name)

    return default


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CACHE_DIR = Path(
    get_cfg(
        [
            "FUSION_FEATURE_CACHE_DIR",
        ],
        default=FUSION_DIR / "cache_features",
    )
).resolve()

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


HSI_BATCH_SIZE = int(
    get_cfg(
        [
            "FUSION_CACHE_BATCH_SIZE",
            "HSI_CACHE_BATCH_SIZE",
        ],
        default=512,
    )
)


EXPECTED_SAR_SCALAR_DIM = 10

EXPECTED_SAR_EMBEDDING_DIM = 32

EXPECTED_SAR_FEATURE_DIM = (
    EXPECTED_SAR_SCALAR_DIM
    + EXPECTED_SAR_EMBEDDING_DIM
)

EXPECTED_HSI_FEATURE_DIM = 64

EXPECTED_HSI_BANDS = 25

SAR_MODEL_SIZE = 256

SAR_WINDOW_SIZE = 5


# =============================================================================
# PRINT HELPER
# =============================================================================

def banner(
    title: str,
):

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# =============================================================================
# BUILD DATASET
# =============================================================================

def build_dataset(
    split: str,
):

    """
    Construct the current FusionPatchDataset.

    The exact arguments here match the dataset implementation used earlier.
    """

    if not hasattr(
        DATASET_MODULE,
        "FusionPatchDataset",
    ):

        raise RuntimeError(
            "\n02_fusion_dataset.py does not contain FusionPatchDataset."
        )

    return DATASET_MODULE.FusionPatchDataset(
        split=split,
        samples_per_scene=int(
            get_cfg(
                [
                    "SAMPLES_PER_SCENE",
                    "FUSION_SAMPLES_PER_SCENE",
                ],
                default=5000,
            )
        ),
        patch_size=int(
            get_cfg(
                [
                    "PATCH_SIZE",
                    "FUSION_PATCH_SIZE",
                ],
                default=5,
            )
        ),
        seed=int(
            get_cfg(
                [
                    "SEED",
                    "RANDOM_SEED",
                ],
                default=42,
            )
        ),
    )


# =============================================================================
# HSI PREPROCESSING
# =============================================================================

def preprocess_hsi_batch(
    hsi_patches: np.ndarray,
    processor: Any,
) -> torch.Tensor:
    """
    Convert raw HSI patches:

        (B, 5, 5, original_bands)

    into HMOSDNet input:

        (B, 1, 25, 5, 5)
    """

    if hsi_patches.ndim != 4:

        raise ValueError(
            "\nExpected HSI batch with shape:\n"
            "    (B, H, W, Bands)\n"
            f"Got:\n    {hsi_patches.shape}"
        )

    batch_size, height, width, _ = hsi_patches.shape

    if height != 5 or width != 5:

        raise RuntimeError(
            "\nHSI patch size mismatch.\n"
            f"Expected 5x5.\n"
            f"Got {height}x{width}"
        )

    # -------------------------------------------------------------
    # Apply the exact saved spectral processor independently to
    # every patch.
    # -------------------------------------------------------------

    transformed = np.stack(
        [
            processor.transform(
                patch
            )
            for patch in hsi_patches
        ],
        axis=0,
    )

    transformed = np.asarray(
        transformed,
        dtype=np.float32,
    )

    if transformed.ndim != 4:

        raise RuntimeError(
            "\nUnexpected spectral processor output.\n"
            f"Got:\n    {transformed.shape}"
        )

    if transformed.shape[-1] != EXPECTED_HSI_BANDS:

        raise RuntimeError(
            "\nHSI PCA dimension mismatch.\n"
            f"Expected: {EXPECTED_HSI_BANDS}\n"
            f"Got: {transformed.shape[-1]}"
        )

    # (B, H, W, 25)
    #       ->
    # (B, 25, H, W)

    transformed = np.transpose(
        transformed,
        (0, 3, 1, 2),
    )

    transformed = np.ascontiguousarray(
        transformed
    )

    tensor = torch.from_numpy(
        transformed
    )

    # (B,25,5,5)
    #       ->
    # (B,1,25,5,5)

    tensor = tensor.unsqueeze(
        1
    ).float()

    expected_shape = (
        batch_size,
        1,
        EXPECTED_HSI_BANDS,
        5,
        5,
    )

    if tuple(tensor.shape) != expected_shape:

        raise RuntimeError(
            "\nIncorrect HSI input shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got: {tuple(tensor.shape)}"
        )

    return tensor


# =============================================================================
# HSI FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def run_hsi_expert(
    hsi_model,
    hsi_input: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    HMOSDNet:

        input:
            (B, 1, 25, 5, 5)

        output:
            logits
            features
            None
            confidence

    Current model features:

        (B, 64, 25)

    Fusion expects:

        (B, 64)

    Therefore:

        mean(features, dim=2)
    """

    hsi_model.eval()

    logits, features, _, confidence = hsi_model(
        hsi_input,
        return_features=True,
    )

    if features.ndim == 3:

        features = features.mean(
            dim=-1
        )

    elif features.ndim != 2:

        raise RuntimeError(
            "\nUnexpected HSI feature shape:\n"
            f"{tuple(features.shape)}"
        )

    if features.shape[1] != EXPECTED_HSI_FEATURE_DIM:

        raise RuntimeError(
            "\nHSI feature dimension mismatch.\n"
            f"Expected: {EXPECTED_HSI_FEATURE_DIM}\n"
            f"Got: {features.shape[1]}"
        )

    logits = logits.reshape(
        logits.shape[0],
        1,
    )

    confidence = confidence.reshape(
        confidence.shape[0],
        1,
    )

    return (
        features.float(),
        logits.float(),
        confidence.float(),
    )


# =============================================================================
# SAR PREPROCESSING
# =============================================================================

def prepare_sar_scene(
    sar_image: np.ndarray,
) -> torch.Tensor:
    """
    Prepare the complete SAR scene for CBDNetSAR.

    Input:

        (H, W)

    Output:

        (1, 3, 256, 256)

    IMPORTANT
    ---------

    This function uses scene-level normalization.

    The SAR expert expects:

        3-channel
        256 x 256

    The same single-channel SAR representation is copied into
    three channels.
    """

    sar = np.asarray(
        sar_image,
        dtype=np.float32,
    )

    if sar.ndim != 2:

        raise ValueError(
            "\nExpected SAR image shape:\n"
            "(H, W)\n"
            f"Got:\n{sar.shape}"
        )

    # -------------------------------------------------------------
    # Numerical safety
    # -------------------------------------------------------------

    sar = np.nan_to_num(
        sar,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # -------------------------------------------------------------
    # Log compression.
    #
    # SAR values may span a large dynamic range.
    # -------------------------------------------------------------

    sar = np.log1p(
        np.abs(sar)
    )

    # -------------------------------------------------------------
    # Scene-level standardization.
    # -------------------------------------------------------------

    mean = float(
        sar.mean()
    )

    std = float(
        sar.std()
    )

    sar = (
        sar - mean
    ) / (
        std + 1e-6
    )

    # -------------------------------------------------------------
    # Convert:

    # (H,W)
    #
    # ->
    #
    # (3,H,W)
    # -------------------------------------------------------------

    sar = np.stack(
        [
            sar,
            sar,
            sar,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    tensor = torch.from_numpy(
        sar
    ).unsqueeze(
        0
    )

    # -------------------------------------------------------------
    # CBDNetSAR input:
    #
    # (B,3,256,256)
    # -------------------------------------------------------------

    tensor = F.interpolate(
        tensor,
        size=(
            SAR_MODEL_SIZE,
            SAR_MODEL_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return tensor


# =============================================================================
# SAR EXPERT
# =============================================================================

@torch.no_grad()
def build_sar_probability_map(
    sar_model,
    sar_image: np.ndarray,
) -> np.ndarray:
    """
    Run CBDNetSAR once for the complete scene.

    CBDNetSAR returns:

        seg_logits,
        boundary_logits,
        encoder_features,
        mask_conf

    We use:

        seg_logits

    and convert:

        logits
        ->
        sigmoid probability map

    Returns:

        (256,256)
    """

    sar_model.eval()

    sar_input = prepare_sar_scene(
        sar_image
    ).to(
        DEVICE,
        non_blocking=True,
    )

    (
        seg_logits,
        _,
        _,
        _,
    ) = sar_model(
        sar_input
    )

    probability = torch.sigmoid(
        seg_logits
    )

    probability = probability[
        0,
        0,
    ]

    return (
        probability
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


@torch.no_grad()
def build_sar_scene_outputs(
    sar_model,
    sar_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the frozen SAR model once and return probability and bottleneck maps."""

    sar_model.eval()

    sar_input = prepare_sar_scene(
        sar_image
    ).to(
        DEVICE,
        non_blocking=True,
    )

    seg_logits, _, encoder_features, _ = sar_model(
        sar_input
    )

    probability = torch.sigmoid(
        seg_logits[0, 0]
    ).detach().cpu().numpy().astype(np.float32)

    bottleneck = encoder_features[0].detach().cpu().numpy().astype(np.float32)

    if bottleneck.ndim != 3 or bottleneck.shape[0] != 512:
        raise RuntimeError(
            "Expected SAR bottleneck shape (512,H,W), got "
            f"{bottleneck.shape}"
        )

    return probability, bottleneck


# =============================================================================
# SAR LOCAL EVIDENCE
# =============================================================================

def extract_sar_evidence(
    probability_map: np.ndarray,
    y_sar: int,
    x_sar: int,
    original_shape: Tuple[int, int],
    window_size: int = SAR_WINDOW_SIZE,
) -> np.ndarray:
    """
    Extract local SAR evidence from the CBDNetSAR probability map.

    The sample coordinate:

        (y_sar, x_sar)

    belongs to the original SAR scene.

    It is mapped into:

        (256,256)

    because CBDNetSAR operates at 256x256.

    Output:

        [center_probability,
         mean_probability,
         std_probability,
         min_probability,
         max_probability,
         contrast]
    """

    original_h, original_w = original_shape

    map_h, map_w = probability_map.shape

    # -------------------------------------------------------------
    # Map original SAR coordinate -> model probability map.
    # -------------------------------------------------------------

    y_map = int(
        round(
            float(y_sar)
            / max(
                original_h - 1,
                1,
            )
            * (
                map_h - 1
            )
        )
    )

    x_map = int(
        round(
            float(x_sar)
            / max(
                original_w - 1,
                1,
            )
            * (
                map_w - 1
            )
        )
    )

    y_map = int(
        np.clip(
            y_map,
            0,
            map_h - 1,
        )
    )

    x_map = int(
        np.clip(
            x_map,
            0,
            map_w - 1,
        )
    )

    radius = window_size // 2

    padded = np.pad(
        probability_map,
        (
            (radius, radius),
            (radius, radius),
        ),
        mode="reflect",
    )

    y_padded = y_map + radius

    x_padded = x_map + radius

    local_patch = padded[
        y_padded - radius:
        y_padded + radius + 1,

        x_padded - radius:
        x_padded + radius + 1,
    ]

    center_probability = float(
        probability_map[
            y_map,
            x_map,
        ]
    )

    mean_probability = float(
        local_patch.mean()
    )

    std_probability = float(
        local_patch.std()
    )

    min_probability = float(
        local_patch.min()
    )

    max_probability = float(
        local_patch.max()
    )

    contrast = (
        max_probability
        -
        min_probability
    )

    features = np.array(
        [
            center_probability,
            mean_probability,
            std_probability,
            min_probability,
            max_probability,
            contrast,
        ],
        dtype=np.float32,
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return features


def fit_sar_projection(
    scene_bottlenecks: Dict[int, np.ndarray],
) -> PCA:
    """Fit a 512-to-32 linear projection using training scenes only."""

    samples = []

    for bottleneck in scene_bottlenecks.values():
        samples.append(
            bottleneck.transpose(1, 2, 0).reshape(-1, bottleneck.shape[0])
        )

    training_features = np.concatenate(
        samples,
        axis=0,
    )

    projector = PCA(
        n_components=EXPECTED_SAR_EMBEDDING_DIM,
        random_state=42,
    )

    projector.fit(training_features)

    print(
        f"[OK] SAR bottleneck projection fitted: "
        f"{training_features.shape[1]} -> "
        f"{EXPECTED_SAR_EMBEDDING_DIM}"
    )

    return projector


def extract_sar_embedding(
    bottleneck: np.ndarray,
    y_map: int,
    x_map: int,
    projector: PCA,
) -> np.ndarray:
    """Project and sample one local SAR bottleneck embedding."""

    channels, height, width = bottleneck.shape

    flattened = bottleneck.transpose(1, 2, 0).reshape(-1, channels)
    projected = projector.transform(flattened).reshape(
        height,
        width,
        EXPECTED_SAR_EMBEDDING_DIM,
    )

    y_feature = int(
        round(
            float(y_map)
            / max(SAR_MODEL_SIZE - 1, 1)
            * max(height - 1, 1)
        )
    )

    x_feature = int(
        round(
            float(x_map)
            / max(SAR_MODEL_SIZE - 1, 1)
            * max(width - 1, 1)
        )
    )

    y_feature = int(np.clip(y_feature, 0, height - 1))
    x_feature = int(np.clip(x_feature, 0, width - 1))

    return projected[
        y_feature,
        x_feature,
    ].astype(np.float32)


def map_sar_coordinate(
    y_sar: int,
    x_sar: int,
    original_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Map an original-scene SAR coordinate onto the 256x256 output grid."""

    original_h, original_w = original_shape

    y_map = int(
        round(
            float(y_sar)
            / max(original_h - 1, 1)
            * (SAR_MODEL_SIZE - 1)
        )
    )

    x_map = int(
        round(
            float(x_sar)
            / max(original_w - 1, 1)
            * (SAR_MODEL_SIZE - 1)
        )
    )

    return (
        int(np.clip(y_map, 0, SAR_MODEL_SIZE - 1)),
        int(np.clip(x_map, 0, SAR_MODEL_SIZE - 1)),
    )


# =============================================================================
# BUILD SAR MAPS ONCE PER SCENE
# =============================================================================

def precompute_sar_scene_maps(
    dataset,
    sar_model,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, Tuple[int, int]],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:
    """
    Run the frozen SAR expert exactly ONCE per scene.
    """

    probability_maps = {}

    sar_shapes = {}

    sar_bottlenecks = {}

    scene_statistics = {}

    num_scenes = len(
        dataset.scene_dataset
    )

    print(
        f"\nRunning CBDNetSAR once per scene ({num_scenes} scenes)..."
    )

    for scene_index in tqdm(
        range(num_scenes),
        desc="SAR scenes",
    ):

        scene = dataset.scene_dataset[
            scene_index
        ]

        sar_image = (
            scene[
                "sar_image"
            ]
            .cpu()
            .numpy()
        )

        probability_map, bottleneck = build_sar_scene_outputs(
            sar_model,
            sar_image,
        )

        probability_maps[scene_index] = probability_map

        sar_bottlenecks[scene_index] = bottleneck

        scene_statistics[scene_index] = np.array(
            [
                float(probability_map.mean()),
                float(probability_map.std()),
                float((probability_map >= 0.5).mean()),
            ],
            dtype=np.float32,
        )

        sar_shapes[
            scene_index
        ] = tuple(
            sar_image.shape
        )

    return (
        probability_maps,
        sar_shapes,
        sar_bottlenecks,
        scene_statistics,
    )


# =============================================================================
# BUILD ONE SPLIT CACHE
# =============================================================================

def build_split_cache(
    split: str,
    sar_model,
    hsi_model,
    processor,
    sar_projection: PCA,
    sar_outputs: Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int]], Dict[int, np.ndarray], Dict[int, np.ndarray]] | None = None,
) -> Path:

    banner(
        f"BUILDING CACHE — {split.upper()}"
    )

    dataset = build_dataset(
        split
    )

    num_samples = len(
        dataset
    )

    print(
        f"\nSamples : {num_samples:,}"
    )

    print(
        f"Scenes  : {len(dataset.scene_dataset)}"
    )

    print(
        f"HSI batch size : {HSI_BATCH_SIZE}"
    )

    # =====================================================================
    # STEP 1
    #
    # RUN SAR EXPERT ONCE PER SCENE
    # =====================================================================

    if sar_outputs is None:
        sar_outputs = precompute_sar_scene_maps(
            dataset,
            sar_model,
        )

    (
        sar_probability_maps,
        sar_shapes,
        sar_bottlenecks,
        scene_statistics,
    ) = sar_outputs

    # =====================================================================
    # STEP 2
    #
    # PREALLOCATE CACHE ARRAYS
    # =====================================================================

    sar_features = np.empty(
        (
            num_samples,
            EXPECTED_SAR_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    hsi_features = np.empty(
        (
            num_samples,
            EXPECTED_HSI_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    hsi_logits = np.empty(
        (
            num_samples,
            1,
        ),
        dtype=np.float32,
    )

    hsi_confidence = np.empty(
        (
            num_samples,
            1,
        ),
        dtype=np.float32,
    )

    labels = np.empty(
        (
            num_samples,
        ),
        dtype=np.int64,
    )

    scene_indices = np.empty(
        (
            num_samples,
        ),
        dtype=np.int64,
    )

    # =====================================================================
    # STEP 3
    #
    # CACHE SAMPLE FEATURES
    # =====================================================================

    print(
        f"\nCaching {num_samples:,} samples..."
    )

    start_time = time.time()

    for batch_start in tqdm(
        range(
            0,
            num_samples,
            HSI_BATCH_SIZE,
        ),
        desc=f"{split} cache",
    ):

        batch_end = min(
            batch_start + HSI_BATCH_SIZE,
            num_samples,
        )

        current_batch_size = (
            batch_end
            -
            batch_start
        )

        hsi_patch_batch = []

        # -------------------------------------------------------------
        # Build HSI batch + SAR evidence.
        # -------------------------------------------------------------

        for i in range(
            batch_start,
            batch_end,
        ):

            (
                scene_index,
                y_hsi,
                x_hsi,
                y_sar,
                x_sar,
            ) = dataset.samples[i]

            scene = dataset.scene_dataset[
                scene_index
            ]

            # ---------------------------------------------------------
            # HSI patch
            # ---------------------------------------------------------

            hsi_cube = (
                scene[
                    "hsi_cube"
                ]
                .cpu()
                .numpy()
            )

            hsi_patch = DATASET_MODULE.extract_hsi_patch(
                hsi_cube,
                y_hsi,
                x_hsi,
                dataset.patch_size,
            )

            hsi_patch_batch.append(
                hsi_patch
            )

            # ---------------------------------------------------------
            # SAR evidence
            #
            # IMPORTANT:
            #
            # Use y_sar/x_sar from registration-free sampling.
            #
            # Do NOT use y_hsi/x_hsi here.
            # ---------------------------------------------------------

            local_sar_features = extract_sar_evidence(
                probability_map=
                    sar_probability_maps[
                        scene_index
                    ],

                y_sar=y_sar,

                x_sar=x_sar,

                original_shape=
                    sar_shapes[
                        scene_index
                    ],
            )

            y_map, x_map = map_sar_coordinate(
                y_sar,
                x_sar,
                sar_shapes[scene_index],
            )

            scene_mean, scene_std, scene_positive_fraction = scene_statistics[
                scene_index
            ]

            sar_embedding = extract_sar_embedding(
                sar_bottlenecks[scene_index],
                y_map,
                x_map,
                sar_projection,
            )

            sar_features[i] = np.concatenate(
                [
                    local_sar_features,
                    np.array(
                        [
                            scene_mean,
                            scene_std,
                            scene_positive_fraction,
                            local_sar_features[0] - scene_mean,
                        ],
                        dtype=np.float32,
                    ),
                    sar_embedding,
                ]
            )

            # ---------------------------------------------------------
            # Label belongs ONLY to HSI coordinate.
            # ---------------------------------------------------------

            gt_mask = (
                scene[
                    "gt_mask"
                ]
                .cpu()
                .numpy()
            )

            labels[i] = int(
                gt_mask[
                    y_hsi,
                    x_hsi,
                ]
            )

            scene_indices[i] = int(
                scene_index
            )

        # -------------------------------------------------------------
        # HSI inference
        # -------------------------------------------------------------

        hsi_patch_batch = np.stack(
            hsi_patch_batch,
            axis=0,
        ).astype(
            np.float32
        )

        hsi_input = preprocess_hsi_batch(
            hsi_patch_batch,
            processor,
        ).to(
            DEVICE,
            non_blocking=True,
        )

        (
            batch_hsi_features,
            batch_hsi_logits,
            batch_hsi_confidence,
        ) = run_hsi_expert(
            hsi_model,
            hsi_input,
        )

        hsi_features[
            batch_start:batch_end
        ] = (
            batch_hsi_features
            .detach()
            .cpu()
            .numpy()
        )

        hsi_logits[
            batch_start:batch_end
        ] = (
            batch_hsi_logits
            .detach()
            .cpu()
            .numpy()
        )

        hsi_confidence[
            batch_start:batch_end
        ] = (
            batch_hsi_confidence
            .detach()
            .cpu()
            .numpy()
        )

        del hsi_input
        del batch_hsi_features
        del batch_hsi_logits
        del batch_hsi_confidence

    elapsed = (
        time.time()
        -
        start_time
    )

    samples_per_second = (
        num_samples
        /
        max(
            elapsed,
            1e-6,
        )
    )

    # =====================================================================
    # VALIDATION
    # =====================================================================

    banner(
        f"VALIDATING {split.upper()} CACHE"
    )

    if sar_features.shape != (
        num_samples,
        EXPECTED_SAR_FEATURE_DIM,
    ):

        raise RuntimeError(
            f"Incorrect SAR cache shape: {sar_features.shape}"
        )

    if hsi_features.shape != (
        num_samples,
        EXPECTED_HSI_FEATURE_DIM,
    ):

        raise RuntimeError(
            f"Incorrect HSI cache shape: {hsi_features.shape}"
        )

    if hsi_logits.shape != (
        num_samples,
        1,
    ):

        raise RuntimeError(
            f"Incorrect HSI logits shape: {hsi_logits.shape}"
        )

    if hsi_confidence.shape != (
        num_samples,
        1,
    ):

        raise RuntimeError(
            f"Incorrect HSI confidence shape: {hsi_confidence.shape}"
        )

    for name, array in [

        (
            "sar_features",
            sar_features,
        ),

        (
            "hsi_features",
            hsi_features,
        ),

        (
            "hsi_logits",
            hsi_logits,
        ),

        (
            "hsi_confidence",
            hsi_confidence,
        ),

    ]:

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    # SAR anchor must be probability.

    anchor = sar_features[
        :,
        0,
    ]

    if (
        anchor.min() < -1e-5
        or
        anchor.max() > 1.00001
    ):

        raise RuntimeError(
            "\nSAR probability anchor is outside [0,1].\n"
            f"Range: [{anchor.min()}, {anchor.max()}]"
        )

    # =====================================================================
    # SAVE CACHE
    # =====================================================================

    output_path = (
        CACHE_DIR
        /
        f"{split}_features.npz"
    )

    np.savez_compressed(
        output_path,

        sar_features=sar_features,

        hsi_features=hsi_features,

        hsi_logits=hsi_logits,

        hsi_confidence=hsi_confidence,

        labels=labels,

        scene_indices=scene_indices,
    )

    # =====================================================================
    # SAVE SUMMARY
    # =====================================================================

    summary = {

        "split":
            split,

        "num_samples":
            int(num_samples),

        "sar_features":
            list(
                sar_features.shape
            ),

        "hsi_features":
            list(
                hsi_features.shape
            ),

        "hsi_logits":
            list(
                hsi_logits.shape
            ),

        "hsi_confidence":
            list(
                hsi_confidence.shape
            ),

        "positive_samples":
            int(
                labels.sum()
            ),

        "positive_fraction":
            float(
                labels.mean()
            ),

        "elapsed_seconds":
            float(
                elapsed
            ),

        "samples_per_second":
            float(
                samples_per_second
            ),

        "cache_path":
            str(
                output_path
            ),
    }

    summary_path = (
        CACHE_DIR
        /
        f"{split}_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # =====================================================================
    # PRINT RESULT
    # =====================================================================

    banner(
        f"{split.upper()} CACHE COMPLETE"
    )

    print(
        f"\nSamples              : {num_samples:,}"
    )

    print(
        f"SAR features         : {sar_features.shape}"
    )

    print(
        f"HSI features         : {hsi_features.shape}"
    )

    print(
        f"HSI logits           : {hsi_logits.shape}"
    )

    print(
        f"HSI confidence       : {hsi_confidence.shape}"
    )

    print(
        f"Positive samples     : {labels.sum():,}"
    )

    print(
        f"Positive fraction    : {labels.mean():.6f}"
    )

    print(
        f"Elapsed              : {elapsed:.2f} seconds"
    )

    print(
        f"Speed                : "
        f"{samples_per_second:.1f} samples/sec"
    )

    print(
        f"\n[OK] Saved:"
    )

    print(
        f"{output_path}"
    )

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "STAGE 6 — BUILD FROZEN EXPERT FEATURE CACHE"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "\nRegistration contract:"
    )

    print(
        "  SAR/HSI pixel registration : NOT REQUIRED"
    )

    print(
        "  HSI coordinates             : independent"
    )

    print(
        "  SAR coordinates             : independent"
    )

    print(
        "  Label source                 : HSI GT coordinate only"
    )

    print(
        "  Frozen SAR recomputation    : ONCE PER SCENE"
    )

    print(
        "  Frozen HSI recomputation    : ONCE PER SAMPLE"
    )

    print(
        "  Expert inference in training: NEVER"
    )

    print(
        "\nCache configuration:"
    )

    print(
        f"  Cache directory : {CACHE_DIR}"
    )

    print(
        f"  HSI batch size  : {HSI_BATCH_SIZE}"
    )

    print(
        f"  SAR model size  : "
        f"{SAR_MODEL_SIZE}x{SAR_MODEL_SIZE}"
    )

    print(
        f"  SAR window      : "
        f"{SAR_WINDOW_SIZE}x{SAR_WINDOW_SIZE}"
    )

    # =====================================================================
    # LOAD FROZEN EXPERTS
    # =====================================================================

    banner(
        "LOADING FROZEN EXPERTS"
    )

    sar_model = EXPERTS.load_sar_expert(
        DEVICE
    )

    hsi_model, processor = EXPERTS.load_hsi_expert(
        DEVICE
    )

    train_dataset = build_dataset(
        "train"
    )

    train_sar_outputs = precompute_sar_scene_maps(
        train_dataset,
        sar_model,
    )

    sar_projection = fit_sar_projection(
        train_sar_outputs[2]
    )

    # =====================================================================
    # BUILD ALL SPLITS
    # =====================================================================

    cache_paths = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        sar_outputs = (
            train_sar_outputs
            if split == "train"
            else None
        )

        cache_paths[
            split
        ] = build_split_cache(

            split=split,

            sar_model=sar_model,

            hsi_model=hsi_model,

            processor=processor,

            sar_projection=sar_projection,

            sar_outputs=sar_outputs,
        )

        gc.collect()

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

    # =====================================================================
    # COMPLETE
    # =====================================================================

    banner(
        "FEATURE CACHE BUILD COMPLETE"
    )

    print(
        "\nGenerated files:"
    )

    for split, path in cache_paths.items():

        print(
            f"  {split:<12} {path}"
        )

    print(
        "\n[SUCCESS] CBDNetSAR was used as the SAR expert."
    )

    print(
        "[SUCCESS] SAR expert was run once per scene."
    )

    print(
        "[SUCCESS] HMOSDNet outputs were cached."
    )

    print(
        "[SUCCESS] HSI PCA/processor will not run every epoch."
    )

    print(
        "[SUCCESS] CBDNetSAR will not run every epoch."
    )

    print(
        "[SUCCESS] Registration-free coordinates were preserved."
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "05_train.py must load only:"
    )

    print(
        "    train_features.npz"
    )

    print(
        "    validation_features.npz"
    )

    print(
        "    test_features.npz"
    )

    print(
        "\n05_train.py must NOT import or execute either expert."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()