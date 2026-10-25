"""
STAGE 3 — REGISTRATION-FREE SAR + HSI FUSION
02_fusion_dataset.py

Purpose
-------
Construct the dataset used by the fusion training stage.

Fusion contract
---------------
SAR and HSI are scene-paired but pixel-level registration is NOT assumed.

Therefore this dataset:

    ALLOWED
    -------
    - Pair SAR and HSI samples by scene ID.
    - Extract SAR patches from the SAR image.
    - Extract HSI spectral patches from the HSI cube.
    - Use scene-level descriptors from either modality.
    - Use independent modality-local coordinates.

    FORBIDDEN
    ---------
    - Treat SAR pixel (x, y) as physically identical to HSI pixel (x, y).
    - Concatenate raw SAR and HSI pixels.
    - Require equal image shapes.
    - Resample one modality merely to create fake pixel correspondence.

The dataset returns independent modality observations belonging to the same
scene. Later fusion learns scene-conditioned reliability/correction without
claiming exact cross-modal pixel correspondence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.io
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with: pip install scipy"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required. Install it with: pip install pillow"
    ) from exc


# =============================================================================
# PATH SETUP
# =============================================================================

THIS_FILE = Path(__file__).resolve()
FUSION_DIR = THIS_FILE.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))


# =============================================================================
# DYNAMIC CONFIG IMPORT
# =============================================================================

from importlib.util import module_from_spec, spec_from_file_location


def load_py_module(name: str, path: Path):
    spec = spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module '{name}' from:\n{path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


CFG = load_py_module(
    "fusion_config",
    FUSION_DIR / "00_config.py",
)


# =============================================================================
# CONSTANTS
# =============================================================================

MANIFEST_PATH = (
    FUSION_DIR
    / "dataset_audit"
    / "osd_manifest.json"
)

LOCKED_SPLITS = {
    "train": [
        "GM01",
        "GM02",
        "GM06",
        "GM07",
        "GM08",
        "GM12",
        "GM13",
        "GM14",
        "GM18",
    ],
    "validation": [
        "GM03",
        "GM09",
        "GM15",
    ],
    "test": [
        "GM04",
        "GM05",
        "GM10",
        "GM11",
        "GM16",
        "GM17",
    ],
}

SPLIT_TO_SAR_ROOT = {
    "train": Path(CFG.SAR_TRAIN_ROOT),
    "validation": Path(CFG.SAR_VAL_ROOT),
    "test": Path(CFG.SAR_TEST_ROOT),
}

DEFAULT_PATCH_SIZE = 5

if hasattr(CFG, "HSI_PATCH_SIZE"):
    DEFAULT_PATCH_SIZE = int(CFG.HSI_PATCH_SIZE)

DEFAULT_BATCH_SIZE = 8

if hasattr(CFG, "BATCH_SIZE"):
    DEFAULT_BATCH_SIZE = int(CFG.BATCH_SIZE)


# =============================================================================
# UTILITY
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def normalize_split_name(split: str) -> str:
    split = str(split).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }

    if split not in aliases:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected train / validation / test."
        )

    return aliases[split]


def find_scene_record(
    manifest: Dict[str, Any],
    scene_id: str,
) -> Dict[str, Any]:

    scenes = manifest.get("scenes", [])

    for record in scenes:

        if record.get("scene_id") == scene_id:
            return record

    raise KeyError(
        f"Scene '{scene_id}' not found in manifest."
    )


# =============================================================================
# MANIFEST
# =============================================================================

def load_manifest() -> Dict[str, Any]:

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Verified OSD manifest not found:\n"
            f"{MANIFEST_PATH}\n\n"
            f"Run:\n"
            f"    fusion_layer/01_explore_osd.py"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    if "scenes" not in manifest:

        raise KeyError(
            "Manifest does not contain a 'scenes' key."
        )

    return manifest


# =============================================================================
# HSI / GT LOADING
# =============================================================================

def load_mat_scene(
    record: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:

    hsi_info = record.get("hsi")
    gt_info = record.get("gt")

    if hsi_info is None:

        raise KeyError(
            f"Scene {record.get('scene_id')} has no 'hsi' entry."
        )

    if gt_info is None:

        raise KeyError(
            f"Scene {record.get('scene_id')} has no 'gt' entry."
        )

    hsi_path_value = hsi_info.get("path")

    if hsi_path_value is None:

        raise KeyError(
            f"Scene {record.get('scene_id')} has no HSI path."
        )

    hsi_path = Path(hsi_path_value)

    if not hsi_path.exists():

        raise FileNotFoundError(
            f"HSI MAT file does not exist:\n{hsi_path}"
        )

    hsi_variable = hsi_info.get(
        "variable",
        "img",
    )

    gt_variable = gt_info.get(
        "variable",
        "map",
    )

    mat = scipy.io.loadmat(hsi_path)

    if hsi_variable not in mat:

        raise KeyError(
            f"HSI variable '{hsi_variable}' "
            f"not found in:\n{hsi_path}\n\n"
            f"Available keys:\n"
            f"{sorted(k for k in mat.keys() if not k.startswith('__'))}"
        )

    if gt_variable not in mat:

        raise KeyError(
            f"GT variable '{gt_variable}' "
            f"not found in:\n{hsi_path}\n\n"
            f"Available keys:\n"
            f"{sorted(k for k in mat.keys() if not k.startswith('__'))}"
        )

    # Keep the stored integer representation compact; individual patches are
    # converted to float32 when returned to the training loader.
    hsi_cube = np.asarray(mat[hsi_variable])

    gt_mask = np.asarray(
        mat[gt_variable],
    )

    gt_mask = (
        gt_mask > 0
    ).astype(np.uint8)

    if hsi_cube.ndim != 3:

        raise RuntimeError(
            f"Expected HSI cube with 3 dimensions.\n"
            f"Got shape: {hsi_cube.shape}"
        )

    if gt_mask.ndim != 2:

        raise RuntimeError(
            f"Expected GT mask with 2 dimensions.\n"
            f"Got shape: {gt_mask.shape}"
        )

    h, w = gt_mask.shape

    if hsi_cube.shape[:2] != (h, w):

        raise RuntimeError(
            f"HSI/GT spatial mismatch for "
            f"{record.get('scene_id')}.\n"
            f"HSI: {hsi_cube.shape}\n"
            f"GT : {gt_mask.shape}"
        )

    return hsi_cube, gt_mask


# =============================================================================
# SAR LOADING
# =============================================================================

def resolve_sar_path(
    split: str,
    scene_id: str,
) -> Path:

    split = normalize_split_name(split)

    root = SPLIT_TO_SAR_ROOT[split]

    if not root.exists():

        raise FileNotFoundError(
            f"SAR root does not exist for split '{split}':\n"
            f"{root}\n\n"
            f"Check SAR_TRAIN_ROOT / SAR_VAL_ROOT / "
            f"SAR_TEST_ROOT in 00_config.py"
        )

    extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    ]

    matches: List[Path] = []

    for extension in extensions:

        candidate = root / f"{scene_id}{extension}"

        if candidate.exists():

            matches.append(candidate)

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple SAR files found for {scene_id}:\n"
            + "\n".join(str(p) for p in matches)
        )

    recursive_matches: List[Path] = []

    for extension in extensions:

        recursive_matches.extend(
            root.rglob(
                f"{scene_id}{extension}"
            )
        )

    recursive_matches = list(
        dict.fromkeys(recursive_matches)
    )

    if len(recursive_matches) == 1:

        return recursive_matches[0]

    if len(recursive_matches) > 1:

        raise RuntimeError(
            f"Ambiguous SAR file for {scene_id}.\n"
            + "\n".join(
                str(p)
                for p in recursive_matches
            )
        )

    raise FileNotFoundError(
        f"No SAR image found for {scene_id} "
        f"under:\n{root}"
    )


def load_sar_image(
    path: Path,
) -> np.ndarray:

    with Image.open(path) as image:

        image = image.convert("L")

        array = np.asarray(
            image,
            dtype=np.float32,
        )

    if array.ndim != 2:

        raise RuntimeError(
            f"SAR image must be 2D after grayscale conversion.\n"
            f"Path : {path}\n"
            f"Shape: {array.shape}"
        )

    return array


# =============================================================================
# PATCH EXTRACTION
# =============================================================================

def _reflect_index_1d(
    indices: np.ndarray,
    length: int,
) -> np.ndarray:
    """
    Map arbitrary 1D indices onto the valid range [0, length-1] using
    NumPy's 'reflect' boundary behavior without allocating a padded array.
    """

    if length <= 0:
        raise ValueError(
            f"length must be positive, got {length}"
        )

    if length == 1:
        return np.zeros_like(
            indices,
            dtype=np.int64,
        )

    last = length - 1
    mapped = np.mod(
        indices,
        2 * last,
    )

    mapped = np.where(
        mapped > last,
        2 * last - mapped,
        mapped,
    )

    return mapped.astype(
        np.int64,
        copy=False,
    )


def extract_hsi_patch(
    cube: np.ndarray,
    y: int,
    x: int,
    patch_size: int,
) -> np.ndarray:

    if cube.ndim != 3:
        raise ValueError(
            f"Expected HSI cube with 3 dims, got {cube.shape}"
        )

    height, width, _ = cube.shape

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            f"patch_size must be a positive odd integer, got {patch_size}"
        )

    radius = patch_size // 2

    y_indices = _reflect_index_1d(
        np.arange(
            y - radius,
            y + radius + 1,
            dtype=np.int64,
        ),
        height,
    )

    x_indices = _reflect_index_1d(
        np.arange(
            x - radius,
            x + radius + 1,
            dtype=np.int64,
        ),
        width,
    )

    return cube[
        np.ix_(y_indices, x_indices)
    ].copy()


def extract_sar_patch(
    image: np.ndarray,
    y: int,
    x: int,
    patch_size: int,
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError(
            f"Expected SAR image with 2 dims, got {image.shape}"
        )

    height, width = image.shape

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            f"patch_size must be a positive odd integer, got {patch_size}"
        )

    radius = patch_size // 2

    y_indices = _reflect_index_1d(
        np.arange(
            y - radius,
            y + radius + 1,
            dtype=np.int64,
        ),
        height,
    )

    x_indices = _reflect_index_1d(
        np.arange(
            x - radius,
            x + radius + 1,
            dtype=np.int64,
        ),
        width,
    )

    return image[
        np.ix_(y_indices, x_indices)
    ].copy()


# =============================================================================
# SCENE DATASET
# =============================================================================

class FusionSceneDataset(Dataset):
    """
    Dataset returning one complete paired scene.

    Important:
        The SAR image and HSI cube are paired ONLY by scene ID.

        Their pixel coordinates are not claimed to correspond physically.
    """

    def __init__(
        self,
        split: str,
    ) -> None:

        self.split = normalize_split_name(split)

        self.manifest = load_manifest()

        self.scene_ids = LOCKED_SPLITS[
            self.split
        ]

        self.records = []
        self._scene_cache: Dict[int, Dict[str, Any]] = {}

        banner(
            f"VALIDATING FUSION SCENES "
            f"({self.split.upper()})"
        )

        for scene_id in self.scene_ids:

            record = find_scene_record(
                self.manifest,
                scene_id,
            )

            sar_path = resolve_sar_path(
                self.split,
                scene_id,
            )

            hsi_path = Path(
                record["hsi"]["path"]
            )

            if not hsi_path.exists():

                raise FileNotFoundError(
                    f"Missing HSI scene:\n"
                    f"{hsi_path}"
                )

            self.records.append(
                {
                    "scene_id": scene_id,
                    "record": record,
                    "sar_path": sar_path,
                }
            )

            print(
                f"[OK] {scene_id}"
            )
            print(
                f"     SAR: {sar_path}"
            )
            print(
                f"     HSI: {hsi_path}"
            )

        print()
        print(
            f"[SUCCESS] {len(self.records)} "
            f"{self.split} scenes verified."
        )

    def __len__(self) -> int:

        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        if index in self._scene_cache:
            return self._scene_cache[index]

        item = self.records[index]

        scene_id = item["scene_id"]

        hsi_cube, gt_mask = load_mat_scene(
            item["record"]
        )

        sar_image = load_sar_image(
            item["sar_path"]
        )

        scene = {
            "scene_id": scene_id,
            "sar_image": torch.from_numpy(
                sar_image.copy()
            ).float(),

            "hsi_cube": torch.from_numpy(hsi_cube.copy()),

            "gt_mask": torch.from_numpy(
                gt_mask.copy()
            ).long(),

            "sar_shape": torch.tensor(
                sar_image.shape,
                dtype=torch.long,
            ),

            "hsi_shape": torch.tensor(
                hsi_cube.shape[:2],
                dtype=torch.long,
            ),
        }
        self._scene_cache[index] = scene
        return scene


# =============================================================================
# PATCH DATASET
# =============================================================================

class FusionPatchDataset(Dataset):
    """
    Registration-free patch dataset.

    A sample contains:

        1. An HSI local patch and its label.

        2. A SAR local patch sampled independently from
           the same scene.

    The SAR coordinate is NOT assumed to equal the HSI coordinate
    in physical space.

    The default pairing strategy is scene-level normalized sampling:

        HSI coordinate:
            (y_hsi, x_hsi)

        SAR coordinate:
            sampled using normalized relative location

    This preserves broad scene position while explicitly NOT claiming
    exact pixel correspondence.

    The fusion model must therefore treat the SAR patch as contextual
    scene evidence, not as an aligned pixel observation.
    """

    def __init__(
        self,
        split: str,
        samples_per_scene: int = 5000,
        patch_size: int = DEFAULT_PATCH_SIZE,
        seed: int = 42,
    ) -> None:

        self.split = normalize_split_name(split)

        self.samples_per_scene = int(
            samples_per_scene
        )

        self.patch_size = int(
            patch_size
        )

        if self.patch_size % 2 == 0:

            raise ValueError(
                "patch_size must be odd."
            )

        self.seed = int(seed)

        self.scene_dataset = FusionSceneDataset(
            self.split
        )

        self.samples: List[
            Tuple[int, int, int, int, int]
        ] = []

        self._build_sample_index()

    def _build_sample_index(
        self,
    ) -> None:

        banner(
            f"BUILDING FUSION SAMPLE INDEX "
            f"({self.split.upper()})"
        )

        rng = np.random.default_rng(
            self.seed
        )

        for scene_index in range(
            len(self.scene_dataset)
        ):

            scene = self.scene_dataset[
                scene_index
            ]

            gt = (
                scene["gt_mask"]
                .cpu()
                .numpy()
            )

            sar_shape = tuple(
                scene["sar_shape"]
                .cpu()
                .numpy()
            )

            hsi_h, hsi_w = gt.shape
            sar_h, sar_w = sar_shape

            positive_coords = np.argwhere(
                gt == 1
            )

            negative_coords = np.argwhere(
                gt == 0
            )

            if len(positive_coords) == 0:

                raise RuntimeError(
                    f"No positive pixels found in "
                    f"{scene['scene_id']}"
                )

            if len(negative_coords) == 0:

                raise RuntimeError(
                    f"No negative pixels found in "
                    f"{scene['scene_id']}"
                )

            n_positive = (
                self.samples_per_scene // 2
            )

            n_negative = (
                self.samples_per_scene -
                n_positive
            )

            pos_indices = rng.choice(
                len(positive_coords),
                size=min(
                    n_positive,
                    len(positive_coords),
                ),
                replace=(
                    len(positive_coords)
                    < n_positive
                ),
            )

            neg_indices = rng.choice(
                len(negative_coords),
                size=min(
                    n_negative,
                    len(negative_coords),
                ),
                replace=(
                    len(negative_coords)
                    < n_negative
                ),
            )

            chosen_coords = np.concatenate(
                [
                    positive_coords[
                        pos_indices
                    ],
                    negative_coords[
                        neg_indices
                    ],
                ],
                axis=0,
            )

            for y_hsi, x_hsi in chosen_coords:

                # -------------------------------------------------------------
                # IMPORTANT:
                #
                # This is NOT pixel registration.
                #
                # We only use normalized scene-relative location to select
                # a SAR context patch from the same scene.
                #
                # The model must not interpret this as exact same-pixel
                # physical correspondence.
                # -------------------------------------------------------------

                y_ratio = (
                    float(y_hsi)
                    / max(hsi_h - 1, 1)
                )

                x_ratio = (
                    float(x_hsi)
                    / max(hsi_w - 1, 1)
                )

                y_sar = int(
                    round(
                        y_ratio
                        * max(sar_h - 1, 1)
                    )
                )

                x_sar = int(
                    round(
                        x_ratio
                        * max(sar_w - 1, 1)
                    )
                )

                y_sar = min(
                    max(y_sar, 0),
                    sar_h - 1,
                )

                x_sar = min(
                    max(x_sar, 0),
                    sar_w - 1,
                )

                self.samples.append(
                    (
                        scene_index,
                        int(y_hsi),
                        int(x_hsi),
                        int(y_sar),
                        int(x_sar),
                    )
                )

            print(
                f"[OK] "
                f"{scene['scene_id']} | "
                f"{len(chosen_coords)} samples | "
                f"HSI={hsi_h}x{hsi_w} | "
                f"SAR={sar_h}x{sar_w}"
            )

        print()
        print(
            f"Total samples: "
            f"{len(self.samples):,}"
        )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        (
            scene_index,
            y_hsi,
            x_hsi,
            y_sar,
            x_sar,
        ) = self.samples[index]

        scene = self.scene_dataset[
            scene_index
        ]

        hsi_cube = (
            scene["hsi_cube"]
            .cpu()
            .numpy()
        )

        sar_image = (
            scene["sar_image"]
            .cpu()
            .numpy()
        )

        gt_mask = (
            scene["gt_mask"]
            .cpu()
            .numpy()
        )

        hsi_patch = extract_hsi_patch(
            hsi_cube,
            y_hsi,
            x_hsi,
            self.patch_size,
        )

        sar_patch = extract_sar_patch(
            sar_image,
            y_sar,
            x_sar,
            self.patch_size,
        )

        label = int(
            gt_mask[
                y_hsi,
                x_hsi
            ]
        )

        return {
            "scene_index": torch.tensor(
                scene_index,
                dtype=torch.long,
            ),

            "hsi_patch": torch.from_numpy(
                hsi_patch.copy()
            ).float(),

            "sar_patch": torch.from_numpy(
                sar_patch.copy()
            ).float(),

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# =============================================================================
# DATALOADER
# =============================================================================

def build_dataloader(
    split: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    samples_per_scene: int = 5000,
    patch_size: int = DEFAULT_PATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[
    FusionPatchDataset,
    DataLoader,
]:

    split = normalize_split_name(split)

    if shuffle is None:

        shuffle = (
            split == "train"
        )

    dataset = FusionPatchDataset(
        split=split,
        samples_per_scene=samples_per_scene,
        patch_size=patch_size,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataset, loader


# =============================================================================
# SELF TEST
# =============================================================================

def self_test() -> None:

    banner(
        "STAGE 3 — FUSION DATASET SELF TEST"
    )

    print()
    print(
        "Registration contract:"
    )

    print(
        "  SAR/HSI pixel registration assumed : NO"
    )

    print(
        "  Pixel-wise SAR+HSI concatenation  : FORBIDDEN"
    )

    print(
        "  Scene-aware pairing                : ALLOWED"
    )

    print(
        "  HSI local spectral patches         : ALLOWED"
    )

    print(
        "  SAR local spatial patches          : ALLOWED"
    )

    print()

    print(
        "SAR roots:"
    )

    for split, root in (
        SPLIT_TO_SAR_ROOT.items()
    ):

        status = (
            "OK"
            if root.exists()
            else "MISSING"
        )

        print(
            f"  [{status}] "
            f"{split:10s}: {root}"
        )

        if not root.exists():

            raise FileNotFoundError(
                f"SAR root missing:\n{root}"
            )

    print()

    print(
        "Building fusion dataset (train)..."
    )

    train_dataset, train_loader = (
        build_dataloader(
            split="train",
            batch_size=4,
            samples_per_scene=32,
            patch_size=DEFAULT_PATCH_SIZE,
            shuffle=True,
            num_workers=0,
            seed=42,
        )
    )

    print()
    print(
        f"Train dataset samples: "
        f"{len(train_dataset)}"
    )

    batch = next(
        iter(train_loader)
    )

    banner(
        "BATCH INSPECTION"
    )

    print(
        "HSI patch shape :",
        tuple(
            batch[
                "hsi_patch"
            ].shape
        ),
    )

    print(
        "SAR patch shape :",
        tuple(
            batch[
                "sar_patch"
            ].shape
        ),
    )

    print(
        "Label shape     :",
        tuple(
            batch[
                "label"
            ].shape
        ),
    )

    print(
        "Labels          :",
        batch[
            "label"
        ].tolist(),
    )

    expected_hsi_shape = (
        DEFAULT_PATCH_SIZE,
        DEFAULT_PATCH_SIZE,
        224,
    )

    if tuple(
        batch[
            "hsi_patch"
        ].shape[1:]
    ) != expected_hsi_shape:

        raise AssertionError(
            f"Unexpected HSI patch shape.\n"
            f"Expected: {expected_hsi_shape}\n"
            f"Got: {tuple(batch['hsi_patch'].shape[1:])}"
        )

    expected_sar_shape = (
        DEFAULT_PATCH_SIZE,
        DEFAULT_PATCH_SIZE,
    )

    if tuple(
        batch[
            "sar_patch"
        ].shape[1:]
    ) != expected_sar_shape:

        raise AssertionError(
            f"Unexpected SAR patch shape.\n"
            f"Expected: {expected_sar_shape}\n"
            f"Got: {tuple(batch['sar_patch'].shape[1:])}"
        )

    banner(
        "FUSION DATASET SELF TEST PASSED"
    )

    print(
        "[SUCCESS] Registration-free dataset is valid."
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "03_load_experts.py will load and freeze "
        "the existing SAR and HSI experts."
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    self_test()