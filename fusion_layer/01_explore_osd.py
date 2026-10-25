"""
STAGE 3 — FUSION
01_explore_osd.py

Purpose
-------
Explore and validate the OSD dataset used by the registration-free
SAR + HSI fusion pipeline.

This script DOES NOT:
    - train a model
    - load the SAR expert
    - load the HSI expert
    - assume pixel-level SAR/HSI registration
    - resize or align images
    - create fusion samples

This script DOES:
    - inspect Train / Validation / Test OSD directories
    - discover scenes
    - identify hyperspectral and ground-truth files
    - verify locked scene splits
    - inspect array shapes, dtypes and value ranges
    - detect duplicate scene IDs
    - save a reproducible manifest

The output manifest is later consumed by the fusion dataset loader.

------------------------------------------------------------------------------
FIXES APPLIED TO THIS VERSION (relative to the original draft)
------------------------------------------------------------------------------
1. `MAX_SUMMARY_VALUES` was referenced but never defined -> added as a
   module-level constant. Without it, the script raised NameError on the
   first .mat scene it inspected.

2. `choose_mask_candidate` excluded candidates whose *file path* matched
   the chosen HSI entry's path. Because the real OSD .mat files store the
   HSI cube ("img") and the GT mask ("map") as two variables inside the
   SAME file, this incorrectly discarded the correct mask candidate too.
   Fixed to exclude by entry identity (the specific classified candidate),
   not by file path.

3. `inspect_hsi_array` / `inspect_mask_labels` reloaded arrays generically
   via `load_array(path)`, which just picks "the largest ndim>=2 array in
   the file." For scenes where the cube and mask share a file, this
   silently returned the HSI cube for BOTH calls, crashing mask inspection
   (a 3D cube cannot be squeezed to 2D). Fixed by loading the specific
   named variable that was actually selected during classification.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# IMPORT CONFIG
# =============================================================================

THIS_FILE = Path(__file__).resolve()
FUSION_DIR = THIS_FILE.parent
PROJECT_ROOT = FUSION_DIR.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from importlib.util import module_from_spec, spec_from_file_location


def load_config_module():
    """
    Load 00_config.py safely even though its filename starts with a number.
    """
    config_path = FUSION_DIR / "00_config.py"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing configuration file:\n{config_path}"
        )

    spec = spec_from_file_location("fusion_config", str(config_path))

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load configuration:\n{config_path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


CFG = load_config_module()


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_first_attr(module: Any, names: List[str]) -> Any:
    """
    Return the first existing attribute from a list of possible names.

    This makes the exploration script robust to minor naming differences
    in 00_config.py.
    """
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(
        "Could not find any expected configuration attribute.\n"
        f"Tried: {names}"
    )


TRAIN_ROOT = Path(
    get_first_attr(
        CFG,
        [
            "OSD_TRAIN_ROOT",
            "TRAIN_ROOT",
            "HSI_TRAIN_ROOT",
        ],
    )
).resolve()

VAL_ROOT = Path(
    get_first_attr(
        CFG,
        [
            "OSD_VAL_ROOT",
            "VAL_ROOT",
            "HSI_VAL_ROOT",
            "VALIDATION_ROOT",
        ],
    )
).resolve()

TEST_ROOT = Path(
    get_first_attr(
        CFG,
        [
            "OSD_TEST_ROOT",
            "TEST_ROOT",
            "HSI_TEST_ROOT",
        ],
    )
).resolve()


TRAIN_SCENES = list(
    get_first_attr(
        CFG,
        [
            "TRAIN_SCENES",
            "LOCKED_TRAIN_SCENES",
        ],
    )
)

VAL_SCENES = list(
    get_first_attr(
        CFG,
        [
            "VAL_SCENES",
            "VALIDATION_SCENES",
            "LOCKED_VAL_SCENES",
        ],
    )
)

TEST_SCENES = list(
    get_first_attr(
        CFG,
        [
            "TEST_SCENES",
            "LOCKED_TEST_SCENES",
        ],
    )
)


OUTPUT_DIR = FUSION_DIR / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_JSON = OUTPUT_DIR / "osd_manifest.json"
MANIFEST_CSV = OUTPUT_DIR / "osd_manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "osd_exploration_summary.json"


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_ARRAY_SUFFIXES = {
    ".npy",
    ".npz",
    ".mat",
    ".tif",
    ".tiff",
}

IMAGE_HINTS = (
    "hsi",
    "hyper",
    "cube",
    "image",
    "img",
    "data",
)

MASK_HINTS = (
    "gt",
    "mask",
    "label",
    "labels",
    "groundtruth",
    "ground_truth",
    "truth",
    "annotation",
    "annotations",
)

# FIX (1): this constant was referenced in classify_candidate_shapes() but
# never defined in the original script, causing an immediate NameError on
# the first .mat file processed. It caps how many values are sampled when
# computing min/max/mean summary statistics on large arrays, so exploration
# stays fast even on multi-million-element HSI cubes.
MAX_SUMMARY_VALUES = 2_000_000


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def normalize_scene_id(value: str) -> str:
    """
    Extract canonical GMxx scene ID where possible.

    Examples:
        GM01
        gm01
        GM01_hsi
        sample_GM01

    all normalize to:
        GM01
    """
    text = str(value).upper()

    match = re.search(r"GM\s*0*(\d+)", text)

    if match:
        number = int(match.group(1))
        return f"GM{number:02d}"

    return text.strip()


def is_hidden_or_system(path: Path) -> bool:
    name = path.name

    if name.startswith("."):
        return True

    if name in {
        "__pycache__",
        "Thumbs.db",
        "desktop.ini",
    }:
        return True

    return False


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def list_scene_candidates(root: Path) -> List[Path]:
    """
    Discover directories or direct files that appear to represent OSD scenes.

    The current OSD Copy roots store scenes as direct files such as GM01.mat.

    If scene folders are nested one level deeper, recursive discovery is used.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist:\n{root}")

    candidates: List[Path] = []

    for path in sorted(root.iterdir()):
        if is_hidden_or_system(path):
            continue

        scene_id = normalize_scene_id(
            path.name if path.is_dir() else path.stem
        )

        if re.fullmatch(r"GM\d{2,}", scene_id):
            candidates.append(path)

    if candidates:
        return candidates

    # Fallback: recursive scene-folder search.
    for path in sorted(root.rglob("*")):
        if not path.is_dir() and not path.is_file():
            continue

        if is_hidden_or_system(path):
            continue

        scene_id = normalize_scene_id(
            path.name if path.is_dir() else path.stem
        )

        if re.fullmatch(r"GM\d{2,}", scene_id):
            candidates.append(path)

    return candidates


def collect_array_files(scene_dir: Path) -> List[Path]:
    """
    Return supported array-like files inside a scene directory.
    """
    files: List[Path] = []

    paths = [scene_dir] if scene_dir.is_file() else sorted(scene_dir.rglob("*"))

    for path in paths:
        if not path.is_file():
            continue

        if is_hidden_or_system(path):
            continue

        if path.suffix.lower() in SUPPORTED_ARRAY_SUFFIXES:
            files.append(path)

    return files


def score_image_candidate(path: Path) -> int:
    """
    Score likely hyperspectral-image files from filename semantics.

    Higher score = more likely to be the HSI cube.
    """
    name = path.stem.lower()

    score = 0

    for hint in IMAGE_HINTS:
        if hint in name:
            score += 10

    for hint in MASK_HINTS:
        if hint in name:
            score -= 20

    return score


def score_mask_candidate(path: Path) -> int:
    """
    Score likely ground-truth mask files from filename semantics.
    """
    name = path.stem.lower()

    score = 0

    for hint in MASK_HINTS:
        if hint in name:
            score += 10

    for hint in IMAGE_HINTS:
        if hint in name:
            score -= 10

    return score


# =============================================================================
# ARRAY LOADING
# =============================================================================

def load_numpy_file(path: Path) -> np.ndarray:
    """
    Load .npy or .npz.

    For NPZ:
        - if one array exists, use it
        - otherwise prefer common data keys
        - otherwise fail explicitly rather than guessing
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path, allow_pickle=False)

    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)

        keys = list(archive.keys())

        if len(keys) == 1:
            return archive[keys[0]]

        preferred = [
            "data",
            "image",
            "hsi",
            "cube",
            "gt",
            "mask",
            "label",
            "labels",
        ]

        for key in preferred:
            if key in archive:
                return archive[key]

        raise RuntimeError(
            f"Ambiguous NPZ file with multiple arrays:\n"
            f"{path}\n"
            f"Keys: {keys}"
        )

    raise ValueError(f"Unsupported NumPy file:\n{path}")


def load_mat_file(path: Path) -> np.ndarray:
    """
    Load MATLAB arrays generically ("best guess" array in the file).

    NOTE: this picks the single largest ndim>=2 array in the file. It is
    only appropriate when the file contains ONE array of interest. When a
    file may contain multiple relevant arrays (e.g. an HSI cube AND a GT
    mask in the same .mat), use load_mat_variable() instead so the correct
    named variable is loaded deterministically.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "Loading .mat files requires scipy.\n"
            "Install it with:\n"
            "pip install scipy"
        ) from exc

    data = loadmat(path)

    candidates: List[Tuple[str, np.ndarray]] = []

    for key, value in data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            candidates.append((key, value))

    if not candidates:
        raise RuntimeError(
            f"No NumPy array found inside MATLAB file:\n{path}"
        )

    # Prefer arrays with dimensionality >= 2 and largest size.
    candidates.sort(
        key=lambda item: (
            item[1].ndim >= 2,
            item[1].size,
        ),
        reverse=True,
    )

    return np.asarray(candidates[0][1])


def load_mat_arrays(path: Path) -> List[Tuple[str, np.ndarray]]:
    """Load named MATLAB variables so HSI and GT can be inspected separately."""
    from scipy.io import loadmat

    data = loadmat(path)
    return [
        (key, np.asarray(value))
        for key, value in data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    ]


def load_mat_variable(path: Path, variable: str) -> np.ndarray:
    """
    FIX (3), part A: load one specific named variable from a MATLAB file.

    This is required because a single .mat scene file can contain BOTH the
    HSI cube (e.g. "img") and the ground-truth mask (e.g. "map") as two
    separate variables. Generic loading (load_mat_file / load_array), which
    just returns "the largest ndim>=2 array", would silently return the
    same (larger) array for both the HSI and GT lookups whenever they share
    a file.
    """
    from scipy.io import loadmat

    data = loadmat(path)

    if variable not in data:
        raise KeyError(
            f"Variable '{variable}' not found in MATLAB file:\n{path}\n"
            f"Available keys: "
            f"{[k for k in data if not k.startswith('__')]}"
        )

    return np.asarray(data[variable])


def load_tiff_file(path: Path) -> np.ndarray:
    """
    Load TIFF using tifffile.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Loading TIFF files requires tifffile.\n"
            "Install it with:\n"
            "pip install tifffile"
        ) from exc

    return np.asarray(tifffile.imread(path))


def load_array(path: Path) -> np.ndarray:
    """
    Load a supported array file using generic ("best guess") selection.

    For .mat files with multiple candidate variables, prefer
    load_entry_array() with a classified entry instead — see FIX (3).
    """
    suffix = path.suffix.lower()

    if suffix in {".npy", ".npz"}:
        return np.asarray(load_numpy_file(path))

    if suffix == ".mat":
        return np.asarray(load_mat_file(path))

    if suffix in {".tif", ".tiff"}:
        return np.asarray(load_tiff_file(path))

    raise ValueError(
        f"Unsupported array format:\n{path}"
    )


def load_entry_array(entry: Dict[str, Any]) -> np.ndarray:
    """
    FIX (3), part B: load the array for a specific classified candidate
    entry, using its exact variable name when the entry came from a .mat
    file with multiple named arrays.

    Without this, re-loading a .mat scene generically for both the HSI
    candidate and the GT candidate returns the SAME array (whichever is
    largest) when both live in one file, which crashes GT inspection
    because a 3D HSI cube cannot be squeezed to a 2D mask.
    """
    path = entry["path"]

    if path.suffix.lower() == ".mat" and entry.get("variable") is not None:
        return load_mat_variable(path, entry["variable"])

    return load_array(path)


# =============================================================================
# CANDIDATE IDENTIFICATION
# =============================================================================

def classify_candidate_shapes(
    files: List[Path],
) -> List[Dict[str, Any]]:
    """
    Load candidate arrays and record their metadata.

    This is exploration only.
    No transformation or resizing occurs.
    """
    results: List[Dict[str, Any]] = []

    for path in files:
        if path.suffix.lower() == ".mat":
            try:
                named_arrays = load_mat_arrays(path)
            except Exception as exc:
                named_arrays = [(path.stem, np.array([]))]
                mat_error = f"{type(exc).__name__}: {exc}"
            else:
                mat_error = None

            for variable_name, array in named_arrays:
                entry = {
                    "path": path,
                    "variable": variable_name,
                    "loaded": mat_error is None,
                    "shape": tuple(int(x) for x in array.shape),
                    "dtype": str(array.dtype),
                    "ndim": int(array.ndim),
                    "size": int(array.size),
                    "min": None,
                    "max": None,
                    "mean": None,
                    "error": mat_error,
                }
                if array.size and np.issubdtype(array.dtype, np.number):
                    values = array.reshape(-1)
                    if values.size > MAX_SUMMARY_VALUES:
                        step = int(np.ceil(values.size / MAX_SUMMARY_VALUES))
                        values = values[::step]
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        entry["min"] = float(np.min(finite))
                        entry["max"] = float(np.max(finite))
                        entry["mean"] = float(np.mean(finite))
                results.append(entry)
            continue

        entry: Dict[str, Any] = {
            "path": path,
            "variable": None,
            "loaded": False,
            "shape": None,
            "dtype": None,
            "ndim": None,
            "size": None,
            "min": None,
            "max": None,
            "mean": None,
            "error": None,
        }

        try:
            array = load_array(path)

            entry["loaded"] = True
            entry["shape"] = tuple(int(x) for x in array.shape)
            entry["dtype"] = str(array.dtype)
            entry["ndim"] = int(array.ndim)
            entry["size"] = int(array.size)

            if array.size > 0 and np.issubdtype(array.dtype, np.number):
                finite = np.asarray(array)[np.isfinite(array)]

                if finite.size > 0:
                    entry["min"] = float(np.min(finite))
                    entry["max"] = float(np.max(finite))
                    entry["mean"] = float(np.mean(finite))

        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"

        results.append(entry)

    return results


def choose_hsi_candidate(
    entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select the most plausible HSI candidate.

    Strong preference:
        3D numeric array

    Filename hints are only tie-breakers.
    """
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    for entry in entries:
        if not entry["loaded"]:
            continue

        ndim = entry["ndim"]

        if ndim is None:
            continue

        score = 0

        if ndim == 3:
            score += 100
        elif ndim > 3:
            score += 40
        else:
            score -= 100

        score += score_image_candidate(entry["path"])

        size = entry["size"] or 0
        score += min(int(np.log10(max(size, 1))), 10)

        candidates.append((score, entry))

    if not candidates:
        return None

    candidates.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    best_score, best_entry = candidates[0]

    if best_entry["ndim"] != 3:
        return None

    return best_entry


def choose_mask_candidate(
    entries: List[Dict[str, Any]],
    hsi_entry: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select the most plausible GT mask candidate.

    Strong preference:
        2D numeric array

    Filename hints are secondary.

    If a 3D array has singleton dimension, it may also represent a mask.
    """
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    for entry in entries:
        if not entry["loaded"]:
            continue

        # FIX (2): exclude the specific HSI *entry* (by identity), not
        # every entry that merely shares the same file path. A single
        # .mat scene commonly stores both the HSI cube ("img") and the
        # GT mask ("map") as separate variables in ONE file, so comparing
        # by path alone would incorrectly discard the correct mask
        # candidate along with the HSI candidate.
        if hsi_entry is not None and entry is hsi_entry:
            continue

        ndim = entry["ndim"]

        if ndim is None:
            continue

        score = 0

        if ndim == 2:
            score += 100
        elif ndim == 3:
            shape = entry["shape"]

            if shape is not None and 1 in shape:
                score += 70
            else:
                score -= 80
        else:
            score -= 100

        score += score_mask_candidate(entry["path"])

        candidates.append((score, entry))

    if not candidates:
        return None

    candidates.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    return candidates[0][1]


# =============================================================================
# SHAPE / LABEL ANALYSIS
# =============================================================================

def infer_hsi_layout(shape: Tuple[int, ...]) -> str:
    """
    Describe possible HSI layout without modifying data.
    """
    if len(shape) != 3:
        return "NOT_3D"

    a, b, c = shape

    # Typical OSD from your existing HSI pipeline is H x W x Bands.
    if c >= 16:
        return "H_W_B"

    if a >= 16:
        return "B_H_W"

    return "AMBIGUOUS_3D"


def normalize_mask_shape_for_analysis(
    array: np.ndarray,
) -> np.ndarray:
    """
    Only removes singleton dimensions for statistics.

    This does NOT alter files or save transformed data.
    """
    squeezed = np.squeeze(array)

    if squeezed.ndim != 2:
        raise ValueError(
            f"Ground-truth candidate is not reducible to 2D. "
            f"Original shape={array.shape}, squeezed={squeezed.shape}"
        )

    return squeezed


def inspect_mask_labels(
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inspect label distribution for a classified GT candidate entry.

    FIX (3): loads via load_entry_array() so the exact named .mat variable
    (e.g. "map") is used, instead of generically reloading the file and
    getting back the largest array (which, for bundled scenes, is the HSI
    cube, not the mask).
    """
    array = load_entry_array(entry)
    mask = normalize_mask_shape_for_analysis(array)

    values, counts = np.unique(mask, return_counts=True)

    result: Dict[str, Any] = {
        "shape": tuple(int(x) for x in mask.shape),
        "dtype": str(mask.dtype),
        "unique_values": [
            float(x) if np.issubdtype(values.dtype, np.floating)
            else int(x)
            for x in values[:50]
        ],
        "num_unique_values": int(len(values)),
    }

    if len(values) <= 50:
        result["unique_counts"] = {
            str(
                float(v)
                if np.issubdtype(values.dtype, np.floating)
                else int(v)
            ): int(c)
            for v, c in zip(values, counts)
        }

    # Conservative oil-ratio interpretation:
    # If labels are binary 0/1, use >0 as oil.
    # Otherwise report positive fraction but do not invent semantics.
    positive_fraction = float(np.mean(mask > 0))

    result["positive_fraction"] = positive_fraction
    result["positive_percent"] = positive_fraction * 100.0

    return result


def inspect_hsi_array(
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inspect HSI cube metadata for a classified HSI candidate entry.

    FIX (3): loads via load_entry_array() so the exact named .mat variable
    (e.g. "img") is used deterministically.
    """
    array = load_entry_array(entry)

    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D HSI array, got shape {array.shape}"
        )

    finite = array[np.isfinite(array)]

    result: Dict[str, Any] = {
        "shape": tuple(int(x) for x in array.shape),
        "dtype": str(array.dtype),
        "layout_guess": infer_hsi_layout(tuple(array.shape)),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "finite_fraction": (
            float(finite.size / array.size)
            if array.size > 0
            else 0.0
        ),
    }

    if result["layout_guess"] == "H_W_B":
        result["height"] = int(array.shape[0])
        result["width"] = int(array.shape[1])
        result["bands"] = int(array.shape[2])

    elif result["layout_guess"] == "B_H_W":
        result["bands"] = int(array.shape[0])
        result["height"] = int(array.shape[1])
        result["width"] = int(array.shape[2])

    else:
        result["height"] = None
        result["width"] = None
        result["bands"] = None

    return result


# =============================================================================
# SCENE EXPLORATION
# =============================================================================

def explore_scene(
    split_name: str,
    scene_dir: Path,
    dataset_root: Path,
) -> Dict[str, Any]:
    """
    Fully inspect one scene directory (or single bundled scene file).
    """
    scene_id = normalize_scene_id(scene_dir.name)

    files = collect_array_files(scene_dir)

    entries = classify_candidate_shapes(files)

    hsi_entry = choose_hsi_candidate(entries)
    mask_entry = choose_mask_candidate(entries, hsi_entry)

    result: Dict[str, Any] = {
        "split": split_name,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir.resolve()),
        "scene_dir_relative": safe_relative(
            scene_dir,
            dataset_root,
        ),
        "num_array_files": len(files),
        "files": [
            {
                "path": str(entry["path"].resolve()),
                "relative_path": safe_relative(
                    entry["path"],
                    scene_dir,
                ),
                "variable": entry.get("variable"),
                "loaded": entry["loaded"],
                "shape": list(entry["shape"])
                if entry["shape"] is not None
                else None,
                "dtype": entry["dtype"],
                "error": entry["error"],
            }
            for entry in entries
        ],
        "hsi": None,
        "gt": None,
        "spatial_shape_match": None,
        "pixel_registration_established": False,
    }

    if hsi_entry is None:
        raise RuntimeError(
            f"Could not identify a 3D HSI cube for {scene_id}\n"
            f"Directory: {scene_dir}\n"
            f"Files: {[str(p) for p in files]}"
        )

    if mask_entry is None:
        raise RuntimeError(
            f"Could not identify a ground-truth mask for {scene_id}\n"
            f"Directory: {scene_dir}\n"
            f"Files: {[str(p) for p in files]}"
        )

    hsi_info = inspect_hsi_array(hsi_entry)
    gt_info = inspect_mask_labels(mask_entry)

    result["hsi"] = {
        "path": str(hsi_entry["path"].resolve()),
        "relative_path": safe_relative(
            hsi_entry["path"],
            scene_dir,
        ),
        "variable": hsi_entry.get("variable"),
        **hsi_info,
    }

    result["gt"] = {
        "path": str(mask_entry["path"].resolve()),
        "relative_path": safe_relative(
            mask_entry["path"],
            scene_dir,
        ),
        "variable": mask_entry.get("variable"),
        **gt_info,
    }

    h = hsi_info.get("height")
    w = hsi_info.get("width")

    gt_shape = gt_info["shape"]

    if h is not None and w is not None:
        result["spatial_shape_match"] = (
            int(h) == int(gt_shape[0])
            and int(w) == int(gt_shape[1])
        )

    # IMPORTANT:
    #
    # Even if HSI and GT shapes match, that only verifies internal
    # HSI/GT array compatibility.
    #
    # This script has NO SAR geometry metadata and therefore cannot
    # establish SAR/HSI physical registration.
    result["pixel_registration_established"] = False

    return result


def explore_split(
    split_name: str,
    root: Path,
) -> List[Dict[str, Any]]:
    """
    Explore all scenes inside one dataset split.
    """
    banner(f"EXPLORING {split_name.upper()} SPLIT")

    print(f"Root: {root}")

    scene_dirs = list_scene_candidates(root)

    if not scene_dirs:
        raise RuntimeError(
            f"No GMxx scene directories found under:\n{root}"
        )

    print(f"Discovered scene directories: {len(scene_dirs)}")

    records: List[Dict[str, Any]] = []

    for index, scene_dir in enumerate(scene_dirs, start=1):
        scene_id = normalize_scene_id(
            scene_dir.name if scene_dir.is_dir() else scene_dir.stem
        )

        print()
        print(
            f"[{index:02d}/{len(scene_dirs):02d}] "
            f"{scene_id} -> {scene_dir}"
        )

        record = explore_scene(
            split_name=split_name,
            scene_dir=scene_dir,
            dataset_root=root,
        )

        hsi = record["hsi"]
        gt = record["gt"]

        print(
            f"  HSI : {tuple(hsi['shape'])} | "
            f"{hsi['dtype']} | "
            f"var={hsi['variable']} | "
            f"layout={hsi['layout_guess']}"
        )

        print(
            f"  GT  : {tuple(gt['shape'])} | "
            f"{gt['dtype']} | "
            f"var={gt['variable']} | "
            f"positive={gt['positive_percent']:.4f}%"
        )

        print(
            f"  HSI/GT spatial shape match: "
            f"{record['spatial_shape_match']}"
        )

        records.append(record)

    return records


# =============================================================================
# SPLIT VALIDATION
# =============================================================================

def extract_scene_ids(
    records: List[Dict[str, Any]],
) -> List[str]:
    return sorted(
        normalize_scene_id(record["scene_id"])
        for record in records
    )


def validate_no_duplicates(
    all_records: List[Dict[str, Any]],
) -> None:
    seen: Dict[str, str] = {}

    for record in all_records:
        scene_id = record["scene_id"]
        split = record["split"]

        if scene_id in seen:
            raise RuntimeError(
                f"Duplicate scene ID detected: {scene_id}\n"
                f"Appears in both {seen[scene_id]} and {split}"
            )

        seen[scene_id] = split


def validate_locked_split(
    split_name: str,
    discovered: List[str],
    expected: List[str],
) -> None:
    """
    Require exact equality between discovered and locked scene IDs.
    """
    discovered_set = set(
        normalize_scene_id(x)
        for x in discovered
    )

    expected_set = set(
        normalize_scene_id(x)
        for x in expected
    )

    missing = sorted(expected_set - discovered_set)
    unexpected = sorted(discovered_set - expected_set)

    if missing or unexpected:
        raise RuntimeError(
            f"{split_name} split does not match locked configuration.\n"
            f"Expected:   {sorted(expected_set)}\n"
            f"Discovered: {sorted(discovered_set)}\n"
            f"Missing:    {missing}\n"
            f"Unexpected: {unexpected}"
        )

    print(
        f"[OK] {split_name:<10} locked split verified: "
        f"{len(discovered_set)} scenes"
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a reproducible exploration summary.
    """
    all_records = (
        train_records
        + val_records
        + test_records
    )

    bands: List[int] = []
    hsi_shapes: List[Tuple[int, ...]] = []
    gt_shapes: List[Tuple[int, ...]] = []
    positive_fractions: List[float] = []

    shape_match_count = 0

    for record in all_records:
        hsi = record["hsi"]
        gt = record["gt"]

        if hsi is not None:
            hsi_shapes.append(tuple(hsi["shape"]))

            if hsi.get("bands") is not None:
                bands.append(int(hsi["bands"]))

        if gt is not None:
            gt_shapes.append(tuple(gt["shape"]))
            positive_fractions.append(
                float(gt["positive_fraction"])
            )

        if record["spatial_shape_match"] is True:
            shape_match_count += 1

    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_roots": {
            "train": str(TRAIN_ROOT),
            "validation": str(VAL_ROOT),
            "test": str(TEST_ROOT),
        },
        "scene_counts": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
            "total": len(all_records),
        },
        "scene_ids": {
            "train": extract_scene_ids(train_records),
            "validation": extract_scene_ids(val_records),
            "test": extract_scene_ids(test_records),
        },
        "hsi_band_counts": sorted(set(bands)),
        "unique_hsi_shapes": sorted(
            {tuple(x) for x in hsi_shapes}
        ),
        "unique_gt_shapes": sorted(
            {tuple(x) for x in gt_shapes}
        ),
        "hsi_gt_spatial_shape_match": {
            "matching_scenes": shape_match_count,
            "total_scenes": len(all_records),
        },
        "positive_label_fraction": {
            "mean": (
                float(np.mean(positive_fractions))
                if positive_fractions
                else None
            ),
            "min": (
                float(np.min(positive_fractions))
                if positive_fractions
                else None
            ),
            "max": (
                float(np.max(positive_fractions))
                if positive_fractions
                else None
            ),
        },
        "registration_contract": {
            "pixel_level_registered": False,
            "pixel_level_fusion_allowed": False,
            "reason": (
                "This exploration verifies only the internal OSD HSI/GT "
                "dataset structure. It does not inspect SAR geolocation, "
                "projection, resolution, acquisition geometry, or physical "
                "co-registration metadata. Therefore matching array "
                "dimensions must not be interpreted as proof that SAR and "
                "HSI pixels represent the same physical location."
            ),
        },
        "next_stage_contract": {
            "allowed": [
                "scene-level fusion",
                "global pooled feature fusion",
                "set/token-based cross-modal attention without positional "
                "same-pixel correspondence",
                "confidence/reliability gating",
                "bounded residual correction",
            ],
            "not_authorized": [
                "raw pixel concatenation based solely on equal dimensions",
                "same-index SAR-HSI pixel correspondence assumptions",
                "pixel-aligned cross-modal attention without independent "
                "registration evidence",
            ],
        },
    }


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def make_json_safe(value: Any) -> Any:
    """
    Convert NumPy values and Paths into JSON-safe Python values.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return [
            make_json_safe(x)
            for x in value
        ]

    if isinstance(value, list):
        return [
            make_json_safe(x)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    return value


def save_manifest(
    all_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """
    Save JSON manifest, CSV summary and exploration summary.
    """
    manifest_payload = {
        "summary": summary,
        "scenes": all_records,
    }

    with MANIFEST_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(manifest_payload),
            file,
            indent=2,
        )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(summary),
            file,
            indent=2,
        )

    fieldnames = [
        "split",
        "scene_id",
        "scene_dir",
        "hsi_path",
        "hsi_variable",
        "hsi_shape",
        "hsi_dtype",
        "hsi_layout",
        "hsi_bands",
        "gt_path",
        "gt_variable",
        "gt_shape",
        "gt_dtype",
        "gt_positive_percent",
        "spatial_shape_match",
        "pixel_registration_established",
    ]

    with MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in all_records:
            hsi = record["hsi"]
            gt = record["gt"]

            writer.writerow(
                {
                    "split": record["split"],
                    "scene_id": record["scene_id"],
                    "scene_dir": record["scene_dir"],
                    "hsi_path": (
                        hsi["path"]
                        if hsi is not None
                        else None
                    ),
                    "hsi_variable": (
                        hsi.get("variable")
                        if hsi is not None
                        else None
                    ),
                    "hsi_shape": (
                        str(tuple(hsi["shape"]))
                        if hsi is not None
                        else None
                    ),
                    "hsi_dtype": (
                        hsi["dtype"]
                        if hsi is not None
                        else None
                    ),
                    "hsi_layout": (
                        hsi["layout_guess"]
                        if hsi is not None
                        else None
                    ),
                    "hsi_bands": (
                        hsi.get("bands")
                        if hsi is not None
                        else None
                    ),
                    "gt_path": (
                        gt["path"]
                        if gt is not None
                        else None
                    ),
                    "gt_variable": (
                        gt.get("variable")
                        if gt is not None
                        else None
                    ),
                    "gt_shape": (
                        str(tuple(gt["shape"]))
                        if gt is not None
                        else None
                    ),
                    "gt_dtype": (
                        gt["dtype"]
                        if gt is not None
                        else None
                    ),
                    "gt_positive_percent": (
                        gt["positive_percent"]
                        if gt is not None
                        else None
                    ),
                    "spatial_shape_match": (
                        record["spatial_shape_match"]
                    ),
                    "pixel_registration_established": (
                        record["pixel_registration_established"]
                    ),
                }
            )

    print()
    print(f"[OK] Manifest JSON : {MANIFEST_JSON}")
    print(f"[OK] Manifest CSV  : {MANIFEST_CSV}")
    print(f"[OK] Summary JSON  : {SUMMARY_JSON}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    banner("STAGE 3 — OSD DATASET EXPLORATION")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Fusion dir   : {FUSION_DIR}")
    print()

    print("Dataset roots:")
    print(f"  Train      : {TRAIN_ROOT}")
    print(f"  Validation : {VAL_ROOT}")
    print(f"  Test       : {TEST_ROOT}")

    for name, path in [
        ("Train", TRAIN_ROOT),
        ("Validation", VAL_ROOT),
        ("Test", TEST_ROOT),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} root does not exist:\n{path}"
            )

    # -------------------------------------------------------------------------
    # Explore exact dataset splits.
    # -------------------------------------------------------------------------

    train_records = explore_split(
        "train",
        TRAIN_ROOT,
    )

    val_records = explore_split(
        "validation",
        VAL_ROOT,
    )

    test_records = explore_split(
        "test",
        TEST_ROOT,
    )

    all_records = (
        train_records
        + val_records
        + test_records
    )

    # -------------------------------------------------------------------------
    # Integrity checks.
    # -------------------------------------------------------------------------

    banner("LOCKED SPLIT INTEGRITY CHECK")

    validate_locked_split(
        "TRAIN",
        extract_scene_ids(train_records),
        TRAIN_SCENES,
    )

    validate_locked_split(
        "VALIDATION",
        extract_scene_ids(val_records),
        VAL_SCENES,
    )

    validate_locked_split(
        "TEST",
        extract_scene_ids(test_records),
        TEST_SCENES,
    )

    validate_no_duplicates(all_records)

    print("[OK] No scene appears in more than one split.")

    # -------------------------------------------------------------------------
    # Dataset summary.
    # -------------------------------------------------------------------------

    summary = build_summary(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )

    banner("OSD DATASET SUMMARY")

    print(
        f"Train scenes      : "
        f"{summary['scene_counts']['train']}"
    )

    print(
        f"Validation scenes : "
        f"{summary['scene_counts']['validation']}"
    )

    print(
        f"Test scenes       : "
        f"{summary['scene_counts']['test']}"
    )

    print(
        f"Total scenes      : "
        f"{summary['scene_counts']['total']}"
    )

    print()
    print(
        f"HSI band counts   : "
        f"{summary['hsi_band_counts']}"
    )

    print(
        f"Unique HSI shapes : "
        f"{len(summary['unique_hsi_shapes'])}"
    )

    print(
        f"Unique GT shapes  : "
        f"{len(summary['unique_gt_shapes'])}"
    )

    print(
        "HSI/GT shape match: "
        f"{summary['hsi_gt_spatial_shape_match']['matching_scenes']}"
        "/"
        f"{summary['hsi_gt_spatial_shape_match']['total_scenes']}"
    )

    positive = summary["positive_label_fraction"]

    if positive["mean"] is not None:
        print()
        print(
            f"Mean positive area: "
            f"{positive['mean'] * 100:.4f}%"
        )

        print(
            f"Min positive area : "
            f"{positive['min'] * 100:.4f}%"
        )

        print(
            f"Max positive area : "
            f"{positive['max'] * 100:.4f}%"
        )

    # -------------------------------------------------------------------------
    # Registration warning.
    # -------------------------------------------------------------------------

    banner("REGISTRATION CONTRACT")

    registration = summary["registration_contract"]

    print(
        "Pixel-level registered : "
        f"{registration['pixel_level_registered']}"
    )

    print(
        "Pixel-level fusion allowed by this exploration : "
        f"{registration['pixel_level_fusion_allowed']}"
    )

    print()
    print(registration["reason"])

    # -------------------------------------------------------------------------
    # Save reproducibility artifacts.
    # -------------------------------------------------------------------------

    save_manifest(
        all_records=all_records,
        summary=summary,
    )

    banner("EXPLORATION COMPLETE")

    print("[SUCCESS] OSD dataset exploration passed.")
    print()
    print("Next stage:")
    print(
        "02_dataset.py will use this verified manifest to construct "
        "scene-aware fusion samples."
    )

    print(
        "IMPORTANT: The next stage must preserve the registration-free "
        "fusion contract."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Dataset exploration stopped.")
        raise

    except Exception as exc:
        banner("EXPLORATION FAILED")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise