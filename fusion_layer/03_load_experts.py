"""
STAGE 3 — LOAD FROZEN EXPERTS
==============================

Purpose
-------
Load the already-trained SAR and HSI experts used by the fusion system.
import joblib

This module:

    1. Loads the existing SAR expert.
    2. Loads the existing HSI expert.
    3. Loads the HSI spectral processor.
    4. Freezes both experts completely.
    5. Puts both experts in eval mode.
    6. Provides safe inference helper functions.

IMPORTANT FUSION CONTRACT
-------------------------
Pixel registration between SAR and HSI is NOT assumed.

Therefore this module does NOT:

    - concatenate SAR and HSI feature maps
    - align SAR pixels with HSI pixels
    - resize one modality to physically match another
    - perform cross-modal spatial attention
    - assume matching (y, x) coordinates represent the same location

The experts are independent frozen feature/probability providers.

Architecture:

    SAR patch/image
          |
          v
    processor = joblib.load(HSI_PROCESSOR_PATH)
                              v
                        Fusion Model
                              ^
                              |
          +-------------------+
          |
    Frozen HSI Expert
          ^
          |
    HSI spectral patch


Only the fusion model is trainable.
"""

from __future__ import annotations

import joblib
import sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# PATH SETUP
# ============================================================================

THIS_FILE = Path(__file__).resolve()
FUSION_DIR = THIS_FILE.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))


def load_python_module(name: str, path: Path):
    """
    Import a Python module directly from a file path.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired Python file not found:\n{path}"
        )

    spec = spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module:\n{path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

CFG = load_python_module(
    "fusion_config",
    FUSION_DIR / "00_config.py",
)


# ============================================================================
# PATHS
# ============================================================================

SAR_MODEL_PATH = Path(CFG.SAR_MODEL_PATH)
SAR_CHECKPOINT_PATH = Path(CFG.SAR_CHECKPOINT_PATH)

HSI_MODEL_PATH = Path(CFG.HSI_MODEL_PATH)
HSI_CHECKPOINT_PATH = Path(CFG.HSI_CHECKPOINT_PATH)
HSI_PROCESSOR_PATH = Path(CFG.HSI_PROCESSOR_PATH)
PCA_BANDS = int(getattr(CFG, "PCA_BANDS", getattr(CFG, "HSI_PCA_COMPONENTS", 25)))


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# UTILITIES
# ============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze every parameter and force evaluation mode.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns:

        total_parameters
        trainable_parameters
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats.

    Accepted:

        checkpoint
        checkpoint["state_dict"]
        checkpoint["model_state_dict"]
        checkpoint["model"]
        checkpoint["net"]
    """

    if isinstance(checkpoint, dict):

        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]

        for key in candidate_keys:

            if key in checkpoint:

                value = checkpoint[key]

                if isinstance(value, dict):
                    return value

        # If dictionary itself looks like a state dict.
        if all(
            isinstance(key, str)
            for key in checkpoint.keys()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not identify a valid model state dictionary "
        "inside checkpoint."
    )


def remove_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Removes DataParallel 'module.' prefix if present.
    """

    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict))

    if first_key.startswith("module."):

        return {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict


# ============================================================================
# SAR EXPERT LOADING
# ============================================================================

def build_sar_model() -> nn.Module:
    """
    Import and construct the SAR model.

    This function tries common construction patterns without changing
    the original SAR model architecture.
    """

    sar_module = load_python_module(
        "sar_expert_model",
        SAR_MODEL_PATH,
    )

    if hasattr(sar_module, "CBDNetSAR"):
        return sar_module.CBDNetSAR(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention="scse",
        )

    candidate_names = [
        "Model",
        "SARModel",
        "UNet",
        "UNetModel",
        "SegmentationModel",
        "build_model",
        "create_model",
    ]

    for name in candidate_names:

        if not hasattr(sar_module, name):
            continue

        candidate = getattr(
            sar_module,
            name,
        )

        try:

            if isinstance(candidate, type):

                model = candidate()

            else:

                model = candidate()

            if isinstance(model, nn.Module):

                return model

        except TypeError:
            continue

    raise RuntimeError(
        "\nCould not construct SAR model.\n"
        f"Model file:\n{SAR_MODEL_PATH}\n\n"
        "Expected one of these constructors:\n"
        "  Model\n"
        "  SARModel\n"
        "  UNet\n"
        "  UNetModel\n"
        "  SegmentationModel\n"
        "  build_model()\n"
        "  create_model()\n\n"
        "If your existing SAR model uses another constructor name, "
        "add that exact name to candidate_names."
    )


def load_sar_expert(
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Load and freeze the SAR expert.
    """

    if not SAR_CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nSAR checkpoint not found:\n"
            f"{SAR_CHECKPOINT_PATH}"
        )

    print(
        "\nLoading SAR expert..."
    )

    model = build_sar_model()

    checkpoint = torch.load(
        SAR_CHECKPOINT_PATH,
        map_location=device,
    )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as strict_error:

        print(
            "\n[WARNING] Strict checkpoint loading failed."
        )

        print(
            "Trying non-strict loading..."
        )

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(
            f"Missing keys   : {len(missing)}"
        )

        print(
            f"Unexpected keys: {len(unexpected)}"
        )

        if len(missing) > 0 or len(unexpected) > 0:

            raise RuntimeError(
                "\nSAR checkpoint is not compatible with "
                "the SAR architecture.\n"
                f"Original error:\n{strict_error}"
            )

    model = model.to(device)

    model = freeze_model(
        model
    )

    total, trainable = count_parameters(
        model
    )

    print(
        f"[OK] SAR expert loaded"
    )

    print(
        f"     Device              : {device}"
    )

    print(
        f"     Total parameters    : {total:,}"
    )

    print(
        f"     Trainable parameters: {trainable:,}"
    )

    return model


# ============================================================================
# HSI EXPERT LOADING
# ============================================================================

def build_hsi_model() -> nn.Module:
    """
    Import and construct the existing HSI model.

    The HSI expert is NOT recreated with a new architecture.
    We use the architecture defined in:

        hsi_layer_corrected/03_model.py
    """

    hsi_module = load_python_module(
        "hsi_expert_model",
        HSI_MODEL_PATH,
    )

    if hasattr(hsi_module, "HMOSDNet"):
        return hsi_module.HMOSDNet(
            in_channels=1,
            num_bands=PCA_BANDS,
            num_classes=1,
        )

    candidate_names = [
        "Model",
        "HSIModel",
        "SpectralModel",
        "CNN",
        "Classifier",
        "build_model",
        "create_model",
    ]

    for name in candidate_names:

        if not hasattr(
            hsi_module,
            name,
        ):
            continue

        candidate = getattr(
            hsi_module,
            name,
        )

        try:

            if isinstance(candidate, type):

                model = candidate()

            else:

                model = candidate()

            if isinstance(
                model,
                nn.Module,
            ):

                return model

        except TypeError:
            continue

    raise RuntimeError(
        "\nCould not construct HSI model.\n"
        f"Model file:\n{HSI_MODEL_PATH}\n\n"
        "Expected one of these constructors:\n"
        "  Model\n"
        "  HSIModel\n"
        "  SpectralModel\n"
        "  CNN\n"
        "  Classifier\n"
        "  build_model()\n"
        "  create_model()\n"
    )


def load_hsi_processor() -> Any:
    """
    Load the exact spectral processor used by the trained HSI expert.
    """

    if not HSI_PROCESSOR_PATH.exists():

        raise FileNotFoundError(
            f"\nHSI processor not found:\n"
            f"{HSI_PROCESSOR_PATH}"
        )

    hsi_data_path = Path(CFG.HSI_DATASET_FILE)
    hsi_data_module = load_python_module(
        "hsi_expert_dataset",
        hsi_data_path,
    )
    bands_to_remove = getattr(
        hsi_data_module,
        "BANDS_TO_REMOVE",
        list(range(103, 113)) + list(range(149, 170)),
    )
    processor = hsi_data_module.SpectralProcessor(
        n_components=PCA_BANDS,
        bands_to_remove=bands_to_remove,
    )
    processor.load(str(HSI_PROCESSOR_PATH))

    print(
        "[OK] HSI spectral processor loaded"
    )

    return processor


def load_hsi_expert(
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, Any]:
    """
    Load and freeze the HSI expert and its processor.
    """

    if not HSI_CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nHSI checkpoint not found:\n"
            f"{HSI_CHECKPOINT_PATH}"
        )

    print(
        "\nLoading HSI expert..."
    )

    model = build_hsi_model()

    checkpoint = torch.load(
        HSI_CHECKPOINT_PATH,
        map_location=device,
    )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as strict_error:

        print(
            "\n[WARNING] Strict checkpoint loading failed."
        )

        print(
            "Trying non-strict loading..."
        )

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(
            f"Missing keys   : {len(missing)}"
        )

        print(
            f"Unexpected keys: {len(unexpected)}"
        )

        if len(missing) > 0 or len(unexpected) > 0:

            raise RuntimeError(
                "\nHSI checkpoint is not compatible with "
                "the HSI architecture.\n"
                f"Original error:\n{strict_error}"
            )

    model = model.to(device)

    model = freeze_model(
        model
    )

    processor = load_hsi_processor()

    total, trainable = count_parameters(
        model
    )

    print(
        f"[OK] HSI expert loaded"
    )

    print(
        f"     Device              : {device}"
    )

    print(
        f"     Total parameters    : {total:,}"
    )

    print(
        f"     Trainable parameters: {trainable:,}"
    )

    return model, processor


# ============================================================================
# EXPERT INFERENCE
# ============================================================================

@torch.no_grad()
def run_sar_expert(
    model: nn.Module,
    sar_input: torch.Tensor,
) -> torch.Tensor:
    """
    Run frozen SAR expert.

    Parameters
    ----------
    sar_input:
        SAR tensor.

        The exact shape must match the SAR expert's original input
        contract.

    Returns
    -------
    torch.Tensor
        Raw expert output.
    """

    model.eval()

    return model(
        sar_input
    )


@torch.no_grad()
def run_hsi_expert(
    model: nn.Module,
    hsi_input: torch.Tensor,
) -> torch.Tensor:
    """
    Run frozen HSI expert.

    Parameters
    ----------
    hsi_input:
        HSI tensor after the exact preprocessing expected by the
        trained HSI expert.

    Returns
    -------
    torch.Tensor
        Raw expert output.
    """

    model.eval()

    return model(
        hsi_input
    )


# ============================================================================
# LOAD BOTH EXPERTS
# ============================================================================

def load_all_experts(
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """
    Load the complete frozen expert bundle.
    """

    banner(
        "STAGE 3 — LOADING FROZEN SAR + HSI EXPERTS"
    )

    print(
        f"\nDevice: {device}"
    )

    sar_model = load_sar_expert(
        device
    )

    hsi_model, hsi_processor = load_hsi_expert(
        device
    )

    bundle = {
        "sar_model": sar_model,
        "hsi_model": hsi_model,
        "hsi_processor": hsi_processor,
        "device": device,
    }

    banner(
        "EXPERT LOADING COMPLETE"
    )

    print(
        "[SUCCESS] SAR expert frozen"
    )

    print(
        "[SUCCESS] HSI expert frozen"
    )

    print(
        "[SUCCESS] Only fusion layers will be trainable"
    )

    return bundle


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:
    """
    Verify that:

        - both model files exist
        - both checkpoints load
        - HSI processor loads
        - both models are completely frozen
    """

    banner(
        "STAGE 3 — FROZEN EXPERT LOADER SELF TEST"
    )

    print(
        "\nInput files:"
    )

    paths = {
        "SAR model": SAR_MODEL_PATH,
        "SAR checkpoint": SAR_CHECKPOINT_PATH,
        "HSI model": HSI_MODEL_PATH,
        "HSI checkpoint": HSI_CHECKPOINT_PATH,
        "HSI processor": HSI_PROCESSOR_PATH,
    }

    for name, path in paths.items():

        status = (
            "[OK]"
            if path.exists()
            else "[MISSING]"
        )

        print(
            f"  {status:<10} {name}"
        )

        print(
            f"             {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"\nMissing required file:\n{path}"
            )

    bundle = load_all_experts(
        DEVICE
    )

    sar_model = bundle[
        "sar_model"
    ]

    hsi_model = bundle[
        "hsi_model"
    ]

    sar_trainable = any(
        parameter.requires_grad
        for parameter in sar_model.parameters()
    )

    hsi_trainable = any(
        parameter.requires_grad
        for parameter in hsi_model.parameters()
    )

    banner(
        "FREEZE VERIFICATION"
    )

    print(
        f"SAR training mode : {sar_model.training}"
    )

    print(
        f"HSI training mode : {hsi_model.training}"
    )

    print(
        f"SAR parameters require grad : {sar_trainable}"
    )

    print(
        f"HSI parameters require grad : {hsi_trainable}"
    )

    if sar_trainable:

        raise RuntimeError(
            "SAR expert is not fully frozen."
        )

    if hsi_trainable:

        raise RuntimeError(
            "HSI expert is not fully frozen."
        )

    if sar_model.training:

        raise RuntimeError(
            "SAR expert is not in eval mode."
        )

    if hsi_model.training:

        raise RuntimeError(
            "HSI expert is not in eval mode."
        )

    banner(
        "EXPERT LOADER SELF TEST PASSED"
    )

    print(
        "[SUCCESS] Both experts loaded correctly."
    )

    print(
        "[SUCCESS] Both experts are frozen."
    )

    print(
        "[SUCCESS] Both experts are in evaluation mode."
    )

    print(
        "\nNext stage:"
    )

    print(
        "04_fusion_model.py will define the ONLY trainable "
        "part of the system."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    self_test()