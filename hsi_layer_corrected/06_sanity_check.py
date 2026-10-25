# ================================================================
# 06_sanity_check.py
# HSI DATA + PROCESSING + MODEL SANITY CHECK
# ================================================================

import os
import sys

import numpy as np
import torch


PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    PROJECT_DIR
)


exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "01_config.py"
        ),
        encoding="utf-8"
    ).read()
)

exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "02_dataset.py"
        ),
        encoding="utf-8"
    ).read()
)

exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "03_model.py"
        ),
        encoding="utf-8"
    ).read()
)


print()
print("=" * 70)
print("HSI COMPLETE SANITY CHECK")
print("=" * 70)


# ================================================================
# PATH CHECK
# ================================================================

for name, path in [

    ("Train", TRAIN_HSI_DIR),

    ("Validation", VAL_HSI_DIR),

    ("Test", TEST_HSI_DIR)

]:

    print(
        f"{name:<12}: {path}"
    )

    assert os.path.isdir(
        path
    )


# ================================================================
# DISCOVER SPLITS
# ================================================================

train_files = discover_scene_files(
    TRAIN_HSI_DIR
)

val_files = discover_scene_files(
    VAL_HSI_DIR
)

test_files = discover_scene_files(
    TEST_HSI_DIR
)


print()
print(
    f"Train scenes: {len(train_files)}"
)

print(
    f"Val scenes:   {len(val_files)}"
)

print(
    f"Test scenes:  {len(test_files)}"
)


# ================================================================
# SPLIT LEAKAGE CHECK
# ================================================================

train_names = {
    scene_name(f)
    for f in train_files
}

val_names = {
    scene_name(f)
    for f in val_files
}

test_names = {
    scene_name(f)
    for f in test_files
}

assert train_names.isdisjoint(
    val_names
)

assert train_names.isdisjoint(
    test_names
)

assert val_names.isdisjoint(
    test_names
)

print()
print(
    "[OK] No scene overlap."
)


# ================================================================
# RAW DATA CHECK
# ================================================================

for path in (
    train_files
    +
    val_files
    +
    test_files
):

    img, gt = load_hsi_scene(
        path
    )

    assert img.ndim == 3

    assert gt.ndim == 2

    assert img.shape[:2] == gt.shape

    assert img.shape[2] in (
        ORIGINAL_BANDS,
        EXPECTED_REMAINING_BANDS
    )

    assert np.isfinite(
        img
    ).all()

    print(
        f"[OK] {scene_name(path)} "
        f"{img.shape} "
        f"GT={gt.shape} "
        f"oil={gt.mean()*100:.2f}%"
    )


# ================================================================
# PCA FIT ONLY ON TRAIN
# ================================================================

processor = SpectralProcessor(
    n_components=PCA_COMPONENTS,
    bands_to_remove=BANDS_TO_REMOVE
)

train_img, _ = load_hsi_scene(
    train_files[0]
)

processor.fit(
    [train_img]
)

pca = processor.transform(
    train_img
)

print()
print(
    f"PCA output: {pca.shape}"
)

assert pca.shape[2] == PCA_COMPONENTS

assert np.isfinite(
    pca
).all()


# ================================================================
# PATCH CHECK
# ================================================================

ds = HOSDDataset(
    train_files[:1],
    processor,
    patch_size=PATCH_SIZE,
    stride=20,
    augment=False
)

x, y = ds[0]

print(
    f"Patch shape: {x.shape}"
)

print(
    f"Label: {y.item()}"
)

assert x.shape == (
    1,
    25,
    5,
    5
)


# ================================================================
# MODEL CHECK
# ================================================================

model = HMOSDNet(
    in_channels=1,
    num_bands=25
).to(DEVICE)

dummy = torch.randn(
    4,
    1,
    25,
    5,
    5,
    device=DEVICE
)

with torch.no_grad():

    (
        logits,
        features,
        none_val,
        conf
    ) = model(dummy)

print(
    f"Logits:      {logits.shape}"
)

print(
    f"Features:    {features.shape}"
)

print(
    f"Confidence:  {conf.shape}"
)

assert logits.shape == (
    4,
    1
)

assert features.shape == (
    4,
    64,
    25
)

assert conf.shape == (
    4,
    1
)

assert none_val is None


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("ALL HSI SANITY CHECKS PASSED")
print("=" * 70)

print()
print("You can now run:")
print()
print("    python 04_train.py")
print()
print("After training:")
print()
print("    python 05_evaluate.py")
print()