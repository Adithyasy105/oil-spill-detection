# # ================================================================
# # 04_train.py
# # FINAL HSI EXPERT TRAINING
# # ================================================================

# import os
# import sys
# import csv
# import json
# import time
# from datetime import datetime

# import numpy as np
# import torch
# import torch.optim as optim

# from tqdm import tqdm


# # ================================================================
# # PATH
# # ================================================================

# PROJECT_DIR = os.path.dirname(
#     os.path.abspath(__file__)
# )

# sys.path.insert(
#     0,
#     PROJECT_DIR
# )


# # ================================================================
# # LOAD PROJECT FILES
# # ================================================================

# exec(
#     open(
#         os.path.join(
#             PROJECT_DIR,
#             "01_config.py"
#         ),
#         encoding="utf-8"
#     ).read()
# )

# exec(
#     open(
#         os.path.join(
#             PROJECT_DIR,
#             "02_dataset.py"
#         ),
#         encoding="utf-8"
#     ).read()
# )

# exec(
#     open(
#         os.path.join(
#             PROJECT_DIR,
#             "03_model.py"
#         ),
#         encoding="utf-8"
#     ).read()
# )


# # ================================================================
# # TRAINING UTILITIES
# # ================================================================

# def save_split_manifest(
#     train_files,
#     val_files
# ):

#     payload = {

#         "seed": SEED,

#         "train": [
#             scene_name(f)
#             for f in train_files
#         ],

#         "validation": [
#             scene_name(f)
#             for f in val_files
#         ],

#         "test": [
#             scene_name(f)
#             for f in discover_scene_files(
#                 TEST_HSI_DIR
#             )
#         ]
#     }

#     with open(
#         SPLIT_FILE,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             payload,
#             f,
#             indent=2
#         )

#     print(
#         f"[OK] Split manifest saved: "
#         f"{SPLIT_FILE}"
#     )


# def evaluate_loader(
#     model,
#     loader,
#     criterion
# ):

#     model.eval()

#     total_loss = 0.0

#     all_probs = []

#     all_targets = []

#     with torch.inference_mode():

#         for images, labels in loader:

#             images = images.to(
#                 DEVICE,
#                 non_blocking=True
#             )

#             labels = (
#                 labels
#                 .to(
#                     DEVICE,
#                     non_blocking=True
#                 )
#                 .unsqueeze(1)
#             )

#             logits = model(
#                 images,
#                 return_features=False
#             )

#             loss = criterion(
#                 logits,
#                 labels
#             )

#             total_loss += (
#                 loss.item()
#                 * images.size(0)
#             )

#             probs = torch.sigmoid(
#                 logits
#             )

#             all_probs.append(
#                 probs.detach().cpu()
#             )

#             all_targets.append(
#                 labels.detach().cpu()
#             )

#     probs = torch.cat(
#         all_probs,
#         dim=0
#     ).flatten()

#     targets = torch.cat(
#         all_targets,
#         dim=0
#     ).flatten()

#     preds = (
#         probs >= DEFAULT_THRESHOLD
#     ).float()

#     metrics = compute_metrics(
#         preds,
#         targets
#     )

#     avg_loss = (
#         total_loss
#         /
#         len(loader.dataset)
#     )

#     return (
#         avg_loss,
#         metrics,
#         probs,
#         targets
#     )


# # ================================================================
# # TRAIN
# # ================================================================

# def train_hsi_expert():

#     print()
#     print("=" * 70)
#     print("STAGE 2 — FINAL HSI EXPERT TRAINING")
#     print("=" * 70)

#     print(
#         f"Device: {DEVICE}"
#     )

#     # ------------------------------------------------------------
#     # Processor
#     # ------------------------------------------------------------

#     processor = SpectralProcessor(
#         n_components=PCA_COMPONENTS,
#         bands_to_remove=BANDS_TO_REMOVE
#     )

#     # ------------------------------------------------------------
#     # Loaders
#     # ------------------------------------------------------------

#     (
#         train_loader,
#         val_loader,
#         train_files,
#         val_files
#     ) = build_train_val_loaders(
#         processor,
#         batch_size=BATCH_SIZE
#     )

#     # ------------------------------------------------------------
#     # LOCK TEST
#     # ------------------------------------------------------------

#     test_files = discover_scene_files(
#         TEST_HSI_DIR
#     )

#     print()
#     print("=" * 70)
#     print("TEST SET — LOCKED / NEVER USED FOR TRAINING")
#     print("=" * 70)

#     for f in test_files:

#         print(
#             f"  {scene_name(f)}"
#         )

#     print("=" * 70)

#     save_split_manifest(
#         train_files,
#         val_files
#     )

#     # ------------------------------------------------------------
#     # MODEL
#     # ------------------------------------------------------------

#     model = HMOSDNet(
#         in_channels=1,
#         num_bands=PCA_COMPONENTS,
#         num_classes=1,
#         mam_reduction=MAM_REDUCTION
#     ).to(DEVICE)

#     total_params = sum(
#         p.numel()
#         for p in model.parameters()
#     )

#     print(
#         f"\nModel parameters: "
#         f"{total_params:,}"
#     )

#     # ------------------------------------------------------------
#     # LOSS
#     # ------------------------------------------------------------

#     criterion = DiceBCELoss(
#         bce_weight=BCE_WEIGHT
#     )

#     # ------------------------------------------------------------
#     # OPTIMIZER
#     # ------------------------------------------------------------

#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY
#     )

#     scheduler = (
#         optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=NUM_EPOCHS
#         )
#     )

#     # ------------------------------------------------------------
#     # LOG
#     # ------------------------------------------------------------

#     timestamp = time.strftime(
#         "%Y%m%d_%H%M%S"
#     )

#     log_path = (
#         HSI_LOGS_DIR
#         /
#         f"hsi_train_{timestamp}.csv"
#     )

#     log_file = open(
#         log_path,
#         "w",
#         newline="",
#         encoding="utf-8"
#     )

#     writer = csv.writer(
#         log_file
#     )

#     writer.writerow([
#         "epoch",
#         "lr",
#         "train_loss",
#         "val_loss",
#         "val_miou",
#         "val_dice",
#         "val_precision",
#         "val_recall",
#         "val_specificity",
#         "val_far",
#         "val_oa"
#     ])

#     # ------------------------------------------------------------
#     # BEST STATE
#     # ------------------------------------------------------------

#     best_miou = -np.inf

#     best_epoch = 0

#     epochs_without_improvement = 0

#     # ============================================================
#     # EPOCH LOOP
#     # ============================================================

#     for epoch in range(
#         1,
#         NUM_EPOCHS + 1
#     ):

#         start_time = time.time()

#         model.train()

#         running_loss = 0.0

#         progress = tqdm(
#             train_loader,
#             desc=(
#                 f"Epoch "
#                 f"{epoch:03d}/{NUM_EPOCHS}"
#             ),
#             leave=False
#         )

#         for images, labels in progress:

#             images = images.to(
#                 DEVICE,
#                 non_blocking=True
#             )

#             labels = (
#                 labels
#                 .to(
#                     DEVICE,
#                     non_blocking=True
#                 )
#                 .unsqueeze(1)
#             )

#             optimizer.zero_grad(
#                 set_to_none=True
#             )

#             logits = model(
#                 images,
#                 return_features=False
#             )

#             loss = criterion(
#                 logits,
#                 labels
#             )

#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(
#                 model.parameters(),
#                 GRAD_CLIP
#             )

#             optimizer.step()

#             running_loss += (
#                 loss.item()
#                 * images.size(0)
#             )

#             progress.set_postfix(
#                 loss=f"{loss.item():.4f}"
#             )

#         train_loss = (
#             running_loss
#             /
#             len(train_loader.dataset)
#         )

#         # --------------------------------------------------------
#         # VALIDATION
#         # --------------------------------------------------------

#         (
#             val_loss,
#             val_metrics,
#             _,
#             _
#         ) = evaluate_loader(
#             model,
#             val_loader,
#             criterion
#         )

#         scheduler.step()

#         lr = optimizer.param_groups[0]["lr"]

#         elapsed = (
#             time.time()
#             -
#             start_time
#         )

#         improved = (
#             val_metrics["miou"]
#             >
#             best_miou + MIN_DELTA
#         )

#         if improved:

#             best_miou = (
#                 val_metrics["miou"]
#             )

#             best_epoch = epoch

#             epochs_without_improvement = 0

#             torch.save(

#                 {
#                     "epoch": epoch,

#                     "model_state_dict":
#                         model.state_dict(),

#                     "optimizer_state_dict":
#                         optimizer.state_dict(),

#                     "scheduler_state_dict":
#                         scheduler.state_dict(),

#                     "val_loss":
#                         val_loss,

#                     "val_metrics":
#                         val_metrics,

#                     "best_val_miou":
#                         best_miou,

#                     "config": {

#                         "pca_components":
#                             PCA_COMPONENTS,

#                         "patch_size":
#                             PATCH_SIZE,

#                         "train_stride":
#                             TRAIN_STRIDE,

#                         "val_stride":
#                             VAL_STRIDE,

#                         "learning_rate":
#                             LEARNING_RATE,

#                         "weight_decay":
#                             WEIGHT_DECAY,

#                         "seed":
#                             SEED
#                     },

#                     "train_scenes": [
#                         scene_name(f)
#                         for f in train_files
#                     ],

#                     "val_scenes": [
#                         scene_name(f)
#                         for f in val_files
#                     ],

#                     "test_scenes": [
#                         scene_name(f)
#                         for f in test_files
#                     ],

#                     "processor_path":
#                         str(
#                             SPECTRAL_PROCESSOR_PATH
#                         ),

#                     "created_at":
#                         datetime.now().isoformat()
#                 },

#                 BEST_CHECKPOINT
#             )

#         else:

#             epochs_without_improvement += 1

#         print(
#             f"E{epoch:03d} | "
#             f"LR={lr:.2e} | "
#             f"Train={train_loss:.5f} | "
#             f"Val={val_loss:.5f} | "
#             f"mIoU={val_metrics['miou']:.5f} | "
#             f"Dice={val_metrics['dice']:.5f} | "
#             f"P={val_metrics['precision']:.5f} | "
#             f"R={val_metrics['recall']:.5f} | "
#             f"FAR={val_metrics['far']:.5f} | "
#             f"{elapsed:.1f}s"
#             +
#             (
#                 "  <-- BEST"
#                 if improved
#                 else ""
#             )
#         )

#         writer.writerow([

#             epoch,

#             lr,

#             train_loss,

#             val_loss,

#             val_metrics["miou"],

#             val_metrics["dice"],

#             val_metrics["precision"],

#             val_metrics["recall"],

#             val_metrics["specificity"],

#             val_metrics["far"],

#             val_metrics["oa"]
#         ])

#         log_file.flush()

#         # --------------------------------------------------------
#         # EARLY STOP
#         # --------------------------------------------------------

#         if (
#             epochs_without_improvement
#             >= EARLY_STOPPING_PATIENCE
#         ):

#             print()
#             print(
#                 f"Early stopping at "
#                 f"epoch {epoch}."
#             )

#             break

#     log_file.close()

#     # ============================================================
#     # LOAD BEST CHECKPOINT
#     # ============================================================

#     print()
#     print("=" * 70)
#     print("LOADING BEST VALIDATION CHECKPOINT")
#     print("=" * 70)

#     if not os.path.exists(
#         BEST_CHECKPOINT
#     ):

#         raise RuntimeError(
#             "No best checkpoint was created."
#         )

#     checkpoint = torch.load(
#         BEST_CHECKPOINT,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     model.load_state_dict(
#         checkpoint[
#             "model_state_dict"
#         ]
#     )

#     print(
#         f"Best epoch: "
#         f"{checkpoint['epoch']}"
#     )

#     print(
#         f"Best validation mIoU: "
#         f"{checkpoint['best_val_miou']:.6f}"
#     )

#     # ============================================================
#     # FREEZE
#     # ============================================================

#     model.eval()

#     for parameter in model.parameters():

#         parameter.requires_grad = False

#     torch.save(

#         {
#             "model_state_dict":
#                 model.state_dict(),

#             "best_epoch":
#                 checkpoint["epoch"],

#             "best_val_miou":
#                 checkpoint[
#                     "best_val_miou"
#                 ],

#             "processor_path":
#                 str(
#                     SPECTRAL_PROCESSOR_PATH
#                 ),

#             "train_scenes": [
#                 scene_name(f)
#                 for f in train_files
#             ],

#             "val_scenes": [
#                 scene_name(f)
#                 for f in val_files
#             ],

#             "test_scenes": [
#                 scene_name(f)
#                 for f in test_files
#             ],

#             "seed":
#                 SEED
#         },

#         FINAL_CHECKPOINT
#     )

#     print(
#         f"[OK] Frozen HSI model saved:\n"
#         f"{FINAL_CHECKPOINT}"
#     )

#     print()
#     print("=" * 70)
#     print("TRAINING COMPLETE")
#     print("=" * 70)

#     print(
#         f"Best validation mIoU: "
#         f"{best_miou:.6f}"
#     )

#     print(
#         f"Best epoch: "
#         f"{best_epoch}"
#     )

#     print(
#         "\nIMPORTANT:"
#     )

#     print(
#         "The test scenes were NOT used "
#         "for training, PCA fitting, "
#         "checkpoint selection, or "
#         "threshold selection."
#     )


# if __name__ == "__main__":

#     train_hsi_expert()



# ================================================================
# 04_train.py
# FINAL HSI EXPERT TRAINING  -- CORRECTED
#
# CHANGES FROM ORIGINAL:
#   1. Scheduler: CosineAnnealingLR(T_max=NUM_EPOCHS) -> a manual
#      linear warmup for WARMUP_EPOCHS, then ReduceLROnPlateau
#      driven by validation mIoU. Cosine annealing over a 60-epoch
#      horizon barely decays during the first ~15-25 epochs, which
#      is all the model was actually getting before early stopping
#      fired -- so LR was effectively ~constant near its max value
#      for the entire observed run. ReduceLROnPlateau reacts to
#      what val mIoU is actually doing.
#   2. scheduler.step() now takes val_metrics["miou"] as required by
#      ReduceLROnPlateau (mode="max"), and is skipped during the
#      warmup phase (warmup sets LR directly).
# ================================================================

import os
import sys
import csv
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm


# ================================================================
# PATH
# ================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


# ================================================================
# LOAD PROJECT FILES
# ================================================================

exec(open(os.path.join(PROJECT_DIR, "01_config.py"), encoding="utf-8").read())
exec(open(os.path.join(PROJECT_DIR, "02_dataset.py"), encoding="utf-8").read())
exec(open(os.path.join(PROJECT_DIR, "03_model.py"), encoding="utf-8").read())


# ================================================================
# TRAINING UTILITIES
# ================================================================

def save_split_manifest(train_files, val_files):

    payload = {
        "seed": SEED,
        "train": [scene_name(f) for f in train_files],
        "validation": [scene_name(f) for f in val_files],
        "test": [scene_name(f) for f in discover_scene_files(TEST_HSI_DIR)],
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Split manifest saved: {SPLIT_FILE}")


def evaluate_loader(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.inference_mode():
        for images, labels in loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            logits = model(images, return_features=False)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    probs = torch.cat(all_probs, dim=0).flatten()
    targets = torch.cat(all_targets, dim=0).flatten()

    preds = (probs >= DEFAULT_THRESHOLD).float()

    metrics = compute_metrics(preds, targets)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics, probs, targets


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ================================================================
# TRAIN
# ================================================================

def train_hsi_expert():

    print()
    print("=" * 70)
    print("STAGE 2 -- FINAL HSI EXPERT TRAINING (CORRECTED)")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    processor = SpectralProcessor(
        n_components=PCA_COMPONENTS, bands_to_remove=BANDS_TO_REMOVE
    )

    (train_loader, val_loader, train_files, val_files) = build_train_val_loaders(
        processor, batch_size=BATCH_SIZE
    )

    test_files = discover_scene_files(TEST_HSI_DIR)

    print()
    print("=" * 70)
    print("TEST SET -- LOCKED / NEVER USED FOR TRAINING")
    print("=" * 70)
    for f in test_files:
        print(f"  {scene_name(f)}")
    print("=" * 70)

    save_split_manifest(train_files, val_files)

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = HMOSDNet(
        in_channels=1,
        num_bands=PCA_COMPONENTS,
        num_classes=1,
        mam_reduction=MAM_REDUCTION,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    criterion = DiceBCELoss(bce_weight=BCE_WEIGHT)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ------------------------------------------------------------
    # SCHEDULER -- CORRECTED
    # Manual linear warmup for WARMUP_EPOCHS, then ReduceLROnPlateau
    # on validation mIoU.
    # ------------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=globals().get("LR_PLATEAU_FACTOR", 0.5),
        patience=globals().get("LR_PLATEAU_PATIENCE", 4),
        min_lr=globals().get("LR_MIN", 1e-6),
    )

    warmup_epochs = globals().get("WARMUP_EPOCHS", 0)

    # ------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = HSI_LOGS_DIR / f"hsi_train_{timestamp}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)

    writer.writerow([
        "epoch", "lr", "train_loss", "val_loss", "val_miou", "val_dice",
        "val_precision", "val_recall", "val_specificity", "val_far", "val_oa",
    ])

    best_miou = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    # ============================================================
    # EPOCH LOOP
    # ============================================================

    for epoch in range(1, NUM_EPOCHS + 1):

        start_time = time.time()

        # --------------------------------------------------------
        # WARMUP: set LR directly for the first warmup_epochs
        # --------------------------------------------------------
        if epoch <= warmup_epochs:
            warmup_lr = LEARNING_RATE * epoch / max(warmup_epochs, 1)
            set_lr(optimizer, warmup_lr)

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS}", leave=False
        )

        for images, labels in progress:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images, return_features=False)
            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val_loss, val_metrics, _, _ = evaluate_loader(model, val_loader, criterion)

        # --------------------------------------------------------
        # SCHEDULER STEP -- only after warmup finishes
        # --------------------------------------------------------
        if epoch > warmup_epochs:
            scheduler.step(val_metrics["miou"])

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        improved = val_metrics["miou"] > best_miou + MIN_DELTA

        if improved:

            best_miou = val_metrics["miou"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "best_val_miou": best_miou,
                    "config": {
                        "pca_components": PCA_COMPONENTS,
                        "patch_size": PATCH_SIZE,
                        "train_stride": TRAIN_STRIDE,
                        "val_stride": VAL_STRIDE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "target_oil_fraction": globals().get("TARGET_OIL_FRACTION", 0.20),
                        "seed": SEED,
                    },
                    "train_scenes": [scene_name(f) for f in train_files],
                    "val_scenes": [scene_name(f) for f in val_files],
                    "test_scenes": [scene_name(f) for f in test_files],
                    "processor_path": str(SPECTRAL_PROCESSOR_PATH),
                    "created_at": datetime.now().isoformat(),
                },
                BEST_CHECKPOINT,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"E{epoch:03d} | LR={lr:.2e} | Train={train_loss:.5f} | "
            f"Val={val_loss:.5f} | mIoU={val_metrics['miou']:.5f} | "
            f"Dice={val_metrics['dice']:.5f} | P={val_metrics['precision']:.5f} | "
            f"R={val_metrics['recall']:.5f} | FAR={val_metrics['far']:.5f} | "
            f"{elapsed:.1f}s" + ("  <-- BEST" if improved else "")
        )

        writer.writerow([
            epoch, lr, train_loss, val_loss, val_metrics["miou"], val_metrics["dice"],
            val_metrics["precision"], val_metrics["recall"], val_metrics["specificity"],
            val_metrics["far"], val_metrics["oa"],
        ])
        log_file.flush()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print()
            print(f"Early stopping at epoch {epoch}.")
            break

    log_file.close()

    # ============================================================
    # LOAD BEST CHECKPOINT
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING BEST VALIDATION CHECKPOINT")
    print("=" * 70)

    if not os.path.exists(BEST_CHECKPOINT):
        raise RuntimeError("No best checkpoint was created.")

    checkpoint = torch.load(BEST_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation mIoU: {checkpoint['best_val_miou']:.6f}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": checkpoint["epoch"],
            "best_val_miou": checkpoint["best_val_miou"],
            "processor_path": str(SPECTRAL_PROCESSOR_PATH),
            "train_scenes": [scene_name(f) for f in train_files],
            "val_scenes": [scene_name(f) for f in val_files],
            "test_scenes": [scene_name(f) for f in test_files],
            "seed": SEED,
        },
        FINAL_CHECKPOINT,
    )

    print(f"[OK] Frozen HSI model saved:\n{FINAL_CHECKPOINT}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation mIoU: {best_miou:.6f}")
    print(f"Best epoch: {best_epoch}")
    print("\nIMPORTANT:")
    print(
        "The test scenes were NOT used for training, PCA fitting, "
        "checkpoint selection, or threshold selection."
    )


if __name__ == "__main__":
    train_hsi_expert()