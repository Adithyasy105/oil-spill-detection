# %% [markdown]
# # 📡 Stage 1: SAR Expert — Full Training Pipeline
# **CBD-Net Training with BCE + MBL Loss, AMP, and Comprehensive Metrics**
#
# This is the main training script. Run this notebook to train the SAR expert.
#
# Training strategy per CBD-Net paper & our architecture:
# - Loss: `L_SAR = BCE + 0.3 × MBL`
# - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - Mixed Precision (AMP) for RTX 4050 6GB VRAM
# - Gradient clipping (max_norm=1.0) for training stability
# - Early stopping (patience=15)

# %%
import os
import sys
import time
import csv
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer")

from importlib import import_module

# %% [markdown]
# ## 1. Load Configuration, Dataset, and Model

# %%
# ═══════════════════════════════════════════════════════
#  LOAD ALL MODULES
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("  📡 SAR Expert (CBD-Net) — Stage 1 Training")
print("  Architecture: ResNet-18 + scSE + Multi-Scale Boundary Supervision")
print("=" * 70)

# Execute config
config_path = os.path.join(r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer", "01_config.py")
exec(open(config_path, encoding='utf-8').read())

# Import dataset module
sys.path.insert(0, r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer")

# We need to import the dataset and model
# Using exec to avoid import issues in notebook environments
SKIP_SELF_TEST = True  # Prevent self-test blocks in dataset/model from running
dataset_path = os.path.join(r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer", "02_dataset.py")
model_path = os.path.join(r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer", "03_model.py")

# Execute dataset module (loads SOSDataset, create_data_loaders, etc.)
exec(open(dataset_path, encoding='utf-8').read())
# Execute model module (loads CBDNetSAR, CBDNetLoss, etc.)
exec(open(model_path, encoding='utf-8').read())

# %% [markdown]
# ## 2. Metrics Computation
#
# Comprehensive metrics for oil spill segmentation evaluation:
# - **mIoU** (mean Intersection over Union) — primary metric
# - **Dice** (F1 score) — overlap with ground truth
# - **FAR** (False Alarm Rate) — fraction of false positives
# - **OA** (Overall Accuracy) — pixel-level accuracy
# - **Precision & Recall** — for class-specific evaluation
# - **Boundary F1** — boundary prediction quality

# %%
def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute comprehensive segmentation metrics.
    
    Bug 2 fix: IoU for water class uses correct perspective mapping.
    
    Args:
        preds: (B, 1, H, W) predicted probabilities (after sigmoid)
        targets: (B, 1, H, W) binary ground truth
        threshold: Classification threshold
    
    Returns:
        dict with all metrics
    """
    # Binarize predictions
    pred_binary = (preds > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Confusion matrix elements (from OIL class perspective)
    tp = (pred_flat * target_flat).sum().item()              # Predicted oil, IS oil
    fp = (pred_flat * (1 - target_flat)).sum().item()         # Predicted oil, IS water
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()   # Predicted water, IS water
    fn = ((1 - pred_flat) * target_flat).sum().item()         # Predicted water, IS oil
    
    eps = 1e-7
    
    # IoU for oil class (class=1)
    # = TP / (TP + FP + FN)
    # = correctly predicted oil / (all predicted oil + missed oil)
    iou_oil = tp / (tp + fp + fn + eps)
    
    # IoU for water class (class=0) — Bug 2 fix with explicit perspective mapping:
    #   TP_water = TN  (correctly predicted water)
    #   FP_water = FN  (predicted water but actually oil — missed oil)
    #   FN_water = FP  (predicted oil but actually water — false alarm)
    # IoU_water = TP_water / (TP_water + FP_water + FN_water)
    #           = TN / (TN + FN + FP)
    iou_water = tn / (tn + fn + fp + eps)
    
    # Mean IoU
    miou = (iou_oil + iou_water) / 2.0
    
    # Dice coefficient (F1) for oil class
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
    # Precision & Recall (oil class)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # False Alarm Rate = FP / (FP + TN) = false alarms / all actual water
    far = fp / (fp + tn + eps)
    
    # Overall Accuracy
    oa = (tp + tn) / (tp + fp + tn + fn + eps)
    
    return {
        'miou': miou,
        'iou_oil': iou_oil,
        'iou_water': iou_water,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'far': far,
        'oa': oa,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

# %% [markdown]
# ## 3. Training Loop
#
# Full training with:
# - Mixed precision (AMP) for speed
# - Gradient clipping for stability
# - Per-epoch validation
# - Best model checkpointing (by mIoU)
# - CSV logging for analysis
# - Early stopping

# %%
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    """Train for one epoch with AMP and gradient clipping."""
    model.train()
    
    running_loss = 0.0
    running_bce = 0.0
    running_mbl = 0.0
    running_conf = 0.0  # Warning 1 fix: track mask_conf
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, masks, boundaries) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boundary_gts = [b.to(device, non_blocking=True) for b in boundaries]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with AMP
        # Bug 1 fix: model ALWAYS returns 4 values, discard features with _
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg_logits, boundary_logits, _, mask_conf = model(images)
            total_loss, bce_loss, mbl_loss = criterion(
                seg_logits, masks, boundary_logits, boundary_gts
            )
        
        # Backward pass with scaler
        scaler.scale(total_loss).backward()
        
        # Gradient clipping (after unscaling)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate metrics
        running_loss += total_loss.item() * images.size(0)
        running_bce += bce_loss.item() * images.size(0)
        running_mbl += mbl_loss.item() * images.size(0)
        running_conf += mask_conf.mean().item() * images.size(0)  # Warning 1
        
        with torch.no_grad():
            probs = torch.sigmoid(seg_logits)
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())
    
    # Compute epoch metrics
    n = len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = running_loss / n
    metrics['bce'] = running_bce / n
    metrics['mbl'] = running_mbl / n
    metrics['mask_conf_mean'] = running_conf / n  # Warning 1: early warning of decoder collapse
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model (no gradients)."""
    model.eval()
    
    running_loss = 0.0
    running_bce = 0.0
    running_mbl = 0.0
    running_conf = 0.0  # Warning 1 fix: track mask_conf
    all_preds = []
    all_targets = []
    
    for images, masks, boundaries in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boundary_gts = [b.to(device, non_blocking=True) for b in boundaries]
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # Bug 1 fix: model ALWAYS returns 4 values
            seg_logits, boundary_logits, _, mask_conf = model(images)
            total_loss, bce_loss, mbl_loss = criterion(
                seg_logits, masks, boundary_logits, boundary_gts
            )
        
        running_loss += total_loss.item() * images.size(0)
        running_bce += bce_loss.item() * images.size(0)
        running_mbl += mbl_loss.item() * images.size(0)
        running_conf += mask_conf.mean().item() * images.size(0)  # Warning 1
        
        probs = torch.sigmoid(seg_logits)
        all_preds.append(probs.cpu())
        all_targets.append(masks.cpu())
    
    n = len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = running_loss / n
    metrics['bce'] = running_bce / n
    metrics['mbl'] = running_mbl / n
    metrics['mask_conf_mean'] = running_conf / n  # Warning 1
    
    return metrics

# %% [markdown]
# ## 4. Main Training Execution

# %%
def main():
    """Full CBD-Net training pipeline."""
    
    # ═══════════════════════════════════════════════════════
    #  SETUP
    # ═══════════════════════════════════════════════════════
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # ═══════════════════════════════════════════════════════
    #  DATA
    # ═══════════════════════════════════════════════════════
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # ═══════════════════════════════════════════════════════
    #  COMPUTE pos_weight FOR CLASS IMBALANCE (Warning 3 fix)
    # ═══════════════════════════════════════════════════════
    # Oil spill datasets are heavily imbalanced (5-15% oil vs 85-95% water).
    # Bug fix: Scanning directory directly with glob (SOSDataset has no .pairs attribute)
    print("\n  Computing class balance (pos_weight) via directory scan...")
    mask_files = glob.glob(os.path.join(SOS_TRAIN_DIR, "*_mask.png"))
    import random
    random.shuffle(mask_files)  # Ensure representative sample
    
    total_oil_pixels = 0
    total_pixels = 0
    
    # Scan first 500 masks for a representative sample (faster than scanning all 3000+)
    scan_limit = 500
    for i, mf in enumerate(mask_files[:scan_limit]):
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            total_oil_pixels += int((mask > 127).sum())
            total_pixels += mask.size
        if (i+1) % 100 == 0:
            print(f"    Scanned {i+1}/{min(len(mask_files), scan_limit)} masks...")
    
    pos_ratio = total_oil_pixels / (total_pixels + 1e-7)
    pos_weight_val = float(np.clip((1.0 - pos_ratio) / (pos_ratio + 1e-7), 1.0, 25.0))
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    print(f"  Oil pixel ratio:  {pos_ratio:.4f} ({pos_ratio*100:.1f}%)")
    print(f"  pos_weight:       {pos_weight_val:.2f}")
    
    # ═══════════════════════════════════════════════════════
    #  MODEL
    # ═══════════════════════════════════════════════════════
    print("\n  Initializing CBD-Net SAR Expert...")
    model = CBDNetSAR(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_attention=DECODER_ATTENTION
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Verify model output shapes — Bug 1 fix: always 4 values
    with torch.no_grad():
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        seg, bounds, feats, conf = model(dummy)
        print(f"  Seg output:       {seg.shape}")
        print(f"  Encoder features: {feats.shape} -> Ready for SQCA")
        print(f"  Boundary scales:  {len(bounds)}")
        print(f"  Mask_conf:        {conf.shape}")
    
    # ═══════════════════════════════════════════════════════
    #  LOSS, OPTIMIZER, SCHEDULER
    # ═══════════════════════════════════════════════════════
    criterion = CBDNetLoss(
        lambda_boundary=LAMBDA_BOUNDARY,
        scale_weights=MBL_SCALE_WEIGHTS,
        pos_weight=pos_weight_tensor  # Warning 3 fix: class imbalance correction
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    print(f"\n  Loss:      BCE(pw={pos_weight_val:.1f}) + {LAMBDA_BOUNDARY}*MBL")
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: CosineAnnealing (T_max={T_MAX})")
    print(f"  AMP:       {'Enabled' if USE_AMP else 'Disabled'}")
    print(f"  Grad Clip: {GRAD_CLIP}")
    
    # ═══════════════════════════════════════════════════════
    #  CSV LOGGER
    # ═══════════════════════════════════════════════════════
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOGS_DIR, f"train_log_{timestamp}.csv")
    csv_fields = [
        'epoch', 'lr',
        'train_loss', 'train_bce', 'train_mbl', 'train_miou', 'train_dice',
        'train_precision', 'train_recall', 'train_far', 'train_oa',
        'train_mask_conf_mean',  # Warning 1 fix: track mask_conf for decoder health
        'val_loss', 'val_bce', 'val_mbl', 'val_miou', 'val_dice',
        'val_precision', 'val_recall', 'val_far', 'val_oa',
        'val_mask_conf_mean',    # Warning 1 fix
        'epoch_time'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
    
    print(f"\n  📊 Logging to: {csv_path}")
    
    # ═══════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═══════════════════════════════════════════════════════
    best_val_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n{'='*90}")
    print(f"  🔥 TRAINING STARTED — {NUM_EPOCHS} epochs")
    print(f"{'='*90}")
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'BCE':>8} | {'MBL':>8} | {'mIoU':>6} | "
          f"{'Dice':>6} | {'Prec':>6} | {'Rec':>6} | {'FAR':>6} | {'OA':>6} | {'Time':>5} | Status")
    print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | "
          f"{'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*5} | ------")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, GRAD_CLIP
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Check for best model
        status = ""
        if val_metrics['miou'] > best_val_miou:
            best_val_miou = val_metrics['miou']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': best_val_miou,
                'val_metrics': val_metrics,
                'config': {
                    'encoder': ENCODER_NAME,
                    'lambda_boundary': LAMBDA_BOUNDARY,
                    'img_size': IMG_SIZE,
                    'lr': LEARNING_RATE
                }
            }
            torch.save(save_dict, os.path.join(WEIGHTS_DIR, "best_sar_expert.pth"))
            status = f"✅ Best (mIoU={best_val_miou:.4f})"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                status = "⏹️ Early Stop"
            
        # Also save last model for resume capability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(WEIGHTS_DIR, "last_sar_expert.pth"))
        
        # Print epoch summary (validation metrics)
        print(f"  {epoch:>5} | {val_metrics['loss']:>8.4f} | {val_metrics['bce']:>8.4f} | "
              f"{val_metrics['mbl']:>8.4f} | {val_metrics['miou']:>.4f} | "
              f"{val_metrics['dice']:>.4f} | {val_metrics['precision']:>.4f} | "
              f"{val_metrics['recall']:>.4f} | {val_metrics['far']:>.4f} | "
              f"{val_metrics['oa']:>.4f} | {epoch_time:>4.0f}s | {status}")
        
        # Log to CSV
        row = {
            'epoch': epoch, 'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'train_bce': train_metrics['bce'],
            'train_mbl': train_metrics['mbl'],
            'train_miou': train_metrics['miou'],
            'train_dice': train_metrics['dice'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_far': train_metrics['far'],
            'train_oa': train_metrics['oa'],
            'train_mask_conf_mean': train_metrics['mask_conf_mean'],  # Warning 1
            'val_loss': val_metrics['loss'],
            'val_bce': val_metrics['bce'],
            'val_mbl': val_metrics['mbl'],
            'val_miou': val_metrics['miou'],
            'val_dice': val_metrics['dice'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_far': val_metrics['far'],
            'val_oa': val_metrics['oa'],
            'val_mask_conf_mean': val_metrics['mask_conf_mean'],      # Warning 1
            'epoch_time': epoch_time
        }
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)
        
        # Early stopping check
        if patience_counter >= PATIENCE:
            print(f"\n  ⏹️ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break
    
    # ═══════════════════════════════════════════════════════
    #  TRAINING COMPLETE
    # ═══════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"  🏁 TRAINING COMPLETE")
    print(f"{'='*90}")
    print(f"  Total Time:     {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"  Best Epoch:     {best_epoch}")
    print(f"  Best Val mIoU:  {best_val_miou:.4f}")
    print(f"  Weights:        {os.path.join(WEIGHTS_DIR, 'best_sar_expert.pth')}")
    print(f"  Training Log:   {csv_path}")
    print(f"{'='*90}")
    
    return model, best_val_miou

# %%
# Run training
if __name__ == "__main__":
    model, best_miou = main()
