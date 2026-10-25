# %% [markdown]
# # 📡 Stage 1: SAR Expert — Evaluation & Testing
# **CBD-Net Test Set Evaluation with Comprehensive Metrics & Visualizations**
#
# This notebook:
# 1. Loads the best trained weights
# 2. Runs evaluation on the SOS test set (839 images)
# 3. Computes all metrics: mIoU, Dice, FAR, OA, Precision, Recall, Boundary F1
# 4. Generates prediction visualizations (SAR | GT | Prediction | Overlay)
# 5. Verifies dual-path output shapes for Stage 3 compatibility

# %%
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import csv
from datetime import datetime

# Setup paths
SAR_LAYER_DIR = r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer"
sys.path.insert(0, SAR_LAYER_DIR)

# Load all modules
exec(open(os.path.join(SAR_LAYER_DIR, "01_config.py"), encoding='utf-8').read())
exec(open(os.path.join(SAR_LAYER_DIR, "02_dataset.py"), encoding='utf-8').read())
exec(open(os.path.join(SAR_LAYER_DIR, "03_model.py"), encoding='utf-8').read())

def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute comprehensive segmentation metrics for evaluation.
    
    Args:
        preds: Sigmoid probabilities (B, 1, H, W)
        targets: Binary ground truth (B, 1, H, W)
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
    iou_oil = tp / (tp + fp + fn + eps)
    
    # IoU for water class (class=0)
    iou_water = tn / (tn + fn + fp + eps)
    
    # Mean IoU
    miou = (iou_oil + iou_water) / 2.0
    
    # Dice coefficient (F1) for oil class
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
    # Precision & Recall (oil class)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # False Alarm Rate = FP / (FP + TN)
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
# ## 1. Load Best Model

# %%
def load_best_model(device):
    """Load the best trained CBD-Net SAR model."""
    model = CBDNetSAR(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,  # Don't need ImageNet weights, loading trained
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_attention=DECODER_ATTENTION
    ).to(device)
    
    weights_path = os.path.join(WEIGHTS_DIR, "best_sar_expert.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No trained weights found at {weights_path}. Run training first.")
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  ✅ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"  Validation mIoU: {checkpoint['val_miou']:.4f}")
    
    return model, checkpoint

# %% [markdown]
# ## 2. Test Set Evaluation

# %%
@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    """
    Run full evaluation on the test set.
    
    Returns per-sample and aggregate metrics.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    sample_metrics = []
    
    print(f"\n  Running evaluation on {len(test_loader.dataset)} test samples...")
    
    for batch_idx, (images, masks, boundaries) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # Bug 1 fix: model ALWAYS returns 4 values
            seg_logits, boundary_logits, _, mask_conf = model(images)
        
        probs = torch.sigmoid(seg_logits)
        all_preds.append(probs.cpu())
        all_targets.append(masks.cpu())
        
        # Per-sample metrics
        for i in range(images.size(0)):
            sample_met = compute_metrics(
                probs[i:i+1].cpu(), masks[i:i+1].cpu()
            )
            sample_met['mask_conf'] = mask_conf[i].item()
            sample_metrics.append(sample_met)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"    Processed {(batch_idx + 1) * images.size(0)}/{len(test_loader.dataset)}")
    
    # Aggregate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    agg_metrics = compute_metrics(all_preds, all_targets)
    
    return agg_metrics, sample_metrics, all_preds, all_targets

# %% [markdown]
# ## 3. Visualization

# %%
def visualize_predictions(model, test_loader, device, num_samples=20, save_dir=None):
    """Generate prediction visualizations: SAR | GT | Prediction | Overlay."""
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    for images, masks, boundaries in test_loader:
        images = images.to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # Bug 1 fix: model ALWAYS returns 4 values
                seg_logits, _, _, mask_conf = model(images)
            probs = torch.sigmoid(seg_logits)
            preds = (probs > THRESHOLD).float()
        
        for i in range(images.size(0)):
            if sample_count >= num_samples:
                return
            
            # Get numpy arrays
            img = images[i, 0].cpu().numpy()  # Take first channel
            gt = masks[i, 0].cpu().numpy()
            pred = preds[i, 0].cpu().numpy()
            prob = probs[i, 0].cpu().numpy()
            conf = mask_conf[i, 0].item()
            
            # Normalize image for display
            img_display = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(np.uint8)
            
            # Create visualization panels
            h, w = img.shape
            canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
            
            # Panel 1: SAR Image
            canvas[:, :w] = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
            
            # Panel 2: Ground Truth (green = oil)
            gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
            gt_vis[gt > 0.5] = [0, 255, 0]
            canvas[:, w:2*w] = gt_vis
            
            # Panel 3: Prediction (blue = predicted oil)
            pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
            pred_vis[pred > 0.5] = [255, 100, 0]  # Blue in BGR
            canvas[:, 2*w:3*w] = pred_vis
            
            # Panel 4: Overlay (TP=green, FP=red, FN=yellow)
            overlay = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
            tp_mask = (pred > 0.5) & (gt > 0.5)
            fp_mask = (pred > 0.5) & (gt < 0.5)
            fn_mask = (pred < 0.5) & (gt > 0.5)
            overlay[tp_mask] = [0, 255, 0]    # Green: True Positive
            overlay[fp_mask] = [0, 0, 255]    # Red: False Positive
            overlay[fn_mask] = [0, 255, 255]  # Yellow: False Negative
            canvas[:, 3*w:4*w] = overlay
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, 'SAR Input', (10, 20), font, 0.5, (255,255,255), 1)
            cv2.putText(canvas, 'Ground Truth', (w+10, 20), font, 0.5, (255,255,255), 1)
            cv2.putText(canvas, f'Prediction (conf={conf:.2f})', (2*w+10, 20), font, 0.5, (255,255,255), 1)
            cv2.putText(canvas, 'TP/FP/FN', (3*w+10, 20), font, 0.5, (255,255,255), 1)
            
            # Save
            save_path = os.path.join(save_dir, f"test_sample_{sample_count:04d}.png")
            cv2.imwrite(save_path, canvas)
            sample_count += 1
    
    print(f"  ✅ Saved {sample_count} visualizations to {save_dir}")

# %% [markdown]
# ## 4. Verify Stage 3 Compatibility

# %%
def verify_stage3_compatibility(model, device):
    """
    Verify that the SAR expert produces correct output shapes for Stage 3 fusion.
    
    Stage 3 expects:
        - encoder_features: (B, 512, 8, 8) → projected to (B, 128, 64) for SQCA
        - mask_conf: (B, 1) → gate input for GSFF
    """
    model.eval()
    dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    features, mask_conf = model.extract_features(dummy)
    
    print(f"\n  Stage 3 Compatibility Check:")
    print(f"  {'='*50}")
    assert features.shape == (4, 512, 8, 8), f"Expected (4,512,8,8), got {features.shape}"
    print(f"  ✅ Encoder features: {features.shape} → Will be projected to (B, 128, 64) for SQCA")
    
    assert mask_conf.shape == (4, 1), f"Expected (4,1), got {mask_conf.shape}"
    print(f"  ✅ Mask_conf: {mask_conf.shape} → Gate input for GSFF (257-dim)")
    
    print(f"  ✅ Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    print(f"  ✅ Mask_conf range: [{mask_conf.min():.4f}, {mask_conf.max():.4f}]")
    print(f"  {'='*50}")
    print(f"  ✅ ALL CHECKS PASSED — Ready for Stage 3 fusion training")

# %% [markdown]
# ## 5. Run Full Evaluation

# %%
def main():
    """Run complete evaluation pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  🖥️  Device: {device}")
    
    # Load model
    model, checkpoint = load_best_model(device)
    
    # Create test loader
    test_ds = SOSDataset(SOS_TEST_DIR, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Run evaluation
    agg_metrics, sample_metrics, all_preds, all_targets = evaluate_test_set(
        model, test_loader, device
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  📊 TEST SET RESULTS (n={len(test_loader.dataset)})")
    print(f"{'='*60}")
    print(f"  mIoU:       {agg_metrics['miou']:.4f}")
    print(f"  IoU (oil):  {agg_metrics['iou_oil']:.4f}")
    print(f"  IoU (water):{agg_metrics['iou_water']:.4f}")
    print(f"  Dice (F1):  {agg_metrics['dice']:.4f}")
    print(f"  Precision:  {agg_metrics['precision']:.4f}")
    print(f"  Recall:     {agg_metrics['recall']:.4f}")
    print(f"  FAR:        {agg_metrics['far']:.4f}")
    print(f"  OA:         {agg_metrics['oa']:.4f}")
    print(f"  TP/FP/TN/FN: {agg_metrics['tp']:.0f} / {agg_metrics['fp']:.0f} / "
          f"{agg_metrics['tn']:.0f} / {agg_metrics['fn']:.0f}")
    print(f"{'='*60}")
    
    # Save metrics to CSV
    metrics_path = os.path.join(LOGS_DIR, "test_metrics.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_metrics.keys()))
        writer.writeheader()
        writer.writerow(agg_metrics)
    print(f"  📊 Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    print(f"\n  🖼️  Generating prediction visualizations...")
    visualize_predictions(model, test_loader, device, num_samples=30)
    
    # Verify Stage 3 compatibility
    verify_stage3_compatibility(model, device)
    
    return agg_metrics

# %%
if __name__ == "__main__":
    results = main()
