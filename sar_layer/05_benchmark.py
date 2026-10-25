# %% [markdown]
# # 📊 Stage 1: SAR Expert — Premium Benchmarking Suite
# **Deep-SAR Oil Spill (SOS) Dataset Evaluation with 8-Fold TTA**
#
# Aligned with: Zhu et al., "Oil Spill Contextual and Boundary-supervised Detection Network", IEEE TGRS 2022.
#
# Metrics tracked: 
# - **mIoU** (Primary Benchmark)
# - **F1-Score / Dice** (Paper Metric)
# - **Precision / Recall** (Paper Table III)
# - **FAR (False Alarm Rate)** (Key project claim)
# - **OA (Overall Accuracy)**

# %%
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\sar_layer"
sys.path.insert(0, PROJECT_DIR)

# Import our custom architecture and logic
# exec logic ensures latest code is used even if module reloading is finicky
SKIP_SELF_TEST = True
exec(open(os.path.join(PROJECT_DIR, "01_config.py"), encoding='utf-8').read())
exec(open(os.path.join(PROJECT_DIR, "02_dataset.py"), encoding='utf-8').read())
exec(open(os.path.join(PROJECT_DIR, "03_model.py"), encoding='utf-8').read())

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(PROJECT_DIR, "weights", "best_sar_expert.pth")
PALSAR_DIR   = r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\SOS_dataset\test\palsar"
SENTINEL_DIR = r"c:\Users\geeth\OneDrive\Pictures\random\OIL_SPILL\SOS_dataset\test\sentinel"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Table III paper results for direct comparison
PAPER_BENCHMARKS = {
    'mIoU': 0.8342,
    'F1':   0.8787,
    'Prec': 0.8675,
    'Rec':  0.8420
}

# ─── 1. TTA ENGINE (8-FOLD ENSEMBLE) ──────────────────────────────────────────
def apply_tta(image_tensor, model):
    """
    Perform 8-fold Test-Time Augmentation:
    - 4 Rotations (0, 90, 180, 270)
    - 2 Flips (Original, Vertical)
    - Total = 8 predictions
    """
    model.eval()
    batch_preds = []
    
    with torch.no_grad():
        # Variation 1-4: Rotations
        for k in range(4):
            x = torch.rot90(image_tensor, k, [2, 3])
            # Variation 5-8: Flips of the rotations
            for flip in [False, True]:
                if flip:
                    x_in = torch.flip(x, [2]) # Vertical flip
                else:
                    x_in = x
                
                # Inference
                seg_logits, _, _, _ = model(x_in)
                pred = torch.sigmoid(seg_logits)
                
                # Undo Augmentations
                if flip:
                    pred = torch.flip(pred, [2])
                pred = torch.rot90(pred, -k, [2, 3])
                
                batch_preds.append(pred)
    
    # Average 8 variations for high-confidence output
    mean_pred = torch.stack(batch_preds).mean(dim=0)
    return mean_pred

# ─── 2. METRIC CALCULATOR ─────────────────────────────────────────────────────
class BenchmarkMetrics:
    def __init__(self):
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        
    def add_batch(self, pred, mask):
        p = (pred > 0.5).float().view(-1)
        g = (mask > 0.5).float().view(-1)
        self.tp += (p * g).sum().item()
        self.fp += (p * (1 - g)).sum().item()
        self.tn += ((1 - p) * (1 - g)).sum().item()
        self.fn += ((1 - p) * g).sum().item()
        
    def compute(self):
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou_oil = self.tp / (self.tp + self.fp + self.fn + eps)
        iou_water = self.tn / (self.tn + self.fn + self.fp + eps)
        miou = (iou_oil + iou_water) / 2.0
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        far = self.fp / (self.fp + self.tn + eps)
        
        return {
            'mIoU': miou, 'F1': f1, 'Prec': precision, 'Rec': recall,
            'OA': oa, 'FAR': far, 'IoU_oil': iou_oil
        }

# ─── 3. EVALUATION RUNNER ─────────────────────────────────────────────────────
def run_benchmark(model, dataset_dir, split_name):
    print(f"\n🚀 Benchmarking {split_name} Split...")
    
    # Find image pairs (sat/gt)
    sat_dir = os.path.join(dataset_dir, "sat")
    gt_dir = os.path.join(dataset_dir, "gt")
    sat_files = sorted(os.listdir(sat_dir))
    
    metrics = BenchmarkMetrics()
    
    for filename in tqdm(sat_files, desc=f"  Evaluating {split_name}"):
        # Load image
        img_path = os.path.join(sat_dir, filename)
        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_raw = cv2.resize(img_raw, (256, 256))
        
        # Load mask
        mask_name = filename.replace("_sat.jpg", "_mask.png") # Handle both SOS conventions
        mask_path = os.path.join(gt_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(gt_dir, filename.replace(".jpg", ".png"))
            
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Preprocess
        img_norm = preprocess_sar(img_raw)
        img_3ch = np.stack([img_norm]*3, axis=0)
        img_tensor = torch.from_numpy(img_3ch).unsqueeze(0).to(DEVICE).float()
        
        # Inference with TTA
        pred = apply_tta(img_tensor, model)
        
        # Update metrics
        mask_tensor = (torch.from_numpy(mask_raw) > 127).float()
        metrics.add_batch(pred.cpu(), mask_tensor)
        
    return metrics.compute()

# ─── 4. MAIN ──────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*80}")
    print(f"  🛰️  SAR EXPERT BENCHMARK — PAPER ALIGNMENT SUITE")
    print(f"{'='*80}")
    print(f"  Device:     {DEVICE}")
    print(f"  Weights:    {WEIGHTS_PATH}")
    print(f"  TTA:        8-Fold Orientation-Invariant Inference")
    print(f"{'='*80}\n")
    
    # 1. Load Model
    model = CBDNetSAR().to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ Best model loaded (Epoch {checkpoint['epoch']})")
    else:
        print(f"  ❌ Error: Weights not found at {WEIGHTS_PATH}")
        return

    results = {}
    
    # 2. Run Evaluatons
    if os.path.exists(PALSAR_DIR):
        results['PALSAR (Mexico)'] = run_benchmark(model, PALSAR_DIR, "PALSAR")
    if os.path.exists(SENTINEL_DIR):
        results['Sentinel-1 (Persian)'] = run_benchmark(model, SENTINEL_DIR, "Sentinel-1")

    # 3. Side-by-Side Comparison
    print("\n\n" + "-"*92)
    print(f"  {'DATASET SPLIT':<25} | {'mIoU':>8} | {'F1':>8} | {'Prec':>8} | {'Rec':>8} | {'FAR':>8}")
    print("-"*92)
    
    # Print Paper Benchmark Row
    print(f"  {'[PAPER] Zhu et al. (2022)':<25} | {PAPER_BENCHMARKS['mIoU']*100:>7.2f}% | "
          f"{PAPER_BENCHMARKS['F1']*100:>7.2f}% | {PAPER_BENCHMARKS['Prec']*100:>7.2f}% | "
          f"{PAPER_BENCHMARKS['Rec']*100:>7.2f}% | {'—':>8}")
    
    # Print Our Results Rows
    for split, m in results.items():
        print(f"  {'[OURS]  ' + split:<25} | {m['mIoU']*100:>7.2f}% | "
              f"{m['F1']*100:>7.2f}% | {m['Prec']*100:>7.2f}% | "
              f"{m['Rec']*100:>7.2f}% | {m['FAR']*100:>7.2f}%")
    
    print("-"*92)
    
    # Delta highlighting
    print("\n  🎯 PERFORMANCE DELTA VS PAPER:")
    for split, m in results.items():
        dm = (m['mIoU'] - PAPER_BENCHMARKS['mIoU']) * 100
        sign = "+" if dm >= 0 else ""
        color = "✅ BEATS PAPER" if dm >= 0 else "❌ BELOW PAPER"
        print(f"    - {split}: {sign}{dm:.2f}% mIoU vs Benchmark — {color}")
    print("\n")

if __name__ == "__main__":
    main()
