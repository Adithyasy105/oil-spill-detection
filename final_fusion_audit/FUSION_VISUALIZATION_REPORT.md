# CAFE-Fusion v2: Multimodal Qualitative Visualization Audit Report

## 1. Overview
This document summarizes the authoritative, high-resolution (300 DPI) qualitative visual panels generated for all 18 benchmark scenes (**GM01 to GM18**) using the **true native 2D spatial resolutions** from the OSD dataset (`new_dataset/OSD/`).

Each figure incorporates:
1. **Ground Truth Mask ($y \in \{0, 1\}$)**: Real binary marine oil spill ground truth at native spatial resolution.
2. **SAR Expert Probability Field ($p_{\text{SAR}}$)**: Inferred using the frozen **CBD-Net** backbone (ResNet-18 + scSE attention + multi-scale boundary supervision), resized from $256 \times 256$ to native $(H \times W)$.
3. **HSI Expert Probability Field ($p_{\text{HSI}}$)**: Inferred via dense $5 \times 5$ patch-based spectral scanning through the frozen **HMOSDNet** (3D-CNN + MAM attention, 25 PCA bands).
4. **CAFE-Fusion v2 Probability Field ($p_{\text{fused}}$)**: Generated via logit-space convex combination $z_{\text{fused}} = (1-g)z_{\text{HSI}} + g z_{\text{SAR}}$ using the 11-D registration-free gate feature vector $u$ and the fold-matched CAFE-Fusion checkpoint.
5. **Fusion Final Binary Prediction ($\hat{y} \in \{0, 1\}$)**: Hard classification thresholded at the optimal validation-selected decision cutoff $\tau_{\text{fold}}$.

---

## 2. Per-Scene Quantitative & Visual Specification

| Scene ID | Native Spatial Grid $(H \times W)$ | Split / Fold | SAR Expert $F_1$ | HSI Expert $F_1$ | CAFE-Fusion $F_1$ | Optimal $\tau$ | Mean Gate $\bar{g}$ | Output Image |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **GM01** | $1243 \times 684$ | Fold 1 | 0.8296 | 0.9545 | **0.9588** | 0.065 | 0.1222 | `final_fusion_audit/figures/fusion/GM01.png` |
| **GM02** | $1786 \times 699$ | Fold 1 | 0.9019 | 0.9706 | **0.9733** | 0.065 | 0.1216 | `final_fusion_audit/figures/fusion/GM02.png` |
| **GM03** | $1386 \times 690$ | Fold 1 | 0.8363 | 0.9357 | **0.9361** | 0.065 | 0.1212 | `final_fusion_audit/figures/fusion/GM03.png` |
| **GM04** | $1466 \times 676$ | Fold 2 | 0.8623 | 0.9536 | **0.9553** | 0.055 | 0.1093 | `final_fusion_audit/figures/fusion/GM04.png` |
| **GM05** | $2085 \times 682$ | Fold 2 | 0.7133 | 0.9410 | **0.9426** | 0.055 | 0.1469 | `final_fusion_audit/figures/fusion/GM05.png` |
| **GM06** | $2088 \times 691$ | Fold 2 | 0.7456 | 0.9656 | **0.9657** | 0.055 | 0.1570 | `final_fusion_audit/figures/fusion/GM06.png` |
| **GM07** | $1965 \times 492$ | Fold 3 | 0.8288 | 0.9442 | **0.9508** | 0.265 | 0.2405 | `final_fusion_audit/figures/fusion/GM07.png` |
| **GM08** | $1532 \times 567$ | Fold 3 | 0.7539 | 0.9568 | **0.9359** | 0.265 | 0.2804 | `final_fusion_audit/figures/fusion/GM08.png` |
| **GM09** | $1569 \times 517$ | Fold 3 | 0.8086 | 0.9517 | **0.9320** | 0.265 | 0.2637 | `final_fusion_audit/figures/fusion/GM09.png` |
| **GM10** | $1084 \times 680$ | Fold 4 | 0.8203 | 0.7989 | **0.8034** | 0.020 | 0.1177 | `final_fusion_audit/figures/fusion/GM10.png` |
| **GM11** | $1185 \times 682$ | Fold 4 | 0.8458 | 0.9280 | **0.9270** | 0.020 | 0.1198 | `final_fusion_audit/figures/fusion/GM11.png` |
| **GM12** | $842 \times 640$ | Fold 4 | 0.8680 | 0.9695 | **0.9697** | 0.020 | 0.1204 | `final_fusion_audit/figures/fusion/GM12.png` |
| **GM13** | $836 \times 572$ | Fold 5 | 0.9046 | 0.9657 | **0.9657** | 0.150 | 0.2081 | `final_fusion_audit/figures/fusion/GM13.png` |
| **GM14** | $1342 \times 527$ | Fold 5 | 0.7984 | 0.9507 | **0.9470** | 0.150 | 0.2538 | `final_fusion_audit/figures/fusion/GM14.png` |
| **GM15** | $1260 \times 523$ | Fold 5 | 0.8096 | 0.8262 | **0.8754** | 0.150 | 0.2604 | `final_fusion_audit/figures/fusion/GM15.png` |
| **GM16** | $1033 \times 437$ | Fold 6 | 0.8423 | 0.9080 | **0.9246** | 0.040 | 0.1188 | `final_fusion_audit/figures/fusion/GM16.png` |
| **GM17** | $600 \times 400$ | Fold 6 | 0.9316 | 0.7491 | **0.7628** | 0.040 | 0.1196 | `final_fusion_audit/figures/fusion/GM17.png` |
| **GM18** | $1175 \times 563$ | Fold 6 | 0.9271 | 0.9745 | **0.9750** | 0.040 | 0.1191 | `final_fusion_audit/figures/fusion/GM18.png` |

---

## 3. Storage Locations
- **Audit Visual Figures Directory**: `final_fusion_audit/figures/fusion/`
- **Source Generator Script**: `generate_fusion_audit_visualizations.py`
- **Experimental Summary**: `fusion_layer/experiments/fusion_visualization_summary.json`
- **Authoritative Checkpoints**: `fusion_layer/experiments/cafe_fusion_v2/primary_results/`
