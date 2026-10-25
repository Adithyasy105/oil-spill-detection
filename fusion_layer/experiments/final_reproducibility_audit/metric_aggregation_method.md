# Metric Aggregation Method Documentation

## 1. Exact Aggregation Formulation
In this research, fold-level F1 and IoU metrics are calculated via:

$$\text{Method A: Global Micro-Aggregation across Held-Out Fold Scenes}$$

Specifically, for each fold $k \in \{1, \dots, 6\}$:
1. All pixels belonging to the 3 held-out test scenes ($3 \times 5000 = 15,000$ test pixels) are concatenated into a single fold-level test prediction vector $\hat{\mathbf{y}}_k$ and ground-truth vector $\mathbf{y}_k$.
2. The classification metrics (F1, IoU, Accuracy, Precision, Recall) are computed once over the entire concatenated 15,000 pixel set using the validation-selected decision threshold $t^*_k$.
3. Mean benchmark scores across the 6 folds represent the unweighted macro-average across folds:
   $$\bar{F}_1 = \frac{1}{6} \sum_{k=1}^6 F_1(\mathbf{y}_k, \hat{\mathbf{y}}_k)$$

## 2. Why Global Aggregation was Selected
- Each satellite scene in the benchmark has an identical sample size ($5,000$ balanced patches/scene).
- Global aggregation prevents extreme distortion from scenes with low positive oil presence while providing an exact representation of deployment segmentation performance across the entire acquisition pass.
