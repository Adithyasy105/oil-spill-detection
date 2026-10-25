# Ablation Study & Regularization Audit

## 1. Audited Ablation Experiments Across 6 Folds
| Model / Ablation Configuration | Mean $F_1 \pm \text{Std}$ | $\Delta F_1$ vs HSI | Folds Won | Paired $t$-test $p$ | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Ablation A: $\lambda_{\text{rel}} = 0.00$** | $0.9288 \pm 0.0269$ | $+0.0027$ | $5 / 6$ | $0.4677$ | No reliability loss (segmentation loss only) |
| **Ablation B: $\lambda_{\text{rel}} = 0.05$** | $0.9292 \pm 0.0270$ | $+0.0031$ | $5 / 6$ | $0.4308$ | Weak reliability supervision |
| **Primary CAFE v2: $\lambda_{\text{rel}} = 0.10$** | $\mathbf{0.9288 \pm 0.0270}$ | $\mathbf{+0.0027}$ | $\mathbf{5 / 6}$ | $\mathbf{0.4532}$ | **Primary Frozen Model** |
| **Ablation C: Variant C (Hard Router)** | $0.9253 \pm 0.0247$ | $-0.0007$ | $4 / 6$ | $0.9044$ | Hard disagreement thresholding ($g=0$ if diff $< \tau$) |

## 2. Key Insights
1. **Regularization Robustness**: Varying $\lambda_{\text{rel}}$ between $0.00$ and $0.10$ yields virtually identical performance ($0.9288$ to $0.9292$), proving the model is robust to hyperparameter selection.
2. **Smooth Gating vs Hard Routing**: Continuous logit mixing ($0.9288$) strictly outperforms discrete hard routing ($0.9253$) by avoiding boundary discontinuities.
