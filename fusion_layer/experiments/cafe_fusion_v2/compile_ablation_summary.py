"""Compile ablation summary table across all finished ablation experiments."""

import csv
import json
from pathlib import Path
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent
ABLATIONS_DIR = EXPERIMENT_ROOT / "ablations"
PRIMARY_DIR = EXPERIMENT_ROOT / "primary_results"

RUNS = [
    {
        "name": "CAFE v2 (Primary)",
        "path": PRIMARY_DIR,
        "fusion_space": "Logit",
        "lambda_rel": 0.10,
        "supervision": "Hard Masked",
        "hsi_reg": "0.01",
    },
    {
        "name": "Ablation A: No Reliability Sup",
        "path": ABLATIONS_DIR / "no_rel_sup",
        "fusion_space": "Logit",
        "lambda_rel": 0.00,
        "supervision": "None",
        "hsi_reg": "0.01",
    },
    {
        "name": "Ablation B: Weak Reliability Sup",
        "path": ABLATIONS_DIR / "lambda_0.05",
        "fusion_space": "Logit",
        "lambda_rel": 0.05,
        "supervision": "Hard Masked",
        "hsi_reg": "0.01",
    },
    {
        "name": "Ablation D: Strong Reliability Sup",
        "path": ABLATIONS_DIR / "lambda_0.20",
        "fusion_space": "Logit",
        "lambda_rel": 0.20,
        "supervision": "Hard Masked",
        "hsi_reg": "0.01",
    },
    {
        "name": "Ablation E: Probability-Space",
        "path": ABLATIONS_DIR / "prob_space",
        "fusion_space": "Probability",
        "lambda_rel": 0.10,
        "supervision": "Hard Masked",
        "hsi_reg": "0.01",
    },
    {
        "name": "Ablation F: No HSI Reg",
        "path": ABLATIONS_DIR / "no_hsi_reg",
        "fusion_space": "Logit",
        "lambda_rel": 0.10,
        "supervision": "Hard Masked",
        "hsi_reg": "0.00",
    },
    {
        "name": "Ablation G: Soft Supervision",
        "path": ABLATIONS_DIR / "soft_targets",
        "fusion_space": "Logit",
        "lambda_rel": 0.10,
        "supervision": "Soft Quality",
        "hsi_reg": "0.01",
    },
]


def compile_summary():
    rows = []
    for run in RUNS:
        summary_file = run["path"] / "summary.json"
        if not summary_file.exists():
            print(f"Skipping {run['name']}: not yet finished ({summary_file})")
            continue

        with open(summary_file, "r") as f:
            d = json.load(f)

        folds = d["folds"]
        f1_fusion = np.array([f["f1_fusion"] for f in folds])
        f1_hsi = np.array([f["f1_hsi"] for f in folds])
        f1_oracle = np.array([f["f1_oracle"] for f in folds])
        
        gate_vals = [f["gate_diagnostics"]["mean_g"] for f in folds if "gate_diagnostics" in f]
        gate_aucs = [f["gate_diagnostics"]["gate_roc_auc"] for f in folds if "gate_diagnostics" in f and not np.isnan(f["gate_diagnostics"]["gate_roc_auc"])]

        mean_f1 = float(np.mean(f1_fusion))
        std_f1 = float(np.std(f1_fusion, ddof=1))
        
        # Load iou from per-fold results if available
        # or calculate delta
        delta_vs_hsi = mean_f1 - float(np.mean(f1_hsi))
        folds_beating = int((f1_fusion > f1_hsi).sum())
        
        oracle_gap = float(np.mean(f1_oracle) - np.mean(f1_hsi))
        gap_recovered = (delta_vs_hsi / (oracle_gap + 1e-8)) * 100.0

        row = {
            "Experiment": run["name"],
            "Fusion Space": run["fusion_space"],
            "lambda_rel": run["lambda_rel"],
            "Hard/Soft/None Supervision": run["supervision"],
            "HSI Regularization": run["hsi_reg"],
            "Mean_F1": f"{mean_f1:.4f}",
            "Std_F1": f"{std_f1:.4f}",
            "Delta_vs_HSI": f"{delta_vs_hsi:+.4f}",
            "Folds_Beating_HSI": f"{folds_beating}/{len(folds)}",
            "Mean_Gate": f"{np.mean(gate_vals):.4f}" if gate_vals else "N/A",
            "Mean_Gate_AUC": f"{np.mean(gate_aucs):.4f}" if gate_aucs else "N/A",
            "Oracle_Gap_Recovery": f"{gap_recovered:.2f}%",
        }
        rows.append(row)

    out_csv = EXPERIMENT_ROOT / "ablation_summary.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Compiled ablation summary table to: {out_csv}")
        print("\nCurrent Ablation Summary Table:")
        for r in rows:
            print(f"  {r['Experiment']:<35} | F1={r['Mean_F1']} +/- {r['Std_F1']} | Delta={r['Delta_vs_HSI']} | Beats={r['Folds_Beating_HSI']}")


if __name__ == "__main__":
    compile_summary()
