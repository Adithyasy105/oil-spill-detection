"""CAFE Fusion v2 — Complete Ablation Suite Runner.

Executes all 6 required ablation experiments into isolated subdirectories:
A. No reliability supervision (lambda = 0.0)
B. Weak reliability supervision (lambda = 0.05)
D. Strong reliability supervision (lambda = 0.20)
E. Probability-space fusion (fusion_mode = prob)
F. No HSI regularization (eps_hsi_reg = 0.0)
G. Soft reliability targets (supervision_mode = soft)
"""

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
EXPERIMENT_ROOT = Path(__file__).resolve().parent
SCRIPT = EXPERIMENT_ROOT / "cafe_fusion.py"
ABLATIONS_DIR = EXPERIMENT_ROOT / "ablations"
ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_CONFIGS = [
    {
        "name": "A_no_reliability_sup",
        "output_dir": ABLATIONS_DIR / "no_rel_sup",
        "args": ["--lambda-rel", "0.0"],
    },
    {
        "name": "B_weak_reliability_sup",
        "output_dir": ABLATIONS_DIR / "lambda_0.05",
        "args": ["--lambda-rel", "0.05"],
    },
    {
        "name": "D_strong_reliability_sup",
        "output_dir": ABLATIONS_DIR / "lambda_0.20",
        "args": ["--lambda-rel", "0.20"],
    },
    {
        "name": "E_prob_space_fusion",
        "output_dir": ABLATIONS_DIR / "prob_space",
        "args": ["--fusion-mode", "prob", "--lambda-rel", "0.10"],
    },
    {
        "name": "F_no_hsi_regularization",
        "output_dir": ABLATIONS_DIR / "no_hsi_reg",
        "args": ["--eps-hsi-reg", "0.0", "--lambda-rel", "0.10"],
    },
    {
        "name": "G_soft_reliability_targets",
        "output_dir": ABLATIONS_DIR / "soft_targets",
        "args": ["--supervision-mode", "soft", "--lambda-rel", "0.10"],
    },
]


def run_single_ablation(config):
    out_dir = config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "summary.json"
    
    if summary_file.exists():
        print(f"Skipping {config['name']} (already complete: {summary_file})")
        return

    print(f"\n" + "=" * 70)
    print(f"STARTING ABLATION: {config['name']}")
    print(f"Output dir: {out_dir}")
    print(f"Args: {config['args']}")
    print(f"=" * 70)

    cmd = [PYTHON, str(SCRIPT), "--output-dir", str(out_dir)] + config["args"]
    res = subprocess.run(cmd, check=True)
    print(f"Completed {config['name']} with return code {res.returncode}")


if __name__ == "__main__":
    for config in ABLATION_CONFIGS:
        run_single_ablation(config)
    print("\nAll ablations completed successfully!")
