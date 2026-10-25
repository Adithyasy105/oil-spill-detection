"""Paired statistical tests for the six-fold scene CV results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "experiments" / "scene_cv_6fold" / "cross_validation_results.csv"
OUTPUT_DIR = ROOT / "experiments" / "scene_cv_6fold"
CSV_PATH = OUTPUT_DIR / "paired_cv_statistics.csv"
JSON_PATH = OUTPUT_DIR / "paired_cv_statistics.json"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    hsi = {
        int(row["fold"]): row
        for row in rows
        if row["model"] == "HSI Only"
    }
    output_rows = []

    for model_name in sorted({row["model"] for row in rows} - {"HSI Only"}):
        model_rows = sorted(
            [row for row in rows if row["model"] == model_name],
            key=lambda row: int(row["fold"]),
        )
        model_f1 = np.array([float(row["f1"]) for row in model_rows])
        hsi_f1 = np.array([float(hsi[int(row["fold"])] ["f1"]) for row in model_rows])
        differences = model_f1 - hsi_f1
        ttest = ttest_rel(model_f1, hsi_f1)
        signed_rank = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        output_rows.append(
            {
                "model": model_name,
                "reference": "HSI Only",
                "fold_count": len(differences),
                "mean_model_f1": float(model_f1.mean()),
                "mean_hsi_f1": float(hsi_f1.mean()),
                "mean_f1_difference": float(differences.mean()),
                "std_f1_difference": float(differences.std(ddof=1)),
                "folds_model_better": int((differences > 0).sum()),
                "folds_model_worse": int((differences < 0).sum()),
                "paired_t_statistic": float(ttest.statistic),
                "paired_t_p_value": float(ttest.pvalue),
                "wilcoxon_statistic": float(signed_rank.statistic),
                "wilcoxon_p_value": float(signed_rank.pvalue),
            }
        )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "reference_model": "HSI Only",
        "fold_count": 6,
        "interpretation": (
            "Fusion is statistically comparable to HSI-only when p >= 0.05; "
            "these tests do not establish fusion superiority."
        ),
        "results": output_rows,
        "files": {
            "input": str(INPUT_PATH),
            "csv": str(CSV_PATH),
        },
    }
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Paired CV statistics")
    for row in output_rows:
        print(
            f"{row['model']:<28} "
            f"mean diff={row['mean_f1_difference']:+.4f} "
            f"t p={row['paired_t_p_value']:.4f} "
            f"Wilcoxon p={row['wilcoxon_p_value']:.4f}"
        )
    print(f"CSV: {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
