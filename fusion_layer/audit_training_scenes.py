"""Compare SAR and HSI baselines independently for all 18 scenes."""

from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import f1_score


CACHE_DIR = Path(__file__).resolve().parent / "cache_features"
OUTPUT_PATH = Path(__file__).resolve().parent / "audit" / "all_scene_modality_comparison.csv"

SCENES_BY_SPLIT = {
    "train": [
        "GM01", "GM02", "GM06", "GM07", "GM08",
        "GM12", "GM13", "GM14", "GM18",
    ],
    "validation": [
        "GM03", "GM09", "GM15",
    ],
    "test": [
        "GM04", "GM05", "GM10", "GM11", "GM16", "GM17",
    ],
}


def main() -> None:
    rows = []

    print(f"{'Scene':<8}{'Split':<12}{'N':>8}{'SAR F1':>10}{'HSI F1':>10}{'SAR - HSI':>12}{'Winner':>10}")
    print("-" * 70)

    for split, scene_names in SCENES_BY_SPLIT.items():
        data = dict(np.load(CACHE_DIR / f"{split}_features.npz", allow_pickle=False))
        labels = data["labels"]
        scene_indices = data["scene_indices"]
        p_sar = data["sar_features"][:, 0]
        p_hsi = 1.0 / (1.0 + np.exp(-data["hsi_logits"].reshape(-1)))

        cache_scenes = sorted(np.unique(scene_indices).tolist())
        if cache_scenes != list(range(len(scene_names))):
            raise RuntimeError(
                f"Unexpected scene indices in {split} cache: {cache_scenes}"
            )

        for scene_index, scene_name in enumerate(scene_names):
            mask = scene_indices == scene_index
            sar_f1 = f1_score(
                labels[mask],
                (p_sar[mask] >= 0.5).astype(int),
                zero_division=0,
            )
            hsi_f1 = f1_score(
                labels[mask],
                (p_hsi[mask] >= 0.5).astype(int),
                zero_division=0,
            )
            gap = sar_f1 - hsi_f1
            if gap > 1e-12:
                winner = "SAR"
            elif gap < -1e-12:
                winner = "HSI"
            else:
                winner = "Tie"

            row = {
                "scene": scene_name,
                "split": split,
                "samples": int(mask.sum()),
                "sar_f1": sar_f1,
                "hsi_f1": hsi_f1,
                "sar_minus_hsi": gap,
                "winner": winner,
            }
            rows.append(row)
            print(
                f"{scene_name:<8}{split:<12}{int(mask.sum()):>8}"
                f"{sar_f1:>10.4f}{hsi_f1:>10.4f}{gap:>+12.4f}{winner:>10}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sar_wins = sum(row["winner"] == "SAR" for row in rows)
    hsi_wins = sum(row["winner"] == "HSI" for row in rows)
    ties = sum(row["winner"] == "Tie" for row in rows)
    sar_dominant = [row for row in rows if row["sar_minus_hsi"] > 0.05]

    print("\nSUMMARY")
    print(f"Scenes audited       : {len(rows)}")
    print(f"SAR wins             : {sar_wins}")
    print(f"HSI wins             : {hsi_wins}")
    print(f"Ties                 : {ties}")
    print(f"Strong SAR wins > .05: {len(sar_dominant)}")
    print(f"Maximum SAR - HSI    : {max(row['sar_minus_hsi'] for row in rows):+.4f}")
    print(f"Minimum SAR - HSI    : {min(row['sar_minus_hsi'] for row in rows):+.4f}")
    print(f"\nCSV saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
