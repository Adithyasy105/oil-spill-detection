from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / "fusion_layer" / "cache_features"
OUTPUT_ROOT = CACHE_ROOT / "scenes"
MANIFEST_PATH = ROOT / "fusion_layer" / "dataset_audit" / "osd_manifest.json"

SCENES_BY_SOURCE = {
    "train": ["GM01", "GM02", "GM06", "GM07", "GM08", "GM12", "GM13", "GM14", "GM18"],
    "validation": ["GM03", "GM09", "GM15"],
    "test": ["GM04", "GM05", "GM10", "GM11", "GM16", "GM17"],
}


def load_manifest_shape_map():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    shape_map = {}
    for record in manifest.get("scenes", []):
        scene_id = record["scene_id"]
        h = int(record["hsi"]["height"])
        w = int(record["hsi"]["width"])
        shape_map[scene_id] = (h, w)
    return shape_map


shape_map = load_manifest_shape_map()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for split, scene_names in SCENES_BY_SOURCE.items():
    cache_path = CACHE_ROOT / f"{split}_features.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache: {cache_path}")

    data = np.load(cache_path, allow_pickle=False)
    for scene_index, scene_name in enumerate(scene_names):
        mask = data["scene_indices"] == scene_index
        if not np.any(mask):
            raise RuntimeError(f"No pixels found for {scene_name} in {split}")

        payload = {
            key: np.asarray(data[key][mask]).copy()
            for key in ("sar_features", "hsi_logits", "hsi_confidence", "labels")
        }
        h, w = shape_map[scene_name]
        payload["H"] = np.asarray(h, dtype=np.int64)
        payload["W"] = np.asarray(w, dtype=np.int64)
        payload["shape"] = np.asarray([h, w], dtype=np.int64)
        payload["scene_id"] = np.asarray(scene_name)

        out_path = OUTPUT_ROOT / f"{scene_name}.npz"
        np.savez_compressed(out_path, **payload)
        print(f"Saved {out_path} | {payload['labels'].shape[0]} pixels | {h}x{w}")

print(f"Done. Scene caches saved under {OUTPUT_ROOT}")
