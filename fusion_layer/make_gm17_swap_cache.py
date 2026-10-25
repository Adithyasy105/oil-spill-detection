"""Create the controlled GM17-train / GM18-test cache swap."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cache_features"
TARGET = ROOT / "experiments" / "gm17_swap" / "cache_features"

TRAIN_SCENES = ["GM01", "GM02", "GM06", "GM07", "GM08", "GM12", "GM13", "GM14", "GM18"]
TEST_SCENES = ["GM04", "GM05", "GM10", "GM11", "GM16", "GM17"]


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE / f"{name}_features.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def select(data: dict[str, np.ndarray], indices: list[int], scene_names: list[str]) -> dict[str, np.ndarray]:
    parts = []
    for index in indices:
        mask = data["scene_indices"] == index
        part = {key: value[mask] for key, value in data.items() if key != "scene_indices"}
        part["scene_indices"] = np.full(int(mask.sum()), len(parts), dtype=np.int64)
        parts.append(part)
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }


def save(name: str, data: dict[str, np.ndarray]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TARGET / f"{name}_features.npz", **data)
    summary = {
        "split": name,
        "num_samples": int(len(data["labels"])),
        "positive_samples": int(data["labels"].sum()),
        "positive_fraction": float(data["labels"].mean()),
        "scene_indices": sorted(np.unique(data["scene_indices"]).tolist()),
    }
    import json
    with (TARGET / f"{name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def relabel_scene_indices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assign contiguous scene IDs after combining source cache portions."""

    result = dict(data)
    original_indices = data["scene_indices"]
    unique_indices = sorted(np.unique(original_indices).tolist())
    remapped = np.empty_like(original_indices)

    for new_index, old_index in enumerate(unique_indices):
        remapped[original_indices == old_index] = new_index

    result["scene_indices"] = remapped
    return result


def main() -> None:
    train = load("train")
    validation = load("validation")
    test = load("test")

    # Existing train index 8 is GM18; existing test index 5 is GM17.
    swapped_train = select(train, list(range(8)), TRAIN_SCENES[:8])
    gm17 = select(test, [5], ["GM17"])
    gm17["scene_indices"] = np.full(
        len(gm17["labels"]),
        8,
        dtype=np.int64,
    )
    swapped_train = {
        key: np.concatenate([swapped_train[key], gm17[key]], axis=0)
        for key in swapped_train
    }
    swapped_train = relabel_scene_indices(swapped_train)

    # Existing test indices 0-4 remain, then GM18 becomes test index 5.
    swapped_test = select(test, list(range(5)), TEST_SCENES[:5])
    gm18 = select(train, [8], ["GM18"])
    gm18["scene_indices"] = np.full(
        len(gm18["labels"]),
        5,
        dtype=np.int64,
    )
    swapped_test = {
        key: np.concatenate([swapped_test[key], gm18[key]], axis=0)
        for key in swapped_test
    }
    swapped_test = relabel_scene_indices(swapped_test)

    save("train", swapped_train)
    save("validation", validation)
    save("test", swapped_test)

    print(f"Created GM17 swap cache: {TARGET}")
    print("train scenes:", TRAIN_SCENES[:8] + ["GM17"])
    print("validation scenes:", ["GM03", "GM09", "GM15"])
    print("test scenes:", TEST_SCENES[:5] + ["GM18"])
    for name, data in (("train", swapped_train), ("validation", validation), ("test", swapped_test)):
        print(name, len(data["labels"]), sorted(np.unique(data["scene_indices"]).tolist()))


if __name__ == "__main__":
    main()
