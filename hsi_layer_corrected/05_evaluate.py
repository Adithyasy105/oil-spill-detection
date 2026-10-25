# ================================================================
# 05_evaluate.py
# FINAL HSI EXPERT EVALUATION
# ================================================================

import os
import sys
import json
import csv

import numpy as np
import scipy.io as sio
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm


# ================================================================
# PATH
# ================================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    PROJECT_DIR
)


# ================================================================
# LOAD MODULES
# ================================================================

exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "01_config.py"
        ),
        encoding="utf-8"
    ).read()
)

exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "02_dataset.py"
        ),
        encoding="utf-8"
    ).read()
)

exec(
    open(
        os.path.join(
            PROJECT_DIR,
            "03_model.py"
        ),
        encoding="utf-8"
    ).read()
)


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    if not os.path.exists(
        BEST_CHECKPOINT
    ):

        raise FileNotFoundError(
            f"Missing checkpoint:\n"
            f"{BEST_CHECKPOINT}"
        )

    model = HMOSDNet(
        in_channels=1,
        num_bands=PCA_COMPONENTS,
        num_classes=1,
        mam_reduction=MAM_REDUCTION
    ).to(DEVICE)

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print()
    print(
        f"[OK] HSI model loaded."
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Best validation mIoU: "
        f"{checkpoint.get('best_val_miou')}"
    )

    return model


# ================================================================
# LOAD PROCESSOR
# ================================================================

def load_processor():

    if not os.path.exists(
        SPECTRAL_PROCESSOR_PATH
    ):

        raise FileNotFoundError(
            f"Missing processor:\n"
            f"{SPECTRAL_PROCESSOR_PATH}"
        )

    processor = SpectralProcessor(
        n_components=PCA_COMPONENTS,
        bands_to_remove=BANDS_TO_REMOVE
    )

    processor.load(
        SPECTRAL_PROCESSOR_PATH
    )

    print(
        "[OK] Exact training "
        "spectral processor loaded."
    )

    return processor


# ================================================================
# HSI INFERENCE
# ================================================================

@torch.inference_mode()
def predict_scene(
    model,
    processor,
    scene_path,
    stride=1,
    batch_size=512
):

    raw, gt = load_hsi_scene(
        scene_path
    )

    pca = processor.transform(
        raw
    )

    h, w, bands = pca.shape

    if bands != PCA_COMPONENTS:

        raise RuntimeError(
            f"Expected {PCA_COMPONENTS} "
            f"PCA bands, got {bands}"
        )

    pad = PATCH_SIZE // 2

    padded = np.pad(
        pca,
        (
            (pad, pad),
            (pad, pad),
            (0, 0)
        ),
        mode="reflect"
    )

    # Prediction grid.
    rows = list(
        range(
            0,
            h,
            stride
        )
    )

    cols = list(
        range(
            0,
            w,
            stride
        )
    )

    prob_grid = np.zeros(
        (
            len(rows),
            len(cols)
        ),
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # ROW-WISE BATCHED INFERENCE
    # ------------------------------------------------------------

    for ri, row in enumerate(
        tqdm(
            rows,
            desc=(
                f"{scene_name(scene_path)} "
                f"stride={stride}"
            ),
            leave=False
        )
    ):

        patches = []

        for col in cols:

            patch = padded[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            # H,W,C -> C,H,W
            patch = patch.transpose(
                2,
                0,
                1
            )

            # C,H,W -> 1,C,H,W
            patches.append(
                patch
            )

        row_probs = []

        for start in range(
            0,
            len(patches),
            batch_size
        ):

            batch_np = np.asarray(
                patches[
                    start:
                    start + batch_size
                ],
                dtype=np.float32
            )

            batch = (
                torch.from_numpy(
                    batch_np
                )
                .unsqueeze(1)
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            logits = model(
                batch,
                return_features=False
            )

            probs = (
                torch.sigmoid(
                    logits
                )
                .flatten()
                .cpu()
                .numpy()
            )

            row_probs.append(
                probs
            )

        prob_grid[
            ri,
            :
        ] = np.concatenate(
            row_probs
        )

    # ------------------------------------------------------------
    # CONVERT SPARSE GRID TO NATIVE IMAGE
    # ------------------------------------------------------------

    if stride == 1:

        prob_map = prob_grid

    else:

        # Nearest-neighbour placement.
        # For standalone reporting, we keep the original
        # sampled-grid result separately.
        prob_map = np.zeros(
            (h, w),
            dtype=np.float32
        )

        for ri, row in enumerate(rows):

            for ci, col in enumerate(cols):

                prob_map[
                    row,
                    col
                ] = prob_grid[
                    ri,
                    ci
                ]

    return (
        prob_map,
        gt,
        prob_grid
    )


# ================================================================
# METRICS FROM PROBABILITY MAP
# ================================================================

def metrics_from_probability(
    probability,
    gt,
    threshold
):

    valid = np.isfinite(
        probability
    )

    prob = probability[
        valid
    ]

    target = gt[
        valid
    ]

    preds = (
        prob >= threshold
    ).astype(
        np.float32
    )

    return compute_metrics(
        torch.from_numpy(
            preds
        ),
        torch.from_numpy(
            target
        )
    )


# ================================================================
# THRESHOLD SELECTION
# ================================================================

def select_threshold(
    model,
    processor,
    val_files
):

    print()
    print("=" * 70)
    print("THRESHOLD SELECTION — VALIDATION ONLY")
    print("=" * 70)

    # Collect validation predictions.
    all_probs = []
    all_targets = []

    for path in val_files:

        prob_map, gt, prob_grid = predict_scene(
            model,
            processor,
            path,
            stride=1
        )

        all_probs.append(
            prob_map.flatten()
        )

        all_targets.append(
            gt.flatten()
        )

    probs = np.concatenate(
        all_probs
    )

    targets = np.concatenate(
        all_targets
    )

    best_threshold = 0.5

    best_miou = -np.inf

    rows = []

    for threshold in THRESHOLD_GRID:

        preds = (
            probs >= threshold
        ).astype(
            np.float32
        )

        metrics = compute_metrics(
            torch.from_numpy(
                preds
            ),
            torch.from_numpy(
                targets
            )
        )

        rows.append(
            {
                "threshold":
                    float(threshold),

                **metrics
            }
        )

        if metrics["miou"] > best_miou:

            best_miou = (
                metrics["miou"]
            )

            best_threshold = (
                float(threshold)
            )

    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation mIoU: "
        f"{best_miou:.6f}"
    )

    return (
        best_threshold,
        rows
    )


# ================================================================
# EVALUATE SCENES
# ================================================================

def evaluate_split(
    model,
    processor,
    scene_files,
    split_name,
    threshold,
    stride=1
):

    print()
    print("=" * 70)

    print(
        f"{split_name.upper()} EVALUATION"
    )

    print("=" * 70)

    results = []

    for path in scene_files:

        prob_map, gt, prob_grid = predict_scene(
            model,
            processor,
            path,
            stride=stride
        )

        if stride == 1:

            metrics = metrics_from_probability(
                prob_map,
                gt,
                threshold
            )

        else:

            # Evaluate only sampled locations.
            rows = list(
                range(
                    0,
                    gt.shape[0],
                    stride
                )
            )

            cols = list(
                range(
                    0,
                    gt.shape[1],
                    stride
                )
            )

            sampled_gt = gt[
                np.ix_(
                    rows,
                    cols
                )
            ]

            metrics = metrics_from_probability(
                prob_grid,
                sampled_gt,
                threshold
            )

        result = {

            "scene":
                scene_name(path),

            "height":
                gt.shape[0],

            "width":
                gt.shape[1],

            "oil_gt_percent":
                float(
                    gt.mean()
                    * 100
                ),

            "threshold":
                threshold,

            **metrics
        }

        results.append(
            result
        )

        print(
            f"{scene_name(path):<8} | "
            f"mIoU={metrics['miou']:.5f} | "
            f"Dice={metrics['dice']:.5f} | "
            f"P={metrics['precision']:.5f} | "
            f"R={metrics['recall']:.5f} | "
            f"FAR={metrics['far']:.5f}"
        )

    return results


# ================================================================
# AGGREGATE
# ================================================================

def aggregate_results(results):

    keys = [

        "miou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "far",
        "oa",
        "iou_oil",
        "iou_water"
    ]

    output = {}

    for key in keys:

        output[
            f"mean_{key}"
        ] = float(
            np.mean(
                [
                    r[key]
                    for r in results
                ]
            )
        )

    return output


# ================================================================
# VISUALIZATION
# ================================================================

def save_visualization(
    scene,
    gt,
    prob,
    threshold,
    output_path
):

    prediction = (
        prob >= threshold
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        gt,
        cmap="gray"
    )

    axes[0].set_title(
        "Ground Truth"
    )

    axes[0].axis("off")

    im = axes[1].imshow(
        prob,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "HSI Probability"
    )

    axes[1].axis("off")

    plt.colorbar(
        im,
        ax=axes[1],
        fraction=0.046
    )

    axes[2].imshow(
        gt,
        cmap="gray",
        alpha=0.5
    )

    axes[2].imshow(
        prediction,
        cmap="Reds",
        alpha=0.5
    )

    axes[2].set_title(
        f"Prediction @ {threshold:.2f}"
    )

    axes[2].axis("off")

    plt.suptitle(
        scene
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print("FINAL HSI EXPERT EVALUATION")
    print("=" * 70)

    model = load_model()

    processor = load_processor()

    train_files = discover_scene_files(
        TRAIN_HSI_DIR
    )

    val_files = discover_scene_files(
        VAL_HSI_DIR
    )

    test_files = discover_scene_files(
        TEST_HSI_DIR
    )

    # ------------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    # ------------------------------------------------------------

    train_names = {
        scene_name(f)
        for f in train_files
    }

    val_names = {
        scene_name(f)
        for f in val_files
    }

    test_names = {
        scene_name(f)
        for f in test_files
    }

    assert train_names.isdisjoint(
        val_names
    )

    assert train_names.isdisjoint(
        test_names
    )

    assert val_names.isdisjoint(
        test_names
    )

    print()
    print("Split integrity:")
    print(
        f"  Train: {sorted(train_names)}"
    )
    print(
        f"  Val  : {sorted(val_names)}"
    )
    print(
        f"  Test : {sorted(test_names)}"
    )

    # ------------------------------------------------------------
    # THRESHOLD
    # ------------------------------------------------------------

    (
        threshold,
        threshold_table
    ) = select_threshold(
        model,
        processor,
        val_files
    )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    val_results = evaluate_split(
        model,
        processor,
        val_files,
        "validation",
        threshold,
        stride=1
    )

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    test_results = evaluate_split(
        model,
        processor,
        test_files,
        "test",
        threshold,
        stride=1
    )

    # ------------------------------------------------------------
    # TEST FUSION-COMPATIBLE STRIDE 3
    # ------------------------------------------------------------

    test_stride3 = evaluate_split(
        model,
        processor,
        test_files,
        "test_stride3",
        threshold,
        stride=3
    )

    # ------------------------------------------------------------
    # AGGREGATES
    # ------------------------------------------------------------

    val_summary = aggregate_results(
        val_results
    )

    test_summary = aggregate_results(
        test_results
    )

    test_stride3_summary = (
        aggregate_results(
            test_stride3
        )
    )

    # ------------------------------------------------------------
    # SAVE JSON
    # ------------------------------------------------------------

    output = {

        "protocol": {

            "train_only_pca":
                True,

            "post_pca_normalization":
                False,

            "threshold_selected_on":
                "validation",

            "test_used_for_selection":
                False,

            "test_stride1":
                True,

            "test_stride3":
                True
        },

        "threshold":
            threshold,

        "validation":
            {
                "scenes":
                    sorted(
                        val_names
                    ),

                "per_scene":
                    val_results,

                "aggregate":
                    val_summary
            },

        "test":
            {
                "scenes":
                    sorted(
                        test_names
                    ),

                "per_scene":
                    test_results,

                "aggregate":
                    test_summary
            },

        "test_stride3":
            {
                "per_scene":
                    test_stride3,

                "aggregate":
                    test_stride3_summary
            }
    }

    json_path = (
        HSI_RESULTS_DIR
        /
        "hsi_final_evaluation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    csv_path = (
        HSI_RESULTS_DIR
        /
        "hsi_test_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        if test_results:

            writer = csv.DictWriter(
                f,
                fieldnames=test_results[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                test_results
            )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL HSI TEST BENCHMARK")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Mean mIoU: "
        f"{test_summary['mean_miou']:.6f}"
    )

    print(
        f"Mean Dice: "
        f"{test_summary['mean_dice']:.6f}"
    )

    print(
        f"Mean Precision: "
        f"{test_summary['mean_precision']:.6f}"
    )

    print(
        f"Mean Recall: "
        f"{test_summary['mean_recall']:.6f}"
    )

    print(
        f"Mean FAR: "
        f"{test_summary['mean_far']:.6f}"
    )

    print(
        f"Mean OA: "
        f"{test_summary['mean_oa']:.6f}"
    )

    print()
    print(
        "Fusion-compatible stride-3:"
    )

    print(
        f"  mIoU: "
        f"{test_stride3_summary['mean_miou']:.6f}"
    )

    print(
        f"  Dice: "
        f"{test_stride3_summary['mean_dice']:.6f}"
    )

    print()
    print(
        f"Saved: {json_path}"
    )

    print(
        f"Saved: {csv_path}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()