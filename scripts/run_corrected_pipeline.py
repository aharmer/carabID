"""
run_corrected_pipeline.py

Corrected split-before-augment pipeline.

PROBLEM WITH ORIGINAL PIPELINE
-------------------------------
The original pipeline augmented the full dataset first, then split
into train/val/test.  Because the split treated every file (original
and augmented copies) as independent, augmented copies of training
images could land in the validation or test set.  This inflates the
reported accuracy.

CORRECTED ORDER
---------------
1. Split *original* specimen images into train / val / test partitions.
2. Augment ONLY the training partition of each split.
3. Val and test sets contain only un-augmented originals throughout.

This script handles Steps 1-2 for both the k-fold CV dataset and the
final single-model dataset.  It then prints the exact training and
assessment commands to run on the GPU machine.

USAGE
-----
Run from the carabID root directory:

    conda run -n ultralytics-env python scripts/run_corrected_pipeline.py

Output directories (all NEW — originals untouched):
    imgs/cv_classification_set_v2/    # k-fold splits
    imgs/final_classification_set_v2/ # final model split

Existing directories are left intact:
    imgs/cv_classification_set/       # original (do not delete)
    imgs/final_classification_set/    # original (do not delete)
"""

import os
import sys
import random
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────── #
#  Configuration                                                       #
# ─────────────────────────────────────────────────────────────────── #

SEED               = 42
IMG_EXTENSIONS     = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Source of original un-augmented images
SOURCE_DIR         = Path("imgs/c1/train")

# Output directories (new — will not overwrite originals)
CV_OUTPUT_DIR      = Path("imgs/cv_classification_set_v2")
FINAL_OUTPUT_DIR   = Path("imgs/final_classification_set_v2")

# Split parameters (match original pipeline)
N_FOLDS            = 5
CV_TEST_RATIO      = 0.10   # 10 % held-out test set before k-fold split
FINAL_TRAIN_RATIO  = 0.80
FINAL_VAL_RATIO    = 0.15
FINAL_TEST_RATIO   = 0.05

# Augmentation parameters (match original: flip + motion_blur, 3× total)
AUG_FACTOR         = 3.0    # includes the original copy
ROTATION_RANGE     = 0      # disabled (matched to original)
ENABLE_NOISE       = False  # disabled (matched to original)
ENABLE_ROTATION    = False  # disabled (matched to original)
ENABLE_FLIP        = True
ENABLE_BLUR        = True


# ─────────────────────────────────────────────────────────────────── #
#  Helpers                                                             #
# ─────────────────────────────────────────────────────────────────── #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_class_images(directory: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} for all class subdirs."""
    result = {}
    for cls_dir in sorted(directory.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
        if imgs:
            result[cls_dir.name] = imgs
    return result


def copy_split(split_dict: dict[str, list[Path]], dest: Path):
    """Copy files from split_dict into dest/<class>/ directories."""
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls, paths in split_dict.items():
        cls_dir = dest / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, cls_dir / src.name)
            total += 1
    return total


# ─────────────────────────────────────────────────────────────────── #
#  Splitting                                                           #
# ─────────────────────────────────────────────────────────────────── #

def stratified_split(class_images: dict, ratios: tuple) -> tuple[dict, ...]:
    """
    Stratified split.  ratios is (train, val) or (train, val, test).
    Returns one dict per ratio entry.
    """
    splits = [defaultdict(list) for _ in ratios]
    for cls, imgs in class_images.items():
        imgs = imgs.copy()
        random.shuffle(imgs)
        n = len(imgs)
        sizes = []
        used = 0
        for i, r in enumerate(ratios[:-1]):
            s = max(1, int(n * r))
            sizes.append(s)
            used += s
        sizes.append(n - used)          # last split gets remainder
        start = 0
        for i, s in enumerate(sizes):
            splits[i][cls] = imgs[start:start + s]
            start += s
    return tuple(dict(s) for s in splits)


def kfold_split(class_images: dict, n_splits: int, test_ratio: float):
    """
    Stratified k-fold split.

    1. Hold out test_ratio from each class (un-augmented test set).
    2. Split the remainder into n_splits folds.

    Returns:
        test_dict : dict  (common to all folds)
        folds     : list of (train_dict, val_dict)
    """
    test_dict    = defaultdict(list)
    fold_pool    = {}

    for cls, imgs in class_images.items():
        imgs = imgs.copy()
        random.shuffle(imgs)
        test_n = max(0, int(len(imgs) * test_ratio))
        test_dict[cls]  = imgs[:test_n]
        fold_pool[cls]  = imgs[test_n:]

    folds = []
    for fold_idx in range(n_splits):
        train_d = defaultdict(list)
        val_d   = defaultdict(list)
        for cls, imgs in fold_pool.items():
            n         = len(imgs)
            fold_size = n // n_splits
            val_start = fold_idx * fold_size
            val_end   = val_start + fold_size if fold_idx < n_splits - 1 else n
            val_d[cls]   = imgs[val_start:val_end]
            train_d[cls] = imgs[:val_start] + imgs[val_end:]
        folds.append((dict(train_d), dict(val_d)))

    return dict(test_dict), folds


# ─────────────────────────────────────────────────────────────────── #
#  Augmentation (flip + motion blur to reach AUG_FACTOR × originals)  #
# ─────────────────────────────────────────────────────────────────── #

def flip_horizontal(img):
    return cv2.flip(img, 1)


def motion_blur(img):
    k = random.choice([5, 7, 9, 11])
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    kernel = np.zeros((k, k))
    if direction == 'horizontal':
        kernel[k // 2, :] = 1.0 / k
    elif direction == 'vertical':
        kernel[:, k // 2] = 1.0 / k
    else:
        np.fill_diagonal(kernel, 1.0 / k)
    return cv2.filter2D(img, -1, kernel)


AUG_OPS = []
if ENABLE_FLIP:
    AUG_OPS.append(('flip',  flip_horizontal))
if ENABLE_BLUR:
    AUG_OPS.append(('mblur', motion_blur))


def augment_directory(src_dir: Path, dst_dir: Path, factor: float = AUG_FACTOR):
    """
    Copy originals into dst_dir and add augmented copies so that the
    total count per original is approximately `factor`.

    Augmented filenames: <stem>_<aug>_<i><suffix>
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    aug_per_image = max(0, int(factor) - 1)   # copies beyond original

    total_orig = total_aug_created = 0

    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
        cls_dst = dst_dir / cls_dir.name
        cls_dst.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue
            # Copy original
            shutil.copy2(img_path, cls_dst / img_path.name)
            total_orig += 1

            # Create augmented copies
            ops = random.choices(AUG_OPS, k=aug_per_image) if AUG_OPS else []
            for i, (tag, fn) in enumerate(ops, start=1):
                aug_img  = fn(img)
                aug_name = f"{img_path.stem}_{tag}_{i}{img_path.suffix}"
                cv2.imwrite(str(cls_dst / aug_name), aug_img)
                total_aug_created += 1

    print(f"  Augmented: {total_orig} originals → "
          f"{total_orig + total_aug_created} total "
          f"({total_orig + total_aug_created / max(total_orig, 1):.1f}× "
          f"[{total_orig} + {total_aug_created} augmented])")
    return total_orig, total_aug_created


# ─────────────────────────────────────────────────────────────────── #
#  Main                                                                #
# ─────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Corrected split-before-augment pipeline")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip augmentation step (split only)")
    parser.add_argument("--cv-only", action="store_true",
                        help="Only build k-fold CV dataset (skip final-model split)")
    parser.add_argument("--final-only", action="store_true",
                        help="Only build final-model dataset (skip k-fold)")
    args = parser.parse_args()

    set_seed(SEED)

    if not SOURCE_DIR.exists():
        sys.exit(f"ERROR: source directory not found: {SOURCE_DIR}")

    class_images = get_class_images(SOURCE_DIR)
    n_classes    = len(class_images)
    n_total      = sum(len(v) for v in class_images.values())
    print(f"\nSource: {SOURCE_DIR}")
    print(f"  {n_total} images across {n_classes} classes")

    # ── K-fold CV ────────────────────────────────────────────────── #
    if not args.final_only:
        print(f"\n{'='*60}")
        print(f"STEP 1a: K-FOLD CV SPLIT  →  {CV_OUTPUT_DIR}")
        print(f"{'='*60}")

        if CV_OUTPUT_DIR.exists():
            print(f"  Directory already exists — removing and recreating.")
            shutil.rmtree(CV_OUTPUT_DIR)

        test_dict, folds = kfold_split(class_images, N_FOLDS, CV_TEST_RATIO)

        test_n = sum(len(v) for v in test_dict.values())
        print(f"  Test set (common): {test_n} original images")

        for fold_idx, (train_d, val_d) in enumerate(folds, start=1):
            fold_dir  = CV_OUTPUT_DIR / f"fold_{fold_idx}"
            train_n   = copy_split(train_d, fold_dir / "train_orig")
            val_n     = copy_split(val_d,   fold_dir / "val")
            test_n_   = copy_split(test_dict, fold_dir / "test")
            print(f"  fold_{fold_idx}: train_orig={train_n}  val={val_n}  test={test_n_}")

        if not args.skip_augment:
            print(f"\n{'='*60}")
            print(f"STEP 1b: AUGMENT K-FOLD TRAIN PARTITIONS")
            print(f"{'='*60}")
            for fold_idx in range(1, N_FOLDS + 1):
                fold_dir  = CV_OUTPUT_DIR / f"fold_{fold_idx}"
                src       = fold_dir / "train_orig"
                dst       = fold_dir / "train"
                print(f"\n  fold_{fold_idx}/train_orig → fold_{fold_idx}/train")
                augment_directory(src, dst, AUG_FACTOR)

    # ── Final model ───────────────────────────────────────────────── #
    if not args.cv_only:
        print(f"\n{'='*60}")
        print(f"STEP 2a: FINAL MODEL SPLIT  →  {FINAL_OUTPUT_DIR}")
        print(f"{'='*60}")

        if FINAL_OUTPUT_DIR.exists():
            print(f"  Directory already exists — removing and recreating.")
            shutil.rmtree(FINAL_OUTPUT_DIR)

        train_d, val_d, test_d = stratified_split(
            class_images,
            (FINAL_TRAIN_RATIO, FINAL_VAL_RATIO, FINAL_TEST_RATIO)
        )
        train_n = copy_split(train_d, FINAL_OUTPUT_DIR / "train_orig")
        val_n   = copy_split(val_d,   FINAL_OUTPUT_DIR / "val")
        test_n  = copy_split(test_d,  FINAL_OUTPUT_DIR / "test")
        print(f"  train_orig={train_n}  val={val_n}  test={test_n}")

        if not args.skip_augment:
            print(f"\n{'='*60}")
            print(f"STEP 2b: AUGMENT FINAL MODEL TRAIN PARTITION")
            print(f"{'='*60}")
            augment_directory(
                FINAL_OUTPUT_DIR / "train_orig",
                FINAL_OUTPUT_DIR / "train",
                AUG_FACTOR
            )

    # ── Training commands ─────────────────────────────────────────── #
    print(f"""
{'='*60}
NEXT STEPS — run these on the GPU machine from the carabID root
{'='*60}

# K-fold CV training (5 folds):
conda activate carabid
python scripts/kfold_train_classification_models.py \\
    --data   imgs/cv_classification_set_v2 \\
    --epochs 30 --dropout 0.2 --lr0 0.001 \\
    --name   carabid_cv_v2_11ncls_ep30_do02_lr001

# K-fold CV assessment:
python scripts/kfold_assess_classification_models.py \\
    --models_dir runs/classify \\
    --name       carabid_cv_v2_11ncls_ep30_do02_lr001 \\
    --data       imgs/cv_classification_set_v2 \\
    --split      test

# Final model training:
python scripts/train_classification_model.py \\
    --data   imgs/final_classification_set_v2 \\
    --epochs 30 --dropout 0.2 --lr0 0.001 \\
    --name   final_carabid_v2_11ncls_ep30_do02_lr001

# Calibration:
python calibrated/calibrate_yolo_classifier.py \\
    --model  runs/classify/final_carabid_v2_11ncls_ep30_do02_lr001/weights/best.pt \\
    --data   imgs/final_classification_set_v2 \\
    --imgsz  640

# Calibrated assessment:
python calibrated/assess_classification_model.py \\
    --model           runs/classify/final_carabid_v2_11ncls_ep30_do02_lr001/weights/best.pt \\
    --data            imgs/final_classification_set_v2 \\
    --split           val \\
    --calibration-dir runs/classify/final_carabid_v2_11ncls_ep30_do02_lr001/weights

# R analysis (after assessment outputs exist):
Rscript scripts/analyse_genus_performance.R   # update paths inside if needed
Rscript scripts/analyse_calibrated_confidence.R
""")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
