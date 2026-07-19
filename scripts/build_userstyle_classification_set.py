"""
build_userstyle_classification_set.py

Build imgs/final_classification_set_v3/ = the lab classification crops
(split-before-augment, 80/15/5) PLUS the staged iNaturalist user-style crops,
with a resolution-jitter augmentation designed to close the magnification gap
that caused known genera to be flagged novel on user photos.

Placement of user-style crops (per project decision):
  - train crops (all genera except the held-out portion) -> train_orig
  - test crops (Laemostenus + Mecodema held-out only)     -> test
  - none go to val (val stays lab-only)

Augmentation (offline, applied to the TRAIN partition only — lab + iNat alike):
  each augmented copy is a random degradation PIPELINE, so the model learns to
  recognise specimens across the capture qualities real users produce:
    - horizontal flip            (p=0.5)
    - resolution jitter          (p=0.8)  downscale to 120-400 px, upsample to 640
    - motion / gaussian blur     (p=0.4)
    - JPEG recompression         (p=0.5)  quality 30-75
  The pristine original is always kept as one copy (keeps high-res capability).

Run from the carabID root:
    conda run -n ultralytics-env python scripts/build_userstyle_classification_set.py
"""

import io
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────── #
#  Configuration                                                       #
# ─────────────────────────────────────────────────────────────────── #

SEED             = 42
LAB_SOURCE       = Path("imgs/c1/train")               # 76 genus folders, lab 640px crops
INAT_CROPS       = Path("imgs/_inat_crops")            # <split>/<genus>/*.jpg  (from prepare step)
OUTPUT_DIR       = Path("imgs/final_classification_set_v4")

TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.15
TEST_RATIO       = 0.05                                # remainder

# Fraction of the (non-test) user-style crops routed into val so the OOD
# threshold and temperature are calibrated on user-style images too.  Genera
# with a single user-style crop stay in train (cannot be split).
USERSTYLE_VAL_RATIO = 0.20

AUG_FACTOR       = 4                                   # 1 original + 3 degraded copies
RES_JITTER_RANGE = (120, 400)                          # px, low effective resolution
JPEG_Q_RANGE     = (30, 75)

IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_class_images(directory: Path) -> dict:
    out = {}
    for cls in sorted(directory.iterdir()):
        if cls.is_dir():
            imgs = sorted(f for f in cls.iterdir()
                          if f.suffix.lower() in IMG_EXTS)
            if imgs:
                out[cls.name] = imgs
    return out


def stratified_split(class_images, ratios):
    splits = [defaultdict(list) for _ in ratios]
    for cls, imgs in class_images.items():
        imgs = imgs.copy()
        random.shuffle(imgs)
        n = len(imgs)
        sizes, used = [], 0
        for r in ratios[:-1]:
            s = max(1, int(n * r)); sizes.append(s); used += s
        sizes.append(n - used)
        start = 0
        for i, s in enumerate(sizes):
            splits[i][cls] = imgs[start:start + s]
            start += s
    return tuple(dict(s) for s in splits)


# ─────────────────────────────────────────────────────────────────── #
#  Augmentation ops                                                    #
# ─────────────────────────────────────────────────────────────────── #

def aug_flip(img):
    return cv2.flip(img, 1)


def aug_resolution_jitter(img):
    """Downsample to a random low resolution, then upsample back — destroys
    fine detail exactly as low-magnification user photos do."""
    h, w = img.shape[:2]
    target = random.randint(*RES_JITTER_RANGE)
    small = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def aug_motion_blur(img):
    k = random.choice([5, 7, 9, 11])
    direction = random.choice(["horizontal", "vertical", "diagonal"])
    kernel = np.zeros((k, k))
    if direction == "horizontal":
        kernel[k // 2, :] = 1.0 / k
    elif direction == "vertical":
        kernel[:, k // 2] = 1.0 / k
    else:
        np.fill_diagonal(kernel, 1.0 / k)
    return cv2.filter2D(img, -1, kernel)


def aug_gaussian_blur(img):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_jpeg(img):
    q = random.randint(*JPEG_Q_RANGE)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def degrade(img):
    """Compose a random degradation pipeline for one augmented copy."""
    out = img.copy()
    if random.random() < 0.5:
        out = aug_flip(out)
    if random.random() < 0.8:
        out = aug_resolution_jitter(out)
    if random.random() < 0.4:
        out = aug_motion_blur(out) if random.random() < 0.5 else aug_gaussian_blur(out)
    if random.random() < 0.5:
        out = aug_jpeg(out)
    return out


# ─────────────────────────────────────────────────────────────────── #
#  Build                                                               #
# ─────────────────────────────────────────────────────────────────── #

def copy_split(split_dict, dest):
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls, paths in split_dict.items():
        d = dest / cls; d.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, d / src.name); n += 1
    return n


def inject_inat(split_name, dest):
    """Copy ALL staged iNat crops for a split into dest/<genus>/."""
    src_root = INAT_CROPS / split_name
    if not src_root.exists():
        return 0
    n = 0
    for genus_dir in sorted(src_root.iterdir()):
        if not genus_dir.is_dir():
            continue
        d = dest / genus_dir.name; d.mkdir(parents=True, exist_ok=True)
        for f in genus_dir.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                shutil.copy2(f, d / f.name); n += 1
    return n


def inject_inat_train_val(train_dest, val_dest, val_ratio):
    """Split the staged iNat *train* crops per genus into train_orig and val.
    Singleton genera stay in train.  Returns (n_train, n_val)."""
    src_root = INAT_CROPS / "train"
    if not src_root.exists():
        return 0, 0
    n_tr = n_va = 0
    for genus_dir in sorted(src_root.iterdir()):
        if not genus_dir.is_dir():
            continue
        crops = sorted(f for f in genus_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        random.shuffle(crops)
        k_val = int(len(crops) * val_ratio) if len(crops) > 1 else 0
        val_crops, train_crops = crops[:k_val], crops[k_val:]
        dtr = train_dest / genus_dir.name; dtr.mkdir(parents=True, exist_ok=True)
        for f in train_crops:
            shutil.copy2(f, dtr / f.name); n_tr += 1
        if val_crops:
            dva = val_dest / genus_dir.name; dva.mkdir(parents=True, exist_ok=True)
            for f in val_crops:
                shutil.copy2(f, dva / f.name); n_va += 1
    return n_tr, n_va


def augment_train(src_dir, dst_dir, factor):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copies = max(0, factor - 1)
    total_orig = total_aug = 0
    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_dst = dst_dir / cls_dir.name
        cls_dst.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  warn: unreadable {img_path}"); continue
            shutil.copy2(img_path, cls_dst / img_path.name)
            total_orig += 1
            for i in range(1, n_copies + 1):
                aug = degrade(img)
                cv2.imwrite(str(cls_dst / f"{img_path.stem}_aug{i}{img_path.suffix}"), aug)
                total_aug += 1
    return total_orig, total_aug


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    class_images = get_class_images(LAB_SOURCE)
    n_total = sum(len(v) for v in class_images.values())
    print(f"Lab source: {n_total} images across {len(class_images)} genera")

    train_d, val_d, test_d = stratified_split(
        class_images, (TRAIN_RATIO, VAL_RATIO, TEST_RATIO))

    # 1. lab partitions
    lab_train = copy_split(train_d, OUTPUT_DIR / "train_orig")
    lab_val   = copy_split(val_d,   OUTPUT_DIR / "val")
    lab_test  = copy_split(test_d,  OUTPUT_DIR / "test")
    print(f"Lab split:  train_orig={lab_train}  val={lab_val}  test={lab_test}")

    # 2. inject iNat user-style crops:
    #      - non-test crops split into train_orig + val (val_ratio)
    #      - held-out Laemostenus/Mecodema crops -> test
    inat_train, inat_val = inject_inat_train_val(
        OUTPUT_DIR / "train_orig", OUTPUT_DIR / "val", USERSTYLE_VAL_RATIO)
    inat_test = inject_inat("test", OUTPUT_DIR / "test")
    print(f"iNat injected:  train_orig+={inat_train}  val+={inat_val}  test+={inat_test}")

    # 3. augment the combined train partition
    print(f"\nAugmenting train_orig -> train (factor {AUG_FACTOR}, resolution jitter)...")
    orig, aug = augment_train(OUTPUT_DIR / "train_orig", OUTPUT_DIR / "train", AUG_FACTOR)
    print(f"  {orig} originals -> {orig + aug} total ({orig} + {aug} augmented)")

    print(f"\nDone. Dataset at {OUTPUT_DIR}")
    print(f"  train (augmented): {orig + aug}")
    print(f"  val:               {lab_val + inat_val}  (incl. {inat_val} user-style)")
    print(f"  test:              {lab_test + inat_test}  (incl. {inat_test} user-style)")


if __name__ == "__main__":
    main()
