"""
build_classification_from_detection.py

Derive the classification dataset from a detection dataset, preserving that
dataset's train/valid/test split.

Cropping per split matters: if crops were pooled and re-split, crops of the
same specimen could land on both sides of the boundary, which is the leakage
the corrected pipeline exists to avoid.  Taking the split from the detection
set also keeps the two models consistent with each other.

Augmentation (train partition only) is a random degradation pipeline rather
than a single op, because the failure mode this addresses is real photographs
being simultaneously lower-resolution, softer and more compressed than the
lab plates:
    horizontal flip            p=0.5
    resolution jitter          p=0.8   downscale to 120-400 px, upsample back
    motion / gaussian blur     p=0.4
    JPEG recompression         p=0.5   quality 30-75
The pristine original is always kept, so high-resolution capability is not
traded away.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/build_classification_from_detection.py \
        --src imgs/detection_set_v3 --out imgs/classification_set_v3
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
CROP = (640, 640)
MIN_CROP_PX = 32
RES_JITTER = (120, 400)
JPEG_Q = (30, 75)


def aug_flip(i):  return cv2.flip(i, 1)


def aug_res(img):
    h, w = img.shape[:2]
    t = random.randint(*RES_JITTER)
    small = cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def aug_motion(img):
    k = random.choice([5, 7, 9, 11])
    ker = np.zeros((k, k))
    d = random.choice("hvd")
    if d == "h":   ker[k//2, :] = 1.0/k
    elif d == "v": ker[:, k//2] = 1.0/k
    else:          np.fill_diagonal(ker, 1.0/k)
    return cv2.filter2D(img, -1, ker)


def aug_gauss(img):
    return cv2.GaussianBlur(img, (random.choice([3, 5, 7]),)*2, 0)


def aug_jpeg(img):
    ok, enc = cv2.imencode(".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, random.randint(*JPEG_Q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def degrade(img):
    o = img.copy()
    if random.random() < 0.5: o = aug_flip(o)
    if random.random() < 0.8: o = aug_res(o)
    if random.random() < 0.4:
        o = aug_motion(o) if random.random() < 0.5 else aug_gauss(o)
    if random.random() < 0.5: o = aug_jpeg(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="imgs/detection_set_v3")
    ap.add_argument("--out", default="imgs/classification_set_v3")
    ap.add_argument("--aug-factor", type=int, default=4,
                    help="total copies per training original, including it")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED)
    src = ROOT / args.src
    out = ROOT / args.out
    names = yaml.safe_load(open(src / "data.yaml"))["names"]

    if out.exists():
        shutil.rmtree(out)

    # detection split -> classification split
    split_map = {"train": "train_orig", "valid": "val", "test": "test"}
    counts = Counter(); skipped = 0

    for det_split, cls_split in split_map.items():
        img_dir = src / det_split / "images"
        lbl_dir = src / det_split / "labels"
        if not img_dir.is_dir():
            continue
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in IMG_EXTS:
                continue
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            with Image.open(img_p) as im:
                im = im.convert("RGB")
                W, H = im.size
                for i, line in enumerate(lbl_p.read_text().splitlines()):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cls = names[int(p[0])]
                    xc, yc, w, h = (float(v) for v in p[1:5])
                    x1 = max(0, int((xc - w/2) * W)); y1 = max(0, int((yc - h/2) * H))
                    x2 = min(W, int((xc + w/2) * W)); y2 = min(H, int((yc + h/2) * H))
                    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
                        skipped += 1
                        continue
                    d = out / cls_split / cls
                    d.mkdir(parents=True, exist_ok=True)
                    im.crop((x1, y1, x2, y2)).resize(CROP, Image.LANCZOS).save(
                        d / f"{img_p.stem}_{i}.jpg", quality=95)
                    counts[cls_split] += 1

    print("crops per split:", dict(counts), f"(skipped tiny: {skipped})")

    # augment the training partition only
    print(f"\naugmenting train_orig -> train (factor {args.aug_factor}) ...")
    n_orig = n_aug = 0
    for cls_dir in sorted((out / "train_orig").iterdir()):
        if not cls_dir.is_dir():
            continue
        dst = out / "train" / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(cls_dir.iterdir()):
            img = cv2.imread(str(f))
            if img is None:
                continue
            shutil.copy2(f, dst / f.name); n_orig += 1
            for k in range(1, args.aug_factor):
                cv2.imwrite(str(dst / f"{f.stem}_aug{k}.jpg"), degrade(img))
                n_aug += 1
    print(f"  {n_orig} originals -> {n_orig + n_aug} total")

    print(f"\ndataset at {out}")
    print(f"  train (augmented): {n_orig + n_aug}")
    print(f"  val:               {counts['val']}")
    print(f"  test:              {counts['test']}")
    print(f"  genera:            {len(list((out/'train').iterdir()))}")


if __name__ == "__main__":
    main()
