"""
prepare_userstyle_data.py

Integrate the iNaturalist user-style images into the detection dataset and
stage per-genus classification crops.

WHAT IT DOES
------------
1. Reads the Roboflow iNaturalist export (243 images, 1 box each), remaps its
   54-class indices onto the project's 76-class detection ordering (by genus
   name), and drops any genus not in the 76-class set (only Trigonothops, 1 img).
2. Assigns each image to a split:
     - Held-out user-style TEST set is drawn ONLY from the two largest genera,
       Laemostenus and Mecodema (per the project decision).  TEST_FRACTION of
       each is reserved for test; the rest go to train.
     - Every other genus goes entirely to train.
     - No user-style images go to val (val stays lab-only).
3. Builds `imgs/detection_set_v2/` = full copy of `detection_set` + the iNat
   images and remapped labels added to the matching train/test split.
   Originals are never modified.
4. Stages 640x640 classification crops (thorax+elytra box, stretched — same as
   convert_detection_to_classification.py) into
   `imgs/_inat_crops/<split>/<genus>/` for the classification builder.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/prepare_userstyle_data.py
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

import yaml
from PIL import Image

# ─────────────────────────────────────────────────────────────────── #
#  Configuration                                                       #
# ─────────────────────────────────────────────────────────────────── #

SEED            = 42
TEST_FRACTION   = 0.30                       # of each held-out-test genus
HELDOUT_GENERA  = {"Laemostenus", "Mecodema"}

INAT_DIR        = Path("D:/Dropbox/downloads/inaturalist_V2.v5i.yolov11")
DET_SRC         = Path("imgs/detection_set")
DET_DST         = Path("imgs/detection_set_v2")
CROP_STAGE      = Path("imgs/_inat_crops")

CROP_SIZE       = (640, 640)
MIN_CROP_PX     = 32
AR_LIMITS       = (0.2, 5.0)                  # drop degenerate boxes

IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def norm_to_px(xc, yc, w, h, W, H):
    x1 = max(0, int((xc - w / 2) * W)); y1 = max(0, int((yc - h / 2) * H))
    x2 = min(W, int((xc + w / 2) * W)); y2 = min(H, int((yc + h / 2) * H))
    return x1, y1, x2, y2


def main():
    random.seed(SEED)

    inat_names = yaml.safe_load(open(INAT_DIR / "data.yaml"))["names"]   # 54
    det_names  = yaml.safe_load(open(DET_SRC / "data.yaml"))["names"]    # 76
    det_index  = {g: i for i, g in enumerate(det_names)}

    img_dir = INAT_DIR / "train" / "images"
    lab_dir = INAT_DIR / "train" / "labels"

    # ---- collect one (image, genus, box) record per usable image ---------- #
    records = []                         # (img_path, genus, (xc,yc,w,h))
    dropped_genus = defaultdict(int)
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lab = lab_dir / (img.stem + ".txt")
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        # one box per image in this export; keep the first valid box
        parts = lines[0].split()
        genus = inat_names[int(parts[0])]
        if genus not in det_index:
            dropped_genus[genus] += 1
            continue
        box = tuple(float(v) for v in parts[1:5])
        records.append((img, genus, box))

    print(f"Usable iNat images: {len(records)}")
    if dropped_genus:
        print(f"Dropped (genus not in 76-class set): {dict(dropped_genus)}")

    # ---- assign splits ---------------------------------------------------- #
    by_genus = defaultdict(list)
    for r in records:
        by_genus[r[1]].append(r)

    split_of = {}                        # img_path -> "train"/"test"
    n_test = 0
    for genus, recs in by_genus.items():
        recs = recs.copy()
        random.shuffle(recs)
        if genus in HELDOUT_GENERA:
            k = int(len(recs) * TEST_FRACTION)
            for r in recs[:k]:
                split_of[r[0]] = "test"
            for r in recs[k:]:
                split_of[r[0]] = "train"
            n_test += k
            print(f"  {genus}: {len(recs)} -> {k} test, {len(recs)-k} train")
        else:
            for r in recs:
                split_of[r[0]] = "train"
    print(f"User-style test images (Laemostenus+Mecodema): {n_test}")
    print(f"User-style train images: {len(records) - n_test}")

    # ---- build detection_set_v2 (copy original, then add iNat) ------------ #
    if DET_DST.exists():
        print(f"\nRemoving existing {DET_DST}")
        shutil.rmtree(DET_DST)
    print(f"Copying {DET_SRC} -> {DET_DST} (preserving original)...")
    shutil.copytree(DET_SRC, DET_DST)

    # update data.yaml paths in the copy (val split lives in valid/)
    yml = yaml.safe_load(open(DET_SRC / "data.yaml"))
    split_dir = {"train": "train", "val": "valid", "test": "test"}
    for k, sub in split_dir.items():
        yml[k] = str((DET_DST / sub / "images").resolve()).replace("\\", "/")
    yaml.safe_dump(yml, open(DET_DST / "data.yaml", "w"), sort_keys=False)

    # ---- write iNat images + remapped labels, and stage crops ------------- #
    if CROP_STAGE.exists():
        shutil.rmtree(CROP_STAGE)

    n_crops = defaultdict(int)
    skipped = 0
    for img_path, genus, (xc, yc, w, h) in records:
        split = split_of[img_path]
        det_idx = det_index[genus]

        # detection: copy image + remapped single-line label
        dst_img = DET_DST / split / "images" / f"inat_{img_path.name}"
        dst_lab = DET_DST / split / "labels" / f"inat_{img_path.stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lab.write_text(f"{det_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # classification: crop thorax+elytra box, stretch to 640x640
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x1, y1, x2, y2 = norm_to_px(xc, yc, w, h, W, H)
            cw, ch = x2 - x1, y2 - y1
            ar = cw / ch if ch else 0
            if cw < MIN_CROP_PX or ch < MIN_CROP_PX or not (AR_LIMITS[0] <= ar <= AR_LIMITS[1]):
                skipped += 1
                continue
            crop = im.crop((x1, y1, x2, y2)).resize(CROP_SIZE, Image.LANCZOS)
        out_dir = CROP_STAGE / split / genus
        out_dir.mkdir(parents=True, exist_ok=True)
        crop.save(out_dir / f"inat_{img_path.stem}_0.jpg", quality=95)
        n_crops[split] += 1

    print(f"\nStaged classification crops: {dict(n_crops)}  (skipped degenerate: {skipped})")
    print(f"detection_set_v2 ready at {DET_DST}")
    print(f"crops staged at {CROP_STAGE}")


if __name__ == "__main__":
    main()
