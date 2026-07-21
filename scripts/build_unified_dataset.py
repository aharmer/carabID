"""
build_unified_dataset.py

Merge the three annotated sources into one detection dataset with a single
class list, bounding-box labels, and a stratified split.

What it reconciles:
  - Class lists differ per source (33 / 76 / 60 classes, different orders, and
    extra_nzac_v1 is lowercase).  Everything is remapped by NAME onto the
    project's 76-genus ordering, so indices can never silently disagree.
  - Known misspellings are corrected rather than becoming new classes.
  - Labels arrive as bounding boxes, 4-corner rectangles, and true
    segmentation polygons with up to ~94 points.  All are reduced to their
    bounding-box extent, which for these sources reproduces the project's
    pronotum+elytra convention (verified: median aspect 1.41-1.42 against the
    lab's 1.41, cover 0.22-0.23 against 0.23).
  - Identical images appearing in more than one source are dropped, since a
    duplicate landing in two different splits would leak.

Split is stratified per genus and made BEFORE any augmentation, matching the
corrected pipeline.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/build_unified_dataset.py
"""
import argparse
import hashlib
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}

SOURCES = [
    Path("D:/Dropbox/downloads/carabids_genus"),
    Path("D:/Dropbox/downloads/carabidae_extra"),
    ROOT / "imgs/extra_nzac_v1",
]

# Misspellings seen in the source class lists. Left uncorrected these become
# separate classes, and in the OOD set they would score a known genus as novel.
TYPOS = {
    "psemagtopterus": "psegmatopterus",
    "molopasida": "molopsida",
}

SEED = 42
RATIOS = (0.80, 0.15, 0.05)      # train / valid / test


def polygon_to_bbox(vals):
    """Return (xc, yc, w, h) from a YOLO label's numeric fields.

    Accepts bbox (4 values) or any polygon (>=6, even count); a polygon is
    reduced to its extent.
    """
    if len(vals) == 4:
        return tuple(vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="imgs/detection_set_v3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proj = yaml.safe_load(open(ROOT / "imgs/detection_set/data.yaml"))["names"]
    canon = {p.lower(): i for i, p in enumerate(proj)}     # name -> unified idx

    records, seen_hash = [], {}
    stats = Counter()
    unmapped = Counter()

    for src in SOURCES:
        names = yaml.safe_load(open(src / "data.yaml"))["names"]
        n_src = 0
        for img_dir in sorted(src.rglob("images")):
            lbl_dir = img_dir.parent / "labels"
            if not lbl_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    stats["no_label"] += 1
                    continue

                boxes, genera = [], set()
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    raw = names[int(p[0])].lower()
                    raw = TYPOS.get(raw, raw)
                    if raw not in canon:
                        unmapped[raw] += 1
                        continue
                    box = polygon_to_bbox([float(v) for v in p[1:]])
                    if box is None:
                        stats["bad_geometry"] += 1
                        continue
                    xc, yc, w, h = box
                    if not (0 < w <= 1 and 0 < h <= 1):
                        stats["bad_geometry"] += 1
                        continue
                    boxes.append((canon[raw], xc, yc, w, h))
                    genera.add(raw)
                if not boxes:
                    stats["no_valid_box"] += 1
                    continue

                digest = hashlib.md5(img.read_bytes()).hexdigest()
                if digest in seen_hash:
                    stats["duplicate_image"] += 1
                    continue
                seen_hash[digest] = img.name

                records.append(dict(img=img, boxes=boxes,
                                    genus=sorted(genera)[0], src=src.name))
                n_src += 1
        print(f"{src.name:<20} usable images: {n_src}")

    print(f"\ntotal usable: {len(records)}")
    for k, v in stats.most_common():
        print(f"  skipped ({k}): {v}")
    if unmapped:
        print(f"  UNMAPPED class names: {dict(unmapped)}")

    # ---- stratified split, per genus, before any augmentation ------------- #
    random.seed(SEED)
    by_genus = defaultdict(list)
    for r in records:
        by_genus[r["genus"]].append(r)

    split_of = {}
    for g, items in by_genus.items():
        items = items[:]
        random.shuffle(items)
        n = len(items)
        n_tr = max(1, int(n * RATIOS[0]))
        n_va = int(n * RATIOS[1])
        for i, r in enumerate(items):
            split_of[id(r)] = ("train" if i < n_tr
                               else "valid" if i < n_tr + n_va else "test")

    counts = Counter(split_of.values())
    print(f"\nsplit: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"genera represented: {len(by_genus)}/{len(proj)}")

    if args.dry_run:
        print("\nDRY RUN - nothing written.")
        return

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)

    for r in records:
        s = split_of[id(r)]
        stem = f"{r['src']}_{r['img'].stem}"
        shutil.copy2(r["img"], out / s / "images" / f"{stem}{r['img'].suffix}")
        (out / s / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                    for c, xc, yc, w, h in r["boxes"]))

    yml = {
        "train": str((out / "train" / "images").resolve()).replace("\\", "/"),
        "val":   str((out / "valid" / "images").resolve()).replace("\\", "/"),
        "test":  str((out / "test" / "images").resolve()).replace("\\", "/"),
        "nc": len(proj),
        "names": proj,
    }
    yaml.safe_dump(yml, open(out / "data.yaml", "w"), sort_keys=False)
    print(f"\nwritten -> {out}")

    per_genus = Counter(r["genus"] for r in records)
    thin = [g for g, n in per_genus.items() if n < 10]
    print(f"genera with fewer than 10 images: {len(thin)}")
    if thin:
        print("  " + ", ".join(sorted(thin)))


if __name__ == "__main__":
    main()
