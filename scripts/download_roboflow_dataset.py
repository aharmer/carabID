"""
download_roboflow_dataset.py

Download a versioned Roboflow dataset and CHECK ITS ANNOTATION CONVENTION
before it is trusted.

The check is the point. An earlier iNaturalist export was annotated
head+pronotum while this project annotates pronotum+elytra; the two overlap
only at the pronotum, and training on the mixture produced a detector that
emitted pronotum-only boxes or nothing at all. The mismatch was invisible in
the metrics and only obvious once boxes were drawn, so this script always
draws them.

The API key is read from ROBOFLOW_API_KEY and never printed. Set it yourself:
    $env:ROBOFLOW_API_KEY = "<your key>"      # PowerShell
    set ROBOFLOW_API_KEY=<your key>           # cmd.exe

Run from the carabID root:
    conda run -n ultralytics-env python scripts/download_roboflow_dataset.py \
        --workspace rainna --project extra_nzac --version 1
    # or, on an already-downloaded dataset:
    conda run -n ultralytics-env python scripts/download_roboflow_dataset.py \
        --verify-only --dir imgs/extra_nzac_v1
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Reference geometry, measured on imgs/detection_set/train (the lab convention).
REF = dict(aspect_median=1.41, cover_median=0.23)


def iter_pairs(root: Path):
    """Yield (image, label) pairs across any YOLO split layout."""
    for img_dir in sorted(root.rglob("images")):
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                yield img, lbl


def verify(root: Path, n_montage: int = 48):
    pairs = list(iter_pairs(root))
    if not pairs:
        sys.exit(f"no image/label pairs found under {root}")

    aspects, covers, counts = [], [], []
    for img, lbl in pairs:
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        counts.append(len(lines))
        for line in lines:
            p = line.split()
            if len(p) < 5:
                continue
            w, h = float(p[3]), float(p[4])
            if h > 0:
                aspects.append(w / h)
            covers.append(w * h)

    a, c = np.array(aspects), np.array(covers)
    print(f"pairs: {len(pairs)}   boxes: {len(a)}   "
          f"images with no box: {sum(1 for x in counts if x == 0)}")
    print(f"\n{'metric':<22}{'this dataset':>14}{'lab reference':>15}")
    print(f"{'aspect w/h (median)':<22}{np.median(a):>14.2f}{REF['aspect_median']:>15.2f}")
    print(f"{'frame cover (median)':<22}{np.median(c):>14.2f}{REF['cover_median']:>15.2f}")
    print(f"{'aspect IQR':<22}{np.percentile(a,25):>7.2f}-{np.percentile(a,75):<6.2f}")

    d_asp = abs(np.median(a) - REF["aspect_median"]) / REF["aspect_median"]
    if d_asp > 0.25:
        print(f"\n  WARNING: median aspect differs from the lab convention by "
              f"{d_asp*100:.0f}%. The previous bad export sat at 1.12 against "
              f"1.41. Inspect the montage before training on this.")
    else:
        print(f"\n  aspect is within {d_asp*100:.0f}% of the lab convention.")

    # Always draw boxes: geometry alone did not reveal the last mismatch.
    out = root / "_convention_check.jpg"
    sample = random.Random(42).sample(pairs, min(n_montage, len(pairs)))
    TH, COLS = 220, 8
    rows = (len(sample) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TH, rows * TH), (255, 255, 255))
    for i, (img_p, lbl_p) in enumerate(sample):
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for line in lbl_p.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            xc, yc, w, h = (float(v) for v in p[1:5])
            d.rectangle([(xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H],
                        outline=(255, 0, 0), width=max(3, W//200))
        im.thumbnail((TH, TH))
        sheet.paste(im, ((i % COLS)*TH + (TH-im.width)//2,
                         (i//COLS)*TH + (TH-im.height)//2))
    sheet.save(out, quality=90)
    print(f"\n  convention check montage: {out}")
    print("  Confirm the boxes cover pronotum+elytra (head and legs excluded).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="rainna")
    ap.add_argument("--project", default="extra_nzac")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--format", default="yolov11")
    ap.add_argument("--dir", default="imgs/extra_nzac_v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    dest = (ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)

    if not args.verify_only:
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            sys.exit("ROBOFLOW_API_KEY is not set in this process.\n"
                     "Set it yourself (do not paste it into a chat):\n"
                     '    $env:ROBOFLOW_API_KEY = "<your key>"   # PowerShell\n'
                     "Note a key set in another terminal is not visible here; "
                     "use setx to persist it, or run this command yourself.")
        from roboflow import Roboflow
        rf = Roboflow(api_key=key)
        version = rf.workspace(args.workspace).project(args.project).version(args.version)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.workspace}/{args.project} v{args.version} "
              f"as {args.format} -> {dest}")
        version.download(args.format, location=str(dest))

    print(f"\n=== annotation convention check: {dest} ===")
    verify(dest)


if __name__ == "__main__":
    main()
