"""
screen_lab_likeness.py

Rank downloaded iNaturalist candidates by how closely they resemble the lab
specimen photographs, so a human can screen the promising ones first instead of
scrolling through field shots.

The goal is more variation *within* the lab domain, not expansion into field
photography, so this scores each candidate on three signals and sorts by them:

  ood        Mahalanobis distance from the training distribution (the same
             measure the app uses).  Low = looks like a training specimen.
             Note it also rises with low resolution, so it is not used alone.
  bg_plain   uniformity of the image border (lab shots sit on plain white or
             pale backgrounds; field shots have leaf litter, soil, moss).
             1.0 = perfectly uniform border.
  det_conf   detection confidence - the detector was trained on lab specimens,
             so a confident box is itself evidence of lab-like framing.

Writes a CSV ranked best-first plus contact-sheet JPEGs for quick visual triage.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/screen_lab_likeness.py \
        --dir imgs/inat_candidates
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import MahalanobisOOD, TemperatureScaler   # noqa: E402
from calibrate_yolo_classifier import get_linear_layer            # noqa: E402

STATIC = ROOT / "app" / "static"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def contrast_stretch(img):
    arr = np.asarray(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 - p2 == 0:
        return img
    return Image.fromarray(
        np.clip((arr - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8))


def border_uniformity(img, frac=0.08):
    """1.0 = perfectly uniform border. Lab plates score high, clutter low."""
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float32)
    b = max(2, int(256 * frac))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                             a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    sd = border.std(axis=0).mean()             # 0 = flat, ~80 = busy
    return float(max(0.0, 1.0 - sd / 60.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="imgs/inat_candidates")
    ap.add_argument("--sheets", type=int, default=60,
                    help="images per contact sheet")
    args = ap.parse_args()

    src = ROOT / args.dir
    det = YOLO(str(STATIC / "detection.pt"))
    clf = YOLO(str(STATIC / "classification.pt")); clf.model.eval()
    ood = MahalanobisOOD.load(STATIC / "ood_model.pkl")
    linear = get_linear_layer(clf)
    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

    @torch.no_grad()
    def ood_score(crop):
        emb = []
        def hook(m, i, o): emb.append(i[0].detach().cpu())
        h = linear.register_forward_hook(hook)
        clf.model(tf(crop.resize((640, 640))).unsqueeze(0)); h.remove()
        return float(ood.score(F.normalize(emb[0], dim=-1).numpy()[0]))

    def best_detection(im):
        best = None
        for rot in (0, 90):
            view = im.rotate(rot, expand=True) if rot else im
            res = det.predict(contrast_stretch(view), conf=0.25, verbose=False)
            b = res[0].boxes
            if not b or len(b) == 0:
                continue
            c = float(max(x.conf[0] for x in b))
            if best is None or c > best[0]:
                bb = max(b, key=lambda x: (x.xyxy[0][2]-x.xyxy[0][0])*(x.xyxy[0][3]-x.xyxy[0][1]))
                best = (c, view.crop(tuple(map(int, bb.xyxy[0].cpu().numpy()))))
        return best or (0.0, None)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"scoring {len(files)} candidates ...")
    rows = []
    for i, f in enumerate(files, 1):
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        conf, crop = best_detection(im)
        score = ood_score(crop) if crop is not None else float("nan")
        rows.append(dict(path=str(f.relative_to(ROOT)), genus=f.parent.name,
                         det_conf=round(conf, 3),
                         ood=None if crop is None else round(score, 1),
                         bg_plain=round(border_uniformity(im), 3),
                         width=im.size[0], height=im.size[1]))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    # rank: plain background and confident detection good, high OOD bad
    def key(r):
        o = r["ood"] if r["ood"] is not None else 9e9
        return (-(r["bg_plain"] * 2 + r["det_conf"]), o)
    rows.sort(key=key)

    out_csv = src / "lab_likeness.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nranked CSV: {out_csv}")

    # contact sheets, best-first
    sheet_dir = src / "_contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    THUMB, COLS = 160, 10
    for s in range(0, len(rows), args.sheets):
        chunk = rows[s:s + args.sheets]
        rowsn = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * THUMB, rowsn * (THUMB + 12)), (255, 255, 255))
        for j, r in enumerate(chunk):
            try:
                t = Image.open(ROOT / r["path"]).convert("RGB")
            except Exception:
                continue
            t.thumbnail((THUMB, THUMB))
            x, y = (j % COLS) * THUMB, (j // COLS) * (THUMB + 12)
            sheet.paste(t, (x + (THUMB - t.width) // 2, y + (THUMB - t.height) // 2))
        p = sheet_dir / f"sheet_{s//args.sheets + 1:03d}.jpg"
        sheet.save(p, quality=88)
    print(f"contact sheets: {sheet_dir} (best-first)")

    ok = [r for r in rows if r["ood"] is not None]
    if ok:
        arr = np.array([r["ood"] for r in ok])
        print(f"\ndetected in {len(ok)}/{len(rows)} images")
        print(f"OOD: median {np.median(arr):.0f}   "
              f"below threshold {ood.threshold:.0f}: "
              f"{(arr <= ood.threshold).sum()} ({(arr <= ood.threshold).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
