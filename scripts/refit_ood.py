"""
refit_ood.py

Refit the Mahalanobis novelty detector on a chosen subset of the training data,
leaving the classifier untouched.

Why this can help: the detector models "how far is this from a typical training
specimen", using per-class centroids and a pooled covariance. Fitting it on the
augmented training set folds heavy resolution jitter, blur and JPEG damage into
that covariance, and fitting it across photographically heterogeneous sources
widens it further. A wider covariance shrinks every Mahalanobis distance, so
genuinely novel specimens stop standing out even when the classifier is fine.

Fitting on un-augmented, visually consistent originals should tighten the
distribution and restore separation without giving up classifier accuracy.

Variants compared:
    augmented   the pipeline default (train/, all sources, 4x augmented)
    originals   train_orig/, un-augmented, all sources
    lab-only    train_orig/, museum-plate sources only

Run from the carabID root:
    conda run -n ultralytics-env python scripts/refit_ood.py \
        --weights runs/classify/classify_v3_unified/weights \
        --data imgs/classification_set_v3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import MahalanobisOOD                 # noqa: E402
from calibrate_yolo_classifier import get_linear_layer       # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}
LAB_SOURCES = ("carabids_genus", "carabidae_extra")   # museum plates


def contrast_stretch(img):
    a = np.asarray(img, dtype=np.float32)
    p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
    if p98 - p2 == 0:
        return img
    return Image.fromarray(np.clip((a-p2)*(255.0/(p98-p2)), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detection", required=True)
    ap.add_argument("--ood-dir", default="imgs/ood_test")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--save", default=None,
                    help="variant to write back as ood_model.pkl "
                         "(augmented|originals|lab-only)")
    args = ap.parse_args()

    w = ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)

    clf = YOLO(str(w / "best.pt")); clf.model.eval()
    det = YOLO(str(args.detection))
    names = clf.names
    name_to_idx = {v: k for k, v in names.items()}
    linear = get_linear_layer(clf)
    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

    @torch.no_grad()
    def embed(img):
        buf = []
        h = linear.register_forward_hook(lambda m, i, o: buf.append(i[0].detach().cpu()))
        clf.model(tf(img.resize((640, 640))).unsqueeze(0)); h.remove()
        return F.normalize(buf[0], dim=-1).numpy()[0]

    def embed_dir(root: Path, source_filter=None):
        X, y = [], []
        for gdir in sorted(root.iterdir()):
            if not gdir.is_dir() or gdir.name not in name_to_idx:
                continue
            for f in gdir.iterdir():
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                if source_filter and not f.name.startswith(source_filter):
                    continue
                X.append(embed(Image.open(f).convert("RGB")))
                y.append(name_to_idx[gdir.name])
        return np.array(X), np.array(y)

    def crop_of(im):
        best = None
        for rot in (0, 90):
            v = im.rotate(rot, expand=True) if rot else im
            r = det.predict(contrast_stretch(v), conf=0.50, verbose=False)
            b = r[0].boxes
            if not b or len(b) == 0:
                continue
            c = float(max(x.conf[0] for x in b))
            if best is None or c > best[0]:
                bb = max(b, key=lambda x: (x.xyxy[0][2]-x.xyxy[0][0])*(x.xyxy[0][3]-x.xyxy[0][1]))
                best = (c, v.crop(tuple(map(int, bb.xyxy[0].cpu().numpy()))))
        return best[1] if best else None

    print("embedding validation set ...")
    Xva, _ = embed_dir(data / "val")
    print(f"  val: {len(Xva)}")

    print("embedding novel-genus set ...")
    Xood = []
    for f in sorted((ROOT / args.ood_dir).iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        c = crop_of(Image.open(f).convert("RGB"))
        if c is not None:
            Xood.append(embed(c))
    Xood = np.array(Xood)
    print(f"  novel: {len(Xood)}")

    variants = {}
    print("\nembedding fit subsets ...")
    if (data / "train").is_dir():
        variants["augmented"] = embed_dir(data / "train")
        print(f"  augmented: {len(variants['augmented'][0])}")
    if (data / "train_orig").is_dir():
        variants["originals"] = embed_dir(data / "train_orig")
        print(f"  originals: {len(variants['originals'][0])}")
        variants["lab-only"] = embed_dir(data / "train_orig", source_filter=LAB_SOURCES)
        print(f"  lab-only:  {len(variants['lab-only'][0])}")

    print(f"\n{'variant':<12}{'fit n':>8}{'thr':>8}{'novel recall':>15}"
          f"{'known fallout':>15}{'separation':>12}")
    results = {}
    for label, (X, y) in variants.items():
        ood = MahalanobisOOD().fit(X, y)
        thr = ood.tune_threshold(Xva, fpr_target=args.fpr)
        sn = ood.score_batch(Xood)
        sk = ood.score_batch(Xva)
        rec = (sn > thr).mean()
        fal = (sk > thr).mean()
        # separation: gap between known 95th pct and novel 5th pct, scaled
        sep = (np.percentile(sn, 5) - np.percentile(sk, 95)) / np.median(sk)
        results[label] = (ood, thr, rec, fal, sep)
        print(f"{label:<12}{len(X):>8}{thr:>8.0f}{f'{rec*100:.0f}%':>15}"
              f"{f'{fal*100:.0f}%':>15}{sep:>12.2f}")

    print("\n(separation > 0 means the distributions are cleanly apart;"
          " higher is better)")

    if args.save:
        ood, thr, rec, fal, _ = results[args.save]
        out = w / "ood_model.pkl"
        ood.save(out)
        print(f"\nsaved '{args.save}' variant -> {out}")
        print(f"  threshold {thr:.0f}, novel recall {rec*100:.0f}%, "
              f"known fallout {fal*100:.0f}%")


if __name__ == "__main__":
    main()
