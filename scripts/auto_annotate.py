"""
auto_annotate.py

Propose YOLO detection labels for new images using the deployed detector, so
they do not have to be boxed by hand.

Why this is reasonable here: the detector was trained on the lab
pronotum+elytra convention, so boxes it proposes are consistent with the
existing 4,715 images by construction.  That is exactly what the Roboflow
iNaturalist export got wrong (it annotated head+pronotum), and the mismatch
made that data unusable.

Two caveats worth keeping in view:
  - Candidates were ranked partly on detection confidence, so the detector
    already handles them well.  Boxes it proposes therefore teach the DETECTOR
    little that it does not know; the value is mostly in the new crops for the
    CLASSIFIER, and in new specimens/backgrounds.
  - Auto-labels are proposals.  Anything below --review-conf, or with an
    implausible box, is routed to a review list instead of being accepted.

Outputs, in --out:
    images/          accepted images
    labels/          YOLO labels (class 0 - single-class detection)
    review/          images needing a human look, with boxes drawn
    _verify/         contact sheets of accepted boxes for spot-checking
    auto_annotation.csv

Run from the carabID root:
    conda run -n ultralytics-env python scripts/auto_annotate.py \
        --src imgs/inat_keepers --out imgs/inat_annotated
"""
import argparse
import csv
import shutil
from pathlib import Path

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import TemperatureScaler          # noqa: E402
from calibrate_yolo_classifier import get_linear_layer   # noqa: E402
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# A lab pronotum+elytra box occupies a limited, fairly consistent slice of the
# frame; boxes far outside this are detector failures, not specimens.
MIN_FRAC, MAX_FRAC = 0.02, 0.80
MIN_AR,  MAX_AR    = 0.35, 3.20


def contrast_stretch(img):
    a = np.asarray(img, dtype=np.float32)
    p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
    if p98 - p2 == 0:
        return img
    return Image.fromarray(np.clip((a - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="imgs/inat_keepers")
    ap.add_argument("--out", default="imgs/inat_annotated")
    ap.add_argument("--accept-conf", type=float, default=0.60,
                    help="accept boxes at or above this confidence")
    ap.add_argument("--review-conf", type=float, default=0.25,
                    help="below this, send to review rather than guess")
    ap.add_argument("--min-crop-prob", type=float, default=0.50,
                    help="minimum top class probability for the cropped region; "
                         "rejects confident boxes that sit on background")
    args = ap.parse_args()

    src, out = ROOT / args.src, ROOT / args.out
    det = YOLO(str(STATIC / "detection.pt"))
    # Detection confidence turns out to be a poor guide to whether a box is
    # actually ON the specimen: it happily returns confident boxes sitting on
    # background. Classifying the crop is a far better check, since a crop of
    # blank paper cannot look like any genus.
    clf = YOLO(str(STATIC / "classification.pt")); clf.model.eval()
    scaler = TemperatureScaler.load(STATIC / "temperature.pt")
    _tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

    # YOLO's Classify head softmaxes internally, so the module output is not
    # logits; hooking the final linear layer is the only way to get them
    # (softmaxing twice flattens every crop to ~1/76 and the check becomes
    # meaningless).
    _linear = get_linear_layer(clf)

    @torch.no_grad()
    def crop_is_beetle(crop):
        """Top calibrated class probability for the crop."""
        buf = []
        h = _linear.register_forward_hook(lambda m, i, o: buf.append(o.detach().cpu()))
        clf.model(_tf(crop.resize((640, 640))).unsqueeze(0))
        h.remove()
        p = scaler.calibrate(buf[0])[0].numpy()
        return float(p.max())
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images", "labels", "review", "_verify"):
        (out / sub).mkdir(parents=True)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"auto-annotating {len(files)} images ...")

    rows, accepted, review = [], [], []
    for f in files:
        im = Image.open(f).convert("RGB")
        # orientation-corrected detection, matching the app
        best = None
        for rot in (0, 90):
            view = im.rotate(rot, expand=True) if rot else im
            res = det.predict(contrast_stretch(view), conf=0.10, verbose=False)
            b = res[0].boxes
            if not b or len(b) == 0:
                continue
            c = float(max(x.conf[0] for x in b))
            if best is None or c > best[0]:
                bb = max(b, key=lambda x: (x.xyxy[0][2]-x.xyxy[0][0])*(x.xyxy[0][3]-x.xyxy[0][1]))
                best = (c, view, tuple(map(float, bb.xyxy[0].cpu().numpy())), rot)

        genus = f.parent.name
        if best is None:
            rows.append(dict(file=f.name, genus=genus, status="no_detection",
                             conf=0, frac="", ar=""))
            review.append((f, None, im))
            continue

        conf, view, (x1, y1, x2, y2), rot = best
        W, H = view.size
        frac = ((x2-x1) * (y2-y1)) / (W*H)
        ar   = (x2-x1) / max(y2-y1, 1e-6)
        plausible = (MIN_FRAC <= frac <= MAX_FRAC) and (MIN_AR <= ar <= MAX_AR)

        crop_p = crop_is_beetle(view.crop((x1, y1, x2, y2)))

        if conf >= args.accept_conf and plausible and crop_p >= args.min_crop_prob:
            status = "accepted"
            # YOLO format, single class, normalised centre/size
            xc, yc = (x1+x2)/2/W, (y1+y2)/2/H
            bw, bh = (x2-x1)/W, (y2-y1)/H
            stem = f"{genus}_{f.stem}"
            view.save(out / "images" / f"{stem}.jpg", quality=95)
            (out / "labels" / f"{stem}.txt").write_text(
                f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            accepted.append((out / "images" / f"{stem}.jpg", (x1, y1, x2, y2)))
        else:
            if conf < args.review_conf:
                status = "review_lowconf"
            elif not plausible:
                status = "review_implausible"
            elif crop_p < args.min_crop_prob:
                status = "review_crop_not_beetle"
            else:
                status = "review_midconf"
            review.append((f, (x1, y1, x2, y2), view))

        rows.append(dict(file=f.name, genus=genus, status=status,
                         conf=round(conf, 3), frac=round(frac, 3), ar=round(ar, 2),
                         crop_prob=round(crop_p, 3)))

    # review images, with the proposed box drawn so it can be judged quickly
    for f, box, view in review:
        v = view.copy()
        if box:
            d = ImageDraw.Draw(v)
            d.rectangle(box, outline=(255, 0, 0), width=max(3, v.width//200))
        v.thumbnail((700, 700))
        v.save(out / "review" / f"{f.parent.name}_{f.stem}.jpg", quality=88)

    # verification sheets of ACCEPTED boxes
    TH, COLS, PER = 200, 8, 48
    for s in range(0, len(accepted), PER):
        chunk = accepted[s:s+PER]
        rn = (len(chunk)+COLS-1)//COLS
        sheet = Image.new("RGB", (COLS*TH, rn*TH), (255, 255, 255))
        for j, (p, box) in enumerate(chunk):
            t = Image.open(p).convert("RGB")
            d = ImageDraw.Draw(t)
            d.rectangle(box, outline=(255, 0, 0), width=max(4, t.width//150))
            t.thumbnail((TH, TH))
            sheet.paste(t, ((j % COLS)*TH + (TH-t.width)//2,
                            (j//COLS)*TH + (TH-t.height)//2))
        sheet.save(out / "_verify" / f"verify_{s//PER+1:02d}.jpg", quality=88)

    with open(out / "auto_annotation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"\n{'status':<22}{'n':>5}")
    for k, v in c.most_common():
        print(f"{k:<22}{v:>5}  ({v/len(rows)*100:.0f}%)")
    print(f"\naccepted -> {out/'images'} + {out/'labels'}")
    print(f"needs review -> {out/'review'}  ({len(review)} images, boxes drawn)")
    print(f"spot-check accepted boxes -> {out/'_verify'}")


if __name__ == "__main__":
    main()
