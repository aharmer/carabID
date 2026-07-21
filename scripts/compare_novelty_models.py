"""
compare_novelty_models.py

Compare two models' novelty detection on identical data.

Three things this controls for that a naive comparison does not:

  same known set   scoring each model against its own validation split compares
                   different images, not different models.
  excluded genera  imgs/detection_set had six class indices pointing at the
                   wrong genus, which also SPLIT six real genera across two
                   class centroids each (Molopsida appears under both
                   'molopsida' and 'gourlayia'). All twelve names are dropped
                   from the known set so neither model is judged on them.
  matched fallout  each model's fitted threshold sits at a different point on
                   its own score distribution. Recall is therefore reported at
                   a threshold chosen per model to hit the SAME false-alarm
                   rate, which is the only way the recalls are comparable.

Caveat that cannot be removed: the sources are the same specimens re-exported,
so any known set is training data for at least one model. Fallout is optimistic
for both; the comparison between them is still like for like.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/compare_novelty_models.py
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
from calibration_utils import MahalanobisOOD                # noqa: E402
from calibrate_yolo_classifier import get_linear_layer      # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# The six misaligned indices and the six real genera they absorbed.
AFFECTED = {
    "gourlayia", "molopsida", "kupetrechus", "tuiharpalus",
    "kiwitrechus", "synteratus", "kenodactylus", "plocamostethus",
    "egadroma", "mecyclothorax", "kupeharpalus", "tarastethus",
}

MODELS = {
    "OLD (deployed)": dict(
        clf=ROOT / "app/static/classification.pt",
        det=ROOT / "app/static/detection.pt",
        ood=ROOT / "app/static/ood_model.pkl"),
    "NEW (v3 unified)": dict(
        clf=ROOT / "runs/classify/classify_v3_unified/weights/best.pt",
        det=ROOT / "runs/detect/detect_v3_unified/weights/best.pt",
        ood=ROOT / "runs/classify/classify_v3_unified/weights/ood_model.pkl"),
}


def contrast_stretch(img):
    a = np.asarray(img, dtype=np.float32)
    p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
    if p98 - p2 == 0:
        return img
    return Image.fromarray(np.clip((a-p2)*(255.0/(p98-p2)), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", default="imgs/classification_set_v3/val")
    ap.add_argument("--ood", default="imgs/ood_test")
    ap.add_argument("--per-genus", type=int, default=3)
    args = ap.parse_args()

    known_dir = ROOT / args.known
    ood_files = sorted(f for f in (ROOT / args.ood).iterdir()
                       if f.suffix.lower() in IMG_EXTS)
    known_files, dropped = [], 0
    for g in sorted(known_dir.iterdir()):
        if not g.is_dir():
            continue
        if g.name.lower() in AFFECTED:
            dropped += 1
            continue
        known_files += list(g.glob("*.jpg"))[:args.per_genus]

    print(f"known set : {len(known_files)} images "
          f"({dropped} genera excluded as mislabel-affected)")
    print(f"novel set : {len(ood_files)} images\n")

    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    results = {}

    for label, m in MODELS.items():
        clf = YOLO(str(m["clf"])); clf.model.eval()
        det = YOLO(str(m["det"]))
        ood = MahalanobisOOD.load(m["ood"])
        linear = get_linear_layer(clf)

        @torch.no_grad()
        def score(crop):
            buf = []
            h = linear.register_forward_hook(lambda mm, i, o: buf.append(i[0].detach().cpu()))
            clf.model(tf(crop.resize((640, 640))).unsqueeze(0)); h.remove()
            return float(ood.score(F.normalize(buf[0], dim=-1).numpy()[0]))

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

        nov = []
        for f in ood_files:
            c = crop_of(Image.open(f).convert("RGB"))
            if c is not None:
                nov.append(score(c))
        kno = [score(Image.open(p).convert("RGB")) for p in known_files]
        results[label] = (np.array(nov), np.array(kno), ood.threshold)
        print(f"{label}: scored {len(nov)} novel, {len(kno)} known")

    print(f"\n{'model':<20}{'known med':>11}{'novel med':>11}{'ratio':>8}"
          f"{'own thr':>9}{'recall':>9}{'fallout':>9}")
    for label, (nov, kno, thr) in results.items():
        print(f"{label:<20}{np.median(kno):>11.0f}{np.median(nov):>11.0f}"
              f"{np.median(nov)/np.median(kno):>8.2f}{thr:>9.0f}"
              f"{f'{(nov>thr).mean()*100:.0f}%':>9}{f'{(kno>thr).mean()*100:.0f}%':>9}")

    print(f"\nMatched-fallout comparison (threshold set per model):")
    print(f"{'target fallout':>15}" + "".join(f"{l:>22}" for l in results))
    for target in (0.02, 0.05, 0.10, 0.20):
        row = f"{f'{target*100:.0f}%':>15}"
        for label, (nov, kno, _) in results.items():
            t = np.quantile(kno, 1 - target)
            row += f"{f'recall {(nov>t).mean()*100:.0f}%  (thr {t:.0f})':>22}"
        print(row)


if __name__ == "__main__":
    main()
