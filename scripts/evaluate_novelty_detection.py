"""
evaluate_novelty_detection.py

Validate the Mahalanobis novelty detector against genuinely novel genera.

Until now the OOD threshold could only be characterised against in-distribution
data, so the cost of raising it was unknown.  imgs/ood_test/ holds specimens
from genera that are NOT among the 76 the model was trained on, photographed in
the same lab style as the training set.  That isolates *semantic* novelty (an
unknown genus) from *covariate* shift (an unusual photograph), which is the
distinction the detector is supposed to make and the one that matters when
choosing a threshold.

Reports, for the deployed model:
  - recall  : novel-genus specimens correctly flagged
  - fallout : known lab specimens incorrectly flagged
  - a threshold sweep showing how the two trade off

Run from the carabID root:
    conda run -n ultralytics-env python scripts/evaluate_novelty_detection.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import MahalanobisOOD, TemperatureScaler      # noqa: E402
from calibrate_yolo_classifier import get_linear_layer               # noqa: E402

STATIC   = ROOT / "app" / "static"
OOD_DIR  = ROOT / "imgs" / "ood_test"
KNOWN_DIR = ROOT / "imgs" / "final_classification_set_v2" / "val"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def contrast_stretch(img):
    arr = np.asarray(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 - p2 == 0:
        return img
    return Image.fromarray(
        np.clip((arr - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8))


def main():
    det = YOLO(str(STATIC / "detection.pt"))
    clf = YOLO(str(STATIC / "classification.pt")); clf.model.eval()
    sc  = TemperatureScaler.load(STATIC / "temperature.pt")
    ood = MahalanobisOOD.load(STATIC / "ood_model.pkl")
    names  = set(clf.names.values())
    linear = get_linear_layer(clf)
    thr = ood.threshold

    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

    @torch.no_grad()
    def score(crop):
        emb = []
        def hook(m, i, o): emb.append(i[0].detach().cpu())
        h = linear.register_forward_hook(hook)
        clf.model(tf(crop.resize((640, 640))).unsqueeze(0)); h.remove()
        return float(ood.score(F.normalize(emb[0], dim=-1).numpy()[0]))

    def detect_crop(im):
        """App pipeline: try both orientations, crop the more confident view."""
        best = None
        for rot in (0, 90):
            view = im.rotate(rot, expand=True) if rot else im
            res  = det.predict(contrast_stretch(view), conf=0.50, verbose=False)
            b = res[0].boxes
            if not b or len(b) == 0:
                continue
            c = float(max(x.conf[0] for x in b))
            if best is None or c > best[0]:
                bb = max(b, key=lambda x: (x.xyxy[0][2]-x.xyxy[0][0])*(x.xyxy[0][3]-x.xyxy[0][1]))
                best = (c, view.crop(tuple(map(int, bb.xyxy[0].cpu().numpy()))))
        return best[1] if best else None

    # ---- novel genera ------------------------------------------------- #
    files = sorted(f for f in OOD_DIR.iterdir() if f.suffix.lower() in IMG_EXTS)
    # filenames are <Genus>-<species>... or nzac_<Genus>_<species>...
    genera = sorted({re.split(r"[-_ ]", f.stem.replace("nzac_", "", 1))[0]
                     for f in files})
    overlap = [g for g in genera if g in names]
    print(f"OOD genera present: {', '.join(genera)}")
    print(f"Any also in the trained 76? {overlap if overlap else 'no - all genuinely novel'}\n")

    novel_scores, undetected = [], 0
    for f in files:
        crop = detect_crop(Image.open(f).convert("RGB"))
        if crop is None:
            undetected += 1
            continue
        novel_scores.append(score(crop))
    novel_scores = np.array(novel_scores)

    # ---- known genera (in-distribution lab crops) --------------------- #
    known_files = [p for g in sorted(KNOWN_DIR.iterdir()) if g.is_dir()
                   for p in list(g.glob("*.jpg"))[:3]]
    known_scores = np.array([score(Image.open(p).convert("RGB")) for p in known_files])

    print(f"{'set':<26}{'n':>5}{'median':>10}{'min':>9}{'max':>9}")
    print(f"{'novel genera (OOD)':<26}{len(novel_scores):>5}"
          f"{np.median(novel_scores):>10.0f}{novel_scores.min():>9.0f}{novel_scores.max():>9.0f}")
    print(f"{'known genera (lab)':<26}{len(known_scores):>5}"
          f"{np.median(known_scores):>10.0f}{known_scores.min():>9.0f}{known_scores.max():>9.0f}")
    if undetected:
        print(f"  ({undetected} OOD image(s) had no detection and were skipped)")

    rec = (novel_scores > thr).mean() * 100
    fal = (known_scores > thr).mean() * 100
    print(f"\nDeployed threshold {thr:.0f}:")
    print(f"  novel genera flagged (recall) : {(novel_scores>thr).sum()}/{len(novel_scores)} ({rec:.0f}%)")
    print(f"  known genera flagged (fallout): {(known_scores>thr).sum()}/{len(known_scores)} ({fal:.0f}%)")

    print(f"\n{'threshold':>10}{'novel flagged':>16}{'known flagged':>16}")
    for t in (1784, 2000, 2331, 2500, 3000, 3067, 3500, 4000):
        print(f"{t:>10}{f'{(novel_scores>t).mean()*100:.0f}%':>16}"
              f"{f'{(known_scores>t).mean()*100:.0f}%':>16}")


if __name__ == "__main__":
    main()
