"""
evaluate_userstyle.py

Report a classification model's performance separately on lab-style and
user-style (iNaturalist) test crops, plus the novelty/domain-flag rate.
Useful for tracking whether added user-style data closes the domain gap.

Assumes the dataset was built by build_userstyle_classification_set.py, i.e.
final_classification_set_v*/test/<genus>/ contains lab crops and
inat_*.jpg user-style crops side by side, and the model weights directory
holds best.pt, temperature.pt and ood_model.pkl.

Usage (from the carabID root):
    conda run -n ultralytics-env python scripts/evaluate_userstyle.py \
        --weights runs/classify/final_carabid_v4_11ncls_ep30_do02_lr001/weights \
        --data    imgs/final_classification_set_v4
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import MahalanobisOOD, TemperatureScaler          # noqa: E402
from calibrate_yolo_classifier import get_linear_layer                   # noqa: E402


def to_tensor(pil):
    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    return tf(pil).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="weights dir with best.pt, temperature.pt, ood_model.pkl")
    ap.add_argument("--data", required=True,
                    help="dataset root with a test/ split")
    args = ap.parse_args()

    w = Path(args.weights)
    clf = YOLO(str(w / "best.pt")); clf.model.eval()
    sc = TemperatureScaler.load(w / "temperature.pt")
    ood = MahalanobisOOD.load(w / "ood_model.pkl")
    names = clf.names
    linear = get_linear_layer(clf)
    test = Path(args.data) / "test"

    @torch.no_grad()
    def infer(pil):
        emb, log = [], []
        def hook(m, i, o):
            emb.append(i[0].detach().cpu()); log.append(o.detach().cpu())
        h = linear.register_forward_hook(hook)
        clf.model(to_tensor(pil.resize((640, 640)))); h.remove()
        probs = sc.calibrate(log[0].cpu())[0].numpy()
        e = F.normalize(emb[0], dim=-1).numpy()[0]
        return probs, float(ood.score(e))

    print(f"Model     : {w}")
    print(f"OOD thresh: {ood.threshold:.0f}   T: {sc.temperature.item():.3f}\n")
    print(f"{'subset':<16}{'n':>5}{'top1':>12}{'top3':>12}{'flagged':>12}")

    for subset, want_inat in (("lab-style", False), ("user-style", True)):
        t1 = t3 = flagged = n = 0
        for gdir in sorted(test.iterdir()):
            if not gdir.is_dir():
                continue
            for f in gdir.glob("*.jpg"):
                if f.name.startswith("inat_") != want_inat:
                    continue
                probs, score = infer(Image.open(f).convert("RGB"))
                order = np.argsort(probs)[::-1]; n += 1
                t1 += names[int(order[0])] == gdir.name
                t3 += gdir.name in [names[int(i)] for i in order[:3]]
                flagged += score > ood.threshold
        if n:
            print(f"{subset:<16}{n:>5}{f'{t1} ({t1/n*100:.0f}%)':>12}"
                  f"{f'{t3} ({t3/n*100:.0f}%)':>12}{f'{flagged} ({flagged/n*100:.0f}%)':>12}")


if __name__ == "__main__":
    main()
