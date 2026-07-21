"""
stage_candidate_model.py

Assemble a candidate model into a staging folder the app can be pointed at via
CARABID_STATIC, so it can be exercised in the real UI without overwriting the
deployed artefacts in app/static.

Also regenerates class_counts.csv from the candidate's own training data: the
app's "limited training data" warning reads that file, so a stale copy would
warn about the wrong genera.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/stage_candidate_model.py \
        --weights runs/classify/classify_v3_unified/weights \
        --detection runs/detect/detect_v3_unified/weights/best.pt \
        --train-dir imgs/classification_set_v3/train_orig \
        --threshold 1658 --out app/static_v3
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibrated"))
from calibration_utils import MahalanobisOOD          # noqa: E402

SHARED = ["carabid_icon.png", "example_image.jpg", "guidance_examples.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--detection", required=True)
    ap.add_argument("--train-dir", required=True,
                    help="un-augmented training crops, for class_counts.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the fitted novelty threshold")
    ap.add_argument("--out", default="app/static_v3")
    args = ap.parse_args()

    w = ROOT / args.weights
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    src_static = ROOT / "app/static"
    for f in SHARED:
        if (src_static / f).exists():
            shutil.copy2(src_static / f, out / f)

    shutil.copy2(ROOT / args.detection, out / "detection.pt")
    shutil.copy2(w / "best.pt", out / "classification.pt")
    shutil.copy2(w / "temperature.pt", out / "temperature.pt")
    shutil.copy2(w / "ood_model.pkl", out / "ood_model.pkl")

    if args.threshold is not None:
        ood = MahalanobisOOD.load(out / "ood_model.pkl")
        old = ood.threshold
        ood.threshold = float(args.threshold)
        ood.save(out / "ood_model.pkl")
        print(f"novelty threshold: {old:.0f} -> {ood.threshold:.0f}")

    # class_counts.csv from the candidate's own training data
    train = ROOT / args.train_dir
    rows = sorted(
        (g.name, len(list(g.glob("*.jpg"))))
        for g in train.iterdir() if g.is_dir()
    )
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["Class Name", "Image Count"])
        wtr.writerows(rows)
    low = [f"{n} ({c})" for n, c in rows if c < 20]
    print(f"class_counts.csv: {len(rows)} genera, "
          f"total {sum(c for _, c in rows)} originals")
    print(f"  genera under 20 images ({len(low)}): {', '.join(low) if low else 'none'}")

    print(f"\nstaged -> {out}")
    print("run with:")
    print(f'  $env:CARABID_STATIC = "{out}"')
    print("  streamlit run app/app.py")


if __name__ == "__main__":
    main()
