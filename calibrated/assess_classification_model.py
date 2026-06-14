"""
Calibration-aware replacement for scripts/assess_classification_model.py.

Adds (all optional, activated by extra flags):
  --calibration-dir   directory containing temperature.pt + ood_model.pkl
  --ood-dir           directory of out-of-distribution images for AUROC

When --calibration-dir is supplied the report shows uncalibrated vs calibrated
metrics side-by-side and saves a reliability diagram.  Without it behaviour
is identical to the original assess script.

Usage
-----
    # Basic (identical to original)
    python calibrated/assess_classification_model.py \\
        --model  runs/classify/my_model/weights/best.pt \\
        --data   imgs/final_classification_set \\
        --split  test

    # With calibration artefacts
    python calibrated/assess_classification_model.py \\
        --model             runs/classify/my_model/weights/best.pt \\
        --data              imgs/final_classification_set \\
        --split             test \\
        --calibration-dir   runs/classify/my_model/weights \\
        --ood-dir           imgs/ood_images
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from calibration_utils import MahalanobisOOD, TemperatureScaler
from calibrate_yolo_classifier import extract_features, yolo_transform


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Assess YOLO classification model (with optional calibration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",           required=True,
                   help="Path to YOLO .pt weights")
    p.add_argument("--data",            required=True,
                   help="Dataset root with train/ val/ test/ subdirs")
    p.add_argument("--split",           default="test",
                   choices=["train", "val", "test"],
                   help="Split to evaluate")
    p.add_argument("--calibration-dir", default=None,
                   help="Directory containing temperature.pt + ood_model.pkl")
    p.add_argument("--ood-dir",         default=None,
                   help="Directory of OOD images for AUROC (flat, no subdirs needed)")
    p.add_argument("--output-dir",      default=None,
                   help="Where to save results (default: model_dir/results)")
    p.add_argument("--batch-size",      type=int, default=32)
    p.add_argument("--img-size",        type=int, default=640)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def resolve_output_dir(model_path: str, custom: str | None) -> Path:
    mp    = Path(model_path)
    parts = mp.parts
    try:
        idx      = list(parts).index("runs")
        base_dir = Path(*parts[:idx + 3])   # runs/classify/model_name
    except ValueError:
        base_dir = mp.parent
    out = Path(custom) if custom else base_dir / "results_calibrated"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_calibration(cal_dir: str | None):
    """Return (TemperatureScaler, MahalanobisOOD) or (None, None)."""
    if cal_dir is None:
        return None, None
    cal_dir = Path(cal_dir)
    tp = cal_dir / "temperature.pt"
    op = cal_dir / "ood_model.pkl"
    if not tp.exists() or not op.exists():
        print(f"[WARN] Calibration files not found in {cal_dir}. "
              f"Run calibrate_yolo_classifier.py first.")
        return None, None
    scaler = TemperatureScaler.load(tp)
    ood    = MahalanobisOOD.load(op)
    print(f"Loaded temperature scaler  T = {scaler.temperature.item():.4f}")
    print(f"Loaded OOD detector  threshold = {ood.threshold:.4f}")
    return scaler, ood


class FlatImageDataset(torch.utils.data.Dataset):
    """Load images from a flat directory (no class subdirs). Label = -1."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root, transform):
        self.paths = sorted(
            p for p in Path(root).iterdir()
            if p.suffix.lower() in self.EXTS
        )
        self.transform = transform

    def __len__(self):  return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), -1


# --------------------------------------------------------------------------- #
#  Core evaluation                                                             #
# --------------------------------------------------------------------------- #

def evaluate(yolo, loader, scaler, ood_detector, device, split_name):
    """
    Return a dict with logits, cal_probs, labels, embeddings, preds, etc.
    """
    logits, embeddings, labels = extract_features(yolo, loader, device)

    raw_probs = torch.softmax(logits, dim=-1)
    cal_probs = scaler.calibrate(logits) if scaler else raw_probs

    preds     = cal_probs.argmax(1).numpy()
    labels_np = labels.numpy()

    # OOD scores
    ood_scores = None
    if ood_detector is not None:
        print(f"  Computing OOD scores for {split_name} ...")
        ood_scores = ood_detector.score_batch(embeddings.numpy())

    return {
        "logits":     logits,
        "raw_probs":  raw_probs.numpy(),
        "cal_probs":  cal_probs.numpy(),
        "embeddings": embeddings.numpy(),
        "labels":     labels_np,
        "preds":      preds,
        "ood_scores": ood_scores,
    }


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_metrics(results, class_names):
    labels, preds = results["labels"], results["preds"]
    macro_acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    per_class = []
    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        acc_c = accuracy_score(labels[mask], preds[mask])
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            labels[mask], preds[mask],
            labels=[idx], average="macro", zero_division=0,
        )
        per_class.append({
            "Class":     name,
            "N":         int(mask.sum()),
            "Accuracy":  round(acc_c, 4),
            "Precision": round(p_c,   4),
            "Recall":    round(r_c,   4),
            "F1":        round(f1_c,  4),
            "Low":       acc_c < 0.9,
        })

    return {
        "accuracy":  macro_acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "cm":        cm,
        "per_class": pd.DataFrame(per_class),
    }


# --------------------------------------------------------------------------- #
#  Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_reliability(raw_probs, cal_probs, labels, preds_raw, preds_cal,
                     out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, probs, preds, title in [
        (axes[0], raw_probs, preds_raw, "Before calibration"),
        (axes[1], cal_probs, preds_cal, "After calibration"),
    ]:
        correct    = (preds == labels).astype(int)
        confidence = probs.max(axis=1)
        try:
            frac_pos, mean_pred = calibration_curve(
                correct, confidence, n_bins=10, strategy="quantile"
            )
        except ValueError:
            frac_pos, mean_pred = calibration_curve(
                correct, confidence, n_bins=5
            )
        ax.plot(mean_pred, frac_pos, "s-", label="Model", color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_title(title)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved reliability diagram → {out_path}")


def plot_confusion_matrix(cm, class_names, out_path):
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n // 4), max(8, n // 4)))
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fs  = max(4, 8 - n // 20)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=fs)
    ax.set_yticklabels(class_names, fontsize=fs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (calibrated predictions)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix   → {out_path}")


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #

def print_report(split, raw_m, cal_m, low_acc_classes, ood_auroc):
    print("\n" + "=" * 65)
    print(f"ASSESSMENT RESULTS — {split.upper()}")
    print("=" * 65)

    if cal_m is not None:
        print(f"  {'Metric':<20} {'Raw':>10} {'Calibrated':>12}")
        print(f"  {'-'*44}")
        for key in ("accuracy", "precision", "recall", "f1"):
            print(f"  {key.capitalize():<20} "
                  f"{raw_m[key]:>10.4f} {cal_m[key]:>12.4f}")
    else:
        print(f"  Accuracy  : {raw_m['accuracy']:.4f}")
        print(f"  Precision : {raw_m['precision']:.4f}")
        print(f"  Recall    : {raw_m['recall']:.4f}")
        print(f"  F1        : {raw_m['f1']:.4f}")

    if ood_auroc is not None:
        print(f"\n  OOD detection AUROC : {ood_auroc:.4f}")

    if low_acc_classes:
        print(f"\n  Classes with accuracy < 0.90 ({len(low_acc_classes)}):")
        for row in low_acc_classes:
            print(f"    {row['Class']:<25}  acc={row['Accuracy']:.3f}  "
                  f"n={row['N']}")
    else:
        print("\n  All classes exceed 0.90 accuracy.")
    print("=" * 65)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args    = parse_args()
    out_dir = resolve_output_dir(args.model, args.output_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model : {args.model}")
    yolo = YOLO(args.model)
    yolo.model.eval().to(device)

    # Calibration artefacts (optional)
    scaler, ood_detector = load_calibration(args.calibration_dir)
    calibrated = scaler is not None

    tf       = yolo_transform(args.img_size)
    eval_dir = Path(args.data) / args.split
    if not eval_dir.exists():
        print(f"Error: {eval_dir} does not exist."); sys.exit(1)

    eval_ds     = datasets.ImageFolder(str(eval_dir), transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    class_names = eval_ds.classes
    print(f"Evaluating on '{args.split}': {len(eval_ds)} images, "
          f"{len(class_names)} classes")

    # ---- evaluate ----
    print(f"\nRunning inference ...")
    res = evaluate(yolo, eval_loader, scaler, ood_detector, device, args.split)

    # Uncalibrated predictions (for comparison)
    raw_preds  = res["raw_probs"].argmax(axis=1)
    cal_preds  = res["preds"]

    raw_metrics = compute_metrics(
        {"labels": res["labels"], "preds": raw_preds}, class_names
    )
    cal_metrics = (compute_metrics(res, class_names) if calibrated else None)

    active_metrics = cal_metrics if calibrated else raw_metrics
    low_acc = active_metrics["per_class"][
        active_metrics["per_class"]["Low"]
    ].to_dict("records")

    # ---- OOD AUROC (optional) ----
    ood_auroc = None
    if args.ood_dir and ood_detector is not None:
        print(f"\nLoading OOD images from {args.ood_dir} ...")
        ood_ds     = FlatImageDataset(args.ood_dir, tf)
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        print(f"  {len(ood_ds)} OOD images found")
        _, ood_emb, _ = extract_features(yolo, ood_loader, device)
        ood_scores_novel = ood_detector.score_batch(ood_emb.numpy())
        in_scores        = res["ood_scores"]
        y_true  = np.array([0]*len(in_scores) + [1]*len(ood_scores_novel))
        y_score = np.concatenate([in_scores, ood_scores_novel])
        ood_auroc = float(roc_auc_score(y_true, y_score))
    elif args.ood_dir and ood_detector is None:
        print("[WARN] --ood-dir supplied but no OOD detector loaded "
              "(missing --calibration-dir). Skipping AUROC.")

    # ---- print report ----
    print_report(args.split, raw_metrics, cal_metrics, low_acc, ood_auroc)

    # ---- save CSVs ----
    active_metrics["per_class"].to_csv(
        out_dir / f"class_metrics_{args.split}.csv", index=False
    )
    overall = {
        "split":    args.split,
        "accuracy": active_metrics["accuracy"],
        "precision":active_metrics["precision"],
        "recall":   active_metrics["recall"],
        "f1":       active_metrics["f1"],
        "ood_auroc":ood_auroc,
    }
    pd.DataFrame([overall]).to_csv(
        out_dir / f"overall_metrics_{args.split}.csv", index=False
    )

    # Per-prediction CSV: one row per image with raw + calibrated confidence.
    # Used for downstream analysis (e.g. comparing confidence scores between
    # correct and incorrect classifications).
    per_pred = pd.DataFrame({
        "image_path":     [eval_ds.samples[i][0] for i in range(len(res["labels"]))],
        "true_class":     [class_names[l] for l in res["labels"]],
        "pred_class":     [class_names[p] for p in cal_preds],
        "correct":        res["labels"] == cal_preds,
        "raw_confidence": res["raw_probs"].max(axis=1).round(6),
        "cal_confidence": res["cal_probs"].max(axis=1).round(6),
    })
    if res["ood_scores"] is not None:
        per_pred["ood_score"] = res["ood_scores"].round(6)
    per_pred.to_csv(
        out_dir / f"per_prediction_{args.split}.csv", index=False
    )
    print(f"  Saved per-prediction CSV  → "
          f"{out_dir / f'per_prediction_{args.split}.csv'}")

    # ---- save plots ----
    if calibrated:
        plot_reliability(
            res["raw_probs"], res["cal_probs"],
            res["labels"], raw_preds, cal_preds,
            out_dir / f"reliability_{args.split}.png",
        )

    plot_confusion_matrix(
        active_metrics["cm"], class_names,
        out_dir / f"confusion_{args.split}.png",
    )

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
