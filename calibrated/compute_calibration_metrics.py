"""
calibrated/compute_calibration_metrics.py

Computes Expected Calibration Error (ECE) and reliability diagram data
for the YOLO classifier, before and after temperature scaling.

Outputs
-------
  calibration_summary.csv        T, accuracy, ECE before/after
  reliability_diagram_data.csv   per-bin data for ggplot2 in R

Usage
-----
    python calibrated/compute_calibration_metrics.py \
        --model           runs/classify/.../weights/best.pt \
        --data            imgs/final_classification_set \
        --calibration-dir runs/classify/.../weights \
        --imgsz           640 \
        --bins            15 \
        --out             calibrated/results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from calibration_utils import TemperatureScaler


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",           required=True,  help="Path to best.pt")
    p.add_argument("--data",            required=True,  help="Dataset root (train/ val/ inside)")
    p.add_argument("--calibration-dir", required=True,  help="Folder containing temperature.pt")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--bins",    type=int,   default=15,  help="Number of equal-width ECE bins")
    p.add_argument("--out",     default="calibrated/results",
                   help="Output directory for CSV files")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Model helpers  (same hook approach as calibrate_yolo_classifier.py)
# --------------------------------------------------------------------------- #

def get_linear_layer(yolo_wrapper):
    classify_head = yolo_wrapper.model.model[-1]
    if hasattr(classify_head, "linear"):
        return classify_head.linear
    last = None
    for m in classify_head.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("Cannot find a Linear layer in model.model[-1].")
    return last


def yolo_transform(imgsz):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def extract_logits(yolo_wrapper, loader, device):
    """Extract pre-softmax logits via forward hook on the final Linear layer."""
    linear = get_linear_layer(yolo_wrapper)
    logit_buffer = []

    def _hook(m, inp, out):
        logit_buffer.append(out.detach().cpu())

    hook = linear.register_forward_hook(_hook)
    all_labels = []
    yolo_wrapper.model.eval()

    for images, labels in loader:
        images = images.to(device)
        yolo_wrapper.model(images)
        all_labels.append(labels)

    hook.remove()
    return torch.cat(logit_buffer), torch.cat(all_labels)


# --------------------------------------------------------------------------- #
#  Calibration metrics
# --------------------------------------------------------------------------- #

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Standard equal-width Expected Calibration Error."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def reliability_rows(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int, label: str) -> list[dict]:
    """Return per-bin data for a reliability diagram."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    rows = []
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "condition":       label,
            "bin_lower":       round(float(edges[i]),     4),
            "bin_upper":       round(float(edges[i + 1]), 4),
            "bin_midpoint":    round(float((edges[i] + edges[i + 1]) / 2), 4),
            "mean_confidence": round(float(conf[mask].mean()),    4),
            "accuracy":        round(float(correct[mask].mean()), 4),
            "count":           count,
            "fraction":        round(float(count / n), 6),
        })
    return rows


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path  = Path(args.data)
    cal_dir    = Path(args.calibration_dir)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    # ---- load model ----
    yolo = YOLO(str(model_path))
    yolo.model.eval().to(device)

    # ---- load temperature scaler ----
    temp_path = cal_dir / "temperature.pt"
    if not temp_path.exists():
        raise FileNotFoundError(f"temperature.pt not found in {cal_dir}")
    scaler = TemperatureScaler.load(temp_path)
    T = scaler.temperature.item()

    # ---- validation set ----
    val_dir = data_path / "val"
    tf      = yolo_transform(args.imgsz)
    val_ds  = datasets.ImageFolder(str(val_dir), transform=tf)
    loader  = DataLoader(val_ds, batch_size=args.batch,
                         shuffle=False, num_workers=0)

    print(f"Val images : {len(val_ds)}")
    print("Extracting logits ...")
    logits, labels = extract_logits(yolo, loader, device)
    labels_np = labels.numpy()

    raw_probs = torch.softmax(logits, dim=-1).numpy()
    cal_probs = scaler.calibrate(logits).numpy()

    # ---- metrics ----
    accuracy  = float((raw_probs.argmax(axis=1) == labels_np).mean())
    ece_raw   = compute_ece(raw_probs, labels_np, args.bins)
    ece_cal   = compute_ece(cal_probs, labels_np, args.bins)
    reduction = (1.0 - ece_cal / ece_raw) * 100 if ece_raw > 0 else 0.0

    print("\n" + "=" * 55)
    print("  Calibration Metrics")
    print("=" * 55)
    print(f"  Temperature T            : {T:.4f}")
    print(f"  Accuracy                 : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  ECE — uncalibrated       : {ece_raw:.4f}  ({ece_raw*100:.2f}%)")
    print(f"  ECE — calibrated (T={T:.2f}): {ece_cal:.4f}  ({ece_cal*100:.2f}%)")
    print(f"  ECE reduction            : {reduction:.1f}%")
    print("=" * 55)

    # ---- export reliability diagram data ----
    rows = (reliability_rows(raw_probs, labels_np, args.bins, "Uncalibrated") +
            reliability_rows(cal_probs, labels_np, args.bins, "Calibrated"))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "reliability_diagram_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nReliability data → {csv_path}")
    print(df.to_string(index=False))

    # ---- export summary ----
    summary = pd.DataFrame([{
        "temperature_T":      round(T, 4),
        "n_val_images":       len(val_ds),
        "n_bins":             args.bins,
        "accuracy":           round(accuracy, 4),
        "ece_uncalibrated":   round(ece_raw, 4),
        "ece_calibrated":     round(ece_cal, 4),
        "ece_reduction_pct":  round(reduction, 1),
    }])
    summary_path = out_dir / "calibration_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary          → {summary_path}")


if __name__ == "__main__":
    main()
