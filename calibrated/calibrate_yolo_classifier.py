"""
Step 14a — Calibrate a trained YOLO classification model.

Fits two post-training artefacts that require no retraining:

  temperature.pt   Scalar T that corrects overconfident softmax probabilities.
  ood_model.pkl    Mahalanobis-distance detector that flags genus predictions
                   when the input looks unlike anything seen during training.

Both files are saved alongside best.pt so the app can find them automatically.

Usage
-----
    python calibrated/calibrate_yolo_classifier.py \\
        --model  runs/classify/my_model/weights/best.pt \\
        --data   imgs/final_classification_set \\
        --imgsz  640

The dataset must have  train/  and  val/  subdirectories in standard
ImageFolder layout (one subfolder per class).

How it works
------------
YOLO's predict() applies softmax internally and never exposes raw logits.
This script bypasses the predictor wrapper and calls the underlying PyTorch
model directly (model.model(batch)), which returns pre-softmax logits.
A forward hook on the final Linear layer captures the D-dimensional embedding
(the globally-averaged feature vector) for OOD fitting.

Temperature scaling is fitted on the *validation* set logits.
The OOD detector is fitted on *training* set embeddings and its threshold is
tuned on *validation* set embeddings.  The test set is never touched.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from calibration_utils import TemperatureScaler, MahalanobisOOD


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit temperature scaler + Mahalanobis OOD for a YOLO classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  required=True,
                   help="Path to trained YOLO weights (.pt)")
    p.add_argument("--data",   required=True,
                   help="Dataset root containing train/ and val/ subdirs")
    p.add_argument("--imgsz",  type=int,   default=640,
                   help="Image size used during training")
    p.add_argument("--batch",  type=int,   default=32,
                   help="Batch size for feature extraction")
    p.add_argument("--fpr",    type=float, default=0.05,
                   help="Target false-novelty rate for OOD threshold tuning")
    p.add_argument("--out",    default=None,
                   help="Output directory (default: same folder as model weights)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  YOLO internals                                                              #
# --------------------------------------------------------------------------- #

def get_linear_layer(yolo_wrapper):
    """
    Locate the final Linear layer inside the YOLO Classify head.
    Works for all yolo11*-cls variants.
    """
    classify_head = yolo_wrapper.model.model[-1]
    if hasattr(classify_head, "linear"):
        return classify_head.linear
    # Fallback: find last Linear in the head
    last = None
    for m in classify_head.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise RuntimeError(
            "Cannot find a Linear layer in model.model[-1]. "
            "Is this a YOLO classification model?"
        )
    return last


def yolo_transform(imgsz: int) -> transforms.Compose:
    """
    Minimal preprocessing matching YOLO's classify predictor:
    resize to square, convert to float tensor in [0, 1].
    """
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),          # uint8 [0,255] → float [0,1]
    ])


# --------------------------------------------------------------------------- #
#  Feature extraction                                                          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_features(yolo_wrapper, loader, device):
    """
    Pass a DataLoader through the YOLO model and return:
        logits     (N, C)  raw pre-softmax scores
        embeddings (N, D)  L2-normalised GAP features (input to final linear)
        labels     (N,)    integer class indices
    """
    linear      = get_linear_layer(yolo_wrapper)
    emb_buffer  = []
    logit_buffer = []

    def _hook(m, inp, out):
        emb_buffer.append(inp[0].detach().cpu())   # embeddings (pre-linear)
        logit_buffer.append(out.detach().cpu())    # logits (post-linear, pre-softmax)

    hook = linear.register_forward_hook(_hook)

    all_labels = []
    yolo_wrapper.model.eval()

    for images, labels in loader:
        images = images.to(device)
        yolo_wrapper.model(images)   # forward pass; outputs captured via hook
        all_labels.append(labels)

    hook.remove()

    logits     = torch.cat(logit_buffer)         # (N, C)
    labels     = torch.cat(all_labels)           # (N,)
    embeddings = torch.cat(emb_buffer)           # (N, D)
    embeddings = F.normalize(embeddings, dim=-1)

    return logits, embeddings, labels


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args       = parse_args()
    model_path = Path(args.model)
    data_path  = Path(args.data)

    # ---- validate paths ----
    for p, label in [(model_path, "model"), (data_path, "data")]:
        if not p.exists():
            print(f"Error: {label} path not found: {p}"); sys.exit(1)

    train_dir = data_path / "train"
    val_dir   = data_path / "val"
    for d, label in [(train_dir, "train/"), (val_dir, "val/")]:
        if not d.exists():
            print(f"Error: {label} not found inside {data_path}"); sys.exit(1)

    out_dir = Path(args.out) if args.out else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("YOLO Classifier Calibration")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Device  : {device}")
    print(f"  Outputs : {out_dir}")
    print(f"  FPR tgt : {args.fpr}")
    print("=" * 60)

    # ---- load model ----
    yolo = YOLO(str(model_path))
    yolo.model.eval().to(device)

    tf          = yolo_transform(args.imgsz)
    train_ds    = datasets.ImageFolder(str(train_dir), transform=tf)
    val_ds      = datasets.ImageFolder(str(val_dir),   transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    print(f"\n  Classes : {len(train_ds.classes)}")
    print(f"  Train   : {len(train_ds)} images")
    print(f"  Val     : {len(val_ds)} images")

    # ---- extract features ----
    print("\nExtracting val logits + embeddings ...")
    val_logits, val_emb, val_labels = extract_features(yolo, val_loader, device)

    print("Extracting train embeddings ...")
    _, train_emb, train_labels = extract_features(yolo, train_loader, device)

    # ---- temperature scaling ----
    print("\nFitting temperature scaler ...")
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    T = scaler.temperature.item()
    direction = "overconfident" if T > 1 else "underconfident"
    print(f"  T = {T:.4f}  ({direction} — {'spreading' if T > 1 else 'sharpening'} probabilities)")

    temp_path = out_dir / "temperature.pt"
    torch.save(scaler.state_dict(), temp_path)
    print(f"  Saved → {temp_path}")

    # ---- Mahalanobis OOD ----
    print("\nFitting Mahalanobis OOD detector ...")
    ood = MahalanobisOOD()
    ood.fit(train_emb.numpy(), train_labels.numpy())
    threshold = ood.tune_threshold(val_emb.numpy(), fpr_target=args.fpr)
    print(f"  Threshold @ FPR {args.fpr*100:.0f}% : {threshold:.4f}")

    ood_path = out_dir / "ood_model.pkl"
    ood.save(ood_path)
    print(f"  Saved → {ood_path}")

    # ---- quick sanity check ----
    raw_probs  = torch.softmax(val_logits, dim=-1)
    cal_probs  = scaler.calibrate(val_logits)
    preds      = cal_probs.argmax(1)

    raw_conf   = raw_probs.max(1).values.mean().item()
    cal_conf   = cal_probs.max(1).values.mean().item()
    accuracy   = (preds == val_labels).float().mean().item()

    print("\n" + "=" * 60)
    print("Val-set sanity check")
    print("=" * 60)
    print(f"  Accuracy (unchanged) : {accuracy:.4f}")
    print(f"  Mean confidence      : {raw_conf:.4f} → {cal_conf:.4f}  (raw → calibrated)")
    print(f"  Temperature          : {T:.4f}")
    print(f"  OOD threshold        : {threshold:.4f}")
    print("=" * 60)
    print("\nCalibration complete.  Place temperature.pt and ood_model.pkl")
    print("in the same folder as your model weights (or app/static/).")


if __name__ == "__main__":
    main()
