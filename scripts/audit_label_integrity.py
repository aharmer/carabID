"""
audit_label_integrity.py

Cross-check a YOLO dataset's class labels against the genus in each filename.

Specimen filenames in this project carry the genus (Molopsida_seriatoporus_
NZAC04006632), so they are an independent record of what each image actually
is.  Comparing them against the class index catches label/class-list
misalignment, which is otherwise invisible: training metrics look excellent
because the labels are internally consistent, just attached to the wrong names.

Reports, per class index, the genus its images actually are, and flags any
index whose declared name disagrees with the majority of its filenames.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/audit_label_integrity.py \
        --dataset imgs/detection_set
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--strip-prefix", default=None,
                    help="regex prefix to strip from filenames first")
    ap.add_argument("--min-purity", type=float, default=0.9)
    args = ap.parse_args()

    root = ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    known = {n.lower() for n in names}

    per_idx = defaultdict(Counter)
    unchecked = 0
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        stem = img.stem
        if args.strip_prefix:
            stem = re.sub(args.strip_prefix, "", stem)
        genus = re.split(r"[-_ ]", stem)[0].lower()
        if genus not in known:
            unchecked += 1
            continue
        for line in lbl.read_text().splitlines():
            if line.strip():
                per_idx[int(line.split()[0])][genus] += 1
                break

    bad, ok, affected = [], 0, 0
    for idx in sorted(per_idx):
        counts = per_idx[idx]
        total = sum(counts.values())
        actual, n = counts.most_common(1)[0]
        declared = names[idx].lower()
        purity = n / total
        if declared == actual:
            ok += 1
        elif purity >= args.min_purity:
            bad.append((idx, declared, actual, total, purity))
            affected += total

    print(f"dataset      : {root}")
    print(f"checkable    : {sum(sum(c.values()) for c in per_idx.values())} images")
    print(f"unchecked    : {unchecked} (filename carries no recognised genus)")
    print(f"indices OK   : {ok}")
    print(f"indices WRONG: {len(bad)}   images affected: {affected}\n")

    if bad:
        print(f"{'idx':>4}  {'declared as':<18}{'actually is':<18}{'n':>5}{'purity':>8}")
        for idx, d, a, n, p in sorted(bad, key=lambda r: -r[3]):
            print(f"{idx:>4}  {d:<18}{a:<18}{n:>5}{p:>8.0%}")
        print("\nEvery image under a wrong index carries the wrong genus. If this "
              "dataset produced classification crops, that model learned those "
              "genera under the wrong names.")
    else:
        print("No misaligned class indices found.")


if __name__ == "__main__":
    main()
