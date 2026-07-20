"""
select_lab_candidates.py

Copy the top-ranked lab-like candidates into a clean folder ready for
annotation, carrying their licence attribution with them.

Reads lab_likeness.csv produced by screen_lab_likeness.py.  Rank rather than an
absolute score is the useful control: the combined score is not calibrated
across pools, so a threshold chosen on one download admits noticeably weaker
images when the pool grows.  Re-check the contact sheets after any big change
in pool size.

Run from the carabID root:
    conda run -n ultralytics-env python scripts/select_lab_candidates.py --top 230
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="imgs/inat_candidates")
    ap.add_argument("--out", default="imgs/inat_keepers")
    ap.add_argument("--top", type=int, default=230,
                    help="how many top-ranked images to keep")
    ap.add_argument("--per-genus-max", type=int, default=None,
                    help="optional cap per genus, to limit imbalance")
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    ranked = list(csv.DictReader(open(src / "lab_likeness.csv", encoding="utf-8")))
    att = {r["filename"]: r for r in
           csv.DictReader(open(src / "attribution.csv", encoding="utf-8"))}

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    kept, per_genus = [], Counter()
    for r in ranked:
        if len(kept) >= args.top:
            break
        g = r["genus"]
        if args.per_genus_max and per_genus[g] >= args.per_genus_max:
            continue
        p = ROOT / r["path"]
        if not p.exists():
            continue
        (out / g).mkdir(exist_ok=True)
        shutil.copy2(p, out / g / p.name)
        per_genus[g] += 1
        kept.append((p.name, g, r))

    with open(out / "attribution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "genus", "licence", "attribution",
                    "observation_url", "bg_plain", "det_conf", "ood"])
        for name, g, r in kept:
            a = att.get(name, {})
            w.writerow([name, g, a.get("licence", ""), a.get("attribution", ""),
                        a.get("observation_url", ""),
                        r["bg_plain"], r["det_conf"], r["ood"]])

    print(f"kept {len(kept)} images across {len(per_genus)} genera -> {out}")
    print(f"attribution: {out/'attribution.csv'}")
    print("\nper genus:")
    for g, n in per_genus.most_common():
        print(f"  {g:<20}{n:>4}")
    print("\nNOTE: still needs pronotum+elytra boxes drawn to the lab convention "
          "before any training use, and a visual pass to drop off-target photos "
          "(research-grade applies to the observation, not to every photo in it).")


if __name__ == "__main__":
    main()
